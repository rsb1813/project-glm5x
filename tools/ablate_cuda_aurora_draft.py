# B-0019 exact CPU와 transient CUDA AURORA draft 실행의 배치 성능을 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    for import_root in (
        repository,
        repository / "converter",
        repository / "reference",
    ):
        sys.path.insert(0, str(import_root))

from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint
from tools.benchmark_synthetic import benchmark_once, write_results


CASES = (
    ("natural-greedy", "none", "token-major", 0, "none", "cpu"),
    (
        "cpu-fixed-2-token", "aurora-persistent", "token-major", 2,
        "fixed", "cpu",
    ),
    (
        "cuda-fixed-2-token", "aurora-persistent", "token-major", 2,
        "fixed", "cuda-custom",
    ),
    (
        "cpu-adaptive-token", "aurora-persistent", "token-major", 4,
        "adaptive", "cpu",
    ),
    (
        "cuda-adaptive-token", "aurora-persistent", "token-major", 4,
        "adaptive", "cuda-custom",
    ),
    (
        "cpu-fixed-2-expert", "aurora-persistent", "expert-major", 2,
        "fixed", "cpu",
    ),
    (
        "cuda-fixed-2-expert", "aurora-persistent", "expert-major", 2,
        "fixed", "cuda-custom",
    ),
    (
        "cpu-adaptive-expert", "aurora-persistent", "expert-major", 4,
        "adaptive", "cpu",
    ),
    (
        "cuda-adaptive-expert", "aurora-persistent", "expert-major", 4,
        "adaptive", "cuda-custom",
    ),
)

PAIRS = (
    ("fixed-2-token", "cpu-fixed-2-token", "cuda-fixed-2-token"),
    ("adaptive-token", "cpu-adaptive-token", "cuda-adaptive-token"),
    ("fixed-2-expert", "cpu-fixed-2-expert", "cuda-fixed-2-expert"),
    ("adaptive-expert", "cpu-adaptive-expert", "cuda-adaptive-expert"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    case: tuple[str, str, str, int, str, str],
) -> dict:
    name, mode, verification, block_size, policy, draft_backend = case
    command = [
        str(runner),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--backend", "cpu",
        "--diagnostics", "true",
        "--speculative-mode", mode,
        "--speculative-verification", verification,
        "--speculative-block-size", str(block_size),
    ]
    if mode == "aurora-persistent":
        command.extend([
            "--aurora-draft-k", "4",
            "--aurora-block-policy", policy,
            "--aurora-draft-backend", draft_backend,
        ])
    command.extend(["--json", str(output)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{name} diagnostic failed")
    return json.loads(output.read_text(encoding="utf-8"))


def _maximum_absolute_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("AURORA final state shape diverged")
    return max(
        (abs(actual - expected) for actual, expected in zip(left, right)),
        default=0.0,
    )


def run_ablation(
    runner: Path,
    *,
    output_dir: Path,
    warmups: int,
    samples: int,
) -> dict:
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be non-negative and samples must be positive")
    runner = Path(runner).resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="k3x-b0019-") as temporary:
        root = Path(temporary)
        config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
        source = root / "source-top16"
        write_source_checkpoint(source, config=config)
        artifact = root / "top16.k3x"
        convert(source, artifact, chunk_bytes=257)
        artifact_sha256 = _sha256(artifact)
        baseline = _diagnostic(
            artifact, runner, root / "natural-diagnostic.json", CASES[0]
        )
        pair_names = {
            name: pair
            for pair, cpu_name, cuda_name in PAIRS
            for name in (cpu_name, cuda_name)
        }
        records: list[dict] = []
        for case in CASES:
            name, mode, verification, block_size, policy, draft_backend = case
            benchmark = benchmark_once(
                artifact,
                runner,
                warmups,
                samples,
                backend="cpu",
                dense_precision="fp32",
                speculative_mode=mode,
                speculative_verification=verification,
                speculative_block_size=block_size,
                aurora_draft_k=4 if mode != "none" else 0,
                aurora_block_policy=policy if mode != "none" else "fixed",
                aurora_draft_backend=draft_backend,
            )
            raw_json = output_dir / f"{name}.json"
            raw_csv = output_dir / f"{name}.csv"
            write_results(benchmark, raw_json, raw_csv)
            diagnostic = baseline if mode == "none" else _diagnostic(
                artifact, runner, root / f"{name}-diagnostic.json", case
            )
            token_parity = diagnostic["token_ids"] == baseline["token_ids"]
            route_parity = (
                diagnostic["routed_experts"] == baseline["routed_experts"]
                and diagnostic["routed_k"] == baseline["routed_k"]
            )
            state_error = _maximum_absolute_error(
                diagnostic["final_state"], baseline["final_state"]
            )
            if not token_parity or not route_parity or state_error > 1.0e-6:
                raise RuntimeError(f"{name} diverged from natural target execution")
            if tuple(benchmark.token_ids) != tuple(baseline["token_ids"]):
                raise RuntimeError(f"{name} benchmark tokens diverged")
            if mode != "none" and (
                benchmark.draft_reader_completed_bytes <= 0
                or benchmark.draft_routing_decisions <= 0
                or benchmark.draft_context_prefill_tokens != 5
            ):
                raise RuntimeError(f"{name} lacks separated draft evidence")
            if draft_backend == "cpu" and mode != "none" and (
                benchmark.draft_kernel_nanoseconds != 0
                or benchmark.draft_host_to_device_bytes != 0
                or benchmark.draft_peak_vram_bytes != 0
            ):
                raise RuntimeError(f"{name} has unexpected CUDA draft evidence")
            if draft_backend == "cuda-custom" and (
                benchmark.draft_kernel_nanoseconds <= 0
                or benchmark.draft_host_to_device_bytes <= 0
                or benchmark.draft_peak_vram_bytes <= 0
                or benchmark.kernel_nanoseconds != 0
                or benchmark.host_to_device_bytes != 0
                or benchmark.peak_vram_bytes != 0
            ):
                raise RuntimeError(f"{name} lacks isolated CUDA draft evidence")

            payload = asdict(benchmark)
            payload.update(
                name=name,
                pair_name=pair_names.get(name),
                token_parity=token_parity,
                final_state_max_abs_error=state_error,
                committed_route_parity=route_parity,
                raw_json_sha256=_sha256(raw_json),
                raw_csv_sha256=_sha256(raw_csv),
            )
            records.append(payload)

    by_name = {record["name"]: record for record in records}
    for field in (
        "paired_decode_delta_percent",
        "paired_draft_kernel_nanoseconds",
        "paired_draft_h2d_bytes",
        "paired_peak_draft_vram_bytes",
    ):
        records[0][field] = None
    for pair_name, cpu_name, cuda_name in PAIRS:
        cpu = by_name[cpu_name]
        cuda = by_name[cuda_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            if cpu[field] != cuda[field]:
                raise RuntimeError(f"{pair_name} {field} diverged")
        cpu["paired_decode_delta_percent"] = 0.0
        cpu["paired_draft_kernel_nanoseconds"] = 0
        cpu["paired_draft_h2d_bytes"] = 0
        cpu["paired_peak_draft_vram_bytes"] = 0
        cuda["paired_decode_delta_percent"] = (
            cuda["decode_tokens_per_second"] / cpu["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        cuda["paired_draft_kernel_nanoseconds"] = cuda[
            "draft_kernel_nanoseconds"
        ]
        cuda["paired_draft_h2d_bytes"] = cuda["draft_host_to_device_bytes"]
        cuda["paired_peak_draft_vram_bytes"] = cuda["draft_peak_vram_bytes"]

    baseline_decode = records[0]["decode_tokens_per_second"]
    for record in records:
        record["decode_delta_percent"] = (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0019",
        "warmups": warmups,
        "samples": samples,
        "artifact_sha256": artifact_sha256,
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    summary["summary_csv_sha256"] = _sha256(summary_csv)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    summary = run_ablation(
        args.runner,
        output_dir=args.output,
        warmups=args.warmups,
        samples=args.samples,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
