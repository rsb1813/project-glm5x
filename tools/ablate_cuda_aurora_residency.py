# B-0020 exact bounded CUDA AURORA draft residency의 성능과 트래픽을 측정합니다.
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


RESIDENT_BYTES = 8 * 1024 * 1024

CASES = (
    ("natural-greedy", "none", "token-major", 0, "none", 0),
    (
        "transient-fixed-2-token", "aurora-persistent", "token-major", 2,
        "fixed", 0,
    ),
    (
        "resident-fixed-2-token", "aurora-persistent", "token-major", 2,
        "fixed", RESIDENT_BYTES,
    ),
    (
        "transient-adaptive-token", "aurora-persistent", "token-major", 4,
        "adaptive", 0,
    ),
    (
        "resident-adaptive-token", "aurora-persistent", "token-major", 4,
        "adaptive", RESIDENT_BYTES,
    ),
    (
        "transient-fixed-2-expert", "aurora-persistent", "expert-major", 2,
        "fixed", 0,
    ),
    (
        "resident-fixed-2-expert", "aurora-persistent", "expert-major", 2,
        "fixed", RESIDENT_BYTES,
    ),
    (
        "transient-adaptive-expert", "aurora-persistent", "expert-major", 4,
        "adaptive", 0,
    ),
    (
        "resident-adaptive-expert", "aurora-persistent", "expert-major", 4,
        "adaptive", RESIDENT_BYTES,
    ),
)

PAIRS = (
    (
        "fixed-2-token",
        "transient-fixed-2-token",
        "resident-fixed-2-token",
    ),
    (
        "adaptive-token",
        "transient-adaptive-token",
        "resident-adaptive-token",
    ),
    (
        "fixed-2-expert",
        "transient-fixed-2-expert",
        "resident-fixed-2-expert",
    ),
    (
        "adaptive-expert",
        "transient-adaptive-expert",
        "resident-adaptive-expert",
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    case: tuple[str, str, str, int, str, int],
) -> dict:
    name, mode, verification, block_size, policy, resident_bytes = case
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
            "--aurora-draft-backend", "cuda-custom",
        ])
        if resident_bytes > 0:
            command.extend([
                "--aurora-draft-resident-bytes", str(resident_bytes),
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


def _validate_cuda_draft(record: dict, resident_bytes: int) -> None:
    if (
        record["draft_kernel_nanoseconds"] <= 0
        or record["draft_host_to_device_bytes"] <= 0
        or record["draft_peak_vram_bytes"] <= 0
        or record["kernel_nanoseconds"] != 0
        or record["host_to_device_bytes"] != 0
        or record["peak_vram_bytes"] != 0
        or record["resident_weight_bytes"] != 0
        or record["peak_resident_weight_bytes"] != 0
    ):
        raise RuntimeError(f"{record['name']} lacks isolated CUDA draft evidence")
    if resident_bytes == 0:
        if (
            record["draft_cuda_weights"] != "transient"
            or record["draft_cuda_resident_bytes"] != 0
            or record["draft_weight_cache_hits"] != 0
            or record["draft_weight_cache_misses"] != 0
            or record["draft_weight_cache_bypasses"] != 0
            or record["draft_resident_weight_bytes"] != 0
            or record["draft_peak_resident_weight_bytes"] != 0
        ):
            raise RuntimeError(f"{record['name']} violates transient identity")
        return
    if (
        record["draft_cuda_weights"] != "resident"
        or record["draft_cuda_resident_bytes"] != RESIDENT_BYTES
        or record["draft_weight_cache_hits"] <= 0
        or record["draft_weight_cache_misses"] <= 0
        or record["draft_weight_cache_bypasses"] != 0
        or not 0 < record["draft_resident_weight_bytes"] <= RESIDENT_BYTES
        or not (
            record["draft_resident_weight_bytes"]
            <= record["draft_peak_resident_weight_bytes"]
            <= RESIDENT_BYTES
        )
    ):
        raise RuntimeError(f"{record['name']} violates resident capacity")


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

    with tempfile.TemporaryDirectory(prefix="k3x-b0020-") as temporary:
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
            for pair, transient_name, resident_name in PAIRS
            for name in (transient_name, resident_name)
        }
        records: list[dict] = []
        for case in CASES:
            name, mode, verification, block_size, policy, resident_bytes = case
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
                aurora_draft_backend=(
                    "cuda-custom" if mode != "none" else "cpu"
                ),
                aurora_draft_resident_bytes=resident_bytes,
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
            if mode != "none":
                _validate_cuda_draft(payload, resident_bytes)
            records.append(payload)

    by_name = {record["name"]: record for record in records}
    for field in (
        "paired_decode_delta_percent",
        "draft_weight_h2d_reduction_percent",
        "draft_resident_hit_rate",
    ):
        records[0][field] = None
    for pair_name, transient_name, resident_name in PAIRS:
        transient = by_name[transient_name]
        resident = by_name[resident_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            if transient[field] != resident[field]:
                raise RuntimeError(f"{pair_name} {field} diverged")
        if resident["draft_weight_h2d_bytes"] >= transient[
            "draft_weight_h2d_bytes"
        ]:
            raise RuntimeError(f"{pair_name} did not reduce draft weight H2D")
        transient["paired_decode_delta_percent"] = 0.0
        transient["draft_weight_h2d_reduction_percent"] = 0.0
        transient["draft_resident_hit_rate"] = 0.0
        resident["paired_decode_delta_percent"] = (
            resident["decode_tokens_per_second"]
            / transient["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        resident["draft_weight_h2d_reduction_percent"] = (
            1.0
            - resident["draft_weight_h2d_bytes"]
            / transient["draft_weight_h2d_bytes"]
        ) * 100.0
        resident["draft_resident_hit_rate"] = (
            resident["draft_weight_cache_hits"]
            / (
                resident["draft_weight_cache_hits"]
                + resident["draft_weight_cache_misses"]
            )
        )

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
        "benchmark": "B-0020",
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
