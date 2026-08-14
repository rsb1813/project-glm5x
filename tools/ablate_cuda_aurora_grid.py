# B-0021 resident expert-grid와 grouped AURORA draft를 동일 조건에서 측정합니다.
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
    for import_root in (repository, repository / "converter", repository / "reference"):
        sys.path.insert(0, str(import_root))

from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint
from tools.benchmark_synthetic import benchmark_once, write_results


RESIDENT_BYTES = 8 * 1024 * 1024
CASES = (
    ("natural-greedy", "none", "token-major", 0, "none", "scalar"),
    ("grouped-fixed-2-token", "aurora-persistent", "token-major", 2, "fixed", "grouped"),
    ("grid-fixed-2-token", "aurora-persistent", "token-major", 2, "fixed", "resident-grid"),
    ("grouped-adaptive-token", "aurora-persistent", "token-major", 4, "adaptive", "grouped"),
    ("grid-adaptive-token", "aurora-persistent", "token-major", 4, "adaptive", "resident-grid"),
    ("grouped-fixed-2-expert", "aurora-persistent", "expert-major", 2, "fixed", "grouped"),
    ("grid-fixed-2-expert", "aurora-persistent", "expert-major", 2, "fixed", "resident-grid"),
    ("grouped-adaptive-expert", "aurora-persistent", "expert-major", 4, "adaptive", "grouped"),
    ("grid-adaptive-expert", "aurora-persistent", "expert-major", 4, "adaptive", "resident-grid"),
)
PAIRS = (
    ("fixed-2-token", "grouped-fixed-2-token", "grid-fixed-2-token"),
    ("adaptive-token", "grouped-adaptive-token", "grid-adaptive-token"),
    ("fixed-2-expert", "grouped-fixed-2-expert", "grid-fixed-2-expert"),
    ("adaptive-expert", "grouped-adaptive-expert", "grid-adaptive-expert"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(artifact: Path, runner: Path, output: Path, case: tuple) -> dict:
    name, mode, verification, block_size, policy, batching = case
    command = [
        str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
        "--generate", "6", "--mode", "incremental", "--backend", "cpu",
        "--diagnostics", "true", "--speculative-mode", mode,
        "--speculative-verification", verification,
        "--speculative-block-size", str(block_size),
    ]
    if mode == "aurora-persistent":
        command.extend([
            "--aurora-draft-k", "4", "--aurora-block-policy", policy,
            "--aurora-draft-backend", "cuda-custom",
            "--aurora-draft-resident-bytes", str(RESIDENT_BYTES),
            "--aurora-draft-batching", batching,
        ])
    command.extend(["--json", str(output)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"{name} diagnostic failed")
    return json.loads(output.read_text(encoding="utf-8"))


def _maximum_absolute_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("AURORA final state shape diverged")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def _validate_cuda_record(record: dict, batching: str) -> None:
    if (
        record["draft_cuda_weights"] != "resident"
        or record["draft_cuda_resident_bytes"] != RESIDENT_BYTES
        or record["draft_cuda_batching"] != batching
        or record["draft_weight_cache_hits"] <= 0
        or record["draft_weight_cache_misses"] <= 0
        or record["draft_weight_cache_bypasses"] != 0
        or record["draft_kernel_nanoseconds"] <= 0
        or record["draft_peak_vram_bytes"] <= 0
        or record["kernel_nanoseconds"] != 0
        or record["host_to_device_bytes"] != 0
    ):
        raise RuntimeError(f"{record['name']} violates isolated CUDA identity")
    if batching == "resident-grid":
        if (
            record["draft_resident_grid_calls"] <= 0
            or record["draft_resident_grid_kernel_launches"]
            != record["draft_resident_grid_calls"] * 4
            or record["draft_resident_grid_fallbacks"] != 0
            or record["draft_resident_grid_descriptor_h2d_bytes"] <= 0
        ):
            raise RuntimeError(f"{record['name']} violates grid launch identity")
    elif record["draft_resident_grid_calls"] != 0:
        raise RuntimeError(f"{record['name']} leaked grid execution")


def run_ablation(
    runner: Path, *, output_dir: Path, warmups: int, samples: int
) -> dict:
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be non-negative and samples must be positive")
    runner = Path(runner).resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="k3x-b0021-") as temporary:
        root = Path(temporary)
        source = root / "source-top16"
        write_source_checkpoint(
            source,
            config=SyntheticK3Config.default().replace(num_experts=24, top_k=16),
        )
        artifact = root / "top16.k3x"
        convert(source, artifact, chunk_bytes=257)
        artifact_sha256 = _sha256(artifact)
        baseline = _diagnostic(artifact, runner, root / "natural.json", CASES[0])
        pair_names = {
            name: pair
            for pair, grouped, grid in PAIRS
            for name in (grouped, grid)
        }
        records: list[dict] = []
        for case in CASES:
            name, mode, verification, block_size, policy, batching = case
            benchmark = benchmark_once(
                artifact, runner, warmups, samples, backend="cpu",
                dense_precision="fp32", speculative_mode=mode,
                speculative_verification=verification,
                speculative_block_size=block_size,
                aurora_draft_k=4 if mode != "none" else 0,
                aurora_block_policy=policy if mode != "none" else "fixed",
                aurora_draft_backend="cuda-custom" if mode != "none" else "cpu",
                aurora_draft_resident_bytes=RESIDENT_BYTES if mode != "none" else 0,
                aurora_draft_batching=batching,
            )
            raw_json = output_dir / f"{name}.json"
            raw_csv = output_dir / f"{name}.csv"
            write_results(benchmark, raw_json, raw_csv)
            diagnostic = baseline if mode == "none" else _diagnostic(
                artifact, runner, root / f"{name}.json", case
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
                _validate_cuda_record(payload, batching)
            records.append(payload)

    by_name = {record["name"]: record for record in records}
    for record in records:
        record["paired_decode_delta_percent"] = None
        record["paired_moe_launch_reduction_percent"] = None
        record["draft_moe_kernel_launches"] = 0
    for pair_name, grouped_name, grid_name in PAIRS:
        grouped = by_name[grouped_name]
        grid = by_name[grid_name]
        for field in (
            "token_ids", "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens", "speculative_committed_tokens",
            "speculative_acceptance_rate", "draft_reader_completed_bytes",
            "draft_routing_decisions", "draft_routing_selected_experts",
        ):
            if grouped[field] != grid[field]:
                raise RuntimeError(f"{pair_name} {field} diverged")
        grouped_launches = grouped["draft_ffn_block_experts"] * 4
        grid_launches = grid["draft_resident_grid_kernel_launches"]
        if grouped_launches <= 0 or grid_launches >= grouped_launches:
            raise RuntimeError(f"{pair_name} did not reduce MoE launches")
        grouped["draft_moe_kernel_launches"] = grouped_launches
        grid["draft_moe_kernel_launches"] = grid_launches
        grouped["paired_decode_delta_percent"] = 0.0
        grouped["paired_moe_launch_reduction_percent"] = 0.0
        grid["paired_decode_delta_percent"] = (
            grid["decode_tokens_per_second"]
            / grouped["decode_tokens_per_second"] - 1.0
        ) * 100.0
        grid["paired_moe_launch_reduction_percent"] = (
            1.0 - grid_launches / grouped_launches
        ) * 100.0
    baseline_decode = records[0]["decode_tokens_per_second"]
    for record in records:
        record["decode_delta_percent"] = (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0021",
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
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    summary["summary_csv_sha256"] = _sha256(summary_csv)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    summary = run_ablation(
        args.runner, output_dir=args.output_dir,
        warmups=args.warmup, samples=args.iterations,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
