# CUDA operation 경계와 FFN block 경계를 동일 조건에서 순차 비교합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

if __package__:
    from tools.benchmark_synthetic import benchmark_once, write_results
else:
    from benchmark_synthetic import benchmark_once, write_results


def ffn_boundary_matrix() -> tuple[dict[str, str], ...]:
    return (
        {"name": "operation-scalar", "cuda_boundary": "operation", "cuda_batching": "scalar"},
        {"name": "operation-grouped", "cuda_boundary": "operation", "cuda_batching": "grouped"},
        {"name": "ffn-block-scalar", "cuda_boundary": "ffn-block", "cuda_batching": "scalar"},
        {"name": "ffn-block-grouped", "cuda_boundary": "ffn-block", "cuda_batching": "grouped"},
    )


def run_ffn_ablation(
    artifact: Path,
    runner: Path,
    *,
    dense_precision: str,
    cuda_resident_bytes: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    if cuda_resident_bytes <= 0:
        raise ValueError("resident capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for configuration in ffn_boundary_matrix():
        record = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cuda-custom",
            dense_precision=dense_precision,
            cuda_allocation="reused",
            cuda_weights="resident",
            cuda_batching=configuration["cuda_batching"],
            cuda_boundary=configuration["cuda_boundary"],
            cuda_resident_bytes=cuda_resident_bytes,
        )
        write_results(
            record,
            output_dir / f"{configuration['name']}.json",
            output_dir / f"{configuration['name']}.csv",
        )
        records.append({"name": configuration["name"], **asdict(record)})

    provenance = (
        "scope", "evidence", "platform", "iterations", "prompt_tokens",
        "generated_tokens", "backend", "device", "dense_precision",
        "cuda_allocation", "cuda_weights", "cuda_resident_bytes",
    )
    baseline = records[0]
    if any(
        any(record[field] != baseline[field] for field in provenance)
        for record in records[1:]
    ):
        raise RuntimeError("FFN ablation provenance changed across cases")
    baseline_tokens = tuple(baseline["token_ids"])
    for record in records:
        record["parity_status"] = (
            "exact" if tuple(record["token_ids"]) == baseline_tokens else "divergent"
        )
    if any(record["parity_status"] != "exact" for record in records):
        raise RuntimeError("FFN ablation token parity failed")

    for record in records[:2]:
        if int(record["ffn_block_calls"]) != 0 or int(record["ffn_block_experts"]) != 0:
            raise RuntimeError("operation boundary reported FFN block counters")
    for record in records[2:]:
        if int(record["ffn_block_calls"]) <= 0 or int(record["ffn_block_experts"]) <= 0:
            raise RuntimeError("FFN block boundary did not report block execution")
    for operation, block in ((records[0], records[2]), (records[1], records[3])):
        if int(block["device_to_host_bytes"]) >= int(operation["device_to_host_bytes"]):
            raise RuntimeError("FFN block did not reduce device-to-host traffic")
        if int(block["stream_synchronization_count"]) >= int(operation["stream_synchronization_count"]):
            raise RuntimeError("FFN block did not reduce stream synchronizations")

    summary: dict[str, object] = {"records": records}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--dense-precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--cuda-resident-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_ffn_ablation(
        args.artifact,
        args.runner,
        dense_precision=args.dense_precision,
        cuda_resident_bytes=args.cuda_resident_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
