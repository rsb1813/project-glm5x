# CUDA allocation, residency, batching 단계를 같은 조건으로 순차 측정합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

if __package__:
    from tools.benchmark_synthetic import benchmark_once, write_results
else:
    from benchmark_synthetic import benchmark_once, write_results


_DELTA_FIELDS = (
    "device_allocation_count",
    "device_free_count",
    "stream_synchronization_count",
    "host_to_device_bytes",
    "weight_h2d_bytes",
    "activation_h2d_bytes",
    "device_to_host_bytes",
    "weight_cache_hits",
    "weight_cache_misses",
    "weight_cache_bypasses",
    "resident_weight_bytes",
    "peak_resident_weight_bytes",
    "scratch_bytes",
    "peak_scratch_bytes",
    "grouped_projection_calls",
    "grouped_projection_members",
)


def cuda_residency_matrix() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "reference",
            "cuda_allocation": "per-operation",
            "cuda_weights": "transient",
            "cuda_batching": "scalar",
        },
        {
            "name": "reuse",
            "cuda_allocation": "reused",
            "cuda_weights": "transient",
            "cuda_batching": "scalar",
        },
        {
            "name": "residency",
            "cuda_allocation": "reused",
            "cuda_weights": "resident",
            "cuda_batching": "scalar",
        },
        {
            "name": "grouped",
            "cuda_allocation": "reused",
            "cuda_weights": "resident",
            "cuda_batching": "grouped",
        },
    )


def run_ablation(
    artifact: Path,
    runner: Path,
    *,
    backend: str,
    dense_precision: str,
    cuda_resident_bytes: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    if backend not in {"cuda-dense", "cuda-custom"}:
        raise ValueError("ablation backend must be cuda-dense or cuda-custom")
    if cuda_resident_bytes <= 0:
        raise ValueError("resident capacity must be positive")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for configuration in cuda_residency_matrix():
        name = str(configuration["name"])
        resident = configuration["cuda_weights"] == "resident"
        record = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend=backend,
            dense_precision=dense_precision,
            cuda_allocation=str(configuration["cuda_allocation"]),
            cuda_weights=str(configuration["cuda_weights"]),
            cuda_batching=str(configuration["cuda_batching"]),
            cuda_resident_bytes=cuda_resident_bytes if resident else 0,
        )
        write_results(
            record,
            output_dir / f"{name}.json",
            output_dir / f"{name}.csv",
        )
        records.append({"name": name, **asdict(record)})

    deltas: list[dict[str, object]] = []
    for previous, current in zip(records, records[1:]):
        deltas.append(
            {
                "from": previous["name"],
                "to": current["name"],
                **{
                    field: int(current[field]) - int(previous[field])
                    for field in _DELTA_FIELDS
                },
            }
        )
    summary: dict[str, object] = {"records": records, "deltas": deltas}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("cuda-dense", "cuda-custom"), required=True
    )
    parser.add_argument(
        "--dense-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument("--cuda-resident-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_ablation(
        args.artifact,
        args.runner,
        backend=args.backend,
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
