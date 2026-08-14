# CUDA routed accumulation fusion을 transfer mode별로 순차 비교합니다.
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

if __package__:
    from tools.benchmark_synthetic import benchmark_once, write_results
else:
    from benchmark_synthetic import benchmark_once, write_results


def fusion_matrix(pinned_bytes: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "synchronous-none",
            "cuda_transfer": "synchronous",
            "cuda_moe_fusion": "none",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "synchronous-routed-accumulate",
            "cuda_transfer": "synchronous",
            "cuda_moe_fusion": "routed-accumulate",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "prefetch-none",
            "cuda_transfer": "prefetch",
            "cuda_moe_fusion": "none",
            "cuda_pinned_bytes": pinned_bytes,
        },
        {
            "name": "prefetch-routed-accumulate",
            "cuda_transfer": "prefetch",
            "cuda_moe_fusion": "routed-accumulate",
            "cuda_pinned_bytes": pinned_bytes,
        },
    )


def run_fusion_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    pinned_bytes: int,
    output_dir: Path,
) -> dict[str, object]:
    if pinned_bytes <= 0:
        raise ValueError("pinned capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for configuration in fusion_matrix(pinned_bytes):
        record = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cuda-custom",
            dense_precision="fp32",
            cuda_allocation="reused",
            cuda_weights="transient",
            cuda_batching="scalar",
            cuda_boundary="ffn-block",
            cuda_transfer=str(configuration["cuda_transfer"]),
            cuda_moe_fusion=str(configuration["cuda_moe_fusion"]),
            cuda_pinned_bytes=int(configuration["cuda_pinned_bytes"]),
        )
        write_results(
            record,
            output_dir / f"{configuration['name']}.json",
            output_dir / f"{configuration['name']}.csv",
        )
        records.append({"name": configuration["name"], **asdict(record)})

    provenance = (
        "scope",
        "evidence",
        "platform",
        "iterations",
        "prompt_tokens",
        "generated_tokens",
        "backend",
        "device",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_boundary",
    )
    baseline = records[0]
    if any(
        any(record[field] != baseline[field] for field in provenance)
        for record in records[1:]
    ):
        raise RuntimeError("fusion ablation provenance changed across cases")

    baseline_tokens = tuple(baseline["token_ids"])
    baseline_routing = tuple(baseline["routed_experts"])
    baseline_routed_k = tuple(baseline["routed_k"])
    for record in records:
        exact = (
            tuple(record["token_ids"]) == baseline_tokens
            and tuple(record["routed_experts"]) == baseline_routing
            and tuple(record["routed_k"]) == baseline_routed_k
        )
        record["parity_status"] = "exact" if exact else "divergent"
        maximum_error = record["max_absolute_error"]
        if (
            not exact
            or maximum_error is None
            or not math.isfinite(float(maximum_error))
            or float(maximum_error) > 1.0e-4
        ):
            raise RuntimeError("fusion ablation correctness parity failed")

    deltas: list[dict[str, object]] = []
    for unfused, fused in ((records[0], records[1]), (records[2], records[3])):
        if int(unfused["fused_moe_calls"]) != 0:
            raise RuntimeError("unfused case reported fused execution")
        if (
            int(fused["fused_moe_calls"]) <= 0
            or int(fused["fused_moe_experts"]) <= 0
        ):
            raise RuntimeError("fused case did not report fused execution")
        reduction = int(unfused["device_to_host_bytes"]) - int(
            fused["device_to_host_bytes"]
        )
        if reduction <= 0:
            raise RuntimeError("routed accumulation did not reduce D2H traffic")
        deltas.append(
            {
                "cuda_transfer": fused["cuda_transfer"],
                "d2h_reduction_bytes": reduction,
                "decode_tokens_per_second_delta": float(
                    fused["decode_tokens_per_second"]
                )
                - float(unfused["decode_tokens_per_second"]),
                "kernel_nanoseconds_delta": int(fused["kernel_nanoseconds"])
                - int(unfused["kernel_nanoseconds"]),
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pinned-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_fusion_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        pinned_bytes=args.pinned_bytes,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
