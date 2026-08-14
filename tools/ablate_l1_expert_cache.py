# persistent L1 expert cache와 L1-to-L0 전송을 교차 측정합니다.
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
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "ttft_ms",
    "peak_rss_bytes",
    "file_read_bytes_per_token",
    "reader_read_calls",
    "reader_requested_bytes",
    "reader_completed_bytes",
    "l1_expert_cache_hits",
    "l1_expert_cache_misses",
    "l1_expert_cache_resident_bytes",
)


def l1_cache_matrix(
    l1_expert_cache_bytes: int, cuda_pinned_bytes: int
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": f"{cache}-{transfer}",
            "l1_expert_cache": cache,
            "l1_expert_cache_bytes": (
                l1_expert_cache_bytes if cache == "static" else 0
            ),
            "cuda_transfer": transfer,
            "cuda_pinned_bytes": (
                cuda_pinned_bytes if transfer == "prefetch" else 0
            ),
        }
        for transfer in ("synchronous", "prefetch")
        for cache in ("disabled", "static")
    )


def run_l1_cache_ablation(
    artifact: Path,
    runner: Path,
    *,
    dense_precision: str,
    l1_expert_cache_bytes: int,
    cuda_pinned_bytes: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    if l1_expert_cache_bytes <= 0:
        raise ValueError("L1 expert cache capacity must be positive")
    if cuda_pinned_bytes <= 0:
        raise ValueError("pinned capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = l1_cache_matrix(l1_expert_cache_bytes, cuda_pinned_bytes)
    records: list[dict[str, object]] = []
    for configuration in matrix:
        name = str(configuration["name"])
        record = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cuda-custom",
            dense_precision=dense_precision,
            cuda_allocation="reused",
            cuda_weights="transient",
            cuda_batching="scalar",
            cuda_boundary="ffn-block",
            cuda_transfer=str(configuration["cuda_transfer"]),
            cuda_resident_bytes=0,
            cuda_pinned_bytes=int(configuration["cuda_pinned_bytes"]),
            l1_expert_cache=str(configuration["l1_expert_cache"]),
            l1_expert_cache_bytes=int(configuration["l1_expert_cache_bytes"]),
        )
        json_path = output_dir / f"{name}.json"
        csv_path = output_dir / f"{name}.csv"
        write_results(record, json_path, csv_path)
        if not json_path.is_file() or not csv_path.is_file():
            raise RuntimeError("L1 cache ablation raw files are missing")
        records.append({"name": name, **asdict(record)})

    fixed_fields = (
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
        "cuda_resident_bytes",
    )
    baseline = records[0]
    if any(
        any(record[field] != baseline[field] for field in fixed_fields)
        for record in records[1:]
    ):
        raise RuntimeError("L1 cache ablation provenance changed")
    if any(
        record["backend"] != "cuda-custom"
        or record["cuda_allocation"] != "reused"
        or record["cuda_weights"] != "transient"
        or record["cuda_batching"] != "scalar"
        or record["cuda_boundary"] != "ffn-block"
        or record["dense_precision"] != dense_precision
        or record["cuda_transfer"] != configuration["cuda_transfer"]
        or record["cuda_pinned_bytes"] != configuration["cuda_pinned_bytes"]
        or record["l1_expert_cache_mode"] != configuration["l1_expert_cache"]
        or record["l1_expert_cache_bytes"]
        != configuration["l1_expert_cache_bytes"]
        for record, configuration in zip(records, matrix, strict=True)
    ):
        raise RuntimeError("L1 cache ablation option identity changed")

    baseline_tokens = tuple(baseline["token_ids"])
    baseline_routing = tuple(baseline["routed_experts"])
    for record in records:
        if tuple(record["token_ids"]) != baseline_tokens:
            raise RuntimeError("L1 cache ablation token parity failed")
        if tuple(record["routed_experts"]) != baseline_routing:
            raise RuntimeError("L1 cache ablation routing parity failed")
        record["parity_status"] = "exact"

    matched_fields = (
        "host_to_device_bytes",
        "weight_h2d_bytes",
        "activation_h2d_bytes",
        "device_to_host_bytes",
        "stream_synchronization_count",
        "ffn_block_calls",
        "ffn_block_experts",
        "async_prefetch_calls",
        "async_prefetch_bytes",
        "transfer_stream_wait_count",
    )
    pairs = ((records[0], records[1]), (records[2], records[3]))
    for disabled, static in pairs:
        if any(int(disabled[field]) != 0 for field in (
            "l1_expert_cache_hits",
            "l1_expert_cache_misses",
            "l1_expert_cache_bypasses",
            "l1_expert_cache_resident_bytes",
            "peak_l1_expert_cache_resident_bytes",
        )):
            raise RuntimeError("disabled L1 cache reported residency")
        if (
            int(static["l1_expert_cache_hits"]) <= 0
            or int(static["l1_expert_cache_misses"]) <= 0
            or int(static["l1_expert_cache_bypasses"]) != 0
            or not 0 < int(static["l1_expert_cache_resident_bytes"])
            <= l1_expert_cache_bytes
            or int(static["peak_l1_expert_cache_resident_bytes"])
            != int(static["l1_expert_cache_resident_bytes"])
        ):
            raise RuntimeError("static L1 cache counters are invalid")
        if any(int(static[field]) >= int(disabled[field]) for field in (
            "reader_read_calls",
            "reader_requested_bytes",
            "reader_completed_bytes",
        )):
            raise RuntimeError("static L1 cache did not reduce logical reads")
        if any(static[field] != disabled[field] for field in matched_fields):
            raise RuntimeError("matched L1 cache GPU execution changed")

    for record in records[:2]:
        if int(record["async_prefetch_calls"]) != 0:
            raise RuntimeError("synchronous L1 case reported prefetch calls")
    for record in records[2:]:
        calls = int(record["async_prefetch_calls"])
        if calls <= 0 or int(record["transfer_stream_wait_count"]) != calls:
            raise RuntimeError("prefetch L1 case reported invalid transfer counts")

    deltas = [
        {
            "from": disabled["name"],
            "to": static["name"],
            **{
                field: float(static[field]) - float(disabled[field])
                for field in _DELTA_FIELDS
            },
        }
        for disabled, static in pairs
    ]
    summary: dict[str, object] = {"records": records, "deltas": deltas}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument(
        "--dense-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument("--l1-expert-cache-bytes", type=int, required=True)
    parser.add_argument("--cuda-pinned-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_l1_cache_ablation(
        args.artifact,
        args.runner,
        dense_precision=args.dense_precision,
        l1_expert_cache_bytes=args.l1_expert_cache_bytes,
        cuda_pinned_bytes=args.cuda_pinned_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
