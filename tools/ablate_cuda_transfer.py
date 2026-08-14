# CUDA synchronous와 exact prefetch 전송 경로를 동일 조건에서 비교합니다.
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
    "host_to_device_bytes",
    "weight_h2d_bytes",
    "activation_h2d_bytes",
    "device_to_host_bytes",
    "peak_vram_bytes",
    "stream_synchronization_count",
    "pinned_host_bytes",
    "peak_pinned_host_bytes",
    "async_prefetch_calls",
    "async_prefetch_bytes",
    "async_prefetch_ready_before_use",
    "async_prefetch_late_at_use",
    "transfer_stream_wait_count",
    "pinned_staging_nanoseconds",
    "transfer_device_nanoseconds",
    "transfer_stall_nanoseconds",
)


def transfer_matrix(cuda_pinned_bytes: int) -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "synchronous-scalar",
            "cuda_transfer": "synchronous",
            "cuda_batching": "scalar",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "prefetch-scalar",
            "cuda_transfer": "prefetch",
            "cuda_batching": "scalar",
            "cuda_pinned_bytes": cuda_pinned_bytes,
        },
        {
            "name": "synchronous-grouped",
            "cuda_transfer": "synchronous",
            "cuda_batching": "grouped",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "prefetch-grouped",
            "cuda_transfer": "prefetch",
            "cuda_batching": "grouped",
            "cuda_pinned_bytes": cuda_pinned_bytes,
        },
    )


def run_transfer_ablation(
    artifact: Path,
    runner: Path,
    *,
    dense_precision: str,
    cuda_pinned_bytes: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    if cuda_pinned_bytes <= 0:
        raise ValueError("pinned capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for configuration in transfer_matrix(cuda_pinned_bytes):
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
            cuda_batching=str(configuration["cuda_batching"]),
            cuda_boundary="ffn-block",
            cuda_transfer=str(configuration["cuda_transfer"]),
            cuda_resident_bytes=0,
            cuda_pinned_bytes=int(configuration["cuda_pinned_bytes"]),
        )
        json_path = output_dir / f"{name}.json"
        csv_path = output_dir / f"{name}.csv"
        write_results(record, json_path, csv_path)
        if not json_path.is_file() or not csv_path.is_file():
            raise RuntimeError("transfer ablation raw files are missing")
        records.append({"name": name, **asdict(record)})

    expected = transfer_matrix(cuda_pinned_bytes)
    for record, configuration in zip(records, expected, strict=True):
        if (
            record["backend"] != "cuda-custom"
            or record["cuda_boundary"] != "ffn-block"
            or record["cuda_allocation"] != "reused"
            or record["cuda_weights"] != "transient"
            or record["cuda_resident_bytes"] != 0
            or record["dense_precision"] != dense_precision
            or record["cuda_transfer"] != configuration["cuda_transfer"]
            or record["cuda_batching"] != configuration["cuda_batching"]
            or record["cuda_pinned_bytes"]
            != configuration["cuda_pinned_bytes"]
        ):
            raise RuntimeError("transfer ablation option identity changed")

    provenance = (
        "scope", "evidence", "platform", "iterations", "prompt_tokens",
        "generated_tokens", "backend", "device", "dense_precision",
        "cuda_allocation", "cuda_weights", "cuda_boundary",
        "cuda_resident_bytes",
    )
    baseline = records[0]
    if any(
        any(record[field] != baseline[field] for field in provenance)
        for record in records[1:]
    ):
        raise RuntimeError("transfer ablation provenance changed")

    baseline_tokens = tuple(baseline["token_ids"])
    baseline_routing = tuple(baseline["routed_experts"])
    for record in records:
        if tuple(record["token_ids"]) != baseline_tokens:
            raise RuntimeError("transfer ablation token parity failed")
        if tuple(record["routed_experts"]) != baseline_routing:
            raise RuntimeError("transfer ablation routing parity failed")
        record["parity_status"] = "exact"
        if (
            int(record["weight_h2d_bytes"])
            + int(record["activation_h2d_bytes"])
            != int(record["host_to_device_bytes"])
        ):
            raise RuntimeError("transfer ablation H2D accounting mismatch")

    asynchronous_fields = (
        "pinned_host_bytes",
        "peak_pinned_host_bytes",
        "async_prefetch_calls",
        "async_prefetch_bytes",
        "async_prefetch_ready_before_use",
        "async_prefetch_late_at_use",
        "transfer_stream_wait_count",
        "pinned_staging_nanoseconds",
        "transfer_device_nanoseconds",
        "transfer_stall_nanoseconds",
    )
    for record in (records[0], records[2]):
        if any(int(record[field]) != 0 for field in asynchronous_fields):
            raise RuntimeError("synchronous transfer reported async counters")
    for record in (records[1], records[3]):
        calls = int(record["async_prefetch_calls"])
        if (
            calls <= 0
            or int(record["async_prefetch_bytes"]) <= 0
            or int(record["pinned_host_bytes"]) != cuda_pinned_bytes
            or int(record["peak_pinned_host_bytes"]) != cuda_pinned_bytes
            or int(record["transfer_stream_wait_count"]) != calls
            or int(record["async_prefetch_ready_before_use"])
            + int(record["async_prefetch_late_at_use"]) != calls
            or int(record["pinned_staging_nanoseconds"]) <= 0
            or int(record["transfer_device_nanoseconds"]) <= 0
            or int(record["async_prefetch_bytes"])
            > int(record["weight_h2d_bytes"])
        ):
            raise RuntimeError("prefetch transfer counters are invalid")
    for synchronous, prefetch in ((records[0], records[1]), (records[2], records[3])):
        if any(
            int(prefetch[field]) != int(synchronous[field])
            for field in (
                "host_to_device_bytes",
                "weight_h2d_bytes",
                "activation_h2d_bytes",
            )
        ):
            raise RuntimeError("matched H2D traffic changed")
        if int(prefetch["stream_synchronization_count"]) != int(
            synchronous["stream_synchronization_count"]
        ):
            raise RuntimeError("matched synchronization count changed")

    deltas: list[dict[str, object]] = []
    for synchronous, prefetch in ((records[0], records[1]), (records[2], records[3])):
        deltas.append(
            {
                "from": synchronous["name"],
                "to": prefetch["name"],
                **{
                    field: float(prefetch[field]) - float(synchronous[field])
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
        "--dense-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument("--cuda-pinned-bytes", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_transfer_ablation(
        args.artifact,
        args.runner,
        dense_precision=args.dense_precision,
        cuda_pinned_bytes=args.cuda_pinned_bytes,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
