# L2 I/O 엔진과 page-cache 모드를 독립적으로 교차 측정합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


def l2_reader_matrix() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "name": f"{io_engine}-{cache_mode}",
            "l2_io": io_engine,
            "l2_cache": cache_mode,
        }
        for cache_mode in ("buffered", "direct")
        for io_engine in ("pread", "io-uring")
    )


def _is_capability_failure(error: RuntimeError) -> bool:
    return "STORAGE_UNAVAILABLE" in str(error)


def _validate_record(record: dict[str, object], configuration: dict[str, str],
                     queue_depth: int) -> None:
    if (
        record["backend"] != "cpu"
        or record["dense_precision"] != "fp32"
        or record["l1_expert_cache_mode"] != "disabled"
        or record["l1_expert_cache_bytes"] != 0
        or record["l2_io_engine"] != configuration["l2_io"]
        or record["l2_cache_mode"] != configuration["l2_cache"]
        or record["l2_queue_depth"] != queue_depth
    ):
        raise RuntimeError("L2 reader ablation option identity changed")
    calls = int(record["reader_read_calls"])
    logical_requested = int(record["reader_requested_bytes"])
    logical_completed = int(record["reader_completed_bytes"])
    batches = int(record["reader_batch_submissions"])
    storage_submitted = int(record["reader_storage_submitted_bytes"])
    storage_completed = int(record["reader_storage_completed_bytes"])
    if logical_requested != logical_completed or logical_completed <= 0:
        raise RuntimeError("L2 reader logical read parity failed")
    if (
        calls <= 0
        or not 0 < batches <= calls
        or int(record["reader_completions"]) != calls
        or int(record["reader_short_reads"]) != 0
        or int(record["reader_failures"]) != 0
    ):
        raise RuntimeError("L2 reader failure counters are invalid")
    if storage_submitted < logical_requested or storage_completed != storage_submitted:
        raise RuntimeError("L2 reader storage accounting is invalid")
    if configuration["l2_cache"] == "buffered":
        if (
            int(record["l2_direct_memory_alignment"]) != 0
            or int(record["l2_direct_offset_alignment"]) != 0
        ):
            raise RuntimeError("L2 buffered alignment must be zero")
        if storage_submitted != logical_requested:
            raise RuntimeError("L2 buffered storage accounting is amplified")
    elif (
        int(record["l2_direct_memory_alignment"]) <= 0
        or int(record["l2_direct_offset_alignment"]) <= 0
    ):
        raise RuntimeError("L2 direct alignment is unavailable")
    if bool(record["process_io_available"]):
        if record["process_rchar_bytes"] is None or record["process_read_bytes"] is None:
            raise RuntimeError("L2 process I/O availability is inconsistent")
    elif record["process_rchar_bytes"] is not None or record["process_read_bytes"] is not None:
        raise RuntimeError("L2 process I/O availability is inconsistent")


def run_l2_reader_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    queue_depth: int,
    output_dir: Path,
    environment_label: str = "local-capability-smoke",
) -> dict[str, object]:
    if queue_depth <= 0:
        raise ValueError("L2 queue depth must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    for configuration in l2_reader_matrix():
        name = configuration["name"]
        try:
            benchmark = benchmark_once(
                artifact,
                runner,
                warmup,
                iterations,
                backend="cpu",
                dense_precision="fp32",
                l1_expert_cache="disabled",
                l1_expert_cache_bytes=0,
                l2_io=configuration["l2_io"],
                l2_cache=configuration["l2_cache"],
                l2_queue_depth=queue_depth,
            )
        except RuntimeError as error:
            if not _is_capability_failure(error):
                raise
            skipped = {
                "name": name,
                "status": "skipped",
                "l2_io_engine": configuration["l2_io"],
                "l2_cache_mode": configuration["l2_cache"],
                "l2_queue_depth": queue_depth,
                "reason": str(error),
            }
            (output_dir / f"{name}.skipped.json").write_text(
                json.dumps(skipped, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cases.append(skipped)
            continue
        write_results(
            benchmark, output_dir / f"{name}.json", output_dir / f"{name}.csv"
        )
        if not (output_dir / f"{name}.json").is_file() or not (
            output_dir / f"{name}.csv"
        ).is_file():
            raise RuntimeError("L2 reader ablation raw files are missing")
        record = asdict(benchmark)
        _validate_record(record, configuration, queue_depth)
        measured.append(record)
        cases.append({"name": name, "status": "measured", **record})

    if not measured:
        raise RuntimeError("L2 reader ablation has no supported baseline")
    baseline = measured[0]
    fixed_fields = (
        "scope", "evidence", "platform", "iterations", "prompt_tokens",
        "generated_tokens", "backend", "device", "dense_precision",
        "reader_read_calls", "reader_requested_bytes", "reader_completed_bytes",
        "reader_batch_submissions", "reader_completions",
    )
    baseline_tokens = tuple(baseline["token_ids"])
    baseline_routing = tuple(baseline["routed_experts"])
    for case in cases:
        if case["status"] != "measured":
            continue
        if any(case[field] != baseline[field] for field in fixed_fields):
            raise RuntimeError("L2 reader ablation logical read parity failed")
        if tuple(case["token_ids"]) != baseline_tokens:
            raise RuntimeError("L2 reader ablation token parity failed")
        if tuple(case["routed_experts"]) != baseline_routing:
            raise RuntimeError("L2 reader ablation routing parity failed")
        case["parity_status"] = "exact"

    summary: dict[str, object] = {
        "benchmark_id": "B-0007",
        "environment_label": environment_label,
        "cases": cases,
        "supported_cases": sum(case["status"] == "measured" for case in cases),
        "skipped_cases": sum(case["status"] == "skipped" for case in cases),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--queue-depth", type=int, default=8)
    parser.add_argument("--environment-label", default="local-capability-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_l2_reader_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        queue_depth=args.queue_depth,
        output_dir=args.output_dir,
        environment_label=args.environment_label,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
