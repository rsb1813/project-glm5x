# B-0008 full-dimension expert storage read 조합을 독립적으로 교차 측정합니다.
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def bounded_slice_matrix() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "name": f"{io_engine}-{cache_mode}",
            "l2_io": io_engine,
            "l2_cache": cache_mode,
        }
        for cache_mode in ("buffered", "direct")
        for io_engine in ("pread", "io-uring")
    )


def _write_raw(record: dict[str, object], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)


def _validate_record(
    record: dict[str, object], configuration: dict[str, str], iterations: int,
    queue_depth: int
) -> None:
    if (
        record.get("artifact_kind") != "storage_fixture"
        or record.get("layer_id") != 1
        or record.get("expert_id") != 0
        or record.get("l2_io_engine") != configuration["l2_io"]
        or record.get("l2_cache_mode") != configuration["l2_cache"]
        or record.get("l2_queue_depth") != queue_depth
        or record.get("iterations") != iterations
        or record.get("expert_payload_bytes") != 17_547_264
    ):
        raise RuntimeError("bounded storage benchmark identity changed")
    expected_calls = iterations * 6
    expected_bytes = iterations * 17_547_264
    if (
        record.get("reader_read_calls") != expected_calls
        or record.get("reader_batch_submissions") != iterations
        or record.get("reader_completions") != expected_calls
        or record.get("reader_requested_bytes") != expected_bytes
        or record.get("reader_completed_bytes") != expected_bytes
        or record.get("reader_short_reads") != 0
        or record.get("reader_failures") != 0
    ):
        raise RuntimeError("bounded storage logical accounting changed")
    submitted = int(record["reader_storage_submitted_bytes"])
    completed = int(record["reader_storage_completed_bytes"])
    if submitted < expected_bytes or completed != submitted:
        raise RuntimeError("bounded storage submitted accounting changed")
    if configuration["l2_cache"] == "buffered":
        if (
            submitted != expected_bytes
            or record.get("l2_direct_memory_alignment") != 0
            or record.get("l2_direct_offset_alignment") != 0
        ):
            raise RuntimeError("bounded buffered storage accounting changed")
    elif (
        int(record["l2_direct_memory_alignment"]) <= 0
        or int(record["l2_direct_offset_alignment"]) <= 0
    ):
        raise RuntimeError("bounded direct alignment is unavailable")
    digest = record.get("ordered_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("bounded storage digest is invalid")
    if not (
        int(record["expert_load_nanoseconds_p05"])
        <= int(record["expert_load_nanoseconds_median"])
        <= int(record["expert_load_nanoseconds_p95"])
    ):
        raise RuntimeError("bounded storage latency percentiles are invalid")
    process_available = bool(record.get("process_io_available"))
    process_values = (
        record.get("process_rchar_bytes"), record.get("process_read_bytes")
    )
    if process_available != all(value is not None for value in process_values):
        raise RuntimeError("bounded process I/O availability is inconsistent")


def _run_case(
    artifact: Path,
    runner: Path,
    configuration: dict[str, str],
    warmup: int,
    iterations: int,
    queue_depth: int,
) -> dict[str, object]:
    result = subprocess.run(
        [
            str(runner),
            "--model", str(artifact),
            "--layer", "1",
            "--expert", "0",
            "--warmup", str(warmup),
            "--iterations", str(iterations),
            "--l2-io", configuration["l2_io"],
            "--l2-cache", configuration["l2_cache"],
            "--l2-queue-depth", str(queue_depth),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(message or f"storage benchmark exited {result.returncode}")
    try:
        record = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("storage benchmark returned invalid JSON") from error
    if not isinstance(record, dict):
        raise RuntimeError("storage benchmark returned a non-object")
    _validate_record(record, configuration, iterations, queue_depth)
    return record


def run_bounded_slice_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    queue_depth: int,
    output_dir: Path,
    environment_label: str = "local-capability-smoke",
) -> dict[str, object]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < queue_depth <= 1024:
        raise ValueError("L2 queue depth must be in [1, 1024]")
    artifact = Path(artifact)
    runner = Path(runner)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []
    for configuration in bounded_slice_matrix():
        name = configuration["name"]
        try:
            record = _run_case(
                artifact, runner, configuration, warmup, iterations, queue_depth
            )
        except RuntimeError as error:
            if "STORAGE_UNAVAILABLE" not in str(error):
                raise
            skipped: dict[str, object] = {
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
        _write_raw(
            record,
            output_dir / f"{name}.json",
            output_dir / f"{name}.csv",
        )
        measured.append(record)
        cases.append({"name": name, "status": "measured", **record})

    if not measured:
        raise RuntimeError("bounded storage ablation has no supported baseline")
    baseline = measured[0]
    parity_fields = (
        "artifact_kind", "layer_id", "expert_id", "l2_queue_depth", "warmup",
        "iterations", "expert_payload_bytes", "ordered_sha256",
        "reader_read_calls", "reader_requested_bytes", "reader_completed_bytes",
        "reader_batch_submissions", "reader_completions",
    )
    for case in cases:
        if case["status"] != "measured":
            continue
        if any(case[field] != baseline[field] for field in parity_fields):
            raise RuntimeError("bounded storage cross-mode parity failed")
        case["parity_status"] = "exact"

    summary: dict[str, object] = {
        "benchmark_id": "B-0008",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-label", default="local-capability-smoke")
    args = parser.parse_args()
    summary = run_bounded_slice_ablation(
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
