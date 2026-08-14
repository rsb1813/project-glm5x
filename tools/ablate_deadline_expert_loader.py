# blocking과 deadline expert loading의 정확성 및 비용을 교차 측정합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


def deadline_loader_matrix() -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "name": f"{schedule}-{io_engine}-{cache_mode}",
            "schedule": schedule,
            "l2_io": io_engine,
            "l2_cache": cache_mode,
        }
        for cache_mode in ("buffered", "direct")
        for io_engine in ("pread", "io-uring")
        for schedule in ("blocking", "deadline")
    )


def _is_capability_failure(error: RuntimeError) -> bool:
    return "STORAGE_UNAVAILABLE" in str(error)


def _validate_reader(record: dict[str, object]) -> None:
    calls = int(record["reader_read_calls"])
    requested = int(record["reader_requested_bytes"])
    completed = int(record["reader_completed_bytes"])
    batches = int(record["reader_batch_submissions"])
    storage_submitted = int(record["reader_storage_submitted_bytes"])
    storage_completed = int(record["reader_storage_completed_bytes"])
    if requested <= 0 or requested != completed:
        raise RuntimeError("deadline loader logical I/O parity failed")
    if (
        calls <= 0
        or not 0 < batches <= calls
        or int(record["reader_completions"]) != calls
        or int(record["reader_short_reads"]) != 0
        or int(record["reader_failures"]) != 0
    ):
        raise RuntimeError("deadline loader logical I/O parity failed")
    if storage_submitted < requested or storage_completed != storage_submitted:
        raise RuntimeError("deadline loader storage accounting failed")


def _validate_scheduler(record: dict[str, object]) -> None:
    counters = (
        "expert_load_submissions",
        "expert_load_inline_resident_hits",
        "expert_load_completions",
        "expert_load_ready_before_use",
        "expert_load_late_at_use",
        "expert_load_estimated_deadline_misses",
        "expert_load_requested_bytes",
        "expert_load_queue_high_water",
        "expert_load_worker_nanoseconds",
        "expert_load_exposed_wait_nanoseconds",
    )
    if record["l2_expert_schedule"] == "blocking":
        if any(int(record[field]) != 0 for field in counters):
            raise RuntimeError("deadline loader blocking counters are nonzero")
        return

    submissions = int(record["expert_load_submissions"])
    inline_hits = int(record["expert_load_inline_resident_hits"])
    completions = int(record["expert_load_completions"])
    ready = int(record["expert_load_ready_before_use"])
    late = int(record["expert_load_late_at_use"])
    misses = int(record["expert_load_estimated_deadline_misses"])
    if submissions <= 0 or completions != submissions or ready + late != submissions:
        raise RuntimeError("deadline loader completion accounting failed")
    if inline_hits <= 0:
        raise RuntimeError("deadline loader resident hits are absent")
    if misses < 0 or misses > submissions:
        raise RuntimeError("deadline loader deadline accounting failed")
    if int(record["expert_load_requested_bytes"]) <= 0:
        raise RuntimeError("deadline loader requested bytes are absent")
    if int(record["expert_load_queue_high_water"]) <= 0:
        raise RuntimeError("deadline loader queue activity is absent")
    if int(record["expert_load_worker_nanoseconds"]) <= 0:
        raise RuntimeError("deadline loader worker timing is absent")


def run_deadline_loader_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    queue_depth: int,
    l1_expert_cache_bytes: int,
    output_dir: Path,
    environment_label: str = "local-capability-smoke",
) -> dict[str, object]:
    if queue_depth <= 0:
        raise ValueError("L2 queue depth must be positive")
    if l1_expert_cache_bytes <= 0:
        raise ValueError("L1 expert cache capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    measured: list[dict[str, object]] = []

    for configuration in deadline_loader_matrix():
        name = configuration["name"]
        try:
            benchmark = benchmark_once(
                artifact,
                runner,
                warmup,
                iterations,
                backend="cpu",
                dense_precision="fp32",
                l1_expert_cache="static",
                l1_expert_cache_bytes=l1_expert_cache_bytes,
                l2_io=configuration["l2_io"],
                l2_cache=configuration["l2_cache"],
                l2_queue_depth=queue_depth,
                l2_expert_schedule=configuration["schedule"],
            )
        except RuntimeError as error:
            if not _is_capability_failure(error):
                raise
            skipped = {
                "name": name,
                "status": "skipped",
                "l2_expert_schedule": configuration["schedule"],
                "l2_io_engine": configuration["l2_io"],
                "l2_cache_mode": configuration["l2_cache"],
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
        record = asdict(benchmark)
        expected_identity = (
            "cpu",
            "fp32",
            "static",
            l1_expert_cache_bytes,
            configuration["schedule"],
            configuration["l2_io"],
            configuration["l2_cache"],
            queue_depth,
        )
        observed_identity = (
            record["backend"],
            record["dense_precision"],
            record["l1_expert_cache_mode"],
            record["l1_expert_cache_bytes"],
            record["l2_expert_schedule"],
            record["l2_io_engine"],
            record["l2_cache_mode"],
            record["l2_queue_depth"],
        )
        if observed_identity != expected_identity:
            raise RuntimeError("deadline loader option identity changed")
        _validate_reader(record)
        _validate_scheduler(record)
        measured.append(record)
        cases.append({"name": name, "status": "measured", **record})

    if not measured:
        raise RuntimeError("deadline loader ablation has no supported baseline")
    baseline = measured[0]
    logical_fields = (
        "scope",
        "evidence",
        "iterations",
        "prompt_tokens",
        "generated_tokens",
        "backend",
        "dense_precision",
        "l1_expert_cache_mode",
        "l1_expert_cache_bytes",
        "l1_expert_cache_hits",
        "l1_expert_cache_misses",
        "l1_expert_cache_bypasses",
        "reader_read_calls",
        "reader_requested_bytes",
        "reader_completed_bytes",
        "reader_batch_submissions",
        "reader_completions",
    )
    for case in cases:
        if case["status"] != "measured":
            continue
        if tuple(case["token_ids"]) != tuple(baseline["token_ids"]):
            raise RuntimeError("deadline loader token parity failed")
        if tuple(case["routed_experts"]) != tuple(baseline["routed_experts"]):
            raise RuntimeError("deadline loader routing parity failed")
        if any(case[field] != baseline[field] for field in logical_fields):
            raise RuntimeError("deadline loader logical I/O parity failed")
        case["parity_status"] = "exact"

    for cache_mode in ("buffered", "direct"):
        for io_engine in ("pread", "io-uring"):
            statuses = {
                case["status"]
                for case in cases
                if case["l2_io_engine"] == io_engine
                and case["l2_cache_mode"] == cache_mode
            }
            if len(statuses) != 1:
                raise RuntimeError("deadline loader capability pair is incomplete")

    summary: dict[str, object] = {
        "benchmark_id": "B-0009",
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
    parser.add_argument("--l1-expert-cache-bytes", type=int, default=65_536)
    parser.add_argument("--environment-label", default="local-capability-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_deadline_loader_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        queue_depth=args.queue_depth,
        l1_expert_cache_bytes=args.l1_expert_cache_bytes,
        output_dir=args.output_dir,
        environment_label=args.environment_label,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
