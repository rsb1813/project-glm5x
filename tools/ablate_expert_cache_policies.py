# L1 expert cache 정책과 용량의 B-0010 교차 측정을 실행하고 검증합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


POLICIES = ("static", "lru", "lfu", "least-stale")


def expert_cache_policy_matrix(
    capacities: tuple[int, ...],
) -> tuple[dict[str, object], ...]:
    cases: list[dict[str, object]] = [
        {"name": "disabled", "policy": "disabled", "capacity_bytes": 0}
    ]
    for capacity in capacities:
        for policy in POLICIES:
            cases.append(
                {
                    "name": f"{policy}-{capacity}",
                    "policy": policy,
                    "capacity_bytes": capacity,
                }
            )
    return tuple(cases)


def _validate_cache_record(
    record: dict[str, object], policy: str, capacity: int
) -> None:
    if (record["l1_expert_cache_mode"], record["l1_expert_cache_bytes"]) != (
        policy,
        capacity,
    ):
        raise RuntimeError("expert cache option identity changed")

    hits = int(record["l1_expert_cache_hits"])
    misses = int(record["l1_expert_cache_misses"])
    bypasses = int(record["l1_expert_cache_bypasses"])
    evictions = int(record["l1_expert_cache_evictions"])
    collisions = int(record["l1_expert_cache_collision_misses"])
    resident = int(record["l1_expert_cache_resident_bytes"])
    peak = int(record["peak_l1_expert_cache_resident_bytes"])
    if policy == "disabled":
        if any((hits, misses, bypasses, evictions, collisions, resident, peak)):
            raise RuntimeError("disabled expert cache accounting changed")
        return
    if misses <= 0 or resident < 0 or peak < resident or peak > capacity:
        raise RuntimeError("expert cache capacity accounting failed")
    if collisions < 0 or collisions > misses:
        raise RuntimeError("expert cache collision accounting failed")
    if policy == "static":
        if evictions != 0 or collisions != 0:
            raise RuntimeError("static expert cache eviction accounting changed")
    elif evictions <= 0:
        raise RuntimeError("dynamic eviction was not exercised")


def _validate_reader(record: dict[str, object]) -> None:
    requested = int(record["reader_requested_bytes"])
    completed = int(record["reader_completed_bytes"])
    if (
        requested <= 0
        or completed != requested
        or int(record["reader_read_calls"]) <= 0
        or int(record["reader_short_reads"]) != 0
        or int(record["reader_failures"]) != 0
    ):
        raise RuntimeError("expert cache reader accounting failed")


def run_expert_cache_policy_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    capacities: tuple[int, ...],
    output_dir: Path,
    environment_label: str = "local-synthetic-smoke",
) -> dict[str, object]:
    if not capacities or any(capacity <= 0 for capacity in capacities):
        raise ValueError("expert cache capacities must be positive")
    if len(set(capacities)) != len(capacities):
        raise ValueError("expert cache capacities must be unique")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    baseline: dict[str, object] | None = None
    for configuration in expert_cache_policy_matrix(capacities):
        name = str(configuration["name"])
        policy = str(configuration["policy"])
        capacity = int(configuration["capacity_bytes"])
        benchmark = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cpu",
            dense_precision="fp32",
            l1_expert_cache=policy,
            l1_expert_cache_bytes=capacity,
            l2_io="pread",
            l2_cache="buffered",
            l2_expert_schedule="blocking",
        )
        write_results(
            benchmark, output_dir / f"{name}.json", output_dir / f"{name}.csv"
        )
        record = asdict(benchmark)
        _validate_cache_record(record, policy, capacity)
        _validate_reader(record)
        if baseline is None:
            baseline = record
        if tuple(record["token_ids"]) != tuple(baseline["token_ids"]):
            raise RuntimeError("expert cache token parity failed")
        if tuple(record["routed_experts"]) != tuple(baseline["routed_experts"]):
            raise RuntimeError("expert cache routing parity failed")
        if (
            record["max_absolute_error"] != baseline["max_absolute_error"]
            or record["max_relative_error"] != baseline["max_relative_error"]
        ):
            raise RuntimeError("expert cache numerical parity failed")
        cases.append(
            {"name": name, "status": "measured", "parity_status": "exact", **record}
        )

    summary: dict[str, object] = {
        "benchmark_id": "B-0010",
        "environment_label": environment_label,
        "capacities_bytes": list(capacities),
        "supported_cases": len(cases),
        "cases": cases,
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
    parser.add_argument(
        "--capacities", type=int, nargs="+", default=(3264, 13056, 26112)
    )
    parser.add_argument("--environment-label", default="local-synthetic-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_expert_cache_policy_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        capacities=tuple(args.capacities),
        output_dir=args.output_dir,
        environment_label=args.environment_label,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
