# adaptive Top-K와 exact cold rescue의 B-0012 교차 측정을 실행합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from tools.benchmark_synthetic import (
    _run_process,
    benchmark_once,
    write_results,
)


def _case(
    name: str,
    mode: str,
    *,
    fixed_k: int = 0,
    mass_target: float = 0.9,
    boundary_gap: float = 0.0,
    agent_failures: int = 0,
    critical: bool = False,
    cache: str = "disabled",
    cache_bytes: int = 0,
) -> dict[str, object]:
    return {
        "name": name,
        "routing_mode": mode,
        "routing_fixed_k": fixed_k,
        "routing_mass_target": mass_target,
        "routing_min_boundary_gap": boundary_gap,
        "routing_agent_failures": agent_failures,
        "routing_critical": critical,
        "l1_expert_cache": cache,
        "l1_expert_cache_bytes": cache_bytes,
    }


def adaptive_routing_matrix(
    rescue_capacity_bytes: int,
) -> tuple[dict[str, object], ...]:
    return (
        _case("natural-k16", "natural"),
        _case("fixed-k4", "fixed", fixed_k=4),
        _case("fixed-k8", "fixed", fixed_k=8),
        _case("fixed-k12", "fixed", fixed_k=12),
        _case("fixed-k16", "fixed", fixed_k=16),
        _case("adaptive-balanced", "adaptive"),
        _case("adaptive-high-mass", "adaptive", mass_target=0.98),
        _case("adaptive-boundary", "adaptive", boundary_gap=0.02),
        _case("adaptive-failure-1", "adaptive", agent_failures=1),
        _case("adaptive-failure-2", "adaptive", agent_failures=2),
        _case("adaptive-critical", "adaptive", critical=True),
        _case("fixed-k4-failure-1", "fixed", fixed_k=4, agent_failures=1),
        _case("fixed-k4-failure-2", "fixed", fixed_k=4, agent_failures=2),
        _case("fixed-k4-critical", "fixed", fixed_k=4, critical=True),
        _case(
            "fixed-k4-rescue",
            "fixed",
            fixed_k=4,
            cache="lru",
            cache_bytes=rescue_capacity_bytes,
        ),
    )


def run_diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    *,
    routing_mode: str,
    routing_fixed_k: int,
    routing_mass_target: float,
    routing_min_boundary_gap: float,
    routing_agent_failures: int,
    routing_critical: bool,
    l1_expert_cache: str,
    l1_expert_cache_bytes: int,
) -> dict[str, object]:
    result, _, _ = _run_process(
        Path(artifact).resolve(),
        Path(runner).resolve(),
        6,
        output,
        backend="cpu",
        dense_precision="fp32",
        cuda_allocation="per-operation",
        cuda_weights="transient",
        cuda_batching="scalar",
        cuda_boundary="operation",
        cuda_transfer="synchronous",
        cuda_resident_bytes=0,
        cuda_pinned_bytes=0,
        l1_expert_cache=l1_expert_cache,
        l1_expert_cache_bytes=l1_expert_cache_bytes,
        l2_io="pread",
        l2_cache="buffered",
        l2_queue_depth=8,
        l2_expert_schedule="blocking",
        routing_mode=routing_mode,
        routing_fixed_k=routing_fixed_k,
        routing_mass_target=routing_mass_target,
        routing_min_boundary_gap=routing_min_boundary_gap,
        routing_agent_failures=routing_agent_failures,
        routing_critical=routing_critical,
        diagnostics=True,
    )
    return result


def _routing_groups(diagnostic: dict[str, object]) -> tuple[tuple[int, ...], ...]:
    routed_k = tuple(int(value) for value in diagnostic["prefill_routed_k"])
    experts = tuple(int(value) for value in diagnostic["prefill_routed_experts"])
    if sum(routed_k) != len(experts):
        raise RuntimeError("routed K does not partition the expert trace")
    groups: list[tuple[int, ...]] = []
    offset = 0
    for selected_k in routed_k:
        if selected_k <= 0:
            raise RuntimeError("routed K must be positive")
        groups.append(experts[offset : offset + selected_k])
        offset += selected_k
    return tuple(groups)


def _max_abs_error(left: object, right: object) -> float:
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if len(left_values) != len(right_values):
        raise RuntimeError("natural comparison tensor length changed")
    return max(
        (abs(actual - expected) for actual, expected in zip(left_values, right_values, strict=True)),
        default=0.0,
    )


def compare_with_natural(
    natural: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    natural_groups = _routing_groups(natural)
    candidate_groups = _routing_groups(candidate)
    if len(natural_groups) != len(candidate_groups):
        raise RuntimeError("candidate routing decision count changed")
    prefix_matches = tuple(
        candidate_group == natural_group[: len(candidate_group)]
        for natural_group, candidate_group in zip(
            natural_groups, candidate_groups, strict=True
        )
    )
    if not prefix_matches or not prefix_matches[0]:
        raise RuntimeError("candidate did not preserve the first natural routing prefix")
    return {
        "natural_token_parity": candidate["token_ids"] == natural["token_ids"],
        "first_decision_natural_prefix": True,
        "natural_routing_prefix_rate": sum(prefix_matches) / len(prefix_matches),
        "natural_prefill_logits_max_abs_error": _max_abs_error(
            candidate["prefill_logits"], natural["prefill_logits"]
        ),
        "natural_prefill_state_max_abs_error": _max_abs_error(
            candidate["prefill_state"], natural["prefill_state"]
        ),
    }


def _validate_record(
    record: dict[str, object], configuration: dict[str, object]
) -> None:
    if int(record["routing_natural_top_k"]) != 16:
        raise RuntimeError("B-0012 requires a natural Top-16 artifact")
    for field in (
        "routing_mode",
        "routing_fixed_k",
        "routing_agent_failures",
        "routing_critical",
        "l1_expert_cache_mode",
        "l1_expert_cache_bytes",
    ):
        expected_field = (
            "l1_expert_cache" if field == "l1_expert_cache_mode" else field
        )
        if record[field] != configuration[expected_field]:
            raise RuntimeError(f"B-0012 option identity changed for {field}")
    for field in ("routing_mass_target", "routing_min_boundary_gap"):
        if not math.isclose(
            float(record[field]), float(configuration[field]),
            rel_tol=1.0e-6, abs_tol=1.0e-7,
        ):
            raise RuntimeError(f"B-0012 option identity changed for {field}")
    decisions = int(record["routing_decisions"])
    selected = int(record["routing_selected_experts"])
    if decisions <= 0 or selected <= 0 or not math.isclose(
        float(record["routing_average_top_k"]), selected / decisions,
        rel_tol=1.0e-9, abs_tol=1.0e-9,
    ):
        raise RuntimeError("B-0012 routing accounting failed")
    if int(record["reader_requested_bytes"]) <= 0 or (
        record["reader_requested_bytes"] != record["reader_completed_bytes"]
    ):
        raise RuntimeError("B-0012 Reader accounting failed")
    if configuration["l1_expert_cache"] == "disabled":
        if int(record["cold_rescue_count"]) != 0:
            raise RuntimeError("disabled cache reported cold rescues")
    elif int(record["cold_rescue_count"]) <= 0:
        raise RuntimeError("bounded cache did not exercise exact rescue")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_adaptive_routing_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    rescue_capacity_bytes: int,
    output_dir: Path,
    environment_label: str = "local-synthetic-smoke",
) -> dict[str, object]:
    if rescue_capacity_bytes <= 0:
        raise ValueError("rescue capacity must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    natural_diagnostic: dict[str, object] | None = None
    for configuration in adaptive_routing_matrix(rescue_capacity_bytes):
        name = str(configuration["name"])
        options = {key: value for key, value in configuration.items() if key != "name"}
        benchmark = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cpu",
            dense_precision="fp32",
            l2_io="pread",
            l2_cache="buffered",
            l2_expert_schedule="blocking",
            **options,
        )
        diagnostic = run_diagnostic(
            artifact, runner, output_dir / f"{name}-diagnostic.json", **options
        )
        if natural_diagnostic is None:
            natural_diagnostic = diagnostic
        quality = compare_with_natural(natural_diagnostic, diagnostic)
        benchmark = replace(benchmark, **quality)
        record = asdict(benchmark)
        _validate_record(record, configuration)
        quality_status = (
            "exact"
            if quality["natural_token_parity"]
            and quality["natural_prefill_logits_max_abs_error"] == 0.0
            and quality["natural_prefill_state_max_abs_error"] == 0.0
            else "diverged"
        )
        write_results(
            benchmark, output_dir / f"{name}.json", output_dir / f"{name}.csv"
        )
        cases.append(
            {
                "name": name,
                "status": "measured",
                "quality_status": quality_status,
                **record,
            }
        )

    by_name = {str(case["name"]): case for case in cases}
    for exact_name in ("natural-k16", "fixed-k16", "fixed-k4-critical"):
        if by_name[exact_name]["quality_status"] != "exact":
            raise RuntimeError(f"{exact_name} diverged from natural execution")
    for name, expected_k in (
        ("fixed-k4-failure-1", 8.0),
        ("fixed-k4-failure-2", 12.0),
        ("fixed-k4-critical", 16.0),
    ):
        if by_name[name]["routing_average_top_k"] != expected_k:
            raise RuntimeError("quality escalation did not select the expected K")
    fixed = by_name["fixed-k4"]
    rescue = by_name["fixed-k4-rescue"]
    for field in (
        "token_ids",
        "routed_experts",
        "routed_k",
        "natural_token_parity",
        "first_decision_natural_prefix",
        "natural_routing_prefix_rate",
        "natural_prefill_logits_max_abs_error",
        "natural_prefill_state_max_abs_error",
    ):
        if rescue[field] != fixed[field]:
            raise RuntimeError("exact rescue changed fixed-K execution")

    summary: dict[str, object] = {
        "benchmark_id": "B-0012",
        "artifact_sha256": _sha256(Path(artifact)),
        "environment_label": environment_label,
        "rescue_capacity_bytes": rescue_capacity_bytes,
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
    parser.add_argument("--rescue-capacity-bytes", type=int, default=6528)
    parser.add_argument("--environment-label", default="local-synthetic-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_adaptive_routing_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        rescue_capacity_bytes=args.rescue_capacity_bytes,
        output_dir=args.output_dir,
        environment_label=args.environment_label,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
