# adaptive Top-K와 exact rescue의 B-0012 측정 계약을 검증합니다.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ablate_adaptive_routing
from tools.benchmark_synthetic import BenchmarkRecord


def _base_record() -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / "disabled-synchronous.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRecord(**payload)


def _selected_k(configuration: dict[str, object]) -> int:
    if configuration["routing_mode"] == "natural":
        return 16
    if configuration["routing_mode"] == "fixed":
        floor = 16 if configuration["routing_critical"] else 12 if configuration["routing_agent_failures"] == 2 else 8 if configuration["routing_agent_failures"] == 1 else 0
        return max(int(configuration["routing_fixed_k"]), floor)
    if configuration["routing_critical"]:
        return 16
    if configuration["routing_agent_failures"] == 2:
        return 12
    if configuration["routing_agent_failures"] == 1:
        return 8
    return 8


def _record(configuration: dict[str, object]) -> BenchmarkRecord:
    selected_k = _selected_k(configuration)
    cache_mode = str(configuration["l1_expert_cache"])
    capacity = int(configuration["l1_expert_cache_bytes"])
    return replace(
        _base_record(),
        token_ids=(43, 32),
        routed_experts=tuple(range(selected_k)) * 2,
        routed_k=(selected_k, selected_k),
        routing_mode=str(configuration["routing_mode"]),
        routing_natural_top_k=16,
        routing_fixed_k=int(configuration["routing_fixed_k"]),
        routing_mass_target=float(configuration["routing_mass_target"]),
        routing_min_boundary_gap=float(configuration["routing_min_boundary_gap"]),
        routing_quality_floor_k=16 if configuration["routing_critical"] else 12 if configuration["routing_agent_failures"] == 2 else 8 if configuration["routing_agent_failures"] == 1 else 0,
        routing_agent_failures=int(configuration["routing_agent_failures"]),
        routing_critical=bool(configuration["routing_critical"]),
        routing_decisions=4,
        routing_selected_experts=selected_k * 4,
        routing_average_top_k=float(selected_k),
        routing_quality_escalated_decisions=4 if configuration["routing_agent_failures"] and selected_k > 8 else 0,
        cold_rescue_count=8 if cache_mode != "disabled" else 0,
        l1_expert_cache_mode=cache_mode,
        l1_expert_cache_bytes=capacity,
        reader_read_calls=10,
        reader_requested_bytes=16_320,
        reader_completed_bytes=16_320,
    )


def _diagnostic(configuration: dict[str, object]) -> dict[str, object]:
    selected_k = _selected_k(configuration)
    exact = selected_k == 16
    return {
        "token_ids": [43, 32],
        "prefill_logits": [1.0, 2.0 if exact else 2.25],
        "prefill_state": [3.0, 4.0 if exact else 4.5],
        "prefill_routed_k": [selected_k, selected_k],
        "prefill_routed_experts": list(range(selected_k)) * 2,
    }


def test_adaptive_routing_matrix_covers_reference_ladder_escalation_and_rescue() -> None:
    cases = ablate_adaptive_routing.adaptive_routing_matrix(6528)
    names = {str(case["name"]) for case in cases}
    assert {"natural-k16", "fixed-k4", "fixed-k8", "fixed-k12", "fixed-k16"} <= names
    assert {"adaptive-balanced", "adaptive-high-mass", "adaptive-boundary"} <= names
    assert {"adaptive-failure-1", "adaptive-failure-2", "adaptive-critical"} <= names
    assert {"fixed-k4-failure-1", "fixed-k4-failure-2", "fixed-k4-critical"} <= names
    rescue = next(case for case in cases if case["name"] == "fixed-k4-rescue")
    assert rescue["l1_expert_cache"] == "lru"
    assert rescue["l1_expert_cache_bytes"] == 6528


def test_b0012_manifest_matches_raw_records() -> None:
    root = Path(__file__).parents[2]
    result_dir = root / "results" / "b0012-adaptive-routing-wsl"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == "B-0012"
    assert summary["environment_label"] == "wsl2-ext4-warm-non-authoritative"
    assert summary["rescue_capacity_bytes"] == 6528
    assert summary["supported_cases"] == 15
    for case in summary["cases"]:
        raw = json.loads(
            (result_dir / f"{case['name']}.json").read_text(encoding="utf-8")
        )
        BenchmarkRecord(**raw)
        assert case["status"] == "measured"
        assert (result_dir / f"{case['name']}.csv").is_file()
        assert (result_dir / f"{case['name']}-diagnostic.json").is_file()
        for field, value in raw.items():
            assert case[field] == value


def test_prefix_validation_rejects_residency_substitution() -> None:
    baseline = _diagnostic(ablate_adaptive_routing.adaptive_routing_matrix(6528)[0])
    candidate = dict(baseline)
    candidate["prefill_routed_k"] = [4, 4]
    candidate["prefill_routed_experts"] = [0, 1, 9, 3, 0, 1, 2, 3]
    with pytest.raises(RuntimeError, match="first natural routing prefix"):
        ablate_adaptive_routing.compare_with_natural(baseline, candidate)


def test_b0012_writes_quality_and_traffic_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "fixture.k3x"
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(
        ablate_adaptive_routing,
        "benchmark_once",
        lambda *_args, **kwargs: _record(kwargs),
    )
    monkeypatch.setattr(
        ablate_adaptive_routing,
        "run_diagnostic",
        lambda *_args, **kwargs: _diagnostic(kwargs),
    )
    summary = ablate_adaptive_routing.run_adaptive_routing_ablation(
        artifact,
        Path("runner"),
        warmup=0,
        iterations=1,
        rescue_capacity_bytes=6528,
        output_dir=tmp_path,
    )
    assert summary["benchmark_id"] == "B-0012"
    assert summary["supported_cases"] == 15
    for case in summary["cases"]:
        raw = json.loads((tmp_path / f"{case['name']}.json").read_text(encoding="utf-8"))
        assert raw["first_decision_natural_prefix"] is True
        assert 0.0 < raw["natural_routing_prefix_rate"] <= 1.0
        assert raw["natural_token_parity"] is True
        assert case["natural_prefill_logits_max_abs_error"] == raw["natural_prefill_logits_max_abs_error"]
        assert (tmp_path / f"{case['name']}.csv").is_file()
    exact = {case["name"]: case for case in summary["cases"]}
    assert exact["natural-k16"]["quality_status"] == "exact"
    assert exact["fixed-k16"]["quality_status"] == "exact"
    assert exact["fixed-k4"]["quality_status"] == "diverged"
    assert exact["fixed-k4-rescue"]["cold_rescue_count"] > 0


def test_b0012_rejects_non_top16_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    artifact = tmp_path / "fixture.k3x"
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(
        ablate_adaptive_routing,
        "benchmark_once",
        lambda *_args, **kwargs: replace(_record(kwargs), routing_natural_top_k=2),
    )
    monkeypatch.setattr(
        ablate_adaptive_routing,
        "run_diagnostic",
        lambda *_args, **kwargs: _diagnostic(kwargs),
    )
    with pytest.raises(RuntimeError, match="natural Top-16"):
        ablate_adaptive_routing.run_adaptive_routing_ablation(
            artifact,
            Path("runner"),
            warmup=0,
            iterations=1,
            rescue_capacity_bytes=6528,
            output_dir=tmp_path,
        )
