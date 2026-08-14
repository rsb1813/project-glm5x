# expert cache policy와 용량을 교차하는 B-0010 ablation 계약을 검증합니다.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ablate_expert_cache_policies
from tools.benchmark_synthetic import BenchmarkRecord


def _base_record() -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / "disabled-synchronous.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRecord(**payload)


def _record(mode: str, capacity: int) -> BenchmarkRecord:
    disabled = mode == "disabled"
    static = mode == "static"
    return replace(
        _base_record(),
        token_ids=(43, 32),
        routed_experts=(1, 2, 3),
        l1_expert_cache_mode=mode,
        l1_expert_cache_bytes=capacity,
        l1_expert_cache_hits=0 if disabled else 2,
        l1_expert_cache_misses=0 if disabled else 12,
        l1_expert_cache_bypasses=0,
        l1_expert_cache_evictions=0 if disabled or static else 9,
        l1_expert_cache_collision_misses=0 if disabled or static else 3,
        l1_expert_cache_resident_bytes=0 if disabled else capacity,
        peak_l1_expert_cache_resident_bytes=0 if disabled else capacity,
        reader_read_calls=100 if disabled else 70,
        reader_requested_bytes=100_000 if disabled else 70_000,
        reader_completed_bytes=100_000 if disabled else 70_000,
    )


def test_policy_matrix_crosses_four_policies_and_capacities() -> None:
    assert ablate_expert_cache_policies.expert_cache_policy_matrix((1632, 3264)) == (
        {"name": "disabled", "policy": "disabled", "capacity_bytes": 0},
        {"name": "static-1632", "policy": "static", "capacity_bytes": 1632},
        {"name": "lru-1632", "policy": "lru", "capacity_bytes": 1632},
        {"name": "lfu-1632", "policy": "lfu", "capacity_bytes": 1632},
        {
            "name": "least-stale-1632",
            "policy": "least-stale",
            "capacity_bytes": 1632,
        },
        {"name": "static-3264", "policy": "static", "capacity_bytes": 3264},
        {"name": "lru-3264", "policy": "lru", "capacity_bytes": 3264},
        {"name": "lfu-3264", "policy": "lfu", "capacity_bytes": 3264},
        {
            "name": "least-stale-3264",
            "policy": "least-stale",
            "capacity_bytes": 3264,
        },
    )


def test_b0010_manifest_matches_all_measured_raw_records() -> None:
    root = Path(__file__).parents[2]
    result_dir = root / "results" / "b0010-expert-cache-policies-wsl"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == "B-0010"
    assert summary["environment_label"] == "wsl2-ext4-warm-non-authoritative"
    assert summary["capacities_bytes"] == [3264, 13056, 26112]
    assert summary["supported_cases"] == 13
    for case in summary["cases"]:
        raw = json.loads(
            (result_dir / f"{case['name']}.json").read_text(encoding="utf-8")
        )
        BenchmarkRecord(**raw)
        assert case["status"] == "measured"
        assert case["parity_status"] == "exact"
        assert (result_dir / f"{case['name']}.csv").is_file()
        for field, value in raw.items():
            assert case[field] == value


def test_b0010_writes_exact_policy_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ablate_expert_cache_policies,
        "benchmark_once",
        lambda *_args, **kwargs: _record(
            str(kwargs["l1_expert_cache"]), int(kwargs["l1_expert_cache_bytes"])
        ),
    )
    summary = ablate_expert_cache_policies.run_expert_cache_policy_ablation(
        Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
        capacities=(1632, 3264), output_dir=tmp_path,
    )
    assert summary["benchmark_id"] == "B-0010"
    assert summary["supported_cases"] == 9
    assert all(case["parity_status"] == "exact" for case in summary["cases"])
    for case in summary["cases"]:
        raw_path = tmp_path / f"{case['name']}.json"
        BenchmarkRecord(**json.loads(raw_path.read_text(encoding="utf-8")))
        assert (tmp_path / f"{case['name']}.csv").is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token_ids", (99,), "token parity"),
        ("routed_experts", (99,), "routing parity"),
        ("max_absolute_error", 1.0, "numerical parity"),
        ("l1_expert_cache_mode", "static", "option identity"),
        ("l1_expert_cache_evictions", 0, "dynamic eviction"),
        ("l1_expert_cache_collision_misses", 13, "collision accounting"),
    ],
)
def test_b0010_rejects_invalid_policy_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    def fake_benchmark(*_args: object, **kwargs: object) -> BenchmarkRecord:
        mode = str(kwargs["l1_expert_cache"])
        capacity = int(kwargs["l1_expert_cache_bytes"])
        record = _record(mode, capacity)
        if mode == "lru" and capacity == 1632:
            return replace(record, **{field: value})
        return record

    monkeypatch.setattr(
        ablate_expert_cache_policies, "benchmark_once", fake_benchmark
    )
    with pytest.raises(RuntimeError, match=message):
        ablate_expert_cache_policies.run_expert_cache_policy_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            capacities=(1632,), output_dir=tmp_path,
        )


@pytest.mark.parametrize("capacities", [(), (0,), (1632, 1632)])
def test_b0010_rejects_invalid_capacities(
    capacities: tuple[int, ...], tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        ablate_expert_cache_policies.run_expert_cache_policy_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            capacities=capacities, output_dir=tmp_path,
        )
