# task/session prior와 live 관측을 비교하는 B-0011 ablation 계약을 검증합니다.
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ablate_task_session_profiles
from tools.benchmark_synthetic import BenchmarkRecord
from k3x_converter.writer import convert
from conftest import cpp_binary


def _base_record() -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / "disabled-synchronous.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRecord(**payload)


def _record(mode: str, profile_kind: str) -> BenchmarkRecord:
    profiled = mode == "profiled"
    has_prior = profile_kind in {"helpful", "conflicting"}
    return replace(
        _base_record(),
        token_ids=(43, 32),
        routed_experts=(1, 2, 3),
        l1_expert_cache_mode=mode,
        l1_expert_cache_bytes=13056,
        l1_expert_cache_hits=23,
        l1_expert_cache_misses=31,
        l1_expert_cache_evictions=23,
        l1_expert_cache_resident_bytes=13056,
        peak_l1_expert_cache_resident_bytes=13056,
        reader_read_calls=31,
        reader_requested_bytes=50_592,
        reader_completed_bytes=50_592,
        runtime_profile_metadata_count=4 if profiled else 0,
        runtime_profile_prior_weight=0.1 if has_prior else 0.0,
        runtime_profile_live_observations=54 if profiled else 0,
        runtime_profile_load_bytes=512 if has_prior else 0,
        runtime_profile_save_bytes=768 if profiled else 0,
        runtime_profile_load_nanoseconds=100 if has_prior else 0,
        runtime_profile_save_nanoseconds=200 if profiled else 0,
    )


def test_profile_matrix_keeps_exact_references_and_prior_cases() -> None:
    assert ablate_task_session_profiles.task_session_profile_matrix() == (
        {"name": "lfu", "policy": "lfu", "profile_kind": "none"},
        {
            "name": "least-stale",
            "policy": "least-stale",
            "profile_kind": "none",
        },
        {
            "name": "profiled-cold",
            "policy": "profiled",
            "profile_kind": "cold",
        },
        {
            "name": "profiled-helpful",
            "policy": "profiled",
            "profile_kind": "helpful",
        },
        {
            "name": "profiled-conflicting",
            "policy": "profiled",
            "profile_kind": "conflicting",
        },
    )


def test_b0011_manifest_matches_raw_records_and_profiles() -> None:
    root = Path(__file__).parents[2]
    result_dir = root / "results" / "b0011-task-session-profiles-wsl"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == "B-0011"
    assert summary["environment_label"] == "wsl2-ext4-warm-non-authoritative"
    assert summary["capacity_bytes"] == 13056
    assert summary["prior_strength"] == 4
    assert summary["supported_cases"] == 5
    assert summary["hot_overlap"] == 5
    for name in ("helpful", "conflicting"):
        profile = result_dir / "profiles" / f"{name}.k3xp"
        digest = hashlib.sha256(profile.read_bytes()).hexdigest()
        assert summary[f"{name}_profile_sha256"] == digest
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


def test_benchmark_materializes_the_full_generation_profile(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    profile = tmp_path / "observed.k3xp"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = ablate_task_session_profiles.benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cpu",
        l1_expert_cache="profiled",
        l1_expert_cache_bytes=13056,
        profile_prior_strength=4,
        runtime_metadata="TASK=coding,REPO=k3x",
        runtime_profile_out=profile,
    )
    assert record.runtime_profile_save_bytes == profile.stat().st_size
    assert record.runtime_profile_live_observations > 0


def test_b0011_writes_exact_profile_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helpful = tmp_path / "helpful.k3xp"
    conflicting = tmp_path / "conflicting.k3xp"
    artifact = tmp_path / "fixture.k3x"
    artifact.write_bytes(b"artifact")
    helpful.write_bytes(b"helpful")
    conflicting.write_bytes(b"conflicting")
    monkeypatch.setattr(
        ablate_task_session_profiles,
        "prepare_task_profiles",
        lambda *_args, **_kwargs: {
            "helpful": helpful,
            "conflicting": conflicting,
            "conflicting_prompt": "2,2,2,2",
            "hot_overlap": 1,
        },
    )
    monkeypatch.setattr(
        ablate_task_session_profiles,
        "benchmark_once",
        lambda *_args, **kwargs: _record(
            str(kwargs["l1_expert_cache"]),
            "helpful"
            if kwargs.get("runtime_profile_in") == helpful
            else "conflicting"
            if kwargs.get("runtime_profile_in") == conflicting
            else "cold",
        ),
    )
    summary = ablate_task_session_profiles.run_task_session_profile_ablation(
        artifact, Path("runner"), warmup=0, iterations=1,
        capacity_bytes=13056, prior_strength=4, output_dir=tmp_path,
    )
    assert summary["benchmark_id"] == "B-0011"
    assert summary["artifact_sha256"] == hashlib.sha256(b"artifact").hexdigest()
    assert summary["supported_cases"] == 5
    assert summary["conflicting_prompt"] == "2,2,2,2"
    assert all(case["parity_status"] == "exact" for case in summary["cases"])
    for case in summary["cases"]:
        BenchmarkRecord(**json.loads(
            (tmp_path / f"{case['name']}.json").read_text(encoding="utf-8")
        ))
        assert (tmp_path / f"{case['name']}.csv").is_file()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("token_ids", (99,), "token parity"),
        ("routed_experts", (99,), "routing parity"),
        ("runtime_profile_prior_weight", 1.1, "prior weight"),
        ("runtime_profile_live_observations", 0, "live observations"),
    ],
)
def test_b0011_rejects_invalid_profile_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    prior = tmp_path / "prior.k3xp"
    artifact = tmp_path / "fixture.k3x"
    prior.write_bytes(b"prior")
    artifact.write_bytes(b"artifact")
    monkeypatch.setattr(
        ablate_task_session_profiles,
        "prepare_task_profiles",
        lambda *_args, **_kwargs: {
            "helpful": prior,
            "conflicting": prior,
            "conflicting_prompt": "2,2,2,2",
            "hot_overlap": 1,
        },
    )

    def fake_benchmark(*_args: object, **kwargs: object) -> BenchmarkRecord:
        kind = "helpful" if kwargs.get("runtime_profile_in") == prior else "cold"
        record = _record(str(kwargs["l1_expert_cache"]), kind)
        if kwargs["l1_expert_cache"] == "profiled":
            return replace(record, **{field: value})
        return record

    monkeypatch.setattr(
        ablate_task_session_profiles, "benchmark_once", fake_benchmark
    )
    with pytest.raises(RuntimeError, match=message):
        ablate_task_session_profiles.run_task_session_profile_ablation(
            artifact, Path("runner"), warmup=0, iterations=1,
            capacity_bytes=13056, prior_strength=4, output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    ("capacity", "strength"), [(0, 4), (13056, 0)]
)
def test_b0011_rejects_invalid_configuration(
    capacity: int, strength: int, tmp_path: Path
) -> None:
    with pytest.raises(ValueError):
        ablate_task_session_profiles.run_task_session_profile_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            capacity_bytes=capacity, prior_strength=strength,
            output_dir=tmp_path,
        )
