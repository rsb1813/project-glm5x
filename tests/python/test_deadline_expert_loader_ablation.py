# deadline expert loader의 8개 조합 ablation 계약을 검증합니다.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ablate_deadline_expert_loader
from tools.benchmark_synthetic import BenchmarkRecord


def _base_record() -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / "disabled-synchronous.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRecord(**payload)


def _record(schedule: str, io_engine: str, cache_mode: str) -> BenchmarkRecord:
    direct = cache_mode == "direct"
    scheduler = schedule == "deadline"
    return replace(
        _base_record(),
        backend="cpu",
        device="CPU",
        routed_experts=(1, 2, 3),
        l1_expert_cache_mode="static",
        l1_expert_cache_bytes=65_536,
        l1_expert_cache_hits=12,
        l1_expert_cache_misses=4,
        l2_expert_schedule=schedule,
        expert_load_submissions=16 if scheduler else 0,
        expert_load_inline_resident_hits=12 if scheduler else 0,
        expert_load_completions=16 if scheduler else 0,
        expert_load_ready_before_use=15 if scheduler else 0,
        expert_load_late_at_use=1 if scheduler else 0,
        expert_load_estimated_deadline_misses=1 if scheduler else 0,
        expert_load_requested_bytes=24_576 if scheduler else 0,
        expert_load_queue_high_water=2 if scheduler else 0,
        expert_load_worker_nanoseconds=100_000 if scheduler else 0,
        expert_load_exposed_wait_nanoseconds=10_000 if scheduler else 0,
        l2_io_engine=io_engine,
        l2_cache_mode=cache_mode,
        l2_queue_depth=8,
        l2_direct_memory_alignment=512 if direct else 0,
        l2_direct_offset_alignment=512 if direct else 0,
        reader_read_calls=20,
        reader_requested_bytes=12_288,
        reader_completed_bytes=12_288,
        reader_batch_submissions=4,
        reader_storage_submitted_bytes=14_336 if direct else 12_288,
        reader_storage_completed_bytes=14_336 if direct else 12_288,
        reader_completions=20,
        reader_short_reads=0,
        reader_failures=0,
    )


def test_deadline_matrix_crosses_schedule_engine_and_cache() -> None:
    assert ablate_deadline_expert_loader.deadline_loader_matrix() == (
        {"name": "blocking-pread-buffered", "schedule": "blocking", "l2_io": "pread", "l2_cache": "buffered"},
        {"name": "deadline-pread-buffered", "schedule": "deadline", "l2_io": "pread", "l2_cache": "buffered"},
        {"name": "blocking-io-uring-buffered", "schedule": "blocking", "l2_io": "io-uring", "l2_cache": "buffered"},
        {"name": "deadline-io-uring-buffered", "schedule": "deadline", "l2_io": "io-uring", "l2_cache": "buffered"},
        {"name": "blocking-pread-direct", "schedule": "blocking", "l2_io": "pread", "l2_cache": "direct"},
        {"name": "deadline-pread-direct", "schedule": "deadline", "l2_io": "pread", "l2_cache": "direct"},
        {"name": "blocking-io-uring-direct", "schedule": "blocking", "l2_io": "io-uring", "l2_cache": "direct"},
        {"name": "deadline-io-uring-direct", "schedule": "deadline", "l2_io": "io-uring", "l2_cache": "direct"},
    )


def test_b0009_manifest_matches_all_measured_raw_records() -> None:
    root = Path(__file__).parents[2]
    result_dir = root / "results" / "b0009-deadline-loader-wsl"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == "B-0009"
    assert summary["environment_label"] == "wsl2-ext4-warm-non-authoritative"
    assert summary["supported_cases"] == 8
    assert summary["skipped_cases"] == 0
    for case in summary["cases"]:
        raw = json.loads(
            (result_dir / f"{case['name']}.json").read_text(encoding="utf-8")
        )
        BenchmarkRecord(**raw)
        assert case["status"] == "measured"
        assert case["parity_status"] == "exact"
        for field, value in raw.items():
            assert case[field] == value


def test_deadline_ablation_validates_all_pairs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        ablate_deadline_expert_loader,
        "benchmark_once",
        lambda *_args, **kwargs: _record(
            str(kwargs["l2_expert_schedule"]),
            str(kwargs["l2_io"]),
            str(kwargs["l2_cache"]),
        ),
    )
    summary = ablate_deadline_expert_loader.run_deadline_loader_ablation(
        Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
        queue_depth=8, l1_expert_cache_bytes=65_536, output_dir=tmp_path,
    )
    assert summary["benchmark_id"] == "B-0009"
    assert summary["supported_cases"] == 8
    assert summary["skipped_cases"] == 0
    assert all(case["parity_status"] == "exact" for case in summary["cases"])
    assert (tmp_path / "summary.json").is_file()
    for configuration in ablate_deadline_expert_loader.deadline_loader_matrix():
        assert (tmp_path / f"{configuration['name']}.json").is_file()
        assert (tmp_path / f"{configuration['name']}.csv").is_file()


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("deadline-pread-buffered", "token_ids", (99,), "token parity"),
        ("deadline-pread-buffered", "routed_experts", (99,), "routing parity"),
        ("deadline-pread-buffered", "reader_completions", 19, "logical I/O parity"),
        ("blocking-pread-buffered", "expert_load_submissions", 1, "blocking counters"),
        ("deadline-pread-buffered", "expert_load_completions", 15, "completion accounting"),
        ("deadline-pread-buffered", "expert_load_inline_resident_hits", 0, "resident hits"),
        ("deadline-pread-buffered", "expert_load_queue_high_water", 0, "queue activity"),
    ],
)
def test_deadline_ablation_rejects_divergence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    def fake_benchmark(*_args: object, **kwargs: object) -> BenchmarkRecord:
        name = f"{kwargs['l2_expert_schedule']}-{kwargs['l2_io']}-{kwargs['l2_cache']}"
        record = _record(
            str(kwargs["l2_expert_schedule"]),
            str(kwargs["l2_io"]),
            str(kwargs["l2_cache"]),
        )
        return replace(record, **{field: value}) if name == target else record

    monkeypatch.setattr(ablate_deadline_expert_loader, "benchmark_once", fake_benchmark)
    with pytest.raises(RuntimeError, match=message):
        ablate_deadline_expert_loader.run_deadline_loader_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            queue_depth=8, l1_expert_cache_bytes=65_536, output_dir=tmp_path,
        )


def test_deadline_ablation_skips_only_capability_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def capability_failure(*_args: object, **kwargs: object) -> BenchmarkRecord:
        if kwargs["l2_io"] == "io-uring" and kwargs["l2_cache"] == "direct":
            raise RuntimeError("runner failed with 3: STORAGE_UNAVAILABLE")
        return _record(
            str(kwargs["l2_expert_schedule"]),
            str(kwargs["l2_io"]),
            str(kwargs["l2_cache"]),
        )

    monkeypatch.setattr(ablate_deadline_expert_loader, "benchmark_once", capability_failure)
    summary = ablate_deadline_expert_loader.run_deadline_loader_ablation(
        Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
        queue_depth=8, l1_expert_cache_bytes=65_536, output_dir=tmp_path,
    )
    assert summary["supported_cases"] == 6
    assert summary["skipped_cases"] == 2

    monkeypatch.setattr(
        ablate_deadline_expert_loader,
        "benchmark_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("BAD_MAGIC")),
    )
    with pytest.raises(RuntimeError, match="BAD_MAGIC"):
        ablate_deadline_expert_loader.run_deadline_loader_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            queue_depth=8, l1_expert_cache_bytes=65_536, output_dir=tmp_path / "bad",
        )
