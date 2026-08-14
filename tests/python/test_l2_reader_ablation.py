# L2 I/O 엔진과 캐시 모드의 독립 ablation 계약을 검증합니다.
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tools import ablate_l2_reader
from tools.ablate_l2_reader import l2_reader_matrix, run_l2_reader_ablation
from tools.benchmark_synthetic import BenchmarkRecord


def _base_record() -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / "disabled-synchronous.json")
        .read_text(encoding="utf-8")
    )
    return BenchmarkRecord(**payload)


def _record(io_engine: str, cache_mode: str) -> BenchmarkRecord:
    direct = cache_mode == "direct"
    logical_bytes = 665_616
    storage_bytes = logical_bytes + (49_920 if direct else 0)
    return replace(
        _base_record(),
        backend="cpu",
        device="CPU",
        routed_experts=(1, 2, 3),
        l2_io_engine=io_engine,
        l2_cache_mode=cache_mode,
        l2_queue_depth=8,
        l2_direct_memory_alignment=512 if direct else 0,
        l2_direct_offset_alignment=512 if direct else 0,
        reader_read_calls=428,
        reader_requested_bytes=logical_bytes,
        reader_completed_bytes=logical_bytes,
        reader_batch_submissions=212,
        reader_storage_submitted_bytes=storage_bytes,
        reader_storage_completed_bytes=storage_bytes,
        reader_completions=428,
        reader_short_reads=0,
        reader_failures=0,
        reader_storage_nanoseconds=1_000_000,
        process_io_available=True,
        process_rchar_bytes=logical_bytes,
        process_read_bytes=storage_bytes if direct else 0,
    )


def test_l2_reader_matrix_crosses_only_engine_and_cache_mode() -> None:
    assert l2_reader_matrix() == (
        {"name": "pread-buffered", "l2_io": "pread", "l2_cache": "buffered"},
        {"name": "io-uring-buffered", "l2_io": "io-uring", "l2_cache": "buffered"},
        {"name": "pread-direct", "l2_io": "pread", "l2_cache": "direct"},
        {"name": "io-uring-direct", "l2_io": "io-uring", "l2_cache": "direct"},
    )


def test_b0007_manifest_matches_all_measured_raw_records() -> None:
    root = Path(__file__).parents[2]
    result_dir = root / "results" / "b0007-l2-reader-wsl"
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark_id"] == "B-0007"
    assert summary["environment_label"] == "wsl2-ext4-smoke-non-authoritative"
    assert summary["supported_cases"] == 4
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


def test_l2_ablation_validates_supported_cases_and_records_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_benchmark(*_args: object, **kwargs: object) -> BenchmarkRecord:
        if kwargs["l2_io"] == "io-uring" and kwargs["l2_cache"] == "direct":
            raise RuntimeError("runner failed with 3: STORAGE_UNAVAILABLE")
        return _record(str(kwargs["l2_io"]), str(kwargs["l2_cache"]))

    monkeypatch.setattr(ablate_l2_reader, "benchmark_once", fake_benchmark)
    summary = run_l2_reader_ablation(
        Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
        queue_depth=8, output_dir=tmp_path,
    )
    assert [item["status"] for item in summary["cases"]] == [
        "measured", "measured", "measured", "skipped"
    ]
    assert all(
        item.get("parity_status") == "exact"
        for item in summary["cases"] if item["status"] == "measured"
    )
    assert "STORAGE_UNAVAILABLE" in summary["cases"][3]["reason"]
    for name in ("pread-buffered", "io-uring-buffered", "pread-direct"):
        assert (tmp_path / f"{name}.json").is_file()
        assert (tmp_path / f"{name}.csv").is_file()
    assert (tmp_path / "io-uring-direct.skipped.json").is_file()
    assert (tmp_path / "summary.json").is_file()


@pytest.mark.parametrize(
    ("case_name", "field", "value", "message"),
    [
        ("io-uring-buffered", "token_ids", (99,), "token parity"),
        ("io-uring-buffered", "routed_experts", (99,), "routing parity"),
        ("pread-direct", "reader_requested_bytes", 1, "logical read parity"),
        ("pread-direct", "reader_failures", 1, "failure counters"),
        ("pread-direct", "reader_storage_submitted_bytes", 1, "storage accounting"),
        ("pread-buffered", "l2_direct_memory_alignment", 512, "buffered alignment"),
        ("pread-direct", "l2_direct_memory_alignment", 0, "direct alignment"),
    ],
)
def test_l2_ablation_rejects_invalid_supported_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case_name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    def fake_benchmark(*_args: object, **kwargs: object) -> BenchmarkRecord:
        name = f"{kwargs['l2_io']}-{kwargs['l2_cache']}"
        record = _record(str(kwargs["l2_io"]), str(kwargs["l2_cache"]))
        return replace(record, **{field: value}) if name == case_name else record

    monkeypatch.setattr(ablate_l2_reader, "benchmark_once", fake_benchmark)
    with pytest.raises(RuntimeError, match=message):
        run_l2_reader_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            queue_depth=8, output_dir=tmp_path,
        )


def test_l2_ablation_does_not_hide_unrelated_runtime_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ablate_l2_reader,
        "benchmark_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("BAD_MAGIC")),
    )
    with pytest.raises(RuntimeError, match="BAD_MAGIC"):
        run_l2_reader_ablation(
            Path("fixture.k3x"), Path("runner"), warmup=0, iterations=1,
            queue_depth=8, output_dir=tmp_path,
        )
