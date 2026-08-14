# B-0008 runner의 네 storage 조합과 raw artifact 교차 검증을 수행합니다.
from __future__ import annotations

import csv
import json
from pathlib import Path

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source
from tools.ablate_bounded_slice import (
    bounded_slice_matrix,
    run_bounded_slice_ablation,
)


def test_bounded_slice_matrix_crosses_independent_reader_axes() -> None:
    assert bounded_slice_matrix() == (
        {"name": "pread-buffered", "l2_io": "pread", "l2_cache": "buffered"},
        {"name": "io-uring-buffered", "l2_io": "io-uring", "l2_cache": "buffered"},
        {"name": "pread-direct", "l2_io": "pread", "l2_cache": "direct"},
        {"name": "io-uring-direct", "l2_io": "io-uring", "l2_cache": "direct"},
    )


def test_bounded_slice_ablation_preserves_digest_bytes_and_raw_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    output = tmp_path / "results"

    summary = run_bounded_slice_ablation(
        artifact,
        cpp_binary("k3x_storage_bench"),
        warmup=0,
        iterations=1,
        queue_depth=8,
        output_dir=output,
        environment_label="pytest-wsl-capability",
    )

    assert summary["benchmark_id"] == "B-0008"
    assert summary["environment_label"] == "pytest-wsl-capability"
    assert summary["supported_cases"] >= 1
    assert summary["supported_cases"] + summary["skipped_cases"] == 4
    measured = [case for case in summary["cases"] if case["status"] == "measured"]
    assert len({case["ordered_sha256"] for case in measured}) == 1
    assert all(case["parity_status"] == "exact" for case in measured)
    for case in summary["cases"]:
        name = case["name"]
        if case["status"] == "skipped":
            assert "STORAGE_UNAVAILABLE" in case["reason"]
            assert (output / f"{name}.skipped.json").is_file()
            continue
        assert case["expert_payload_bytes"] == 17_547_264
        assert case["reader_requested_bytes"] == 17_547_264
        assert case["reader_completed_bytes"] == 17_547_264
        assert case["reader_read_calls"] == 6
        assert case["reader_batch_submissions"] == 1
        assert case["reader_completions"] == 6
        assert case["reader_short_reads"] == 0
        assert case["reader_failures"] == 0
        raw_json = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        with (output / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
            raw_csv = next(csv.DictReader(stream))
        assert raw_json["ordered_sha256"] == case["ordered_sha256"]
        assert raw_csv["ordered_sha256"] == case["ordered_sha256"]
        assert int(raw_csv["expert_payload_bytes"]) == 17_547_264
    persisted = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
