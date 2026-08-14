# 전용 C++ storage benchmark가 실제 expert 여섯 extent를 정확히 읽는지 검증합니다.
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from conftest import cpp_binary
from k3x_converter.safetensors_reader import inspect_shard, iter_tensor_chunks
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _source_digest(source: Path) -> str:
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    shard_name = next(iter(manifest["weight_map"].values()))
    tensors = inspect_shard(source / shard_name)
    base = "model.layers.1.feed_forward.experts.0"
    digest = hashlib.sha256()
    for role in ("gate", "up", "down"):
        for suffix in ("weight_packed", "weight_scale"):
            tensor = tensors[f"{base}.{role}.{suffix}"]
            for chunk in iter_tensor_chunks(tensor, 193 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _bounded_artifact(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    return source, artifact


def test_storage_bench_loads_one_exact_full_dimension_expert(
    tmp_path: Path,
) -> None:
    runner = cpp_binary("k3x_storage_bench")
    assert runner.is_file(), "build k3x_storage_bench before running storage tests"
    source, artifact = _bounded_artifact(tmp_path)

    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--layer",
            "1",
            "--expert",
            "0",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--l2-io",
            "pread",
            "--l2-cache",
            "buffered",
            "--l2-queue-depth",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["artifact_kind"] == "storage_fixture"
    assert record["layer_id"] == 1
    assert record["expert_id"] == 0
    assert record["l2_io_engine"] == "pread"
    assert record["l2_cache_mode"] == "buffered"
    assert record["l2_queue_depth"] == 8
    assert record["warmup"] == 1
    assert record["iterations"] == 2
    assert record["expert_payload_bytes"] == 17_547_264
    assert record["ordered_sha256"] == _source_digest(source)
    assert record["reader_read_calls"] == 12
    assert record["reader_batch_submissions"] == 2
    assert record["reader_completions"] == 12
    assert record["reader_requested_bytes"] == 35_094_528
    assert record["reader_completed_bytes"] == 35_094_528
    assert record["reader_storage_submitted_bytes"] == 35_094_528
    assert record["reader_storage_completed_bytes"] == 35_094_528
    assert record["reader_short_reads"] == 0
    assert record["reader_failures"] == 0
    assert record["l2_direct_memory_alignment"] == 0
    assert record["l2_direct_offset_alignment"] == 0
    assert record["expert_load_nanoseconds_median"] > 0
    assert (
        record["expert_load_nanoseconds_p05"]
        <= record["expert_load_nanoseconds_median"]
        <= record["expert_load_nanoseconds_p95"]
    )
    assert record["expert_loads_per_second"] > 0
    assert "token_ids" not in record
    assert "decode_tok_s" not in record


def test_storage_bench_rejects_non_fixture_missing_expert_and_zero_iterations(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_storage_bench")
    assert runner.is_file(), "build k3x_storage_bench before running storage tests"
    tiny = tmp_path / "tiny.k3x"
    convert(synthetic_source, tiny, chunk_bytes=257)
    non_fixture = subprocess.run(
        [str(runner), "--model", str(tiny)], capture_output=True, text=True
    )
    assert non_fixture.returncode == 4
    assert non_fixture.stderr.strip() == (
        "INVALID_EXTENT: artifact is not a storage fixture"
    )

    _, bounded = _bounded_artifact(tmp_path / "bounded-case")
    missing = subprocess.run(
        [str(runner), "--model", str(bounded), "--expert", "1"],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 4
    assert missing.stderr.strip() == "TENSOR_NOT_FOUND: storage expert tensor missing"

    invalid = subprocess.run(
        [str(runner), "--model", str(bounded), "--iterations", "0"],
        capture_output=True,
        text=True,
    )
    assert invalid.returncode == 2
    assert invalid.stderr.strip() == "iterations must be positive"
