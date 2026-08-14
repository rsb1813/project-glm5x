# 실제 K3 차원의 bounded expert source fixture를 검증합니다.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from k3x_converter.format import K3XError
from k3x_converter.reader import K3XReader
from k3x_converter.safetensors_reader import inspect_shard
from k3x_converter.writer import convert
import k3x_ref.storage_fixture as storage_fixture
from k3x_ref.storage_fixture import write_bounded_expert_source


GATE = "model.layers.1.feed_forward.experts.0.gate"
UP = "model.layers.1.feed_forward.experts.0.up"
DOWN = "model.layers.1.feed_forward.experts.0.down"


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_edges(path: Path, offset: int, length: int) -> tuple[bytes, bytes]:
    with path.open("rb") as stream:
        stream.seek(offset)
        first = stream.read(16)
        stream.seek(offset + length - 16)
        last = stream.read(16)
    return first, last


def test_bounded_expert_source_materializes_exact_extents_with_bounded_chunks(
    tmp_path: Path,
) -> None:
    first = write_bounded_expert_source(
        tmp_path / "first", chunk_bytes=257 * 1024
    )
    second = write_bounded_expert_source(
        tmp_path / "second", chunk_bytes=257 * 1024
    )

    assert first.payload_bytes == 17_547_264
    assert first.maximum_chunk_bytes <= 257 * 1024
    assert first.source_sha256 == second.source_sha256
    assert _file_sha256(first.shard_path) == _file_sha256(second.shard_path)
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()

    tensors = inspect_shard(first.shard_path)
    assert len(tensors) == 6
    assert tensors[f"{GATE}.weight_packed"].shape == (5_505_024,)
    assert tensors[f"{GATE}.weight_scale"].shape == (344_064,)
    assert tensors[f"{UP}.weight_packed"].shape == (5_505_024,)
    assert tensors[f"{UP}.weight_scale"].shape == (344_064,)
    assert tensors[f"{DOWN}.weight_packed"].shape == (5_505_024,)
    assert tensors[f"{DOWN}.weight_scale"].shape == (344_064,)

    gate = tensors[f"{GATE}.weight_packed"]
    gate_first, gate_last = _read_edges(gate.path, gate.offset, gate.length)
    assert gate_first == bytes(range(201, 217))
    assert gate_last == bytes(range(185, 201))
    gate_scale = tensors[f"{GATE}.weight_scale"]
    assert _read_edges(gate_scale.path, gate_scale.offset, gate_scale.length) == (
        bytes([120]) * 16,
        bytes([120]) * 16,
    )

    ranges = sorted((item.offset, item.offset + item.length) for item in tensors.values())
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert ranges[-1][1] == first.shard_path.stat().st_size

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "k3-storage-slice-v1"
    assert manifest["artifact_kind"] == "storage_fixture"
    assert manifest["config"]["hidden_size"] == 7168
    assert manifest["config"]["num_experts"] == 896
    assert manifest["config"]["top_k"] == 16
    assert manifest["packed_shapes"][GATE] == [3072, 3584]
    assert manifest["packed_shapes"][DOWN] == [3584, 3072]
    assert set(manifest["tensor_sha256"]) == set(tensors)


def test_bounded_expert_source_rejects_invalid_arguments_before_finalizing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="chunk_bytes"):
        write_bounded_expert_source(output, chunk_bytes=0)
    assert not (output / "bounded-expert.safetensors").exists()
    assert not (output / "source-manifest.json").exists()

    with pytest.raises(ValueError, match="layer_id"):
        write_bounded_expert_source(output, layer_id=0)
    with pytest.raises(ValueError, match="expert_id"):
        write_bounded_expert_source(output, expert_id=896)


def test_storage_fixture_conversion_rejects_mutated_shard(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    report = write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    with report.shard_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(K3XError, match="SOURCE_SHARD_SHA256_MISMATCH"):
        convert(source, tmp_path / "mutated.k3x", chunk_bytes=193 * 1024)


def test_storage_fixture_conversion_rejects_tensor_digest_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    report = write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    manifest["tensor_sha256"][f"{GATE}.weight_packed"] = "0" * 64
    report.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(K3XError, match="SOURCE_TENSOR_SHA256_MISMATCH"):
        convert(source, tmp_path / "bad-tensor.k3x", chunk_bytes=193 * 1024)


def test_manifest_publish_failure_keeps_previous_fixture_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    first = write_bounded_expert_source(source, seed=1, chunk_bytes=257 * 1024)
    original_manifest = first.manifest_path.read_bytes()

    def fail_manifest_publish(path: Path, value: dict[str, object]) -> None:
        raise RuntimeError("simulated manifest publish failure")

    monkeypatch.setattr(storage_fixture, "_write_json_atomic", fail_manifest_publish)
    with pytest.raises(RuntimeError, match="simulated manifest publish failure"):
        write_bounded_expert_source(source, seed=2, chunk_bytes=257 * 1024)

    assert first.manifest_path.read_bytes() == original_manifest
    manifest = json.loads(original_manifest)
    referenced = source / next(iter(manifest["weight_map"].values()))
    assert _file_sha256(referenced) == manifest["source_sha256"]


def test_unreferenced_shard_does_not_change_source_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    first = tmp_path / "first.k3x"
    second = tmp_path / "second.k3x"
    convert(source, first, chunk_bytes=193 * 1024)

    (source / "unreferenced.safetensors").write_bytes(b"not a source shard")
    convert(source, second, chunk_bytes=193 * 1024)

    assert K3XReader.open(first).superblock.source_sha256 == (
        K3XReader.open(second).superblock.source_sha256
    )
