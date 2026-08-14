# K3X 고정 레이아웃과 corruption rejection을 검증합니다.
import os
import json
import struct
from pathlib import Path

import google_crc32c
import pytest

from k3x_converter.format import (
    OPTIONAL_STORAGE_FIXTURE,
    SUPERBLOCK_BYTES,
    DType,
    ExpertRecord,
    K3XError,
    LayerRecord,
    Quantization,
    Superblock,
    TensorRecord,
    fnv1a64,
    validate_extent_layout,
)
from k3x_converter.reader import K3XReader
from k3x_converter.safetensors_reader import inspect_shard, iter_tensor_chunks
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def test_superblock_has_literal_offsets_and_zero_reserved_bytes() -> None:
    block = Superblock.empty(source_sha256=bytes(range(32)), file_uuid=bytes(range(16)))
    encoded = block.encode()
    assert len(encoded) == SUPERBLOCK_BYTES
    assert encoded[:8] == b"K3XCHKPT"
    assert struct.unpack_from("<HHII", encoded, 8) == (1, 0, 4096, 4096)
    assert encoded[232:4092] == bytes(4092 - 232)
    assert google_crc32c.value(encoded[:4092]) == struct.unpack_from("<I", encoded, 4092)[0]
    assert Superblock.decode(encoded) == block


def test_tensor_record_is_exactly_128_bytes() -> None:
    record = TensorRecord(
        tensor_id=7,
        role=0,
        dtype=DType.UINT8,
        quantization=Quantization.MXFP4,
        dimensions=(32, 32),
        layer_id=2,
        expert_id=3,
        data_offset=4096,
        data_length=512,
        logical_length=4096,
        auxiliary_offset=8192,
        auxiliary_length=32,
        data_crc32c=1,
        auxiliary_crc32c=2,
    )
    encoded = record.encode()
    assert len(encoded) == 128
    assert encoded[112:] == bytes(16)
    assert TensorRecord.decode(encoded) == record


def test_directory_records_reject_unsupported_enums_and_flags() -> None:
    layer = bytearray(LayerRecord(0, 1, 1, 0, 0, 0, 0, 0).encode())
    struct.pack_into("<H", layer, 4, 99)
    with pytest.raises(K3XError, match="INVALID_LAYER_ENUM"):
        LayerRecord.decode(bytes(layer))

    expert = bytearray(ExpertRecord(1, 2, 0, 0, 3, 4, 5).encode())
    struct.pack_into("<I", expert, 12, 1)
    with pytest.raises(K3XError, match="UNSUPPORTED_EXPERT_FLAGS"):
        ExpertRecord.decode(bytes(expert))

    tensor = bytearray(TensorRecord(
        1, 0, DType.FP32, Quantization.NONE, (1,), -1, -1,
        4096, 4, 4, 0, 0, 0, 0,
    ).encode())
    struct.pack_into("<I", tensor, 8, 1)
    with pytest.raises(K3XError, match="UNSUPPORTED_TENSOR_ROLE"):
        TensorRecord.decode(bytes(tensor))


def test_extent_validation_rejects_unaligned_overlap_and_truncation() -> None:
    base = TensorRecord(1, 0, DType.UINT8, Quantization.NONE, (32,), -1, -1, 4096, 32, 32, 0, 0, 0, 0)
    with pytest.raises(K3XError, match="UNALIGNED_EXTENT"):
        validate_extent_layout((base.replace(data_offset=4097),), 16384, 4096)
    with pytest.raises(K3XError, match="OVERLAPPING_EXTENT"):
        validate_extent_layout((base, base.replace(tensor_id=2)), 16384, 4096)
    with pytest.raises(K3XError, match="TRUNCATED_FILE"):
        validate_extent_layout((base.replace(data_offset=16384),), 16384, 4096)


def test_reader_rejects_payload_corruption(synthetic_source: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    reader = K3XReader.open(artifact)
    first = reader.tensor_records[0]
    with artifact.open("r+b") as stream:
        stream.seek(first.data_offset)
        original = stream.read(1)
        stream.seek(first.data_offset)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(K3XError, match="DATA_CRC_MISMATCH"):
        K3XReader.open(artifact)


def test_reader_rejects_unknown_required_feature(synthetic_source: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    with artifact.open("r+b") as stream:
        block = bytearray(stream.read(SUPERBLOCK_BYTES))
        struct.pack_into("<Q", block, 24, 1)
        struct.pack_into("<I", block, 4092, google_crc32c.value(bytes(block[:4092])))
        stream.seek(0)
        stream.write(block)
    with pytest.raises(K3XError, match="UNSUPPORTED_REQUIRED_FEATURE"):
        K3XReader.open(artifact, verify_root=False)


def test_reader_rejects_truncated_file(synthetic_source: Path, tmp_path: Path) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    os.truncate(artifact, artifact.stat().st_size - 1)
    with pytest.raises(K3XError, match="TRUNCATED_FILE"):
        K3XReader.open(artifact)


def test_native_mxfp4_payload_round_trips_byte_for_byte(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    reader = K3XReader.open(artifact)
    record = next(
        item for item in reader.tensor_records if item.quantization == Quantization.MXFP4
    )
    manifest = json.loads(
        (synthetic_source / "source-manifest.json").read_text(encoding="utf-8")
    )
    base = next(
        name.removesuffix(".weight_packed")
        for name in sorted(manifest["weight_map"])
        if name.endswith(".weight_packed") and fnv1a64(name.removesuffix(".weight_packed")) == record.tensor_id
    )
    packed_name, scale_name = base + ".weight_packed", base + ".weight_scale"
    packed = inspect_shard(synthetic_source / manifest["weight_map"][packed_name])[packed_name]
    scale = inspect_shard(synthetic_source / manifest["weight_map"][scale_name])[scale_name]
    actual_data, actual_auxiliary = reader.read_tensor_extents(record)
    assert actual_data == b"".join(iter_tensor_chunks(packed, 257))
    assert actual_auxiliary == b"".join(iter_tensor_chunks(scale, 257))


def test_full_dimension_storage_fixture_round_trips_with_optional_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"

    report = convert(source, artifact, chunk_bytes=193 * 1024)
    reader = K3XReader.open(artifact)

    assert report.completed is True
    assert report.maximum_source_read_bytes <= 193 * 1024
    assert reader.superblock.optional_features == OPTIONAL_STORAGE_FIXTURE
    assert len(reader.tensor_records) == 3
    assert len(reader.expert_records) == 1
    assert sum(
        record.data_length + record.auxiliary_length
        for record in reader.tensor_records
    ) == 17_547_264
    assert {
        record.dimensions for record in reader.tensor_records
    } == {(3072, 3584), (3584, 3072)}
    assert all(
        record.quantization == Quantization.MXFP4
        for record in reader.tensor_records
    )
    base = "model.layers.1.feed_forward.experts.0"
    by_id = {record.tensor_id: record for record in reader.tensor_records}
    execution_order = [
        by_id[fnv1a64(f"{base}.{role}")]
        for role in ("gate", "up", "down")
    ]
    physical_offsets = [
        offset
        for record in execution_order
        for offset in (record.data_offset, record.auxiliary_offset)
    ]
    assert physical_offsets == sorted(physical_offsets)


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    (
        ("artifact_kind", "INVALID_STORAGE_FIXTURE_KIND"),
        ("missing_shape", "INCOMPLETE_MXFP4_TENSOR"),
        ("wrong_shape", "INVALID_STORAGE_FIXTURE_SHAPE"),
        ("unsupported_format", "UNSUPPORTED_SOURCE_FORMAT"),
    ),
)
def test_storage_fixture_manifest_rejects_malformed_identity(
    tmp_path: Path, mutation: str, error_code: str
) -> None:
    source = tmp_path / mutation
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    manifest_path = source / "source-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    down = "model.layers.1.feed_forward.experts.0.down"
    if mutation == "artifact_kind":
        manifest["artifact_kind"] = "executable"
    elif mutation == "missing_shape":
        del manifest["packed_shapes"][down]
    elif mutation == "wrong_shape":
        manifest["packed_shapes"][down] = [3584, 3040]
    else:
        manifest["format"] = "unknown-source-v9"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(K3XError, match=error_code):
        convert(source, tmp_path / f"{mutation}.k3x", chunk_bytes=193 * 1024)
