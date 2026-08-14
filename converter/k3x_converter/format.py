# K3X v1 바이너리 레이아웃과 무결성 계약을 정의합니다.
from __future__ import annotations

import dataclasses
import enum
import hashlib
import struct
from dataclasses import dataclass
from typing import Iterable

import google_crc32c

SUPERBLOCK_BYTES = 4096
EXTENT_ALIGNMENT = 4096
TENSOR_RECORD_BYTES = 128
LAYER_RECORD_BYTES = 64
EXPERT_RECORD_BYTES = 64
MODEL_CONFIG_BYTES = 256
MAGIC = b"K3XCHKPT"
SUPPORTED_REQUIRED_FEATURES = 0
OPTIONAL_STORAGE_FIXTURE = 1 << 0


class K3XError(ValueError):
    """검증 가능한 안정적 오류 코드를 포함하는 K3X 예외입니다."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class DType(enum.IntEnum):
    FP32 = 1
    UINT8 = 2


class Quantization(enum.IntEnum):
    NONE = 0
    MXFP4 = 1


def align_up(value: int, alignment: int = EXTENT_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise K3XError("INVALID_ALIGNMENT")
    return (value + alignment - 1) & ~(alignment - 1)


def fnv1a64(value: str) -> int:
    result = 0xCBF29CE484222325
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return result


@dataclass(frozen=True)
class Superblock:
    source_sha256: bytes
    file_uuid: bytes
    state: int = 0
    required_features: int = 0
    optional_features: int = 0
    tensor_directory_offset: int = 0
    tensor_directory_length: int = 0
    layer_directory_offset: int = 0
    layer_directory_length: int = 0
    expert_directory_offset: int = 0
    expert_directory_length: int = 0
    model_config_offset: int = 0
    model_config_length: int = 0
    payload_offset: int = SUPERBLOCK_BYTES
    file_length: int = 0
    directory_sha256: bytes = bytes(32)
    root_sha256: bytes = bytes(32)

    @classmethod
    def empty(cls, source_sha256: bytes, file_uuid: bytes) -> "Superblock":
        return cls(source_sha256=source_sha256, file_uuid=file_uuid)

    def encode(self) -> bytes:
        if len(self.source_sha256) != 32 or len(self.file_uuid) != 16:
            raise K3XError("INVALID_SUPERBLOCK_IDENTITY")
        if len(self.directory_sha256) != 32 or len(self.root_sha256) != 32:
            raise K3XError("INVALID_SUPERBLOCK_DIGEST")
        block = bytearray(SUPERBLOCK_BYTES)
        struct.pack_into("<8sHHIIIQQ", block, 0, MAGIC, 1, 0, SUPERBLOCK_BYTES,
                         EXTENT_ALIGNMENT, self.state, self.required_features,
                         self.optional_features)
        block[40:56] = self.file_uuid
        block[56:88] = self.source_sha256
        struct.pack_into(
            "<10Q", block, 88,
            self.tensor_directory_offset, self.tensor_directory_length,
            self.layer_directory_offset, self.layer_directory_length,
            self.expert_directory_offset, self.expert_directory_length,
            self.model_config_offset, self.model_config_length,
            self.payload_offset, self.file_length,
        )
        block[168:200] = self.directory_sha256
        block[200:232] = self.root_sha256
        struct.pack_into("<I", block, 4092, google_crc32c.value(bytes(block[:4092])))
        return bytes(block)

    @classmethod
    def decode(cls, data: bytes) -> "Superblock":
        if len(data) != SUPERBLOCK_BYTES:
            raise K3XError("TRUNCATED_SUPERBLOCK")
        if data[:8] != MAGIC:
            raise K3XError("BAD_MAGIC")
        major, minor, block_size, alignment, state = struct.unpack_from("<HHIII", data, 8)
        if major != 1 or minor != 0:
            raise K3XError("UNSUPPORTED_VERSION")
        if block_size != SUPERBLOCK_BYTES or alignment != EXTENT_ALIGNMENT:
            raise K3XError("INVALID_SUPERBLOCK_LAYOUT")
        expected_crc = struct.unpack_from("<I", data, 4092)[0]
        if google_crc32c.value(data[:4092]) != expected_crc:
            raise K3XError("SUPERBLOCK_CRC_MISMATCH")
        if any(data[232:4092]):
            raise K3XError("NONZERO_RESERVED_BYTES")
        required, optional = struct.unpack_from("<QQ", data, 24)
        if required & ~SUPPORTED_REQUIRED_FEATURES:
            raise K3XError("UNSUPPORTED_REQUIRED_FEATURE")
        values = struct.unpack_from("<10Q", data, 88)
        return cls(
            source_sha256=data[56:88], file_uuid=data[40:56], state=state,
            required_features=required, optional_features=optional,
            tensor_directory_offset=values[0], tensor_directory_length=values[1],
            layer_directory_offset=values[2], layer_directory_length=values[3],
            expert_directory_offset=values[4], expert_directory_length=values[5],
            model_config_offset=values[6], model_config_length=values[7],
            payload_offset=values[8], file_length=values[9],
            directory_sha256=data[168:200], root_sha256=data[200:232],
        )


@dataclass(frozen=True)
class TensorRecord:
    tensor_id: int
    role: int
    dtype: DType
    quantization: Quantization
    dimensions: tuple[int, ...]
    layer_id: int
    expert_id: int
    data_offset: int
    data_length: int
    logical_length: int
    auxiliary_offset: int
    auxiliary_length: int
    data_crc32c: int
    auxiliary_crc32c: int
    flags: int = 0

    def replace(self, **changes: object) -> "TensorRecord":
        return dataclasses.replace(self, **changes)

    def encode(self) -> bytes:
        if len(self.dimensions) > 4:
            raise K3XError("INVALID_TENSOR_RANK")
        dims = self.dimensions + (0,) * (4 - len(self.dimensions))
        record = bytearray(TENSOR_RECORD_BYTES)
        struct.pack_into("<QIHHBBHiiI", record, 0, self.tensor_id, self.role,
                         int(self.dtype), int(self.quantization), len(self.dimensions),
                         self.flags, 0, self.layer_id, self.expert_id, 0)
        struct.pack_into("<4Q5QII", record, 32, *dims, self.data_offset,
                         self.data_length, self.logical_length, self.auxiliary_offset,
                         self.auxiliary_length, self.data_crc32c, self.auxiliary_crc32c)
        return bytes(record)

    @classmethod
    def decode(cls, data: bytes) -> "TensorRecord":
        if len(data) != TENSOR_RECORD_BYTES:
            raise K3XError("INVALID_TENSOR_RECORD")
        head = struct.unpack_from("<QIHHBBHiiI", data, 0)
        if head[1]:
            raise K3XError("UNSUPPORTED_TENSOR_ROLE")
        if head[5]:
            raise K3XError("UNSUPPORTED_TENSOR_FLAGS")
        if head[6] or head[9] or any(data[112:]):
            raise K3XError("NONZERO_RESERVED_BYTES")
        if head[4] > 4:
            raise K3XError("INVALID_TENSOR_RANK")
        values = struct.unpack_from("<4Q5QII", data, 32)
        dims = values[:4]
        if any(dims[head[4]:]):
            raise K3XError("NONZERO_UNUSED_DIMENSION")
        try:
            dtype = DType(head[2])
            quantization = Quantization(head[3])
        except ValueError as error:
            raise K3XError("INVALID_TENSOR_ENUM") from error
        return cls(head[0], head[1], dtype, quantization, tuple(dims[:head[4]]),
                   head[7], head[8], values[4], values[5], values[6], values[7],
                   values[8], values[9], values[10], head[5])


@dataclass(frozen=True)
class LayerRecord:
    layer_index: int
    attention_kind: int
    ffn_kind: int
    first_tensor: int
    tensor_count: int
    first_expert: int
    expert_count: int
    residual_write_index: int
    flags: int = 0

    def encode(self) -> bytes:
        record = bytearray(LAYER_RECORD_BYTES)
        struct.pack_into("<IHHIIIIiI", record, 0, self.layer_index, self.attention_kind,
                         self.ffn_kind, self.first_tensor, self.tensor_count,
                         self.first_expert, self.expert_count, self.residual_write_index,
                         self.flags)
        return bytes(record)

    @classmethod
    def decode(cls, data: bytes) -> "LayerRecord":
        if len(data) != LAYER_RECORD_BYTES or any(data[32:]):
            raise K3XError("INVALID_LAYER_RECORD")
        values = struct.unpack_from("<IHHIIIIiI", data, 0)
        if values[1] not in (1, 2) or values[2] not in (1, 2):
            raise K3XError("INVALID_LAYER_ENUM")
        if values[8]:
            raise K3XError("UNSUPPORTED_LAYER_FLAGS")
        return cls(*values)


@dataclass(frozen=True)
class ExpertRecord:
    layer_index: int
    expert_id: int
    physical_order: int
    flags: int
    gate_tensor_id: int
    up_tensor_id: int
    down_tensor_id: int
    profile_frequency_q32: int = 0

    def encode(self) -> bytes:
        record = bytearray(EXPERT_RECORD_BYTES)
        struct.pack_into("<IIIIQQQQ", record, 0, self.layer_index, self.expert_id,
                         self.physical_order, self.flags, self.gate_tensor_id,
                         self.up_tensor_id, self.down_tensor_id,
                         self.profile_frequency_q32)
        return bytes(record)

    @classmethod
    def decode(cls, data: bytes) -> "ExpertRecord":
        if len(data) != EXPERT_RECORD_BYTES or any(data[48:]):
            raise K3XError("INVALID_EXPERT_RECORD")
        values = struct.unpack_from("<IIIIQQQQ", data, 0)
        if values[3]:
            raise K3XError("UNSUPPORTED_EXPERT_FLAGS")
        return cls(*values)


def encode_directory(tag: bytes, record_size: int, records: Iterable[bytes]) -> bytes:
    items = tuple(records)
    if len(tag) != 4 or any(len(item) != record_size for item in items):
        raise K3XError("INVALID_DIRECTORY")
    return struct.pack("<4sIQ", tag, record_size, len(items)) + b"".join(items)


def decode_directory(data: bytes, tag: bytes, record_size: int) -> tuple[bytes, ...]:
    if len(data) < 16:
        raise K3XError("TRUNCATED_DIRECTORY")
    actual_tag, actual_size, count = struct.unpack_from("<4sIQ", data, 0)
    if actual_tag != tag or actual_size != record_size:
        raise K3XError("INVALID_DIRECTORY_HEADER")
    if count > (len(data) - 16) // record_size or len(data) != 16 + count * record_size:
        raise K3XError("INVALID_DIRECTORY_LENGTH")
    return tuple(data[16 + i * record_size:16 + (i + 1) * record_size] for i in range(count))


def validate_extent_layout(
    records: Iterable[TensorRecord], file_length: int, alignment: int
) -> None:
    ranges: list[tuple[int, int]] = []
    for record in records:
        for offset, length in ((record.data_offset, record.data_length),
                               (record.auxiliary_offset, record.auxiliary_length)):
            if not length:
                if offset:
                    raise K3XError("INVALID_EMPTY_EXTENT")
                continue
            if offset % alignment:
                raise K3XError("UNALIGNED_EXTENT")
            end = offset + length
            if end < offset or end > file_length:
                raise K3XError("TRUNCATED_FILE")
            ranges.append((offset, end))
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise K3XError("OVERLAPPING_EXTENT")


def root_sha256(stream, file_length: int, chunk_bytes: int = 1024 * 1024) -> bytes:
    digest = hashlib.sha256()
    stream.seek(0)
    position = 0
    while position < file_length:
        chunk = bytearray(stream.read(min(chunk_bytes, file_length - position)))
        if not chunk:
            raise K3XError("TRUNCATED_FILE")
        for start, end in ((200, 232), (4092, 4096)):
            left = max(start - position, 0)
            right = min(end - position, len(chunk))
            if left < right:
                chunk[left:right] = bytes(right - left)
        digest.update(chunk)
        position += len(chunk)
    return digest.digest()
