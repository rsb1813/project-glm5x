# GLM5X C++ 런타임용 고정 레코드 shard/tensor index를 생성합니다.

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import google_crc32c

from k3x_converter.format import K3XError

from .bundle import GLM5XExpertBundle


_HEADER = struct.Struct("<8sHHIIIIIQQQQQQQ32sII")
_ARTIFACT = struct.Struct("<QII32s16s")
_TENSOR = struct.Struct("<QIIII")


@dataclass(frozen=True)
class GLM5XRuntimeIndexReport:
    completed: bool
    output_path: Path
    artifact_count: int
    tensor_count: int
    file_bytes: int


def _validated_relative_path(value: str) -> bytes:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or "\x00" in value
    ):
        raise K3XError("GLM5X_RUNTIME_INDEX_PATH", value)
    encoded = value.encode("utf-8")
    if len(encoded) > 0xFFFFFFFF:
        raise K3XError("GLM5X_RUNTIME_INDEX_PATH", value)
    return encoded


def build_glm5x_runtime_index(
    bundle_path: str | Path,
    output_path: str | Path,
) -> GLM5XRuntimeIndexReport:
    bundle_path, output_path = Path(bundle_path), Path(output_path)
    bundle = GLM5XExpertBundle.open(
        bundle_path,
        verify_payloads=False,
        verify_root=False,
    )
    artifact_count = len(bundle.artifact_paths)
    if artifact_count == 0 or artifact_count > 0xFFFFFFFF:
        raise K3XError("GLM5X_RUNTIME_INDEX_ARTIFACT_COUNT")

    artifact_records = bytearray()
    string_table = bytearray()
    tensor_locators: list[tuple[int, int, int, int, int]] = []
    seen_tensor_ids: set[int] = set()
    for artifact_index, relative in enumerate(bundle.artifact_paths):
        encoded_path = _validated_relative_path(relative)
        reader = bundle.readers[relative]
        records = reader.tensor_records
        if len(records) > 0xFFFFFFFF:
            raise K3XError("GLM5X_RUNTIME_INDEX_TENSOR_COUNT", relative)
        path_offset = len(string_table)
        string_table.extend(encoded_path)
        root_sha256 = bytes(reader.superblock.root_sha256)
        if len(root_sha256) != 32:
            raise K3XError("GLM5X_RUNTIME_INDEX_ROOT", relative)
        artifact_records.extend(
            _ARTIFACT.pack(
                path_offset,
                len(encoded_path),
                len(records),
                root_sha256,
                bytes(16),
            )
        )
        for record_index, record in enumerate(records):
            if record.tensor_id in seen_tensor_ids:
                raise K3XError(
                    "GLM5X_RUNTIME_INDEX_DUPLICATE_TENSOR",
                    f"{record.tensor_id:016x}",
                )
            seen_tensor_ids.add(record.tensor_id)
            tensor_locators.append(
                (
                    record.tensor_id,
                    artifact_index,
                    record_index,
                    record.data_crc32c,
                    0,
                )
            )
    tensor_locators.sort()
    if len(tensor_locators) > 0xFFFFFFFFFFFFFFFF:
        raise K3XError("GLM5X_RUNTIME_INDEX_TENSOR_COUNT")
    tensor_records = b"".join(_TENSOR.pack(*item) for item in tensor_locators)

    artifact_offset = _HEADER.size
    artifact_length = len(artifact_records)
    tensor_offset = artifact_offset + artifact_length
    tensor_length = len(tensor_records)
    string_offset = tensor_offset + tensor_length
    string_length = len(string_table)
    body = bytes(artifact_records) + tensor_records + bytes(string_table)
    body_sha256 = hashlib.sha256(body).digest()
    header = _HEADER.pack(
        b"GLM5XIDX",
        1,
        0,
        _HEADER.size,
        _ARTIFACT.size,
        _TENSOR.size,
        artifact_count,
        0,
        len(tensor_locators),
        artifact_offset,
        artifact_length,
        tensor_offset,
        tensor_length,
        string_offset,
        string_length,
        body_sha256,
        0,
        0,
    )
    header_crc32c = google_crc32c.value(header[:124])
    header = header[:124] + struct.pack("<I", header_crc32c)
    payload = header + body

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, output_path)
    return GLM5XRuntimeIndexReport(
        completed=True,
        output_path=output_path,
        artifact_count=artifact_count,
        tensor_count=len(tensor_locators),
        file_bytes=len(payload),
    )

