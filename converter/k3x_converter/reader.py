# K3X 파일의 경계와 모든 계층 무결성을 검증해 엽니다.
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import google_crc32c

from .format import (
    EXPERT_RECORD_BYTES,
    LAYER_RECORD_BYTES,
    MODEL_CONFIG_BYTES,
    SUPERBLOCK_BYTES,
    TENSOR_RECORD_BYTES,
    ExpertRecord,
    K3XError,
    LayerRecord,
    Superblock,
    TensorRecord,
    decode_directory,
    root_sha256,
    validate_extent_layout,
)


def _read_exact(stream, offset: int, length: int, file_length: int) -> bytes:
    end = offset + length
    if end < offset or end > file_length:
        raise K3XError("TRUNCATED_FILE")
    stream.seek(offset)
    data = stream.read(length)
    if len(data) != length:
        raise K3XError("TRUNCATED_FILE")
    return data


@dataclass(frozen=True)
class K3XReader:
    path: Path
    superblock: Superblock
    tensor_records: tuple[TensorRecord, ...]
    layer_records: tuple[LayerRecord, ...]
    expert_records: tuple[ExpertRecord, ...]
    model_config: bytes

    @classmethod
    def open(cls, path: Path, *, verify_root: bool = True) -> "K3XReader":
        path = Path(path)
        actual_length = path.stat().st_size
        with path.open("rb") as stream:
            superblock = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
            if superblock.state != 1:
                raise K3XError("UNFINALIZED_FILE")
            if superblock.file_length != actual_length:
                raise K3XError("TRUNCATED_FILE")
            tensor_bytes = _read_exact(stream, superblock.tensor_directory_offset,
                                       superblock.tensor_directory_length, actual_length)
            layer_bytes = _read_exact(stream, superblock.layer_directory_offset,
                                      superblock.layer_directory_length, actual_length)
            expert_bytes = _read_exact(stream, superblock.expert_directory_offset,
                                       superblock.expert_directory_length, actual_length)
            if superblock.model_config_length != MODEL_CONFIG_BYTES:
                raise K3XError("INVALID_MODEL_CONFIG_LENGTH")
            config = _read_exact(stream, superblock.model_config_offset,
                                 superblock.model_config_length, actual_length)
            tensor_records = tuple(TensorRecord.decode(item) for item in
                                   decode_directory(tensor_bytes, b"TENS", TENSOR_RECORD_BYTES))
            layer_records = tuple(LayerRecord.decode(item) for item in
                                  decode_directory(layer_bytes, b"LAYR", LAYER_RECORD_BYTES))
            expert_records = tuple(ExpertRecord.decode(item) for item in
                                   decode_directory(expert_bytes, b"EXPT", EXPERT_RECORD_BYTES))
            validate_extent_layout(tensor_records, actual_length, 4096)
            for record in tensor_records:
                data = _read_exact(stream, record.data_offset, record.data_length, actual_length)
                if google_crc32c.value(data) != record.data_crc32c:
                    raise K3XError("DATA_CRC_MISMATCH")
                if record.auxiliary_length:
                    auxiliary = _read_exact(stream, record.auxiliary_offset,
                                            record.auxiliary_length, actual_length)
                    if google_crc32c.value(auxiliary) != record.auxiliary_crc32c:
                        raise K3XError("AUXILIARY_CRC_MISMATCH")
            directory_digest = hashlib.sha256(
                tensor_bytes + layer_bytes + expert_bytes + config
            ).digest()
            if directory_digest != superblock.directory_sha256:
                raise K3XError("DIRECTORY_SHA256_MISMATCH")
            if verify_root and root_sha256(stream, actual_length) != superblock.root_sha256:
                raise K3XError("ROOT_SHA256_MISMATCH")
        return cls(path, superblock, tensor_records, layer_records, expert_records, config)

    def read_tensor_extents(self, record: TensorRecord) -> tuple[bytes, bytes]:
        with self.path.open("rb") as stream:
            data = _read_exact(stream, record.data_offset, record.data_length,
                               self.superblock.file_length)
            auxiliary = (
                _read_exact(stream, record.auxiliary_offset, record.auxiliary_length,
                            self.superblock.file_length)
                if record.auxiliary_length else b""
            )
        return data, auxiliary
