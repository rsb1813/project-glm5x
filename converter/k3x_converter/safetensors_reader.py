# safetensors shard를 전체 적재 없이 제한된 chunk로 읽습니다.
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .format import K3XError


@dataclass(frozen=True)
class SourceTensor:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


def inspect_shard(path: Path) -> dict[str, SourceTensor]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise K3XError("TRUNCATED_SOURCE_SHARD")
        header_length = struct.unpack("<Q", raw)[0]
        if header_length > size - 8:
            raise K3XError("INVALID_SOURCE_HEADER")
        try:
            header = json.loads(stream.read(header_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise K3XError("INVALID_SOURCE_HEADER") from error
    data_start = 8 + header_length
    result: dict[str, SourceTensor] = {}
    ranges: list[tuple[int, int]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        start, end = metadata["data_offsets"]
        absolute_start, absolute_end = data_start + start, data_start + end
        if start < 0 or end < start or absolute_end > size:
            raise K3XError("INVALID_SOURCE_EXTENT")
        ranges.append((absolute_start, absolute_end))
        result[name] = SourceTensor(path, name, metadata["dtype"],
                                    tuple(metadata["shape"]), absolute_start,
                                    absolute_end - absolute_start)
    ranges.sort()
    if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
        raise K3XError("OVERLAPPING_SOURCE_EXTENT")
    return result


def iter_tensor_chunks(tensor: SourceTensor, chunk_bytes: int) -> Iterator[bytes]:
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    with tensor.path.open("rb") as stream:
        stream.seek(tensor.offset)
        remaining = tensor.length
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise K3XError("TRUNCATED_SOURCE_SHARD")
            remaining -= len(chunk)
            yield chunk
