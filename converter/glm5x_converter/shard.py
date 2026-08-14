# GLM-5.2 safetensors shard를 bounded K3X extent artifact로 스트리밍 변환합니다.

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import google_crc32c

from glm5x_ref.manifest import GLM5XTensorManifest
from k3x_converter.format import (
    SUPERBLOCK_BYTES,
    DType,
    ExpertRecord,
    K3XError,
    LayerRecord,
    Quantization,
    Superblock,
    TensorRecord,
    align_up,
    encode_directory,
    fnv1a64,
    root_sha256,
)
from k3x_converter.safetensors_reader import SourceTensor, inspect_shard, iter_tensor_chunks


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class GLM5XShardConversionReport:
    completed: bool
    output_path: Path
    sidecar_path: Path
    tensor_count: int
    tensor_ids: Mapping[str, int]
    maximum_source_read_bytes: int
    source_sha256: str


@dataclass(frozen=True)
class _Plan:
    name: str
    source: SourceTensor
    dtype: DType
    layer_id: int
    expert_id: int


def _dtype(value: str) -> DType:
    try:
        return {
            "BF16": DType.BF16,
            "F32": DType.FP32,
        }[value]
    except KeyError as exc:
        raise K3XError("UNSUPPORTED_GLM_DTYPE", value) from exc


def _layer_and_expert(name: str) -> tuple[int, int]:
    layer = _LAYER_RE.match(name)
    expert = _EXPERT_RE.match(name)
    return (
        int(layer.group(1)) if layer else -1,
        int(expert.group(2)) if expert else -1,
    )


def _config_bytes(manifest: GLM5XTensorManifest) -> bytes:
    descriptor = manifest.descriptor
    block = bytearray(256)
    struct.pack_into(
        "<12I",
        block,
        0,
        descriptor.vocab_size,
        descriptor.hidden_size,
        descriptor.hidden_layers,
        descriptor.top_k,
        descriptor.shared_experts,
        descriptor.mtp_layers,
        descriptor.moe_intermediate_size,
        descriptor.index_topk,
        descriptor.index_topk_freq,
        descriptor.index_n_heads,
        descriptor.index_head_dim,
        descriptor.max_position_embeddings,
    )
    struct.pack_into("<I", block, 48, descriptor.routed_experts)
    return bytes(block)


def _source_sha256(path: Path, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    maximum = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            maximum = max(maximum, len(chunk))
            digest.update(chunk)
    return digest.hexdigest(), maximum


def _plans(
    source: Path, manifest: GLM5XTensorManifest, shard_name: str
) -> list[_Plan]:
    expected = {
        name for name, mapped_shard in manifest.tensor_shards if mapped_shard == shard_name
    }
    tensors = inspect_shard(source)
    if set(tensors) != expected:
        raise K3XError("GLM5X_SHARD_MANIFEST_MISMATCH")
    plans = []
    for name in sorted(tensors, key=lambda value: _layer_and_expert(value) + (value,)):
        layer_id, expert_id = _layer_and_expert(name)
        plans.append(_Plan(name, tensors[name], _dtype(tensors[name].dtype), layer_id, expert_id))
    return plans


def _directories(
    plans: list[_Plan], manifest: GLM5XTensorManifest
) -> tuple[list[LayerRecord], list[ExpertRecord]]:
    layers: list[LayerRecord] = []
    for layer_id in range(manifest.descriptor.hidden_layers):
        indices = [index for index, plan in enumerate(plans) if plan.layer_id == layer_id]
        has_expert = any(plans[index].expert_id >= 0 for index in indices)
        layers.append(
            LayerRecord(
                layer_id,
                1,
                2 if has_expert else 1,
                min(indices) if indices else 0,
                len(indices),
                0,
                0,
                0,
            )
        )
    # Raw BF16 expert tensors are kept in the tensor directory; expert records are
    # intentionally deferred until an exact MXFP4 role bundle is available.
    return layers, []


def convert_glm5x_shard(
    source: str | Path,
    output: str | Path,
    manifest: GLM5XTensorManifest,
    shard_name: str,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    dry_run: bool = False,
) -> GLM5XShardConversionReport:
    source, output = Path(source), Path(output)
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    if output.exists() and not dry_run:
        raise K3XError("OUTPUT_EXISTS", str(output))
    if not source.is_file():
        raise K3XError("SOURCE_SHARD_NOT_FOUND", str(source))
    manifest.validate_safetensors_shard(source, shard_name)
    plans = _plans(source, manifest, shard_name)
    source_sha256, maximum_read = _source_sha256(source, chunk_bytes)
    tensor_ids = {plan.name: fnv1a64(plan.name) for plan in plans}
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if dry_run:
        return GLM5XShardConversionReport(
            False, output, sidecar, len(plans), tensor_ids, maximum_read, source_sha256
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    sidecar_partial = sidecar.with_suffix(sidecar.suffix + ".partial")
    for path in (partial, sidecar_partial):
        if path.exists():
            path.unlink()
    file_uuid = uuid.uuid4().bytes
    records: list[TensorRecord] = []
    with partial.open("w+b") as stream:
        stream.write(bytes(SUPERBLOCK_BYTES))
        for plan in plans:
            offset = align_up(stream.tell())
            if offset > stream.tell():
                stream.write(bytes(offset - stream.tell()))
            checksum = google_crc32c.Checksum()
            length = 0
            for chunk in iter_tensor_chunks(plan.source, chunk_bytes):
                maximum_read = max(maximum_read, len(chunk))
                stream.write(chunk)
                checksum.update(chunk)
                length += len(chunk)
            if length != plan.source.length:
                raise K3XError("GLM5X_SOURCE_LENGTH_MISMATCH", plan.name)
            crc = int.from_bytes(checksum.digest(), "big")
            records.append(
                TensorRecord(
                    tensor_ids[plan.name],
                    0,
                    plan.dtype,
                    Quantization.NONE,
                    plan.source.shape,
                    plan.layer_id,
                    plan.expert_id,
                    offset,
                    length,
                    length,
                    0,
                    0,
                    crc,
                    0,
                )
            )
        layers, experts = _directories(plans, manifest)
        directories = (
            encode_directory(b"TENS", 128, (record.encode() for record in records)),
            encode_directory(b"LAYR", 64, (record.encode() for record in layers)),
            encode_directory(b"EXPT", 64, (record.encode() for record in experts)),
            _config_bytes(manifest),
        )
        offsets: list[tuple[int, bytes]] = []
        for data in directories:
            offset = align_up(stream.tell())
            if offset > stream.tell():
                stream.write(bytes(offset - stream.tell()))
            stream.write(data)
            offsets.append((offset, data))
        file_length = stream.tell()
        directory_digest = hashlib.sha256(b"".join(data for _, data in offsets)).digest()
        block = Superblock(
            bytes.fromhex(source_sha256),
            file_uuid,
            state=1,
            tensor_directory_offset=offsets[0][0],
            tensor_directory_length=len(offsets[0][1]),
            layer_directory_offset=offsets[1][0],
            layer_directory_length=len(offsets[1][1]),
            expert_directory_offset=offsets[2][0],
            expert_directory_length=len(offsets[2][1]),
            model_config_offset=offsets[3][0],
            model_config_length=len(offsets[3][1]),
            file_length=file_length,
            directory_sha256=directory_digest,
        )
        stream.seek(0)
        stream.write(block.encode())
        stream.flush()
        os.fsync(stream.fileno())
        digest = root_sha256(stream, file_length)
        stream.seek(0)
        stream.write(replace(block, root_sha256=digest).encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, output)

    sidecar_data = {
        "format": "glm5x-bounded-shard-v1",
        "source_shard": shard_name,
        "source_sha256": source_sha256,
        "tensor_count": len(records),
        "tensors": [
            {
                "name": plan.name,
                "tensor_id": tensor_ids[plan.name],
                "dtype": plan.source.dtype,
                "shape": list(plan.source.shape),
                "layer_id": plan.layer_id,
                "expert_id": plan.expert_id,
            }
            for plan in plans
        ],
    }
    sidecar_partial.write_text(
        json.dumps(sidecar_data, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(sidecar_partial, sidecar)
    return GLM5XShardConversionReport(
        True, output, sidecar, len(records), tensor_ids, maximum_read, source_sha256
    )
