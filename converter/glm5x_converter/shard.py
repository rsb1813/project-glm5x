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
from k3x_converter.resume import (
    CompletedExtent,
    ResumeManifest,
    read_resume_manifest,
    write_resume_manifest,
)
from k3x_converter.safetensors_reader import SourceTensor, inspect_shard, iter_tensor_chunks


_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
CONVERTER_VERSION = "glm5x-shard-converter-0.1.0"


@dataclass(frozen=True)
class GLM5XShardConversionReport:
    completed: bool
    output_path: Path
    sidecar_path: Path
    tensor_count: int
    tensor_ids: Mapping[str, int]
    maximum_source_read_bytes: int
    source_sha256: str
    reused_extent_ids: tuple[str, ...] = ()


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


def _crc_tensor(tensor: SourceTensor, chunk_bytes: int) -> int:
    checksum = google_crc32c.Checksum()
    for chunk in iter_tensor_chunks(tensor, chunk_bytes):
        checksum.update(chunk)
    return int.from_bytes(checksum.digest(), "big")


def _crc_at(stream, offset: int, length: int, chunk_bytes: int) -> int:
    checksum = google_crc32c.Checksum()
    stream.seek(offset)
    remaining = length
    while remaining:
        chunk = stream.read(min(chunk_bytes, remaining))
        if not chunk:
            raise K3XError("TRUNCATED_FILE")
        checksum.update(chunk)
        remaining -= len(chunk)
    return int.from_bytes(checksum.digest(), "big")


def _expected_extents(plans: list[_Plan], tensor_ids: Mapping[str, int]) -> list[tuple[str, SourceTensor]]:
    return [
        (f"{tensor_ids[plan.name]:016x}:data", plan.source)
        for plan in plans
    ]


def _validate_resume_extents(
    completed: tuple[CompletedExtent, ...],
    expected: list[tuple[str, SourceTensor]],
    chunk_bytes: int,
) -> None:
    if len(completed) > len(expected):
        raise K3XError("INVALID_RESUME_EXTENT")
    expected_offset = align_up(SUPERBLOCK_BYTES)
    for item, (extent_id, source_tensor) in zip(completed, expected):
        if (
            item.extent_id != extent_id
            or item.offset != expected_offset
            or item.length != source_tensor.length
            or item.length <= 0
        ):
            raise K3XError("INVALID_RESUME_EXTENT", item.extent_id)
        if item.crc32c != _crc_tensor(source_tensor, chunk_bytes):
            raise K3XError("RESUME_SOURCE_EXTENT_MISMATCH", item.extent_id)
        expected_offset = align_up(item.offset + item.length)


def _configuration_fingerprint(manifest: GLM5XTensorManifest, shard_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(_config_bytes(manifest))
    digest.update(shard_name.encode("utf-8"))
    return digest.hexdigest()


def _sidecar_data(
    plans: list[_Plan], tensor_ids: Mapping[str, int], shard_name: str, source_sha256: str
) -> dict[str, object]:
    return {
        "format": "glm5x-bounded-shard-v1",
        "source_shard": shard_name,
        "source_sha256": source_sha256,
        "tensor_count": len(plans),
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


def _write_sidecar(
    sidecar: Path,
    plans: list[_Plan],
    tensor_ids: Mapping[str, int],
    shard_name: str,
    source_sha256: str,
) -> None:
    sidecar_partial = sidecar.with_suffix(sidecar.suffix + ".partial")
    sidecar_partial.write_text(
        json.dumps(
            _sidecar_data(plans, tensor_ids, shard_name, source_sha256),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    with sidecar_partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(sidecar_partial, sidecar)


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
    plans: list[_Plan], manifest: GLM5XTensorManifest, tensor_ids: Mapping[str, int]
) -> tuple[list[LayerRecord], list[ExpertRecord]]:
    layers: list[LayerRecord] = []
    experts: list[ExpertRecord] = []
    for layer_id in range(manifest.descriptor.hidden_layers):
        indices = [index for index, plan in enumerate(plans) if plan.layer_id == layer_id]
        has_expert = any(plans[index].expert_id >= 0 for index in indices)
        first_expert = len(experts)
        grouped: dict[int, dict[str, _Plan]] = {}
        for index in indices:
            plan = plans[index]
            match = _EXPERT_RE.match(plan.name)
            if match is not None:
                grouped.setdefault(int(match.group(2)), {})[match.group(3)] = plan
        for expert_id in sorted(grouped):
            roles = grouped[expert_id]
            if {"gate_proj", "up_proj", "down_proj"}.issubset(roles):
                experts.append(
                    ExpertRecord(
                        layer_id,
                        expert_id,
                        len(experts),
                        0,
                        tensor_ids[roles["gate_proj"].name],
                        tensor_ids[roles["up_proj"].name],
                        tensor_ids[roles["down_proj"].name],
                    )
                )
        layers.append(
            LayerRecord(
                layer_id,
                1,
                2 if has_expert else 1,
                min(indices) if indices else 0,
                len(indices),
                first_expert,
                len(experts) - first_expert,
                0,
            )
        )
    return layers, experts


def convert_glm5x_shard(
    source: str | Path,
    output: str | Path,
    manifest: GLM5XTensorManifest,
    shard_name: str,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    dry_run: bool = False,
    stop_after_tensors: int | None = None,
) -> GLM5XShardConversionReport:
    source, output = Path(source), Path(output)
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    if stop_after_tensors is not None and stop_after_tensors <= 0:
        raise K3XError("INVALID_STOP_LIMIT")
    if not source.is_file():
        raise K3XError("SOURCE_SHARD_NOT_FOUND", str(source))
    manifest.validate_safetensors_shard(source, shard_name)
    plans = _plans(source, manifest, shard_name)
    source_sha256, maximum_read = _source_sha256(source, chunk_bytes)
    tensor_ids = {plan.name: fnv1a64(plan.name) for plan in plans}
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    partial = output.with_suffix(output.suffix + ".partial")
    resume_path = output.with_suffix(output.suffix + ".resume.json")
    if dry_run:
        return GLM5XShardConversionReport(
            False, output, sidecar, len(plans), tensor_ids, maximum_read, source_sha256, ()
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    config_bytes = _config_bytes(manifest)
    config_fingerprint = _configuration_fingerprint(manifest, shard_name)
    expected = _expected_extents(plans, tensor_ids)
    completed: list[CompletedExtent] = []
    reused: list[str] = []
    if output.exists() and not resume_path.exists():
        raise K3XError("OUTPUT_EXISTS", str(output))

    if resume_path.exists():
        try:
            ledger = read_resume_manifest(resume_path)
            file_uuid = bytes.fromhex(ledger.file_uuid)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise K3XError("INVALID_RESUME_MANIFEST") from exc
        if (
            ledger.source_fingerprint != source_sha256
            or ledger.converter_version != CONVERTER_VERSION
            or ledger.configuration_fingerprint != config_fingerprint
        ):
            raise K3XError("RESUME_CONFIGURATION_MISMATCH")
        if len(file_uuid) != 16:
            raise K3XError("INVALID_RESUME_MANIFEST")
        _validate_resume_extents(ledger.completed, expected, chunk_bytes)
        if partial.exists() and output.exists():
            raise K3XError("AMBIGUOUS_RESUME_STATE")
        if not partial.exists():
            if not output.exists():
                raise K3XError("MISSING_PARTIAL_FILE")
            from k3x_converter.reader import K3XReader

            finalized = K3XReader.open(output)
            if (
                finalized.superblock.source_sha256 != bytes.fromhex(source_sha256)
                or finalized.superblock.file_uuid != file_uuid
                or finalized.model_config != config_bytes
                or len(finalized.tensor_records) != len(plans)
            ):
                raise K3XError("FINAL_ARTIFACT_MISMATCH")
            _write_sidecar(sidecar, plans, tensor_ids, shard_name, source_sha256)
            resume_path.unlink()
            return GLM5XShardConversionReport(
                True,
                output,
                sidecar,
                len(plans),
                tensor_ids,
                maximum_read,
                source_sha256,
                tuple(item.extent_id for item in ledger.completed),
            )
        with partial.open("r+b") as stream:
            expected_end = align_up(SUPERBLOCK_BYTES)
            required_end = SUPERBLOCK_BYTES
            for item in ledger.completed:
                if _crc_at(stream, item.offset, item.length, chunk_bytes) != item.crc32c:
                    raise K3XError("RESUME_EXTENT_CRC_MISMATCH", item.extent_id)
                completed.append(item)
                reused.append(item.extent_id)
                required_end = item.offset + item.length
                expected_end = align_up(item.offset + item.length)
            stream.seek(0, os.SEEK_END)
            if stream.tell() < required_end:
                raise K3XError("TRUNCATED_FILE")
            stream.truncate(expected_end)
    else:
        file_uuid = uuid.uuid4().bytes
        with partial.open("wb") as stream:
            stream.write(bytes(SUPERBLOCK_BYTES))
            stream.flush()
            os.fsync(stream.fileno())
        write_resume_manifest(
            resume_path,
            ResumeManifest(
                source_sha256,
                CONVERTER_VERSION,
                config_fingerprint,
                file_uuid.hex(),
                (),
            ),
        )

    completed_map = {item.extent_id: item for item in completed}
    records: list[TensorRecord] = []
    newly_written = 0
    with partial.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        for plan in plans:
            extent_id = f"{tensor_ids[plan.name]:016x}:data"
            if extent_id in completed_map:
                item = completed_map[extent_id]
                offset, length, crc = item.offset, item.length, item.crc32c
            else:
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
                stream.flush()
                os.fsync(stream.fileno())
                crc = int.from_bytes(checksum.digest(), "big")
                if _crc_at(stream, offset, length, chunk_bytes) != crc:
                    raise K3XError("EXTENT_READBACK_MISMATCH", plan.name)
                item = CompletedExtent(extent_id, offset, length, crc)
                completed.append(item)
                completed_map[extent_id] = item
                write_resume_manifest(
                    resume_path,
                    ResumeManifest(
                        source_sha256,
                        CONVERTER_VERSION,
                        config_fingerprint,
                        file_uuid.hex(),
                        tuple(completed),
                    ),
                )
                newly_written += 1
                if stop_after_tensors is not None and newly_written >= stop_after_tensors:
                    return GLM5XShardConversionReport(
                        False,
                        output,
                        sidecar,
                        len(plans),
                        tensor_ids,
                        maximum_read,
                        source_sha256,
                        tuple(reused),
                    )
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
        layers, experts = _directories(plans, manifest, tensor_ids)
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
    _write_sidecar(sidecar, plans, tensor_ids, shard_name, source_sha256)
    resume_path.unlink()
    return GLM5XShardConversionReport(
        True,
        output,
        sidecar,
        len(records),
        tensor_ids,
        maximum_read,
        source_sha256,
        tuple(reused),
    )
