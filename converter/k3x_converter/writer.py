# 합성 shard를 K3X 실행 순서 extent로 스트리밍 변환합니다.
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import google_crc32c

from .format import (
    EXPERT_RECORD_BYTES,
    LAYER_RECORD_BYTES,
    MODEL_CONFIG_BYTES,
    OPTIONAL_STORAGE_FIXTURE,
    SUPERBLOCK_BYTES,
    TENSOR_RECORD_BYTES,
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
from .resume import CompletedExtent, ResumeManifest, read_resume_manifest, write_resume_manifest
from .safetensors_reader import SourceTensor, inspect_shard, iter_tensor_chunks

CONVERTER_VERSION = "k3x-converter-0.1.0"
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.(gate|up|down)$")


@dataclass(frozen=True)
class ConversionReport:
    completed: bool
    reused_extent_ids: tuple[str, ...]
    maximum_source_read_bytes: int
    output_path: Path


@dataclass(frozen=True)
class _TensorPlan:
    name: str
    data: SourceTensor
    auxiliary: SourceTensor | None
    dimensions: tuple[int, ...]
    dtype: DType
    quantization: Quantization
    layer_id: int
    expert_id: int


def _fingerprint_source(
    source: Path, manifest: dict, chunk_bytes: int
) -> tuple[bytes, int]:
    digest = hashlib.sha256()
    maximum = 0
    shard_names = sorted(set(manifest["weight_map"].values()))
    paths = [source / "source-manifest.json", *(source / name for name in shard_names)]
    for path in paths:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                maximum = max(maximum, len(chunk))
                digest.update(chunk)
    return digest.digest(), maximum


def _sha256_path(path: Path, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    maximum = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            maximum = max(maximum, len(chunk))
            digest.update(chunk)
    return digest.hexdigest(), maximum


def _sha256_tensor(tensor: SourceTensor, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    maximum = 0
    for chunk in iter_tensor_chunks(tensor, chunk_bytes):
        maximum = max(maximum, len(chunk))
        digest.update(chunk)
    return digest.hexdigest(), maximum


def _crc_tensor(tensor: SourceTensor, chunk_bytes: int) -> int:
    checksum = google_crc32c.Checksum()
    for chunk in iter_tensor_chunks(tensor, chunk_bytes):
        checksum.update(chunk)
    return int.from_bytes(checksum.digest(), "big")


def _load_plans(source: Path) -> tuple[dict, list[_TensorPlan]]:
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") not in (
        "synthetic-k3-source-v1",
        "k3-storage-slice-v1",
    ):
        raise K3XError("UNSUPPORTED_SOURCE_FORMAT")
    tensors: dict[str, SourceTensor] = {}
    for shard_name in sorted(set(manifest["weight_map"].values())):
        tensors.update(inspect_shard(source / shard_name))
    if set(tensors) != set(manifest["weight_map"]):
        raise K3XError("SOURCE_MANIFEST_MISMATCH")
    plans: list[_TensorPlan] = []
    consumed: set[str] = set()
    for name in sorted(tensors):
        if name in consumed or name.endswith(".weight_scale"):
            continue
        if name.endswith(".weight_packed"):
            base = name.removesuffix(".weight_packed")
            auxiliary_name = base + ".weight_scale"
            if auxiliary_name not in tensors or base not in manifest.get("packed_shapes", {}):
                raise K3XError("INCOMPLETE_MXFP4_TENSOR", base)
            canonical, auxiliary = base, tensors[auxiliary_name]
            dimensions = tuple(manifest["packed_shapes"][base])
            dtype, quantization = DType.UINT8, Quantization.MXFP4
            consumed.add(auxiliary_name)
        else:
            canonical, auxiliary = name, None
            dimensions = tensors[name].shape
            if tensors[name].dtype != "F32":
                raise K3XError("UNSUPPORTED_SOURCE_DTYPE", tensors[name].dtype)
            dtype, quantization = DType.FP32, Quantization.NONE
        layer_match = _LAYER_RE.match(canonical)
        expert_match = _EXPERT_RE.match(canonical)
        plans.append(_TensorPlan(canonical, tensors[name], auxiliary, dimensions, dtype,
                                 quantization,
                                 int(layer_match.group(1)) if layer_match else -1,
                                 int(expert_match.group(2)) if expert_match else -1))
        consumed.add(name)
    if manifest["format"] == "k3-storage-slice-v1":
        role_order = {"gate": 0, "up": 1, "down": 2}
        plans.sort(
            key=lambda plan: role_order.get(
                match.group(3) if (match := _EXPERT_RE.match(plan.name)) else "",
                len(role_order),
            )
        )
    ids = [fnv1a64(item.name) for item in plans]
    if len(ids) != len(set(ids)):
        raise K3XError("TENSOR_ID_COLLISION")
    if manifest["format"] == "k3-storage-slice-v1":
        _validate_storage_fixture(manifest, plans)
    return manifest, plans


def _validate_storage_fixture(manifest: dict, plans: list[_TensorPlan]) -> None:
    if manifest.get("artifact_kind") != "storage_fixture":
        raise K3XError("INVALID_STORAGE_FIXTURE_KIND")
    config = manifest.get("config", {})
    expected_config = {
        "hidden_size": 7168,
        "num_experts": 896,
        "top_k": 16,
        "routed_latent_size": 3584,
        "expert_intermediate_size": 3072,
        "mxfp4_group_size": 32,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise K3XError("INVALID_STORAGE_FIXTURE_CONFIG")
    if len(config.get("layer_kinds", ())) != 93:
        raise K3XError("INVALID_STORAGE_FIXTURE_CONFIG")
    layer_id = manifest.get("layer_id")
    expert_id = manifest.get("expert_id")
    if not isinstance(layer_id, int) or not 1 <= layer_id < 93:
        raise K3XError("INVALID_STORAGE_FIXTURE_IDENTITY")
    if not isinstance(expert_id, int) or not 0 <= expert_id < 896:
        raise K3XError("INVALID_STORAGE_FIXTURE_IDENTITY")
    base = f"model.layers.{layer_id}.feed_forward.experts.{expert_id}"
    expected_shapes = {
        f"{base}.gate": (3072, 3584),
        f"{base}.up": (3072, 3584),
        f"{base}.down": (3584, 3072),
    }
    if {plan.name for plan in plans} != set(expected_shapes):
        raise K3XError("INCOMPLETE_STORAGE_FIXTURE_EXPERT")
    for plan in plans:
        shape = expected_shapes[plan.name]
        values = shape[0] * shape[1]
        if (
            plan.dimensions != shape
            or plan.quantization != Quantization.MXFP4
            or plan.data.dtype != "U8"
            or plan.data.length != values // 2
            or plan.auxiliary is None
            or plan.auxiliary.dtype != "U8"
            or plan.auxiliary.length != values // 32
        ):
            raise K3XError("INVALID_STORAGE_FIXTURE_SHAPE", plan.name)
    if manifest.get("payload_bytes") != 17_547_264:
        raise K3XError("INVALID_STORAGE_FIXTURE_LENGTH")


def _validate_storage_fixture_hashes(
    manifest: dict,
    plans: list[_TensorPlan],
    source: Path,
    chunk_bytes: int,
) -> int:
    shard_names = set(manifest["weight_map"].values())
    if len(shard_names) != 1:
        raise K3XError("INVALID_STORAGE_FIXTURE_SHARD_SET")
    shard_digest, maximum = _sha256_path(source / next(iter(shard_names)), chunk_bytes)
    if manifest.get("source_sha256") != shard_digest:
        raise K3XError("SOURCE_SHARD_SHA256_MISMATCH")

    tensors: dict[str, SourceTensor] = {}
    for plan in plans:
        tensors[plan.data.name] = plan.data
        if plan.auxiliary is not None:
            tensors[plan.auxiliary.name] = plan.auxiliary
    expected = manifest.get("tensor_sha256")
    if not isinstance(expected, dict) or set(expected) != set(tensors):
        raise K3XError("SOURCE_TENSOR_SHA256_MISMATCH")
    for name, tensor in tensors.items():
        digest, observed = _sha256_tensor(tensor, chunk_bytes)
        maximum = max(maximum, observed)
        if expected.get(name) != digest:
            raise K3XError("SOURCE_TENSOR_SHA256_MISMATCH", name)
    return maximum


def _expected_extents(plans: list[_TensorPlan]) -> list[tuple[str, SourceTensor]]:
    result: list[tuple[str, SourceTensor]] = []
    for plan in plans:
        for suffix, tensor in (("data", plan.data), ("auxiliary", plan.auxiliary)):
            if tensor is not None:
                result.append((f"{fnv1a64(plan.name):016x}:{suffix}", tensor))
    return result


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


def _configuration_bytes(config: dict) -> bytes:
    integers = (
        config["vocab_size"], config["hidden_size"], len(config["layer_kinds"]),
        config["kda_heads"], config["kda_head_dim"], config["short_conv_kernel_size"],
        config["mla_heads"], config["q_lora_rank"], config["kv_lora_rank"],
        config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"],
        config["num_experts"], config["top_k"], config["num_shared_experts"],
        config["routed_latent_size"], config["expert_intermediate_size"],
        config["dense_intermediate_size"], config["attn_res_block_size"],
        config["mxfp4_group_size"],
    )
    floats = (
        config["rms_norm_eps"], config["kda_gate_lower_bound"],
        config["routed_scaling_factor"], config["activation_situ_beta"],
        config["activation_situ_linear_beta"], 1.0e-6, 1.0e-6,
    )
    flags = int(config["mla_use_nope"]) | (int(config["mla_use_output_gate"]) << 1)
    import struct
    result = bytearray(MODEL_CONFIG_BYTES)
    struct.pack_into("<20I7fI", result, 0, *integers, *floats, flags)
    return bytes(result)


def _write_padding(stream, offset: int) -> None:
    current = stream.tell()
    if offset < current:
        raise K3XError("OVERLAPPING_EXTENT")
    stream.write(bytes(offset - current))


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


def _directory_records(plans: list[_TensorPlan], records: list[TensorRecord], config: dict):
    layers: list[LayerRecord] = []
    experts: list[ExpertRecord] = []
    for layer_index, attention in enumerate(config["layer_kinds"]):
        indices = [index for index, plan in enumerate(plans) if plan.layer_id == layer_index]
        layer_experts = sorted({plan.expert_id for plan in plans
                                if plan.layer_id == layer_index and plan.expert_id >= 0})
        first_expert = len(experts)
        for expert_id in layer_experts:
            ids = {match.group(3): fnv1a64(plan.name) for plan in plans
                   if (match := _EXPERT_RE.match(plan.name))
                   and int(match.group(1)) == layer_index
                   and int(match.group(2)) == expert_id}
            experts.append(ExpertRecord(layer_index, expert_id, len(experts), 0,
                                        ids["gate"], ids["up"], ids["down"]))
        layers.append(LayerRecord(
            layer_index, 1 if attention == "kda" else 2,
            1 if layer_index in config["dense_layers"] else 2,
            min(indices) if indices else 0, len(indices), first_expert,
            len(layer_experts), layer_index // config["attn_res_block_size"], 0,
        ))
    return layers, experts


def convert(
    source: Path,
    output: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    stop_after_extents: int | None = None,
    dry_run: bool = False,
) -> ConversionReport:
    source, output = Path(source), Path(output)
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    manifest, plans = _load_plans(source)
    optional_features = (
        OPTIONAL_STORAGE_FIXTURE
        if manifest["format"] == "k3-storage-slice-v1"
        else 0
    )
    maximum_read = 0
    if optional_features:
        maximum_read = _validate_storage_fixture_hashes(
            manifest, plans, source, chunk_bytes
        )
    source_fingerprint, observed = _fingerprint_source(source, manifest, chunk_bytes)
    maximum_read = max(maximum_read, observed)
    config_bytes = _configuration_bytes(manifest["config"])
    fingerprint_bytes = config_bytes
    if optional_features:
        fingerprint_bytes += struct.pack("<Q", optional_features)
    config_fingerprint = hashlib.sha256(fingerprint_bytes).hexdigest()
    if dry_run:
        return ConversionReport(False, (), maximum_read, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    resume_path = output.with_suffix(output.suffix + ".resume.json")
    reused: list[str] = []
    completed: list[CompletedExtent] = []
    if resume_path.exists():
        ledger = read_resume_manifest(resume_path)
        if ledger.source_fingerprint != source_fingerprint.hex():
            raise K3XError("SOURCE_FINGERPRINT_MISMATCH")
        if ledger.converter_version != CONVERTER_VERSION or ledger.configuration_fingerprint != config_fingerprint:
            raise K3XError("RESUME_CONFIGURATION_MISMATCH")
        _validate_resume_extents(
            ledger.completed, _expected_extents(plans), chunk_bytes
        )
        if not partial.exists():
            if output.exists():
                from .reader import K3XReader

                finalized = K3XReader.open(output)
                if (
                    finalized.superblock.source_sha256 != source_fingerprint
                    or finalized.superblock.file_uuid != bytes.fromhex(ledger.file_uuid)
                    or finalized.superblock.optional_features != optional_features
                    or finalized.model_config != config_bytes
                ):
                    raise K3XError("FINAL_ARTIFACT_MISMATCH")
                resume_path.unlink()
                return ConversionReport(
                    True,
                    tuple(item.extent_id for item in ledger.completed),
                    maximum_read,
                    output,
                )
            raise K3XError("MISSING_PARTIAL_FILE")
        file_uuid = bytes.fromhex(ledger.file_uuid)
        with partial.open("rb") as stream:
            for item in ledger.completed:
                if _crc_at(stream, item.offset, item.length, chunk_bytes) != item.crc32c:
                    raise K3XError("RESUME_EXTENT_CRC_MISMATCH")
                completed.append(item)
                reused.append(item.extent_id)
    else:
        file_uuid = uuid.uuid4().bytes
        with partial.open("wb") as stream:
            stream.write(bytes(SUPERBLOCK_BYTES))
            stream.flush()
            os.fsync(stream.fileno())
        write_resume_manifest(resume_path, ResumeManifest(
            source_fingerprint.hex(), CONVERTER_VERSION, config_fingerprint,
            file_uuid.hex(), (),
        ))
    completed_map = {item.extent_id: item for item in completed}
    records: list[TensorRecord] = []
    newly_written = 0
    with partial.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        for plan in plans:
            extent_values: list[tuple[int, int, int]] = []
            for suffix, source_tensor in (("data", plan.data), ("auxiliary", plan.auxiliary)):
                if source_tensor is None:
                    extent_values.append((0, 0, 0))
                    continue
                extent_id = f"{fnv1a64(plan.name):016x}:{suffix}"
                if extent_id in completed_map:
                    item = completed_map[extent_id]
                    extent_values.append((item.offset, item.length, item.crc32c))
                    continue
                offset = align_up(stream.seek(0, os.SEEK_END))
                _write_padding(stream, offset)
                checksum = google_crc32c.Checksum()
                length = 0
                for chunk in iter_tensor_chunks(source_tensor, chunk_bytes):
                    maximum_read = max(maximum_read, len(chunk))
                    stream.write(chunk)
                    checksum.update(chunk)
                    length += len(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                crc = int.from_bytes(checksum.digest(), "big")
                if _crc_at(stream, offset, length, chunk_bytes) != crc:
                    raise K3XError("EXTENT_READBACK_MISMATCH")
                item = CompletedExtent(extent_id, offset, length, crc)
                completed.append(item)
                completed_map[extent_id] = item
                write_resume_manifest(resume_path, ResumeManifest(
                    source_fingerprint.hex(), CONVERTER_VERSION, config_fingerprint,
                    file_uuid.hex(), tuple(completed),
                ))
                newly_written += 1
                extent_values.append((offset, length, crc))
                if stop_after_extents is not None and newly_written >= stop_after_extents:
                    return ConversionReport(False, tuple(reused), maximum_read, output)
            data, auxiliary = extent_values
            logical_length = (
                plan.dimensions[0] * plan.dimensions[1] * 4
                if plan.quantization == Quantization.MXFP4 else data[1]
            )
            records.append(TensorRecord(
                fnv1a64(plan.name), 0, plan.dtype, plan.quantization, plan.dimensions,
                plan.layer_id, plan.expert_id, data[0], data[1], logical_length,
                auxiliary[0], auxiliary[1], data[2], auxiliary[2],
            ))
        layers, experts = _directory_records(plans, records, manifest["config"])
        tensor_directory = encode_directory(b"TENS", TENSOR_RECORD_BYTES,
                                            (item.encode() for item in records))
        layer_directory = encode_directory(b"LAYR", LAYER_RECORD_BYTES,
                                           (item.encode() for item in layers))
        expert_directory = encode_directory(b"EXPT", EXPERT_RECORD_BYTES,
                                            (item.encode() for item in experts))
        offsets: list[tuple[int, bytes]] = []
        for data in (tensor_directory, layer_directory, expert_directory, config_bytes):
            offset = align_up(stream.seek(0, os.SEEK_END))
            _write_padding(stream, offset)
            stream.write(data)
            offsets.append((offset, data))
        file_length = stream.tell()
        directory_digest = hashlib.sha256(b"".join(data for _, data in offsets)).digest()
        block = Superblock(
            source_fingerprint, file_uuid, state=1,
            optional_features=optional_features,
            tensor_directory_offset=offsets[0][0], tensor_directory_length=len(tensor_directory),
            layer_directory_offset=offsets[1][0], layer_directory_length=len(layer_directory),
            expert_directory_offset=offsets[2][0], expert_directory_length=len(expert_directory),
            model_config_offset=offsets[3][0], model_config_length=len(config_bytes),
            file_length=file_length, directory_sha256=directory_digest,
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
    resume_path.unlink()
    return ConversionReport(True, tuple(reused), maximum_read, output)
