# 실제 K3 expert 크기의 bounded storage fixture를 스트리밍 생성합니다.
from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_SHARD_STEM = "bounded-expert"
_MANIFEST_NAME = "source-manifest.json"
_PACKED_BYTES = 5_505_024
_SCALE_BYTES = 344_064
_EXPERT_PAYLOAD_BYTES = 17_547_264


@dataclass(frozen=True)
class StorageSliceReport:
    shard_path: Path
    manifest_path: Path
    maximum_chunk_bytes: int
    payload_bytes: int
    source_sha256: str


def _released_config() -> dict[str, object]:
    layer_kinds = [
        "mla" if (index < 92 and index % 4 == 3) or index == 92 else "kda"
        for index in range(93)
    ]
    return {
        "vocab_size": 163_840,
        "hidden_size": 7_168,
        "layer_kinds": layer_kinds,
        "dense_layers": [0],
        "kda_heads": 96,
        "kda_head_dim": 128,
        "short_conv_kernel_size": 4,
        "mla_heads": 96,
        "q_lora_rank": 1_536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "num_experts": 896,
        "top_k": 16,
        "num_shared_experts": 2,
        "routed_latent_size": 3_584,
        "expert_intermediate_size": 3_072,
        "dense_intermediate_size": 33_792,
        "attn_res_block_size": 12,
        "rms_norm_eps": 1.0e-5,
        "kda_gate_lower_bound": -5.0,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "routed_scaling_factor": 1.0,
        "mxfp4_group_size": 32,
    }


def _pattern_chunk(start: int, offset: int, length: int) -> bytes:
    first = (start + offset) & 0xFF
    cycle = bytes((first + index) & 0xFF for index in range(256))
    return (cycle * ((length + 255) // 256))[:length]


def _write_tensor(
    stream: BinaryIO,
    *,
    length: int,
    chunk_bytes: int,
    packed_start: int | None,
    scale_byte: int | None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    maximum = 0
    offset = 0
    while offset < length:
        count = min(chunk_bytes, length - offset)
        if packed_start is not None:
            chunk = _pattern_chunk(packed_start, offset, count)
        else:
            assert scale_byte is not None
            chunk = bytes([scale_byte]) * count
        stream.write(chunk)
        digest.update(chunk)
        maximum = max(maximum, len(chunk))
        offset += len(chunk)
    return digest.hexdigest(), maximum


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def write_bounded_expert_source(
    root: Path,
    *,
    seed: int = 20260809,
    chunk_bytes: int = 1 << 20,
    layer_id: int = 1,
    expert_id: int = 0,
) -> StorageSliceReport:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not 1 <= layer_id < 93:
        raise ValueError("layer_id must select a routed layer in [1, 92]")
    if not 0 <= expert_id < 896:
        raise ValueError("expert_id must be in [0, 895]")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / _MANIFEST_NAME
    shard_partial = root / f".{_SHARD_STEM}-{uuid.uuid4().hex}.safetensors.partial"

    base = f"model.layers.{layer_id}.feed_forward.experts.{expert_id}"
    tensor_specs: list[tuple[str, int, int | None, int | None]] = []
    packed_shapes: dict[str, list[int]] = {}
    for matrix_index, (role, shape) in enumerate(
        (("gate", (3072, 3584)), ("up", (3072, 3584)), ("down", (3584, 3072)))
    ):
        name = f"{base}.{role}"
        tensor_specs.extend(
            (
                (f"{name}.weight_packed", _PACKED_BYTES, (seed + matrix_index * 37) & 0xFF, None),
                (f"{name}.weight_scale", _SCALE_BYTES, None, 120 + matrix_index),
            )
        )
        packed_shapes[name] = list(shape)

    header: dict[str, dict[str, object]] = {}
    data_offset = 0
    for name, length, _, _ in tensor_specs:
        header[name] = {
            "dtype": "U8",
            "shape": [length],
            "data_offsets": [data_offset, data_offset + length],
        }
        data_offset += length
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((-len(header_bytes)) % 8)

    tensor_digests: dict[str, str] = {}
    maximum_chunk = 0
    try:
        with shard_partial.open("wb") as stream:
            stream.write(struct.pack("<Q", len(header_bytes)))
            stream.write(header_bytes)
            for name, length, packed_start, scale_byte in tensor_specs:
                digest, observed = _write_tensor(
                    stream,
                    length=length,
                    chunk_bytes=chunk_bytes,
                    packed_start=packed_start,
                    scale_byte=scale_byte,
                )
                tensor_digests[name] = digest
                maximum_chunk = max(maximum_chunk, observed)
            stream.flush()
            os.fsync(stream.fileno())
        expected_size = 8 + len(header_bytes) + _EXPERT_PAYLOAD_BYTES
        if shard_partial.stat().st_size != expected_size:
            raise RuntimeError("bounded source shard length mismatch")
        with shard_partial.open("rb") as stream:
            source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        shard_name = f"{_SHARD_STEM}-{source_sha256}.safetensors"
        shard_path = root / shard_name
        os.replace(shard_partial, shard_path)
    except BaseException:
        shard_partial.unlink(missing_ok=True)
        raise

    weight_map = {name: shard_name for name, _, _, _ in tensor_specs}
    manifest: dict[str, object] = {
        "format": "k3-storage-slice-v1",
        "artifact_kind": "storage_fixture",
        "seed": seed,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "config": _released_config(),
        "weight_map": weight_map,
        "packed_shapes": packed_shapes,
        "tensor_sha256": tensor_digests,
        "source_sha256": source_sha256,
        "payload_bytes": _EXPERT_PAYLOAD_BYTES,
    }
    _write_json_atomic(manifest_path, manifest)
    return StorageSliceReport(
        shard_path,
        manifest_path,
        maximum_chunk,
        _EXPERT_PAYLOAD_BYTES,
        source_sha256,
    )
