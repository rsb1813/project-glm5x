# GLM5X CUDA INT4 packed expert를 NVMe sidecar로 원자적으로 저장하고 재사용합니다.
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import google_crc32c
import torch

from .int4 import GLM5XInt4Weight


_MAGIC = b"GLM5XPI4"
_VERSION = 1
_HEADER = struct.Struct("<8sII")
_PRECISIONS = {"int4", "fp8"}


@dataclass(frozen=True)
class GLM5XPackedExpertCacheStats:
    hits: int
    misses: int
    writes: int


class GLM5XPackedExpertCache:
    """Persistent, fingerprint-bound packed expert cache."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._lock = Lock()

    @staticmethod
    def _path(root: Path, key: tuple[int, int], precision: str = "int4") -> Path:
        if precision not in _PRECISIONS:
            raise ValueError("GLM5X_PACKED_CACHE_PRECISION")
        layer_id, expert_id = (int(value) for value in key)
        if layer_id < 0 or expert_id < 0:
            raise ValueError("GLM5X_PACKED_CACHE_KEY")
        suffix = "pi4" if precision == "int4" else "pf8"
        return root / f"layer-{layer_id:04d}-expert-{expert_id:04d}.{suffix}"

    @staticmethod
    def _weight_payload(
        weight: GLM5XInt4Weight | torch.Tensor,
        scale: torch.Tensor | None,
        precision: str,
    ) -> tuple[dict[str, Any], bytes, bytes]:
        if precision == "int4":
            if not isinstance(weight, GLM5XInt4Weight) or scale is not None:
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_INT4")
            packed = weight.packed.detach().to(device="cpu").contiguous()
            qparams = weight.scale_and_zero.detach().to(device="cpu").contiguous()
            packed_bytes = packed.numpy().tobytes(order="C")
            qparam_bytes = qparams.view(torch.int16).numpy().tobytes(order="C")
            shape = list(weight.shape)
            group_size = int(weight.group_size)
            inner_k_tiles = int(weight.inner_k_tiles)
            qparams_dtype = "bfloat16"
        else:
            if (
                not isinstance(weight, torch.Tensor)
                or weight.dtype != torch.float8_e4m3fn
                or scale is None
            ):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_FP8")
            packed = weight.detach().to(device="cpu").contiguous()
            qparams = scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
            packed_bytes = packed.view(torch.uint8).numpy().tobytes(order="C")
            qparam_bytes = qparams.numpy().tobytes(order="C")
            shape = list(weight.shape)
            group_size = 0
            inner_k_tiles = 0
            qparams_dtype = "float32"
        return (
            {
                "shape": shape,
                "group_size": group_size,
                "inner_k_tiles": inner_k_tiles,
                "packed_shape": list(packed.shape),
                "qparams_shape": list(qparams.shape),
                "qparams_dtype": qparams_dtype,
                "packed_bytes": len(packed_bytes),
                "qparams_bytes": len(qparam_bytes),
                "packed_crc32c": int(google_crc32c.value(packed_bytes)),
                "qparams_crc32c": int(google_crc32c.value(qparam_bytes)),
            },
            packed_bytes,
            qparam_bytes,
        )

    def put(
        self,
        key: tuple[int, int],
        source_digest: str,
        expert: Any,
        *,
        precision: str = "int4",
    ) -> None:
        if not isinstance(source_digest, str) or len(source_digest) < 16:
            raise ValueError("GLM5X_PACKED_CACHE_SOURCE_DIGEST")
        if precision not in _PRECISIONS:
            raise ValueError("GLM5X_PACKED_CACHE_PRECISION")
        weights = (expert.gate_proj, expert.up_proj, expert.down_proj)
        scales = (None, None, None)
        if precision == "int4":
            if not all(isinstance(weight, GLM5XInt4Weight) for weight in weights):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_INT4")
        else:
            scales = (expert.gate_scale, expert.up_scale, expert.down_scale)
            if not expert.is_fp8:
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_FP8")
        records: list[dict[str, Any]] = []
        payload = bytearray()
        for weight, scale in zip(weights, scales):
            record, packed_bytes, qparam_bytes = self._weight_payload(
                weight, scale, precision
            )
            record["packed_offset"] = len(payload)
            payload.extend(packed_bytes)
            record["qparams_offset"] = len(payload)
            payload.extend(qparam_bytes)
            records.append(record)
        metadata = {
            "format": f"glm5x-{precision}-expert-v1",
            "source_digest": source_digest,
            "roles": records,
            "payload_bytes": len(payload),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        destination = self._path(self.root, key, precision)
        with self._lock:
            with NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                try:
                    temporary.write(_HEADER.pack(_MAGIC, _VERSION, len(encoded)))
                    temporary.write(encoded)
                    temporary.write(payload)
                    temporary.flush()
                    os.fsync(temporary.fileno())
                except BaseException:
                    temporary_path.unlink(missing_ok=True)
                    raise
            os.replace(temporary_path, destination)
            self._writes += 1

    @staticmethod
    def _decode_role(
        metadata: dict[str, Any], payload: bytes, *, device: torch.device,
        precision: str,
    ) -> GLM5XInt4Weight | tuple[torch.Tensor, torch.Tensor]:
        shape = tuple(int(value) for value in metadata["shape"])
        packed_shape = tuple(int(value) for value in metadata["packed_shape"])
        qparams_shape = tuple(int(value) for value in metadata["qparams_shape"])
        packed_bytes = int(metadata["packed_bytes"])
        qparams_bytes = int(metadata["qparams_bytes"])
        packed_offset = int(metadata["packed_offset"])
        qparams_offset = int(metadata["qparams_offset"])
        packed_data = payload[packed_offset : packed_offset + packed_bytes]
        qparams_data = payload[qparams_offset : qparams_offset + qparams_bytes]
        if len(packed_data) != packed_bytes or len(qparams_data) != qparams_bytes:
            raise ValueError("GLM5X_PACKED_CACHE_PAYLOAD_EXTENT")
        if google_crc32c.value(packed_data) != int(metadata["packed_crc32c"]):
            raise ValueError("GLM5X_PACKED_CACHE_PACKED_CRC")
        if google_crc32c.value(qparams_data) != int(metadata["qparams_crc32c"]):
            raise ValueError("GLM5X_PACKED_CACHE_QPARAM_CRC")
        if precision == "int4":
            packed = torch.frombuffer(bytearray(packed_data), dtype=torch.int32).reshape(
                packed_shape
            )
            qparams = torch.frombuffer(bytearray(qparams_data), dtype=torch.int16).view(
                torch.bfloat16
            ).reshape(qparams_shape)
            return GLM5XInt4Weight(
                packed=packed.to(device=device),
                scale_and_zero=qparams.to(device=device),
                shape=shape,
                group_size=int(metadata["group_size"]),
                inner_k_tiles=int(metadata["inner_k_tiles"]),
            )
        packed = torch.frombuffer(bytearray(packed_data), dtype=torch.uint8).reshape(
            packed_shape
        ).view(torch.float8_e4m3fn)
        return packed.to(device=device), torch.frombuffer(
            bytearray(qparams_data), dtype=torch.float32
        ).reshape(qparams_shape).to(device=device)

    def get(
        self,
        key: tuple[int, int],
        source_digest: str,
        *,
        device: torch.device | str,
        precision: str = "int4",
    ) -> Any | None:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GLM5X_PACKED_CACHE_CUDA_REQUIRED")
        if precision not in _PRECISIONS:
            raise ValueError("GLM5X_PACKED_CACHE_PRECISION")
        path = self._path(self.root, key, precision)
        with self._lock:
            try:
                data = path.read_bytes()
                if len(data) < _HEADER.size:
                    raise ValueError("GLM5X_PACKED_CACHE_HEADER")
                magic, version, metadata_length = _HEADER.unpack_from(data)
                if magic != _MAGIC or version != _VERSION:
                    raise ValueError("GLM5X_PACKED_CACHE_FORMAT")
                metadata_start = _HEADER.size
                metadata_end = metadata_start + metadata_length
                metadata = json.loads(data[metadata_start:metadata_end].decode("utf-8"))
                if metadata.get("format") != f"glm5x-{precision}-expert-v1":
                    raise ValueError("GLM5X_PACKED_CACHE_FORMAT")
                if metadata.get("source_digest") != source_digest:
                    raise ValueError("GLM5X_PACKED_CACHE_SOURCE_MISMATCH")
                roles = metadata.get("roles")
                if not isinstance(roles, list) or len(roles) != 3:
                    raise ValueError("GLM5X_PACKED_CACHE_ROLES")
                payload = data[metadata_end:]
                if len(payload) != int(metadata.get("payload_bytes", -1)):
                    raise ValueError("GLM5X_PACKED_CACHE_PAYLOAD_EXTENT")
                decoded = tuple(
                    self._decode_role(role, payload, device=target, precision=precision)
                    for role in roles
                )
                from .layer10_moe import GLM5XExpertWeights

                self._hits += 1
                if precision == "fp8":
                    gate, gate_scale = decoded[0]
                    up, up_scale = decoded[1]
                    down, down_scale = decoded[2]
                    return GLM5XExpertWeights(
                        gate_proj=gate,
                        up_proj=up,
                        down_proj=down,
                        gate_scale=gate_scale,
                        up_scale=up_scale,
                        down_scale=down_scale,
                    )
                return GLM5XExpertWeights(
                    gate_proj=decoded[0], up_proj=decoded[1], down_proj=decoded[2]
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                self._misses += 1
                return None

    @property
    def stats(self) -> GLM5XPackedExpertCacheStats:
        with self._lock:
            return GLM5XPackedExpertCacheStats(
                hits=self._hits, misses=self._misses, writes=self._writes
            )
