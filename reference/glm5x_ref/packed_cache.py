# GLM5X CUDA INT4 packed expert를 NVMe sidecar로 원자적으로 저장하고 재사용합니다.
from __future__ import annotations

import json
import math
import os
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Mapping

import google_crc32c
import torch

from k3x_ref.mxfp4 import decode_mxfp4, encode_mxfp4
from .int4 import GLM5XInt4Weight
from .nvfp4 import GLM5XNVFP4Weight


_MAGIC = b"GLM5XPI4"
_VERSION = 1
_HEADER = struct.Struct("<8sII")
_PRECISIONS = {"int4", "fp8", "mxfp4", "nvfp4", "nvfp4_gate_up"}


@dataclass(frozen=True)
class GLM5XPackedExpertCacheStats:
    hits: int
    misses: int
    writes: int
    host_hits: int = 0
    host_misses: int = 0
    host_resident_bytes: int = 0
    host_capacity_bytes: int = 0
    pinned_staging_bytes: int = 0
    pinned_staging_capacity_bytes: int = 0
    pinned_staging_hits: int = 0


@dataclass(frozen=True)
class _HostPayload:
    metadata: dict[str, Any]
    payload: bytes


@dataclass
class _PinnedPayload:
    sections: dict[tuple[int, int], torch.Tensor]
    bytes: int
    ready: torch.cuda.Event | None = None


class GLM5XPackedExpertCache:
    """Persistent, fingerprint-bound packed expert cache."""

    def __init__(
        self,
        root: str | Path,
        *,
        host_cache_capacity_bytes: int = 0,
        pinned_staging_capacity_bytes: int = 0,
        non_blocking: bool = False,
    ) -> None:
        if (
            not isinstance(host_cache_capacity_bytes, int)
            or isinstance(host_cache_capacity_bytes, bool)
            or host_cache_capacity_bytes < 0
        ):
            raise ValueError("GLM5X_PACKED_CACHE_HOST_CAPACITY")
        if (
            not isinstance(pinned_staging_capacity_bytes, int)
            or isinstance(pinned_staging_capacity_bytes, bool)
            or pinned_staging_capacity_bytes < 0
        ):
            raise ValueError("GLM5X_PACKED_CACHE_PINNED_CAPACITY")
        if not isinstance(non_blocking, bool):
            raise ValueError("GLM5X_PACKED_CACHE_NON_BLOCKING")
        if non_blocking and pinned_staging_capacity_bytes == 0:
            raise ValueError("GLM5X_PACKED_CACHE_PINNED_CAPACITY_REQUIRED")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._host_capacity_bytes = host_cache_capacity_bytes
        self._host_resident_bytes = 0
        self._host_hits = 0
        self._host_misses = 0
        self._host_payloads: OrderedDict[Path, _HostPayload] = OrderedDict()
        self._pinned_capacity_bytes = pinned_staging_capacity_bytes
        self._non_blocking = non_blocking
        self._pinned_resident_bytes = 0
        self._pinned_hits = 0
        self._pinned_payloads: OrderedDict[Path, _PinnedPayload] = OrderedDict()
        self._lock = Lock()

    @property
    def non_blocking(self) -> bool:
        return self._non_blocking

    def _host_get(self, path: Path) -> _HostPayload | None:
        if self._host_capacity_bytes == 0:
            return None
        with self._lock:
            cached = self._host_payloads.pop(path, None)
            if cached is None:
                self._host_misses += 1
                return None
            self._host_payloads[path] = cached
            self._host_hits += 1
            return cached

    def _host_put(self, path: Path, metadata: dict[str, Any], payload: bytes) -> None:
        if self._host_capacity_bytes == 0:
            return
        size = len(payload)
        if size > self._host_capacity_bytes:
            return
        with self._lock:
            previous = self._host_payloads.pop(path, None)
            if previous is not None:
                self._host_resident_bytes -= len(previous.payload)
            while (
                self._host_payloads
                and self._host_resident_bytes + size > self._host_capacity_bytes
            ):
                _, evicted = self._host_payloads.popitem(last=False)
                self._host_resident_bytes -= len(evicted.payload)
            self._host_payloads[path] = _HostPayload(metadata, payload)
            self._host_resident_bytes += size

    def _pinned_get(self, path: Path) -> _PinnedPayload | None:
        if self._pinned_capacity_bytes == 0:
            return None
        with self._lock:
            cached = self._pinned_payloads.pop(path, None)
            if cached is None:
                return None
            self._pinned_payloads[path] = cached
            self._pinned_hits += 1
            return cached

    def _pinned_put(
        self, path: Path, sections: dict[tuple[int, int], torch.Tensor]
    ) -> _PinnedPayload | None:
        if self._pinned_capacity_bytes == 0:
            return None
        size = sum(int(value.numel()) for value in sections.values())
        if size > self._pinned_capacity_bytes:
            return None
        cached = _PinnedPayload(sections=sections, bytes=size)
        with self._lock:
            previous = self._pinned_payloads.pop(path, None)
            if previous is not None:
                self._pinned_resident_bytes -= previous.bytes
            while (
                self._pinned_payloads
                and self._pinned_resident_bytes + size > self._pinned_capacity_bytes
            ):
                _, evicted = self._pinned_payloads.popitem(last=False)
                if evicted.ready is not None:
                    evicted.ready.synchronize()
                self._pinned_resident_bytes -= evicted.bytes
            self._pinned_payloads[path] = cached
            self._pinned_resident_bytes += size
        return cached

    @staticmethod
    def _pinned_sections(
        metadata: list[dict[str, Any]], payload: bytes
    ) -> dict[tuple[int, int], torch.Tensor]:
        sections: dict[tuple[int, int], torch.Tensor] = {}
        for role in metadata:
            for kind in ("packed", "qparams"):
                offset = int(role[f"{kind}_offset"])
                size = int(role[f"{kind}_bytes"])
                if size == 0:
                    continue
                data = payload[offset : offset + size]
                if len(data) != size:
                    raise ValueError("GLM5X_PACKED_CACHE_PAYLOAD_EXTENT")
                sections[(offset, size)] = torch.frombuffer(
                    bytearray(data), dtype=torch.uint8
                ).pin_memory()
        return sections

    @staticmethod
    def _path(root: Path, key: tuple[int, int], precision: str = "int4") -> Path:
        if precision not in _PRECISIONS:
            raise ValueError("GLM5X_PACKED_CACHE_PRECISION")
        layer_id, expert_id = (int(value) for value in key)
        if layer_id < 0 or expert_id < 0:
            raise ValueError("GLM5X_PACKED_CACHE_KEY")
        suffix = {
            "int4": "pi4",
            "fp8": "pf8",
            "mxfp4": "pm4",
            "nvfp4": "pn4",
            "nvfp4_gate_up": "pgu",
        }[precision]
        return root / f"layer-{layer_id:04d}-expert-{expert_id:04d}.{suffix}"

    @staticmethod
    def _weight_payload(
        weight: GLM5XInt4Weight | GLM5XNVFP4Weight | torch.Tensor,
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
        elif precision == "fp8":
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
        elif precision == "mxfp4":
            if (
                not isinstance(weight, torch.Tensor)
                or weight.ndim != 2
                or weight.shape[1] % 32
                or not weight.is_floating_point()
            ):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_MXFP4")
            packed_bytes, qparam_bytes = encode_mxfp4(
                weight, group_size=32, scale_mode="max_abs"
            )
            shape = list(weight.shape)
            packed = torch.empty(len(packed_bytes), dtype=torch.uint8)
            qparams = torch.empty(len(qparam_bytes), dtype=torch.uint8)
            group_size = 32
            inner_k_tiles = 0
            qparams_dtype = "uint8"
        else:
            if precision == "nvfp4_gate_up" and isinstance(weight, torch.Tensor):
                packed = weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                packed_bytes = packed.view(torch.int16).numpy().tobytes(order="C")
                return (
                    {
                        "representation": "bf16",
                        "shape": list(weight.shape),
                        "group_size": 0,
                        "inner_k_tiles": 0,
                        "packed_shape": list(packed.shape),
                        "qparams_shape": [],
                        "qparams_dtype": "none",
                        "packed_bytes": len(packed_bytes),
                        "qparams_bytes": 0,
                        "packed_crc32c": int(google_crc32c.value(packed_bytes)),
                        "qparams_crc32c": int(google_crc32c.value(b"")),
                    },
                    packed_bytes,
                    b"",
                )
            if not isinstance(weight, GLM5XNVFP4Weight) or scale is not None:
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_NVFP4")
            packed = weight.packed.detach().to(device="cpu").contiguous()
            scales = weight.scales.detach().to(device="cpu").contiguous()
            global_scale = weight.global_scale.detach().to(device="cpu", dtype=torch.float32).contiguous()
            packed_bytes = packed.numpy().tobytes(order="C")
            scale_bytes = scales.view(torch.uint8).numpy().tobytes(order="C")
            qparam_bytes = scale_bytes + global_scale.numpy().tobytes(order="C")
            shape = list(weight.shape)
            packed_shape = packed.shape
            qparams_shape = scales.shape
            group_size = 16
            inner_k_tiles = 0
            qparams_dtype = "float8_e4m3fn"
            return (
                {
                    "representation": "nvfp4",
                    "shape": shape,
                    "group_size": group_size,
                    "inner_k_tiles": inner_k_tiles,
                    "packed_shape": list(packed_shape),
                    "qparams_shape": list(qparams_shape),
                    "qparams_dtype": qparams_dtype,
                    "packed_bytes": len(packed_bytes),
                    "qparams_bytes": len(qparam_bytes),
                    "scale_bytes": len(scale_bytes),
                    "global_scale_bytes": 4,
                    "packed_crc32c": int(google_crc32c.value(packed_bytes)),
                    "qparams_crc32c": int(google_crc32c.value(qparam_bytes)),
                },
                packed_bytes,
                qparam_bytes,
            )
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
        elif precision == "fp8":
            scales = (expert.gate_scale, expert.up_scale, expert.down_scale)
            if not expert.is_fp8:
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_FP8")
        elif precision == "nvfp4":
            if not all(isinstance(weight, GLM5XNVFP4Weight) for weight in weights):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_NVFP4")
        elif precision == "nvfp4_gate_up":
            if not all(isinstance(weight, GLM5XNVFP4Weight) for weight in weights[:2]):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_NVFP4")
            if not isinstance(weights[2], torch.Tensor):
                raise ValueError("GLM5X_PACKED_CACHE_REQUIRES_BF16_DOWN")
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
            previous = self._host_payloads.pop(destination, None)
            if previous is not None:
                self._host_resident_bytes -= len(previous.payload)
            previous_pinned = self._pinned_payloads.pop(destination, None)
            if previous_pinned is not None:
                if previous_pinned.ready is not None:
                    previous_pinned.ready.synchronize()
                self._pinned_resident_bytes -= previous_pinned.bytes
            self._writes += 1

    @staticmethod
    def _decode_role(
        metadata: dict[str, Any], payload: bytes, *, device: torch.device,
        precision: str,
        non_blocking: bool = False,
        pinned_sections: dict[tuple[int, int], torch.Tensor] | None = None,
    ) -> GLM5XInt4Weight | GLM5XNVFP4Weight | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
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

        def transfer(
            data: bytes, *, offset: int, dtype: torch.dtype, shape: tuple[int, ...]
        ) -> torch.Tensor:
            if pinned_sections is None:
                source = torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)
            else:
                source = pinned_sections[(offset, len(data))].view(dtype).reshape(shape)
            return source.to(device=device, non_blocking=non_blocking)

        if precision == "nvfp4_gate_up" and metadata.get("representation") == "bf16":
            if qparams_bytes != 0:
                raise ValueError("GLM5X_PACKED_CACHE_BF16_QPARAM_EXTENT")
            return transfer(
                packed_data,
                offset=packed_offset,
                dtype=torch.int16,
                shape=(math.prod(packed_shape),),
            ).view(torch.bfloat16).reshape(packed_shape)
        if precision == "int4":
            packed = transfer(
                packed_data, offset=packed_offset, dtype=torch.int32, shape=packed_shape
            )
            qparams = transfer(
                qparams_data, offset=qparams_offset, dtype=torch.int16, shape=qparams_shape
            ).view(torch.bfloat16)
            return GLM5XInt4Weight(
                packed=packed.to(device=device),
                scale_and_zero=qparams.to(device=device),
                shape=shape,
                group_size=int(metadata["group_size"]),
                inner_k_tiles=int(metadata["inner_k_tiles"]),
            )
        if precision == "fp8":
            packed = transfer(
                packed_data, offset=packed_offset, dtype=torch.uint8, shape=packed_shape
            ).view(torch.float8_e4m3fn)
            return packed, transfer(
                qparams_data,
                offset=qparams_offset,
                dtype=torch.float32,
                shape=qparams_shape,
            )
        if precision in {"nvfp4", "nvfp4_gate_up"}:
            scale_bytes = int(metadata["scale_bytes"])
            global_scale_bytes = int(metadata["global_scale_bytes"])
            if scale_bytes + global_scale_bytes != qparams_bytes:
                raise ValueError("GLM5X_PACKED_CACHE_NVFP4_QPARAM_EXTENT")
            packed = transfer(
                packed_data, offset=packed_offset, dtype=torch.uint8, shape=packed_shape
            )
            if pinned_sections is None:
                qparams_source = torch.frombuffer(
                    bytearray(qparams_data), dtype=torch.uint8
                )
            else:
                qparams_source = pinned_sections[(qparams_offset, qparams_bytes)]
            scales = qparams_source[:scale_bytes].to(
                device=device, non_blocking=non_blocking
            ).view(torch.float8_e4m3fn).reshape(qparams_shape)
            global_scale = qparams_source[scale_bytes:].view(torch.float32).to(
                device=device, non_blocking=non_blocking
            ).reshape(())
            return GLM5XNVFP4Weight(
                packed=packed.to(device=device),
                scales=scales.to(device=device),
                global_scale=global_scale.to(device=device),
                shape=shape,
            )
        decoded = decode_mxfp4(
            packed_data,
            qparams_data,
            shape[0],
            shape[1],
            int(metadata["group_size"]),
        ).to(dtype=torch.bfloat16, device=device)
        return decoded

    def get(
        self,
        key: tuple[int, int],
        source_digest: str,
        *,
        device: torch.device | str,
        precision: str = "int4",
        non_blocking: bool = False,
    ) -> Any | None:
        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("GLM5X_PACKED_CACHE_CUDA_REQUIRED")
        if precision not in _PRECISIONS:
            raise ValueError("GLM5X_PACKED_CACHE_PRECISION")
        if not isinstance(non_blocking, bool):
            raise ValueError("GLM5X_PACKED_CACHE_NON_BLOCKING")
        if non_blocking and self._pinned_capacity_bytes == 0:
            raise ValueError("GLM5X_PACKED_CACHE_PINNED_CAPACITY_REQUIRED")
        path = self._path(self.root, key, precision)
        pinned_is_new = False
        try:
            cached = self._host_get(path)
            if cached is None:
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
            else:
                metadata = cached.metadata
                payload = cached.payload
                roles = metadata.get("roles")
                if not isinstance(roles, list) or len(roles) != 3:
                    raise ValueError("GLM5X_PACKED_CACHE_ROLES")
                if metadata.get("format") != f"glm5x-{precision}-expert-v1":
                    raise ValueError("GLM5X_PACKED_CACHE_FORMAT")
                if metadata.get("source_digest") != source_digest:
                    raise ValueError("GLM5X_PACKED_CACHE_SOURCE_MISMATCH")
            pinned = None
            if non_blocking:
                pinned = self._pinned_get(path)
                if pinned is None:
                    sections = self._pinned_sections(roles, payload)
                    pinned = _PinnedPayload(
                        sections=sections,
                        bytes=sum(int(value.numel()) for value in sections.values()),
                    )
                    if pinned.bytes > self._pinned_capacity_bytes:
                        raise ValueError("GLM5X_PACKED_CACHE_PINNED_CAPACITY")
                    pinned_is_new = True
            decoded = tuple(
                self._decode_role(
                    role,
                    payload,
                    device=target,
                    precision=precision,
                    non_blocking=non_blocking,
                    pinned_sections=None if pinned is None else pinned.sections,
                )
                for role in roles
            )
            if pinned_is_new:
                stored = self._pinned_put(path, pinned.sections)
                if stored is None:
                    raise ValueError("GLM5X_PACKED_CACHE_PINNED_CAPACITY")
                pinned = stored
            if pinned is not None:
                pinned.ready = torch.cuda.Event()
                pinned.ready.record(torch.cuda.current_stream(target))
            if cached is None:
                self._host_put(path, metadata, payload)
            from .layer10_moe import GLM5XExpertWeights

            with self._lock:
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
                gate_proj=decoded[0],
                up_proj=decoded[1],
                down_proj=decoded[2],
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            if non_blocking and pinned_is_new:
                torch.cuda.current_stream(target).synchronize()
            with self._lock:
                self._misses += 1
            return None

    def get_many(
        self,
        source_digests: Mapping[tuple[int, int], str],
        *,
        device: torch.device | str,
        precision: str = "int4",
        workers: int = 1,
        non_blocking: bool = False,
    ) -> dict[tuple[int, int], Any]:
        """Read independent sidecars concurrently without serializing file I/O."""
        if (
            not isinstance(workers, int)
            or isinstance(workers, bool)
            or workers <= 0
        ):
            raise ValueError("GLM5X_PACKED_CACHE_WORKERS")
        if not isinstance(non_blocking, bool):
            raise ValueError("GLM5X_PACKED_CACHE_NON_BLOCKING")
        if non_blocking and workers != 1:
            raise ValueError("GLM5X_PACKED_CACHE_NON_BLOCKING_WORKERS")
        items = list(source_digests.items())
        if not items:
            return {}
        if workers == 1 or len(items) == 1:
            return {
                key: cached
                for key, digest in items
                if (cached := self.get(
                    key,
                    digest,
                    device=device,
                    precision=precision,
                    non_blocking=non_blocking,
                ))
                is not None
            }
        result: dict[tuple[int, int], Any] = {}
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
            futures = {
                key: executor.submit(
                    self.get,
                    key,
                    digest,
                    device=device,
                    precision=precision,
                    non_blocking=non_blocking,
                )
                for key, digest in items
            }
            for key, future in futures.items():
                cached = future.result()
                if cached is not None:
                    result[key] = cached
        return result

    @property
    def stats(self) -> GLM5XPackedExpertCacheStats:
        with self._lock:
            return GLM5XPackedExpertCacheStats(
                hits=self._hits,
                misses=self._misses,
                writes=self._writes,
                host_hits=self._host_hits,
                host_misses=self._host_misses,
                host_resident_bytes=self._host_resident_bytes,
                host_capacity_bytes=self._host_capacity_bytes,
                pinned_staging_bytes=self._pinned_resident_bytes,
                pinned_staging_capacity_bytes=self._pinned_capacity_bytes,
                pinned_staging_hits=self._pinned_hits,
            )
