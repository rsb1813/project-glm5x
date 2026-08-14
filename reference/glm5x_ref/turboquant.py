# TurboQuant 계열 KV 압축의 CPU/reference 경로와 paged cache 계약을 구현합니다.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


_SUPPORTED_BITS = (2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0, 16.0)
RotationKind = Literal["hadamard", "none"]


def _validate_bits(bits: float) -> float:
    normalized = float(bits)
    if normalized not in _SUPPORTED_BITS:
        raise ValueError(f"UNSUPPORTED_TURBOQUANT_BITS={bits!r}")
    return normalized


def _bit_schedule(bits: float) -> tuple[int, ...]:
    normalized = _validate_bits(bits)
    if normalized.is_integer():
        return (int(normalized),)
    lower = math.floor(normalized)
    if normalized != lower + 0.5:
        raise ValueError(f"UNSUPPORTED_TURBOQUANT_BITS={bits!r}")
    return (lower, lower + 1)


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _hadamard(values: torch.Tensor) -> torch.Tensor:
    width = values.shape[-1]
    if width & (width - 1):
        raise ValueError("HADAMARD_WIDTH_MUST_BE_POWER_OF_TWO")
    result = values.contiguous()
    stride = 1
    while stride < width:
        result = result.reshape(*result.shape[:-1], -1, stride * 2)
        left = result[..., :stride]
        right = result[..., stride:]
        result = torch.cat((left + right, left - right), dim=-1)
        result = result.reshape(*values.shape)
        stride *= 2
    return result / math.sqrt(width)


def _rotation_signs(width: int, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) & 0x7FFFFFFF)
    bits = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int8)
    return torch.where(bits == 0, -torch.ones((), dtype=torch.float32), torch.ones((), dtype=torch.float32)).to(device)


def _rotate(values: torch.Tensor, seed: int, rotation: RotationKind) -> tuple[torch.Tensor, int]:
    original_width = values.shape[-1]
    if rotation == "none":
        return values, original_width
    padded_width = _next_power_of_two(original_width)
    if padded_width != original_width:
        values = torch.nn.functional.pad(values, (0, padded_width - original_width))
    values = values * _rotation_signs(padded_width, seed, values.device)
    return _hadamard(values), original_width


def _inverse_rotate(values: torch.Tensor, original_width: int, seed: int, rotation: RotationKind) -> torch.Tensor:
    if rotation == "none":
        return values[..., :original_width]
    restored = _hadamard(values)
    restored = restored * _rotation_signs(restored.shape[-1], seed, restored.device)
    return restored[..., :original_width]


def _qmax_for_width(schedule: tuple[int, ...], width: int, device: torch.device) -> torch.Tensor:
    qmax = torch.tensor(
        [2 ** (bits - 1) - 1 for bits in schedule],
        dtype=torch.float32,
        device=device,
    )
    repeats = (width + len(schedule) - 1) // len(schedule)
    return qmax.repeat(repeats)[:width]


@dataclass(frozen=True)
class TurboQuantConfig:
    """KV cache 압축 정책입니다. key/value에 서로 다른 bit-width를 줄 수 있습니다."""

    bits: float = 4.0
    key_bits: float | None = None
    value_bits: float | None = None
    seed: int = 0
    rotation: RotationKind = "hadamard"

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        if self.key_bits is not None:
            _validate_bits(self.key_bits)
        if self.value_bits is not None:
            _validate_bits(self.value_bits)
        if self.rotation not in {"hadamard", "none"}:
            raise ValueError(f"UNSUPPORTED_TURBOQUANT_ROTATION={self.rotation!r}")

    @property
    def effective_key_bits(self) -> float:
        return self.bits if self.key_bits is None else self.key_bits

    @property
    def effective_value_bits(self) -> float:
        return self.bits if self.value_bits is None else self.value_bits


def estimate_kv_storage_bytes(
    *,
    tokens: int,
    key_width: int,
    value_width: int,
    config: TurboQuantConfig,
    block_tokens: int = 256,
) -> int:
    """압축 KV의 논리 payload와 row scale metadata의 저장량을 계산합니다."""

    if tokens < 0 or key_width <= 0 or value_width <= 0 or block_tokens <= 0:
        raise ValueError("INVALID_TURBOQUANT_CAPACITY_SHAPE")
    payload_bits = tokens * (
        key_width * config.effective_key_bits
        + value_width * config.effective_value_bits
    )
    scale_bytes = tokens * 2 * 4
    block_metadata_bytes = math.ceil(tokens / block_tokens) * 16
    if config.effective_key_bits == 16.0 and config.effective_value_bits == 16.0:
        return tokens * (key_width + value_width) * 2
    return math.ceil(payload_bits / 8) + scale_bytes + block_metadata_bytes


@dataclass(frozen=True)
class QuantizedVector:
    """회전된 벡터의 양자화 payload와 복원 메타데이터입니다."""

    codes: torch.Tensor
    scale: torch.Tensor | None
    original_shape: tuple[int, ...]
    original_width: int
    bit_schedule: tuple[int, ...]
    seed: int
    rotation: RotationKind
    original_dtype: torch.dtype
    lossless: bool = False

    @property
    def effective_bits(self) -> float:
        return sum(self.bit_schedule) / len(self.bit_schedule)

    @property
    def storage_bytes(self) -> int:
        if self.lossless:
            return self.codes.numel() * self.codes.element_size()
        rows = self.codes.shape[0]
        logical_bits = self.codes.numel() * self.effective_bits
        scale_bytes = rows * 4
        metadata_bytes = 16
        return math.ceil(logical_bits / 8) + scale_bytes + metadata_bytes

    def dequantize(self) -> torch.Tensor:
        if self.lossless:
            return self.codes.reshape(self.original_shape).clone()
        assert self.scale is not None
        qmax = _qmax_for_width(
            self.bit_schedule,
            self.codes.shape[-1],
            self.codes.device,
        )
        rotated = self.codes.to(torch.float32) / qmax
        rotated = rotated * self.scale.to(torch.float32).unsqueeze(-1)
        restored = _inverse_rotate(
            rotated,
            self.original_width,
            self.seed,
            self.rotation,
        )
        return restored.reshape(self.original_shape).to(self.original_dtype)


def quantize_vector(
    values: torch.Tensor,
    *,
    bits: float,
    seed: int = 0,
    rotation: RotationKind = "hadamard",
) -> QuantizedVector:
    """마지막 차원을 기준으로 벡터 batch를 압축합니다."""

    tensor = torch.as_tensor(values)
    if tensor.ndim == 0:
        raise ValueError("TURBOQUANT_INPUT_MUST_HAVE_VECTOR_DIMENSION")
    schedule = _bit_schedule(bits)
    original_shape = tuple(tensor.shape)
    original_width = tensor.shape[-1]
    if bits == 16.0:
        return QuantizedVector(
            codes=tensor.clone(),
            scale=None,
            original_shape=original_shape,
            original_width=original_width,
            bit_schedule=schedule,
            seed=seed,
            rotation="none",
            original_dtype=tensor.dtype,
            lossless=True,
        )

    rows = tensor.to(torch.float32).reshape(-1, original_width)
    rotated, _ = _rotate(rows, seed, rotation)
    qmax = _qmax_for_width(schedule, rotated.shape[-1], rotated.device)
    scale = rotated.abs().amax(dim=-1)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = torch.round(rotated / scale.unsqueeze(-1) * qmax)
    codes = codes.clamp(-qmax, qmax).to(torch.int16)
    return QuantizedVector(
        codes=codes,
        scale=scale.to(torch.float32),
        original_shape=original_shape,
        original_width=original_width,
        bit_schedule=schedule,
        seed=seed,
        rotation=rotation,
        original_dtype=tensor.dtype,
    )


class TurboQuantKVCache:
    """토큰 block을 압축 저장하고 incremental attention을 수행하는 reference cache입니다."""

    def __init__(self, config: TurboQuantConfig) -> None:
        self.config = config
        self._key_blocks: list[QuantizedVector] = []
        self._value_blocks: list[QuantizedVector] = []
        self._key_width: int | None = None
        self._value_width: int | None = None
        self._token_count = 0

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def storage_bytes(self) -> int:
        return sum(block.storage_bytes for block in self._key_blocks) + sum(
            block.storage_bytes for block in self._value_blocks
        )

    def append(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        keys = torch.as_tensor(keys)
        values = torch.as_tensor(values)
        if keys.ndim != 2 or values.ndim != 2:
            raise ValueError("TURBOQUANT_KV_BLOCKS_MUST_BE_RANK_TWO")
        if keys.shape[0] != values.shape[0]:
            raise ValueError("TURBOQUANT_KV_TOKEN_COUNT_MISMATCH")
        if keys.shape[0] == 0:
            return
        if self._key_width is None:
            self._key_width = keys.shape[1]
            self._value_width = values.shape[1]
        if keys.shape[1] != self._key_width or values.shape[1] != self._value_width:
            raise ValueError("TURBOQUANT_KV_WIDTH_CHANGED")
        block_id = len(self._key_blocks)
        self._key_blocks.append(
            quantize_vector(
                keys,
                bits=self.config.effective_key_bits,
                seed=self.config.seed + block_id * 2,
                rotation=self.config.rotation,
            )
        )
        self._value_blocks.append(
            quantize_vector(
                values,
                bits=self.config.effective_value_bits,
                seed=self.config.seed + block_id * 2 + 1,
                rotation=self.config.rotation,
            )
        )
        self._token_count += keys.shape[0]

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._key_blocks:
            key_width = self._key_width or 0
            value_width = self._value_width or 0
            return torch.empty((0, key_width)), torch.empty((0, value_width))
        return (
            torch.cat([block.dequantize() for block in self._key_blocks], dim=0),
            torch.cat([block.dequantize() for block in self._value_blocks], dim=0),
        )

    def attend(self, query: torch.Tensor) -> torch.Tensor:
        keys, values = self.materialize()
        query = torch.as_tensor(query)
        if query.ndim != 1 or query.shape[0] != keys.shape[1]:
            raise ValueError("TURBOQUANT_QUERY_WIDTH_MISMATCH")
        if keys.shape[0] == 0:
            return torch.zeros(values.shape[1], dtype=query.dtype, device=query.device)
        scores = query.to(keys.dtype) @ keys.T / math.sqrt(keys.shape[1])
        return torch.softmax(scores, dim=-1) @ values
