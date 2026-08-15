# RTX 5080 NVFP4 양자화와 Blackwell scaled GEMM 실행을 담당합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
_FP4_MAX = 6.0
_FP8_MAX = 448.0
_BLOCK_SIZE = 16


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """Convert row-major [M, K/16] scales to the cuBLAS blocked layout."""
    rows, cols = (int(scales.shape[0]), int(scales.shape[1]))
    row_blocks = _ceil_div(rows, 128)
    col_blocks = _ceil_div(cols, 4)
    padded = torch.zeros(
        (row_blocks * 128, col_blocks * 4),
        dtype=scales.dtype,
        device=scales.device,
    )
    padded[:rows, :cols] = scales
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    return (
        blocks.reshape(-1, 4, 32, 4)
        .transpose(1, 2)
        .reshape(-1, 32, 16)
        .flatten()
    )


def _from_blocked(
    blocked: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    row_blocks = _ceil_div(rows, 128)
    col_blocks = _ceil_div(cols, 4)
    expected = row_blocks * col_blocks * 32 * 16
    if int(blocked.numel()) != expected:
        raise ValueError("GLM5X_NVFP4_SCALE_LENGTH")
    rearranged = blocked.reshape(row_blocks * col_blocks, 32, 16)
    temp = rearranged.reshape(row_blocks * col_blocks, 32, 4, 4).transpose(1, 2)
    blocks = temp.reshape(row_blocks, col_blocks, 128, 4)
    padded = blocks.permute(0, 2, 1, 3).reshape(row_blocks * 128, col_blocks * 4)
    return padded[:rows, :cols]


@dataclass(frozen=True)
class GLM5XNVFP4Weight:
    """Packed NVFP4 [out, in] weight with blocked FP8 scales."""

    packed: torch.Tensor
    scales: torch.Tensor
    global_scale: torch.Tensor
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        if self.packed.dtype != torch.uint8 or self.packed.ndim != 2:
            raise ValueError("GLM5X_NVFP4_PACKED_SHAPE")
        if self.scales.dtype != torch.float8_e4m3fn or self.scales.ndim != 1:
            raise ValueError("GLM5X_NVFP4_SCALE_DTYPE")
        if self.global_scale.dtype != torch.float32 or self.global_scale.numel() != 1:
            raise ValueError("GLM5X_NVFP4_GLOBAL_SCALE")
        if len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape):
            raise ValueError("GLM5X_NVFP4_WEIGHT_SHAPE")
        rows, cols = (int(self.shape[0]), int(self.shape[1]))
        if cols % _BLOCK_SIZE or cols % 2 or tuple(self.packed.shape) != (rows, cols // 2):
            raise ValueError("GLM5X_NVFP4_WEIGHT_ALIGNMENT")
        if self.packed.device != self.scales.device or self.packed.device != self.global_scale.device:
            raise ValueError("GLM5X_NVFP4_DEVICE_MISMATCH")
        expected = _ceil_div(rows, 128) * _ceil_div(cols // _BLOCK_SIZE, 4) * 32 * 16
        if int(self.scales.numel()) != expected:
            raise ValueError("GLM5X_NVFP4_SCALE_SHAPE")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def ndim(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def num_bytes(self) -> int:
        return int(self.packed.numel()) + int(self.scales.numel()) + int(self.global_scale.numel() * 4)


def _quantize_payload(
    values: torch.Tensor,
    *,
    chunk_rows: int = 128,
    scale_mode: str = "mse",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.as_tensor(values)
    if source.ndim != 2 or not source.is_floating_point():
        raise ValueError("GLM5X_NVFP4_SOURCE_MATRIX")
    rows, cols = (int(source.shape[0]), int(source.shape[1]))
    if rows <= 0 or cols <= 0 or cols % _BLOCK_SIZE or cols % 2:
        raise ValueError("GLM5X_NVFP4_SOURCE_ALIGNMENT")
    if chunk_rows <= 0:
        raise ValueError("GLM5X_NVFP4_CHUNK_ROWS")
    if scale_mode not in {"max_abs", "mse"}:
        raise ValueError("GLM5X_NVFP4_SCALE_MODE")
    source = source.detach().to(dtype=torch.float32).contiguous()
    if not torch.isfinite(source).all():
        raise ValueError("GLM5X_NVFP4_SOURCE_FINITE")
    grouped = source.reshape(rows, cols // _BLOCK_SIZE, _BLOCK_SIZE)
    block_max = grouped.abs().amax(dim=-1)
    maximum = block_max.max()
    global_scale = torch.maximum(
        maximum / (_FP8_MAX * _FP4_MAX),
        torch.tensor(1e-12, dtype=torch.float32, device=source.device),
    )
    base_scales = (block_max / (_FP4_MAX * global_scale)).clamp_(
        min=torch.finfo(torch.float8_e4m3fn).tiny,
        max=_FP8_MAX,
    )
    codebook = E2M1_VALUES.to(device=source.device)
    fp8_scales = torch.empty_like(base_scales, dtype=torch.float8_e4m3fn)
    factors = torch.tensor(
        [0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.5, 2.0],
        dtype=torch.float32,
        device=source.device,
    )
    packed_chunks: list[torch.Tensor] = []
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        grouped_chunk = source[start:stop].reshape(
            stop - start, cols // _BLOCK_SIZE, _BLOCK_SIZE
        )
        if scale_mode == "mse":
            candidates = (
                base_scales[start:stop].unsqueeze(-1) * factors
            ).clamp_(
                min=torch.finfo(torch.float8_e4m3fn).tiny,
                max=_FP8_MAX,
            ).to(torch.float8_e4m3fn).to(torch.float32)
            normalized = grouped_chunk.unsqueeze(2) / (
                candidates * global_scale
            ).unsqueeze(-1)
            candidate_codes = (
                normalized.unsqueeze(-1) - codebook
            ).abs().argmin(dim=-1)
            candidate_reconstructed = codebook[candidate_codes] * (
                candidates * global_scale
            ).unsqueeze(-1)
            errors = (candidate_reconstructed - grouped_chunk.unsqueeze(2)).square().mean(dim=-1)
            best = errors.argmin(dim=2)
            selected = candidates.gather(2, best.unsqueeze(-1)).squeeze(-1)
            codes = candidate_codes.gather(
                2, best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, _BLOCK_SIZE)
            ).squeeze(2)
            fp8_scales[start:stop] = selected.to(torch.float8_e4m3fn)
        else:
            selected = base_scales[start:stop].to(torch.float8_e4m3fn).to(torch.float32)
            fp8_scales[start:stop] = selected.to(torch.float8_e4m3fn)
            normalized = grouped_chunk / (selected * global_scale).unsqueeze(-1)
            codes = (normalized.unsqueeze(-1) - codebook).abs().argmin(dim=-1)
        codes = codes.to(torch.uint8)
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(
            stop - start, cols // 2
        )
        packed_chunks.append(packed.contiguous())
    return torch.cat(packed_chunks, dim=0), fp8_scales, global_scale.reshape(())


def quantize_nvfp4_weight(
    weight: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    chunk_rows: int = 128,
    scale_mode: str = "mse",
) -> GLM5XNVFP4Weight:
    """Quantize a row-major weight and retain its global decoding scale."""
    source = torch.as_tensor(weight)
    target = source.device if device is None else torch.device(device)
    packed, raw_scales, global_scale = _quantize_payload(
        source.to(device=target), chunk_rows=chunk_rows, scale_mode=scale_mode
    )
    return GLM5XNVFP4Weight(
        packed=packed,
        scales=_to_blocked(raw_scales),
        global_scale=global_scale,
        shape=(int(source.shape[0]), int(source.shape[1])),
    )


def dequantize_nvfp4(weight: GLM5XNVFP4Weight) -> torch.Tensor:
    """Decode an NVFP4 weight for correctness checks or CPU fallback."""
    rows, cols = weight.shape
    codes = torch.stack(
        (
            weight.packed.bitwise_and(0x0F),
            weight.packed.bitwise_right_shift(4),
        ),
        dim=-1,
    ).reshape(rows, cols)
    values = E2M1_VALUES.to(device=weight.device)[codes.long()]
    raw_scales = _from_blocked(weight.scales, rows, cols // _BLOCK_SIZE)
    decoded_scales = raw_scales.to(torch.float32) * weight.global_scale
    return (
        values.reshape(rows, cols // _BLOCK_SIZE, _BLOCK_SIZE)
        * decoded_scales.unsqueeze(-1)
    ).reshape(rows, cols)


def _scaled_mm_nvfp4(
    activation_packed: torch.Tensor,
    activation_blocked_scales: torch.Tensor,
    activation_global_scale: torch.Tensor,
    weight: GLM5XNVFP4Weight,
) -> torch.Tensor:
    result = torch._scaled_mm(
        activation_packed.view(torch.float4_e2m1fn_x2),
        weight.packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=activation_blocked_scales,
        scale_b=weight.scales,
        out_dtype=torch.bfloat16,
    )
    return result * (activation_global_scale * weight.global_scale).to(torch.bfloat16)


def _quantize_cuda_activation(
    flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed, raw_scales, global_scale = _quantize_payload(
        flat, scale_mode="max_abs"
    )
    return packed, _to_blocked(raw_scales), global_scale


def linear_nvfp4(
    values: torch.Tensor,
    weight: GLM5XNVFP4Weight,
) -> torch.Tensor:
    """Run dynamic-activation NVFP4 GEMM on Blackwell or a CPU reference fallback."""
    if values.shape[-1] != weight.shape[1]:
        raise ValueError("GLM5X_NVFP4_LINEAR_SHAPE")
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, weight.shape[1])
    if flat.device.type != "cuda" or weight.device.type != "cuda":
        decoded = dequantize_nvfp4(weight).to(device=flat.device, dtype=flat.dtype)
        return F.linear(flat, decoded).reshape(*original_shape, weight.shape[0])
    activation_packed, activation_blocked, activation_global = _quantize_cuda_activation(flat)
    return _scaled_mm_nvfp4(
        activation_packed, activation_blocked, activation_global, weight
    ).reshape(*original_shape, weight.shape[0])


def linear_nvfp4_pair(
    values: torch.Tensor,
    gate_weight: GLM5XNVFP4Weight,
    up_weight: GLM5XNVFP4Weight,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run gate/up NVFP4 GEMMs while quantizing the shared activation once."""
    if values.shape[-1] != gate_weight.shape[1] or values.shape[-1] != up_weight.shape[1]:
        raise ValueError("GLM5X_NVFP4_LINEAR_PAIR_SHAPE")
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    if (
        flat.device.type != "cuda"
        or gate_weight.device.type != "cuda"
        or up_weight.device.type != "cuda"
    ):
        return linear_nvfp4(values, gate_weight), linear_nvfp4(values, up_weight)
    activation_packed, activation_blocked, activation_global = _quantize_cuda_activation(flat)
    gate = _scaled_mm_nvfp4(
        activation_packed, activation_blocked, activation_global, gate_weight
    ).reshape(*original_shape, gate_weight.shape[0])
    up = _scaled_mm_nvfp4(
        activation_packed, activation_blocked, activation_global, up_weight
    ).reshape(*original_shape, up_weight.shape[0])
    return gate, up
