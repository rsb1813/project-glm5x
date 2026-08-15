# RTX 5080 TinyGEMM INT4 weight-only packing과 선형 연산을 제공합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .nvfp4 import GLM5XNVFP4Weight, linear_nvfp4


@dataclass(frozen=True)
class GLM5XInt4Weight:
    """CUDA TinyGEMM packed weight with original linear shape."""

    packed: torch.Tensor
    scale_and_zero: torch.Tensor
    shape: tuple[int, int]
    group_size: int
    inner_k_tiles: int

    def __post_init__(self) -> None:
        if self.packed.device.type != "cuda":
            raise ValueError("GLM5X_INT4_CUDA_REQUIRED")
        if self.packed.dtype != torch.int32 or self.packed.ndim != 4:
            raise ValueError("GLM5X_INT4_PACKED_SHAPE")
        if self.scale_and_zero.device != self.packed.device:
            raise ValueError("GLM5X_INT4_QPARAM_DEVICE")
        if self.scale_and_zero.dtype != torch.bfloat16:
            raise ValueError("GLM5X_INT4_QPARAM_DTYPE")
        if self.scale_and_zero.ndim != 3 or self.scale_and_zero.shape[-1] != 2:
            raise ValueError("GLM5X_INT4_QPARAM_SHAPE")
        if len(self.shape) != 2 or any(value <= 0 for value in self.shape):
            raise ValueError("GLM5X_INT4_WEIGHT_SHAPE")
        if self.shape[0] % 8 != 0:
            raise ValueError("GLM5X_INT4_OUTPUT_ALIGNMENT")
        if self.shape[1] % self.group_size != 0:
            raise ValueError("GLM5X_INT4_GROUP_ALIGNMENT")
        if self.shape[1] % (self.group_size * self.inner_k_tiles) != 0:
            raise ValueError("GLM5X_INT4_TILE_ALIGNMENT")
        expected_groups = self.shape[1] // self.group_size
        if tuple(self.scale_and_zero.shape) != (
            expected_groups,
            self.shape[0],
            2,
        ):
            raise ValueError("GLM5X_INT4_QPARAM_SHAPE")

    @property
    def device(self) -> torch.device:
        return self.packed.device

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def ndim(self) -> int:
        return 2

    def num_bytes(self) -> int:
        return int(self.packed.numel() * self.packed.element_size()) + int(
            self.scale_and_zero.numel() * self.scale_and_zero.element_size()
        )


def _choose_qparams(
    weight: torch.Tensor,
    group_size: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = weight.to(device=device, dtype=torch.float32)
    grouped = source.reshape(source.shape[0], -1, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scale_fp32 = ((maximum - minimum) / 15.0).clamp_min(1e-6)
    scale = scale_fp32.to(dtype=torch.bfloat16)
    zero = (minimum + scale_fp32 * 8.0).to(dtype=torch.bfloat16)
    return source, scale, zero


def quantize_int4_weight(
    weight: torch.Tensor,
    *,
    group_size: int = 128,
    inner_k_tiles: int | None = None,
    device: torch.device | str | None = None,
) -> GLM5XInt4Weight:
    """Quantize a BF16/FP32 [out, in] matrix and pack it for CUDA TinyGEMM."""
    source = torch.as_tensor(weight).detach()
    if source.ndim != 2:
        raise ValueError("GLM5X_INT4_WEIGHT_RANK")
    out_features, in_features = (int(source.shape[0]), int(source.shape[1]))
    if out_features % 8 or in_features % group_size:
        raise ValueError("GLM5X_INT4_WEIGHT_ALIGNMENT")
    if inner_k_tiles is None:
        candidates = (8, 4, 2)
        inner_k_tiles = next(
            (value for value in candidates if in_features % (group_size * value) == 0),
            0,
        )
    if inner_k_tiles not in (2, 4, 8) or in_features % (group_size * inner_k_tiles):
        raise ValueError("GLM5X_INT4_TILE_ALIGNMENT")
    if not torch.cuda.is_available():
        raise RuntimeError("GLM5X_INT4_CUDA_UNAVAILABLE")
    target = torch.device("cuda" if device is None else device)
    if target.type != "cuda":
        raise ValueError("GLM5X_INT4_CUDA_REQUIRED")

    source_device, scale, zero = _choose_qparams(
        source, group_size, device=target
    )
    grouped = source_device.reshape(out_features, -1, group_size)
    q = torch.round(
        (grouped - zero.to(dtype=torch.float32).unsqueeze(-1))
        / scale.to(dtype=torch.float32).unsqueeze(-1)
        + 8.0
    ).clamp_(0, 15).to(dtype=torch.uint8)
    q = q.reshape(out_features, in_features)
    # TinyGEMM expects the first source nibble in the high half of each byte.
    packed_bytes = ((q[:, 0::2] << 4) | q[:, 1::2]).contiguous()
    packed = torch._convert_weight_to_int4pack(
        packed_bytes.to(device=target), int(inner_k_tiles)
    )
    scale_and_zero = torch.cat(
        (scale.unsqueeze(-1), zero.unsqueeze(-1)), dim=-1
    ).transpose(0, 1).contiguous().to(device=target)
    return GLM5XInt4Weight(
        packed=packed,
        scale_and_zero=scale_and_zero,
        shape=(out_features, in_features),
        group_size=group_size,
        inner_k_tiles=int(inner_k_tiles),
    )


def linear(
    values: torch.Tensor,
    weight: torch.Tensor | GLM5XInt4Weight | GLM5XNVFP4Weight,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run BF16 TinyGEMM linear or the exact eager fallback for normal tensors."""
    if isinstance(weight, GLM5XNVFP4Weight):
        output = linear_nvfp4(values, weight)
        if bias is not None:
            output = output + torch.as_tensor(bias, device=output.device, dtype=output.dtype)
        return output
    if isinstance(weight, GLM5XInt4Weight):
        if values.device != weight.device:
            raise ValueError("GLM5X_INT4_VALUE_DEVICE")
        original_shape = values.shape[:-1]
        flat = values.reshape(-1, weight.shape[1]).to(dtype=torch.bfloat16)
        output = torch._weight_int4pack_mm(
            flat,
            weight.packed,
            weight.group_size,
            weight.scale_and_zero,
        )
        if bias is not None:
            output = output + torch.as_tensor(bias, device=output.device, dtype=output.dtype)
        return output.reshape(*original_shape, weight.shape[0])
    tensor = torch.as_tensor(weight).to(device=values.device)
    if values.dtype != tensor.dtype:
        values = values.to(tensor.dtype)
    if bias is not None:
        bias = torch.as_tensor(bias).to(device=values.device, dtype=tensor.dtype)
    return F.linear(values, tensor, bias)


def weight_shape(
    weight: torch.Tensor | GLM5XInt4Weight | GLM5XNVFP4Weight,
) -> tuple[int, ...]:
    if isinstance(weight, (GLM5XInt4Weight, GLM5XNVFP4Weight)):
        return weight.shape
    return tuple(torch.as_tensor(weight).shape)
