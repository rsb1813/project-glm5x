# 여러 GLM5X expert의 NVFP4 projection을 하나의 grouped scaled GEMM으로 실행합니다.
from __future__ import annotations

from collections.abc import Sequence

import torch

from .nvfp4 import (
    GLM5XNVFP4Weight,
    _from_blocked,
    _to_blocked,
    linear_nvfp4,
    quantize_nvfp4_activation,
)


Activation = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _as_weights(
    weights: Sequence[GLM5XNVFP4Weight],
) -> tuple[GLM5XNVFP4Weight, ...]:
    selected = tuple(weights)
    if not selected:
        raise ValueError("GLM5X_NVFP4_BATCH_EMPTY")
    if any(not isinstance(weight, GLM5XNVFP4Weight) for weight in selected):
        raise ValueError("GLM5X_NVFP4_BATCH_TYPE")
    first_shape = selected[0].shape
    if any(weight.shape != first_shape for weight in selected[1:]):
        raise ValueError("GLM5X_NVFP4_BATCH_SHAPE")
    first_device = selected[0].device
    if any(weight.device != first_device for weight in selected[1:]):
        raise ValueError("GLM5X_NVFP4_BATCH_DEVICE")
    return selected


def _validate_activation(
    activation: Activation,
    weights: tuple[GLM5XNVFP4Weight, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activation_packed, activation_scales, activation_global = activation
    if activation_packed.device.type != "cuda":
        raise ValueError("GLM5X_NVFP4_ACTIVATION_CUDA_REQUIRED")
    if any(weight.device.type != "cuda" for weight in weights):
        raise ValueError("GLM5X_NVFP4_BATCH_CUDA_REQUIRED")
    if activation_packed.ndim != 2:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_SHAPE")
    if activation_scales.ndim != 1 or activation_global.numel() != 1:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_SCALE")
    rows, packed_cols = map(int, activation_packed.shape)
    if rows <= 0 or packed_cols <= 0:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_SHAPE")
    if activation_packed.dtype != torch.uint8:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_DTYPE")
    if activation_scales.dtype != torch.float8_e4m3fn:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_SCALE_DTYPE")
    if activation_global.dtype != torch.float32:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_GLOBAL_DTYPE")
    if packed_cols * 2 != weights[0].shape[1]:
        raise ValueError("GLM5X_NVFP4_ACTIVATION_SHAPE")
    return activation_packed, activation_scales, activation_global


def _combined_weight_payload(
    weights: tuple[GLM5XNVFP4Weight, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = weights[0].shape
    packed = torch.cat(tuple(weight.packed for weight in weights), dim=0)
    raw_scales = torch.cat(
        tuple(_from_blocked(weight.scales, rows, cols // 16) for weight in weights),
        dim=0,
    )
    blocked_scales = _to_blocked(raw_scales)
    global_scales = torch.stack(
        tuple(weight.global_scale.reshape(()).to(dtype=torch.float32) for weight in weights)
    )
    return packed, blocked_scales, global_scales


def _grouped_scaled_mm_from_activation(
    activation: Activation,
    weights: tuple[GLM5XNVFP4Weight, ...],
) -> torch.Tensor:
    activation_packed, activation_scales, activation_global = _validate_activation(
        activation, weights
    )
    packed, weight_scales, global_scales = _combined_weight_payload(weights)
    result = torch._scaled_mm(
        activation_packed.view(torch.float4_e2m1fn_x2),
        packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=activation_scales,
        scale_b=weight_scales,
        out_dtype=torch.bfloat16,
    )
    output_scale = torch.repeat_interleave(global_scales, weights[0].shape[0])
    result = result * (activation_global * output_scale).to(torch.bfloat16)
    return result.reshape(
        int(activation_packed.shape[0]), len(weights), weights[0].shape[0]
    )


def linear_nvfp4_batched_from_activation(
    activation: Activation,
    weights: Sequence[GLM5XNVFP4Weight],
) -> torch.Tensor:
    """하나의 양자화 activation으로 여러 expert weight를 계산합니다.

    반환 형상은 ``[tokens, experts, out_features]``이며 CUDA에서 단일
    ``torch._scaled_mm`` 호출을 사용합니다.
    """
    selected = _as_weights(weights)
    return _grouped_scaled_mm_from_activation(activation, selected)


def linear_nvfp4_batched(
    values: torch.Tensor,
    weights: Sequence[GLM5XNVFP4Weight],
) -> torch.Tensor:
    """값 입력을 받아 여러 expert projection을 실행합니다."""
    selected = _as_weights(weights)
    if values.shape[-1] != selected[0].shape[1]:
        raise ValueError("GLM5X_NVFP4_BATCH_INPUT_SHAPE")
    if (
        values.device.type != "cuda"
        or any(weight.device.type != "cuda" for weight in selected)
    ):
        return torch.stack(
            tuple(linear_nvfp4(values, weight) for weight in selected), dim=-2
        )
    activation = quantize_nvfp4_activation(values)
    result = _grouped_scaled_mm_from_activation(activation, selected)
    return result.reshape(*values.shape[:-1], len(selected), selected[0].shape[0])


def linear_nvfp4_gate_up_batched_from_activation(
    activation: Activation,
    gate_weights: Sequence[GLM5XNVFP4Weight],
    up_weights: Sequence[GLM5XNVFP4Weight],
) -> tuple[torch.Tensor, torch.Tensor]:
    """선택된 모든 gate/up weight를 한 번의 grouped scaled GEMM으로 계산합니다."""
    gate = _as_weights(gate_weights)
    up = _as_weights(up_weights)
    if len(gate) != len(up):
        raise ValueError("GLM5X_NVFP4_GATE_UP_BATCH_LENGTH")
    if gate[0].shape != up[0].shape:
        raise ValueError("GLM5X_NVFP4_GATE_UP_BATCH_SHAPE")
    if gate[0].device != up[0].device:
        raise ValueError("GLM5X_NVFP4_GATE_UP_BATCH_DEVICE")
    combined = _grouped_scaled_mm_from_activation(activation, gate + up)
    count = len(gate)
    return combined[:, :count, :], combined[:, count:, :]


def linear_nvfp4_gate_up_batched(
    values: torch.Tensor,
    gate_weights: Sequence[GLM5XNVFP4Weight],
    up_weights: Sequence[GLM5XNVFP4Weight],
) -> tuple[torch.Tensor, torch.Tensor]:
    """값 입력 gate/up 경로를 실행하고 CUDA에서 activation을 한 번 양자화합니다."""
    gate = _as_weights(gate_weights)
    up = _as_weights(up_weights)
    if len(gate) != len(up):
        raise ValueError("GLM5X_NVFP4_GATE_UP_BATCH_LENGTH")
    if values.shape[-1] != gate[0].shape[1] or values.shape[-1] != up[0].shape[1]:
        raise ValueError("GLM5X_NVFP4_GATE_UP_BATCH_SHAPE")
    if (
        values.device.type != "cuda"
        or any(weight.device.type != "cuda" for weight in gate + up)
    ):
        return (
            linear_nvfp4_batched(values, gate),
            linear_nvfp4_batched(values, up),
        )
    activation = quantize_nvfp4_activation(values)
    result_gate, result_up = linear_nvfp4_gate_up_batched_from_activation(
        activation, gate, up
    )
    original_shape = values.shape[:-1]
    shape = (*original_shape, len(gate), gate[0].shape[0])
    return result_gate.reshape(shape), result_up.reshape(shape)


__all__ = [
    "linear_nvfp4_batched",
    "linear_nvfp4_batched_from_activation",
    "linear_nvfp4_gate_up_batched",
    "linear_nvfp4_gate_up_batched_from_activation",
]
