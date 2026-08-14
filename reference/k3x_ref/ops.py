# Kimi K3 참조 그래프의 공통 FP32 수치 연산을 제공합니다.
from __future__ import annotations

import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    source = x.to(torch.float32)
    gain = weight.to(device=x.device, dtype=torch.float32)
    normalized = source * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * gain).to(x.dtype)


def situ_glu(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate_fp32 = gate.to(torch.float32)
    up_fp32 = up.to(torch.float32)
    bounded_gate = beta * torch.tanh(gate_fp32 / beta) * torch.sigmoid(gate_fp32)
    if linear_beta is not None:
        up_fp32 = linear_beta * torch.tanh(up_fp32 / linear_beta)
    return (bounded_gate * up_fp32).to(torch.promote_types(gate.dtype, up.dtype))

