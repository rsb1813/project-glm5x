# GLM5X CUDA INT4 선형 경로의 형태와 수치 오차를 검증합니다.
from __future__ import annotations

import pytest
import torch

from glm5x_ref.int4 import GLM5XInt4Weight, linear, quantize_int4_weight
from glm5x_ref.layer10_moe import GLM5XExpertWeights, GLM5XLayer10MoEReference


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_int4_linear_matches_dequantized_reference() -> None:
    torch.manual_seed(7)
    weight = torch.randn(128, 512, dtype=torch.bfloat16)
    values = torch.randn(1, 512, dtype=torch.bfloat16, device="cuda")
    packed = quantize_int4_weight(weight, group_size=128)
    assert isinstance(packed, GLM5XInt4Weight)
    result = linear(values, packed)
    scale, zero = packed.scale_and_zero.transpose(0, 1).unbind(-1)
    # Compare against the quantizer's direct affine reconstruction on CPU.
    source = weight.to(dtype=torch.float32).reshape(128, 4, 128)
    scale_cpu = scale.cpu().to(dtype=torch.float32)
    zero_cpu = zero.cpu().to(dtype=torch.float32)
    reconstructed = (
        torch.round(
            (source - zero_cpu.unsqueeze(-1)) / scale_cpu.unsqueeze(-1) + 8.0
        ).clamp(0, 15)
        - 8.0
    ) * scale_cpu.unsqueeze(-1) + zero_cpu.unsqueeze(-1)
    reference = values.float().cpu() @ reconstructed.reshape(128, 512).cpu().t()
    assert result.shape == (1, 128)
    assert torch.allclose(result.float().cpu(), reference, atol=0.25, rtol=0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_int4_expert_mlp_uses_packed_projections() -> None:
    torch.manual_seed(11)
    expert = GLM5XExpertWeights(
        gate_proj=torch.randn(256, 256, dtype=torch.bfloat16),
        up_proj=torch.randn(256, 256, dtype=torch.bfloat16),
        down_proj=torch.randn(256, 256, dtype=torch.bfloat16),
    )
    packed = GLM5XLayer10MoEReference._quantize_expert_int4(expert, device="cuda")
    hidden = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
    actual = GLM5XLayer10MoEReference._mlp(hidden, packed)
    gate = torch.nn.functional.linear(hidden, expert.gate_proj.to("cuda"))
    up = torch.nn.functional.linear(hidden, expert.up_proj.to("cuda"))
    reference = torch.nn.functional.linear(
        torch.nn.functional.silu(gate) * up, expert.down_proj.to("cuda")
    )
    assert isinstance(packed.gate_proj, GLM5XInt4Weight)
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
    relative = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert float(relative) < 0.25
