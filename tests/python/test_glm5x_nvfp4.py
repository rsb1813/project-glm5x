# GLM5X NVFP4 양자화와 RTX 5080 scaled GEMM 경로를 검증하는 테스트입니다.
from __future__ import annotations

import pytest
import torch

from glm5x_ref.nvfp4 import (
    GLM5XNVFP4Weight,
    dequantize_nvfp4,
    linear_nvfp4,
    quantize_nvfp4_weight,
)


def test_nvfp4_cpu_round_trip_has_expected_shape_and_finite_values() -> None:
    torch.manual_seed(41)
    source = torch.randn(64, 128, dtype=torch.bfloat16)
    quantized = quantize_nvfp4_weight(source, device="cpu")
    assert isinstance(quantized, GLM5XNVFP4Weight)
    assert quantized.shape == (64, 128)
    decoded = dequantize_nvfp4(quantized)
    assert decoded.shape == source.shape
    assert torch.isfinite(decoded).all()
    assert float((decoded.float() - source.float()).norm() / source.float().norm()) < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_cuda_linear_matches_dequantized_reference() -> None:
    torch.manual_seed(43)
    source = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
    quantized = quantize_nvfp4_weight(source, device="cuda")
    actual = linear_nvfp4(hidden, quantized)
    activation_quantized = quantize_nvfp4_weight(
        hidden, device="cuda", scale_mode="max_abs"
    )
    reference = torch.nn.functional.linear(
        dequantize_nvfp4(activation_quantized).to(dtype=hidden.dtype),
        dequantize_nvfp4(quantized).to(dtype=hidden.dtype),
    )
    relative = float((actual.float() - reference.float()).norm() / reference.float().norm())
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()
    assert relative < 0.01
