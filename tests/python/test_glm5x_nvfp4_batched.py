# GLM5X 선택 expert NVFP4 gate/up grouped projection의 parity를 검증합니다.
from __future__ import annotations

import pytest
import torch

from glm5x_ref.nvfp4 import (
    GLM5XNVFP4Weight,
    linear_nvfp4,
    linear_nvfp4_from_activation,
    quantize_nvfp4_activation,
    quantize_nvfp4_weight,
)
from glm5x_ref.nvfp4_batched import (
    linear_nvfp4_batched,
    linear_nvfp4_batched_from_activation,
    linear_nvfp4_gate_up_batched,
    linear_nvfp4_gate_up_batched_from_activation,
)


def _weights(count: int, *, device: str) -> tuple[GLM5XNVFP4Weight, ...]:
    torch.manual_seed(101)
    return tuple(
        quantize_nvfp4_weight(
            torch.randn(32, 64, dtype=torch.bfloat16, device=device),
            device=device,
        )
        for _ in range(count)
    )


def test_nvfp4_batched_cpu_preserves_expert_order_and_shape() -> None:
    torch.manual_seed(103)
    weights = _weights(3, device="cpu")
    hidden = torch.randn(2, 64, dtype=torch.bfloat16)

    actual = linear_nvfp4_batched(hidden, weights)
    expected = torch.stack(tuple(linear_nvfp4(hidden, weight) for weight in weights), dim=1)

    assert actual.shape == (2, 3, 32)
    assert torch.equal(actual, expected)


def test_nvfp4_gate_up_batched_cpu_matches_two_per_expert_groups() -> None:
    torch.manual_seed(107)
    gate = _weights(3, device="cpu")
    up = _weights(3, device="cpu")
    hidden = torch.randn(2, 64, dtype=torch.bfloat16)

    actual_gate, actual_up = linear_nvfp4_gate_up_batched(hidden, gate, up)
    expected_gate = torch.stack(tuple(linear_nvfp4(hidden, weight) for weight in gate), dim=1)
    expected_up = torch.stack(tuple(linear_nvfp4(hidden, weight) for weight in up), dim=1)

    assert torch.equal(actual_gate, expected_gate)
    assert torch.equal(actual_up, expected_up)


def test_nvfp4_batched_rejects_empty_or_inconsistent_experts() -> None:
    hidden = torch.randn(1, 64, dtype=torch.bfloat16)
    weight = _weights(1, device="cpu")[0]
    other = quantize_nvfp4_weight(torch.randn(16, 64, dtype=torch.bfloat16), device="cpu")

    with pytest.raises(ValueError, match="GLM5X_NVFP4_BATCH_EMPTY"):
        linear_nvfp4_batched(hidden, ())
    with pytest.raises(ValueError, match="GLM5X_NVFP4_BATCH_SHAPE"):
        linear_nvfp4_batched(hidden, (weight, other))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_batched_cuda_matches_per_expert_relative_and_bit_parity() -> None:
    torch.manual_seed(109)
    weights = _weights(4, device="cuda")
    hidden = torch.randn(3, 64, dtype=torch.bfloat16, device="cuda")
    activation = quantize_nvfp4_activation(hidden)

    actual = linear_nvfp4_batched_from_activation(activation, weights)
    expected = torch.stack(
        tuple(linear_nvfp4_from_activation(activation, weight) for weight in weights),
        dim=1,
    )
    exact = torch.equal(actual, expected)
    relative = float((actual.float() - expected.float()).norm() / expected.float().norm())

    assert actual.shape == expected.shape == (3, 4, 32)
    assert torch.isfinite(actual).all()
    assert exact or relative < 1e-2
    assert relative < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nvfp4_gate_up_batched_cuda_uses_shared_activation_and_matches_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(113)
    gate = _weights(3, device="cuda")
    up = _weights(3, device="cuda")
    hidden = torch.randn(2, 64, dtype=torch.bfloat16, device="cuda")
    activation = quantize_nvfp4_activation(hidden)

    calls = 0
    original_scaled_mm = torch._scaled_mm

    def counted_scaled_mm(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_scaled_mm(*args, **kwargs)

    # gate와 up 전체가 하나의 native scaled GEMM으로 결합되는지 확인합니다.
    monkeypatch.setattr(torch, "_scaled_mm", counted_scaled_mm)
    actual_gate, actual_up = linear_nvfp4_gate_up_batched_from_activation(
        activation, gate, up
    )
    assert calls == 1
    expected_gate = torch.stack(
        tuple(linear_nvfp4_from_activation(activation, weight) for weight in gate),
        dim=1,
    )
    expected_up = torch.stack(
        tuple(linear_nvfp4_from_activation(activation, weight) for weight in up),
        dim=1,
    )

    for actual, expected in ((actual_gate, expected_gate), (actual_up, expected_up)):
        relative = float((actual.float() - expected.float()).norm() / expected.float().norm())
        assert actual.shape == expected.shape == (2, 3, 32)
        assert torch.isfinite(actual).all()
        assert relative < 1e-2
