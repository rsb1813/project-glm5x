# GLM5X INT4 packed expert sidecar의 원자적 저장과 재로딩을 검증합니다.
from __future__ import annotations

import pytest
import torch

from glm5x_ref.int4 import GLM5XInt4Weight, quantize_int4_weight
from glm5x_ref.layer10_moe import GLM5XExpertWeights
from glm5x_ref.packed_cache import GLM5XPackedExpertCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip(tmp_path) -> None:
    torch.manual_seed(23)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
        up_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
        down_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-0000000000000000000000000000000000000000"
    cache.put((4, 7), digest, expert)
    loaded = cache.get((4, 7), digest, device="cuda")
    assert loaded is not None
    assert isinstance(loaded.gate_proj, GLM5XInt4Weight)
    assert loaded.gate_proj.shape == expert.gate_proj.shape
    torch.testing.assert_close(loaded.gate_proj.packed, expert.gate_proj.packed)
    torch.testing.assert_close(
        loaded.gate_proj.scale_and_zero, expert.gate_proj.scale_and_zero
    )
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_fp8(tmp_path) -> None:
    torch.manual_seed(29)
    weights = tuple(
        torch.randn(256, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        for _ in range(3)
    )
    scales = tuple(torch.rand(256, 1, dtype=torch.float32) + 0.25 for _ in range(3))
    expert = GLM5XExpertWeights(
        gate_proj=weights[0],
        up_proj=weights[1],
        down_proj=weights[2],
        gate_scale=scales[0],
        up_scale=scales[1],
        down_scale=scales[2],
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-fp8-000000000000000000000000000000000000"
    cache.put((4, 8), digest, expert, precision="fp8")
    loaded = cache.get((4, 8), digest, device="cuda", precision="fp8")
    assert loaded is not None
    assert loaded.is_fp8
    torch.testing.assert_close(loaded.gate_proj, expert.gate_proj.cuda())
    torch.testing.assert_close(loaded.gate_scale, expert.gate_scale.cuda())
    assert (tmp_path / "layer-0004-expert-0008.pf8").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_mxfp4(tmp_path) -> None:
    torch.manual_seed(37)
    expert = GLM5XExpertWeights(
        gate_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
        up_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
        down_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-mxfp4-000000000000000000000000000000000"
    cache.put((4, 9), digest, expert, precision="mxfp4")
    loaded = cache.get((4, 9), digest, device="cuda", precision="mxfp4")
    assert loaded is not None
    assert (tmp_path / "layer-0004-expert-0009.pm4").exists()
    assert loaded.gate_proj.dtype == torch.bfloat16
    assert loaded.gate_proj.shape == expert.gate_proj.shape
    assert torch.isfinite(loaded.gate_proj).all()
    assert float(
        (loaded.gate_proj.float() - expert.gate_proj.float()).norm()
        / expert.gate_proj.float().norm()
    ) < 0.25
