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
