# GLM-5.2 DSA 인덱서와 압축 KV 상태의 reference 계약을 검증합니다.

import pytest
import torch

from glm5x_ref.dsa import (
    GLM5XDSAConfig,
    GLM5XDSAState,
    estimate_dsa_state_bytes,
)
from glm5x_ref.model import GLM5XModelDescriptor
from glm5x_ref.turboquant import TurboQuantConfig


def _descriptor() -> GLM5XModelDescriptor:
    return GLM5XModelDescriptor.from_config(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "vocab_size": 32,
            "num_nextn_predict_layers": 1,
            "moe_intermediate_size": 24,
            "index_topk": 3,
            "index_topk_freq": 2,
            "index_n_heads": 2,
            "index_head_dim": 2,
            "index_share_for_mtp_iteration": True,
            "max_position_embeddings": 1_000_000,
        }
    )


def test_dsa_state_from_descriptor_keeps_exact_reference_selection() -> None:
    descriptor = _descriptor()
    config = GLM5XDSAConfig.from_descriptor(
        descriptor,
        kv_config=TurboQuantConfig(bits=16, rotation="none"),
        index_dtype=torch.float32,
    )
    state = GLM5XDSAState(config)
    index_keys = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    keys = index_keys.clone()
    values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    state.append(index_keys, keys, values)

    query = torch.tensor([0.0, 1.0, 0.0, 0.0])
    selected = state.select(query, reference_mode=True)
    expected = torch.topk(query @ index_keys.T / 4**0.5, k=3).indices

    assert torch.equal(selected, expected)
    actual = state.attend(query, reference_mode=True)
    selected_keys = keys[expected]
    weights = torch.softmax(query @ selected_keys.T / 4**0.5, dim=-1)
    assert torch.allclose(actual, weights @ values[expected], atol=1e-6)
    assert state.token_count == 4
    assert state.index_width == 4


def test_dsa_fast_selection_refreshes_at_declared_frequency() -> None:
    descriptor = _descriptor()
    config = GLM5XDSAConfig.from_descriptor(
        descriptor,
        kv_config=TurboQuantConfig(bits=16, rotation="none"),
        index_dtype=torch.float32,
    )
    state = GLM5XDSAState(config)
    index_keys = torch.eye(4)
    state.append(index_keys[:2], index_keys[:2], torch.ones((2, 4)))

    first = state.select(torch.tensor([1.0, 0.0, 0.0, 0.0]), reference_mode=False)
    assert state.last_selection_refreshed
    state.append(index_keys[2:3], index_keys[2:3], torch.ones((1, 4)))
    second = state.select(torch.tensor([0.0, 0.0, 1.0, 0.0]), reference_mode=False)
    assert not state.last_selection_refreshed
    assert torch.equal(second, first)

    state.append(index_keys[3:4], index_keys[3:4], torch.ones((1, 4)))
    third = state.select(torch.tensor([0.0, 0.0, 0.0, 1.0]), reference_mode=False)
    assert state.last_selection_refreshed
    assert int(third[0]) == 3


def test_dsa_capacity_estimate_is_formula_only_and_scales_to_one_million() -> None:
    descriptor = _descriptor()
    config = GLM5XDSAConfig.from_descriptor(
        descriptor,
        kv_config=TurboQuantConfig(key_bits=6, value_bits=4, rotation="none"),
        index_dtype=torch.bfloat16,
    )
    at_600k = estimate_dsa_state_bytes(
        tokens=600_000,
        index_width=config.index_width,
        key_width=256,
        value_width=256,
        config=config,
    )
    at_1m = estimate_dsa_state_bytes(
        tokens=1_000_000,
        index_width=config.index_width,
        key_width=256,
        value_width=256,
        config=config,
    )

    assert at_600k > 0
    assert at_1m > at_600k
    assert at_1m / at_600k == pytest.approx(1_000_000 / 600_000, rel=0.01)
