# GLM-5.x 모델 descriptor의 형상과 routing 계약을 검증합니다.

import pytest

from glm5x_ref.model import GLM5XModelDescriptor


def _glm52_config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "hidden_size": 6144,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2048,
        "index_topk": 2048,
        "index_topk_freq": 4,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_share_for_mtp_iteration": True,
        "max_position_embeddings": 1048576,
        "vocab_size": 154880,
        "num_nextn_predict_layers": 1,
    }


def test_glm52_descriptor_reads_dsa_top8_shape() -> None:
    descriptor = GLM5XModelDescriptor.from_config(_glm52_config())

    assert descriptor.model_family == "glm5"
    assert descriptor.attention_kind == "dsa"
    assert descriptor.routed_experts == 256
    assert descriptor.top_k == 8
    assert descriptor.mtp_layers == 1
    assert descriptor.moe_intermediate_size == 2048
    assert descriptor.index_topk == 2048
    assert descriptor.index_topk_freq == 4
    assert descriptor.index_n_heads == 32
    assert descriptor.index_head_dim == 128
    assert descriptor.index_share_for_mtp_iteration is True
    assert descriptor.max_position_embeddings == 1048576


def test_descriptor_rejects_k3_architecture() -> None:
    config = _glm52_config()
    config["architectures"] = ["SyntheticK3ForCausalLM"]

    with pytest.raises(ValueError, match="GLM_ARCHITECTURE_REQUIRED"):
        GLM5XModelDescriptor.from_config(config)


def test_descriptor_rejects_invalid_top_k() -> None:
    config = _glm52_config()
    config["num_experts_per_tok"] = 0

    with pytest.raises(ValueError, match="INVALID_TOP_K"):
        GLM5XModelDescriptor.from_config(config)
