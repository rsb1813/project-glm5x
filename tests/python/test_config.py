# 합성 K3 설정의 실제 위상과 패킹 제약을 검증합니다.
import pytest

from k3x_ref.config import SyntheticK3Config


def test_default_config_reproduces_minimal_k3_topology() -> None:
    cfg = SyntheticK3Config.default()
    assert cfg.layer_kinds == ("kda", "kda", "kda", "mla")
    assert cfg.dense_layers == (0,)
    assert cfg.hidden_size == 64
    assert cfg.num_experts == 8
    assert cfg.top_k == 2
    assert cfg.attn_res_block_size == 2
    assert cfg.mla_use_nope is True
    assert cfg.mla_use_output_gate is True
    assert cfg.activation_situ_beta == 4.0
    assert cfg.activation_situ_linear_beta == 25.0


def test_config_rejects_mxfp4_incompatible_expert_width() -> None:
    cfg = SyntheticK3Config.default().replace(expert_intermediate_size=31)
    with pytest.raises(ValueError, match="expert_intermediate_size"):
        cfg.validate()


def test_config_rejects_rope_enabled_mla() -> None:
    cfg = SyntheticK3Config.default().replace(mla_use_nope=False)
    with pytest.raises(ValueError, match="mla_use_nope"):
        cfg.validate()
