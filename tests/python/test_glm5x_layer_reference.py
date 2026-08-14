# GLM5X decoder layer의 input norm, DSA, MLA, MoE residual 연결을 검증합니다.
from __future__ import annotations

import torch

from glm5x_ref.layer10_moe import GLM5XExpertWeights, GLM5XLayer10MoEReference
from glm5x_ref.layer_reference import GLM5XDecoderLayerReference
from glm5x_ref.mla_dsa import GLM5XMLAReference, GLM5XMLAWeights
from glm5x_ref.official_dsa import GLM5XOfficialDSAIndexer, build_glm_indexer_rope


def _make_layer() -> tuple[GLM5XDecoderLayerReference, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(41)
    hidden_size = 8
    heads = 2
    q_rank = 4
    kv_rank = 3
    nope = 2
    rope = 2
    value = 2
    rand = lambda *shape: torch.randn(shape, dtype=torch.float32)  # noqa: E731
    attention = GLM5XMLAReference(
        GLM5XMLAWeights(
            q_a_proj=rand(q_rank, hidden_size),
            q_a_norm=rand(q_rank),
            q_b_proj=rand(heads * (nope + rope), q_rank),
            kv_a_proj=rand(kv_rank + rope, hidden_size),
            kv_a_norm=rand(kv_rank),
            kv_b_proj=rand(heads * (nope + value), kv_rank),
            o_proj=rand(hidden_size, heads * value),
            num_heads=heads,
            qk_nope_head_dim=nope,
            qk_rope_head_dim=rope,
            v_head_dim=value,
        )
    )
    experts = {
        expert_id: GLM5XExpertWeights(
            gate_proj=rand(3, hidden_size),
            up_proj=rand(3, hidden_size),
            down_proj=rand(hidden_size, 3),
        )
        for expert_id in range(4)
    }
    moe = GLM5XLayer10MoEReference(
        router_weight=rand(4, hidden_size),
        correction_bias=torch.zeros(4),
        expert_loader=lambda expert_id: experts[expert_id],
        shared_expert=experts[0],
        top_k=2,
        routed_scaling_factor=2.5,
    )
    indexer = GLM5XOfficialDSAIndexer(
        wq_b=rand(8, q_rank),
        wk=rand(4, hidden_size),
        k_norm_weight=rand(4),
        k_norm_bias=rand(4),
        weights_proj=rand(2, hidden_size),
        qk_rope_head_dim=rope,
        index_topk=1,
    )
    positions = torch.arange(4, dtype=torch.float32).view(1, 4)
    inverse = 1.0 / (10000.0 ** (torch.arange(0, rope, 2) / rope))
    frequencies = torch.cat((positions[..., None] * inverse,) * 2, dim=-1)
    hidden = rand(1, 4, hidden_size)
    return (
        GLM5XDecoderLayerReference(
            input_layernorm=torch.ones(hidden_size),
            attention=attention,
            post_attention_layernorm=torch.ones(hidden_size),
            moe=moe,
            dsa_indexer=indexer,
        ),
        hidden,
        (frequencies.cos(), frequencies.sin()),
    )


def test_decoder_layer_incremental_matches_prefill_with_dsa_state() -> None:
    layer, hidden, position_embeddings = _make_layer()
    full = layer(
        hidden,
        position_embeddings,
        position_ids=torch.arange(4).view(1, 4),
    )
    first = layer(
        hidden[:, :3],
        (position_embeddings[0][:, :3], position_embeddings[1][:, :3]),
        position_ids=torch.arange(3).view(1, 3),
    )
    last = layer(
        hidden[:, 3:],
        (position_embeddings[0][:, 3:], position_embeddings[1][:, 3:]),
        position_ids=torch.tensor([[3]]),
        attention_state=first.attention_state,
        dsa_state=first.dsa_state,
    )
    torch.testing.assert_close(last.output, full.output[:, 3:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(last.topk_indices, full.topk_indices[:, 3:])
    assert last.attention_state.length == 4
    assert last.dsa_state is not None and last.dsa_state.length == 4
