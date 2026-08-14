# GLM5X decoder layer의 input norm, DSA, MLA, MoE residual 연결을 검증합니다.
from __future__ import annotations

import torch
from safetensors.torch import save_file

from glm5x_converter.bundle import GLM5XExpertBundle, assemble_glm5x_expert_bundle
from glm5x_converter.shard import convert_glm5x_shard
from glm5x_ref.manifest import GLM5XTensorManifest
from glm5x_ref.layer10_moe import GLM5XExpertWeights, GLM5XLayer10MoEReference
from glm5x_ref.layer_reference import GLM5XDecoderLayerReference
from glm5x_ref.mla_dsa import GLM5XMLAReference, GLM5XMLAWeights
from glm5x_ref.official_dsa import GLM5XOfficialDSAIndexer


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


def test_decoder_layer_bundle_loader_reads_attention_and_experts(monkeypatch, tmp_path) -> None:
    hidden_size = 8
    heads = 2
    q_rank = 4
    kv_rank = 3
    nope = 2
    rope = 2
    value = 2
    intermediate = 3
    prefix = "model.layers.0"
    names: dict[str, torch.Tensor] = {
        f"{prefix}.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
        f"{prefix}.self_attn.q_a_proj.weight": torch.randn(q_rank, hidden_size).bfloat16(),
        f"{prefix}.self_attn.q_a_layernorm.weight": torch.ones(q_rank, dtype=torch.bfloat16),
        f"{prefix}.self_attn.q_b_proj.weight": torch.randn(heads * (nope + rope), q_rank).bfloat16(),
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(kv_rank + rope, hidden_size).bfloat16(),
        f"{prefix}.self_attn.kv_a_layernorm.weight": torch.ones(kv_rank, dtype=torch.bfloat16),
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(heads * (nope + value), kv_rank).bfloat16(),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden_size, heads * value).bfloat16(),
        f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(heads * 4, q_rank).bfloat16(),
        f"{prefix}.self_attn.indexer.wk.weight": torch.randn(4, hidden_size).bfloat16(),
        f"{prefix}.self_attn.indexer.k_norm.weight": torch.ones(4, dtype=torch.bfloat16),
        f"{prefix}.self_attn.indexer.k_norm.bias": torch.zeros(4, dtype=torch.bfloat16),
        f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(heads, hidden_size).bfloat16(),
        f"{prefix}.mlp.gate.weight": torch.randn(2, hidden_size).bfloat16(),
        f"{prefix}.mlp.gate.e_score_correction_bias": torch.zeros(2, dtype=torch.bfloat16),
        f"{prefix}.mlp.shared_experts.gate_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
        f"{prefix}.mlp.shared_experts.up_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
        f"{prefix}.mlp.shared_experts.down_proj.weight": torch.randn(hidden_size, intermediate).bfloat16(),
    }
    for expert_id in range(2):
        names.update(
            {
                f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.experts.{expert_id}.up_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.experts.{expert_id}.down_proj.weight": torch.randn(hidden_size, intermediate).bfloat16(),
            }
        )
    source = tmp_path / "model.safetensors"
    save_file(names, str(source))
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 1,
        "hidden_size": hidden_size,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_intermediate_size": intermediate,
        "index_topk": 1,
        "index_n_heads": heads,
        "index_head_dim": 4,
        "indexer_types": ["full"],
        "max_position_embeddings": 64,
        "vocab_size": 16,
        "num_nextn_predict_layers": 1,
    }
    manifest = GLM5XTensorManifest.from_json(
        config,
        {
            "metadata": {"total_size": source.stat().st_size},
            "weight_map": {name: source.name for name in names},
        },
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    convert_glm5x_shard(source, artifacts / "model.k3x", manifest, source.name)
    bundle_path = artifacts / "experts.json"
    report = assemble_glm5x_expert_bundle(artifacts, bundle_path)
    assert report.completed is True
    original_open = GLM5XExpertBundle.open
    open_count = 0

    def count_bundle_open(path):
        nonlocal open_count
        open_count += 1
        return original_open(path)

    monkeypatch.setattr(GLM5XExpertBundle, "open", staticmethod(count_bundle_open))
    layer = GLM5XDecoderLayerReference.from_bundle(
        bundle_path,
        layer_id=0,
        num_heads=heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=value,
        index_topk=1,
        top_k=1,
        expert_intermediate_size=intermediate,
        hidden_size=hidden_size,
    )
    hidden = torch.randn(1, 2, hidden_size)
    positions = torch.arange(2, dtype=torch.float32).view(1, 2)
    inverse = 1.0 / (10000.0 ** (torch.arange(0, rope, 2) / rope))
    frequencies = torch.cat((positions[..., None] * inverse,) * 2, dim=-1)
    result = layer(
        hidden,
        (frequencies.cos(), frequencies.sin()),
        position_ids=torch.arange(2).view(1, 2),
    )
    assert result.output.shape == hidden.shape
    assert result.attention_state.length == 2
    assert result.dsa_state is not None and result.dsa_state.length == 2
    assert result.moe.expert_load_count > 0
    assert open_count == 1
