# GLM5X 다층 decoder 및 incremental greedy 생성 경로를 검증합니다.
from __future__ import annotations

import torch
from safetensors.torch import save_file

from glm5x_converter.bundle import assemble_glm5x_expert_bundle
from glm5x_converter.shard import convert_glm5x_shard
from glm5x_ref import GLM5XDecoderModelReference
from glm5x_ref.layer10_moe import GLM5XDenseMlpReference, GLM5XLayer10MoEReference
from glm5x_ref.manifest import GLM5XTensorManifest
from test_glm5x_layer_reference import _make_layer


def test_model_prefill_incremental_and_greedy_generation_match() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(73)
    model = GLM5XDecoderModelReference(
        embedding=torch.randn(16, 8),
        layers=(layer, layer),
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
    )
    prompt = [1, 4, 7]
    full = model.forward_tokens(torch.tensor(prompt))
    state = model.empty_state()
    for index, token in enumerate(prompt):
        step = model.forward_token(token, state)
        torch.testing.assert_close(step.logits, full.logits[:, index : index + 1], rtol=1e-5, atol=1e-5)
        state = step.state

    expected = prompt[:]
    state = model.empty_state()
    forward = model.forward_tokens(torch.tensor(expected))
    state = forward.state
    for _ in range(2):
        token = int(torch.argmax(forward.logits[:, -1, :], dim=-1).item())
        expected.append(token)
        forward = model.forward_token(token, state)
        state = forward.state
    assert model.generate(prompt, 2) == expected


def test_model_reuses_prepared_fp32_lm_head() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(74)
    model = GLM5XDecoderModelReference(
        embedding=torch.randn(16, 8),
        layers=(layer,),
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
    )

    assert model.prepared_lm_head is None
    first = model.forward_tokens(torch.tensor([1]))
    prepared = model.prepared_lm_head
    assert prepared is not None
    second = model.forward_tokens(torch.tensor([2]))
    assert model.prepared_lm_head is prepared
    assert prepared.dtype == torch.float32
    assert prepared.device == first.logits.device
    assert second.logits.shape == first.logits.shape


def test_model_reference_can_load_one_layer_at_a_time() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(79)
    calls: list[int] = []

    def load_layer(layer_id: int):
        calls.append(layer_id)
        return layer

    model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=torch.randn(16, 8),
        layer_count=2,
        layer_loader=load_layer,
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
        rope_dim=2,
    )
    forward = model.forward_tokens(torch.tensor([1, 2]))
    assert calls == [0, 1]
    assert model.layer_count == 2
    assert len(forward.layers) == 2


def test_model_reference_can_retain_trunk_layers_between_forwards() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(83)
    calls: list[int] = []

    def load_layer(layer_id: int):
        calls.append(layer_id)
        return layer

    model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=torch.randn(16, 8),
        layer_count=2,
        layer_loader=load_layer,
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
        rope_dim=2,
        layer_cache_capacity=2,
    )
    first = model.forward_tokens(torch.tensor([1, 2]))
    second = model.forward_tokens(torch.tensor([1, 2]))
    torch.testing.assert_close(first.logits, second.logits)
    assert calls == [0, 1]
    assert model.layer_cache_capacity == 2
    assert model.cached_layer_count == 2

    evicted_calls: list[int] = []

    def load_evicted(layer_id: int):
        evicted_calls.append(layer_id)
        return layer

    evicted_model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=model.embedding,
        layer_count=2,
        layer_loader=load_evicted,
        final_norm=model.final_norm,
        lm_head=model.lm_head,
        rope_dim=2,
        layer_cache_capacity=1,
    )
    evicted_model.forward_tokens(torch.tensor([1, 2]))
    evicted_model.forward_tokens(torch.tensor([1, 2]))
    assert evicted_calls == [0, 1, 0, 1]
    assert evicted_model.cached_layer_count == 1


def _add_tiny_bundle_layer(
    names: dict[str, torch.Tensor],
    layer_id: int,
    *,
    dense: bool,
    include_indexer: bool,
) -> None:
    torch.manual_seed(200 + layer_id)
    hidden_size = 8
    heads = 2
    q_rank = 4
    kv_rank = 3
    nope = 2
    rope = 2
    value = 2
    intermediate = 3
    prefix = f"model.layers.{layer_id}"
    names.update(
        {
            f"{prefix}.input_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
            f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden_size, dtype=torch.bfloat16),
            f"{prefix}.self_attn.q_a_proj.weight": torch.randn(q_rank, hidden_size).bfloat16(),
            f"{prefix}.self_attn.q_a_layernorm.weight": torch.ones(q_rank, dtype=torch.bfloat16),
            f"{prefix}.self_attn.q_b_proj.weight": torch.randn(heads * (nope + rope), q_rank).bfloat16(),
            f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(kv_rank + rope, hidden_size).bfloat16(),
            f"{prefix}.self_attn.kv_a_layernorm.weight": torch.ones(kv_rank, dtype=torch.bfloat16),
            f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(heads * (nope + value), kv_rank).bfloat16(),
            f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden_size, heads * value).bfloat16(),
        }
    )
    if include_indexer:
        names.update(
            {
                f"{prefix}.self_attn.indexer.wq_b.weight": torch.randn(heads * 4, q_rank).bfloat16(),
                f"{prefix}.self_attn.indexer.wk.weight": torch.randn(4, hidden_size).bfloat16(),
                f"{prefix}.self_attn.indexer.k_norm.weight": torch.ones(4, dtype=torch.bfloat16),
                f"{prefix}.self_attn.indexer.k_norm.bias": torch.zeros(4, dtype=torch.bfloat16),
                f"{prefix}.self_attn.indexer.weights_proj.weight": torch.randn(heads, hidden_size).bfloat16(),
            }
        )
    if dense:
        names.update(
            {
                f"{prefix}.mlp.gate_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.up_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.down_proj.weight": torch.randn(hidden_size, intermediate).bfloat16(),
            }
        )
        return
    names.update(
        {
            f"{prefix}.mlp.gate.weight": torch.randn(2, hidden_size).bfloat16(),
            f"{prefix}.mlp.gate.e_score_correction_bias": torch.zeros(2, dtype=torch.bfloat16),
            f"{prefix}.mlp.shared_experts.gate_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
            f"{prefix}.mlp.shared_experts.up_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
            f"{prefix}.mlp.shared_experts.down_proj.weight": torch.randn(hidden_size, intermediate).bfloat16(),
        }
    )
    for expert_id in range(2):
        names.update(
            {
                f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.experts.{expert_id}.up_proj.weight": torch.randn(intermediate, hidden_size).bfloat16(),
                f"{prefix}.mlp.experts.{expert_id}.down_proj.weight": torch.randn(hidden_size, intermediate).bfloat16(),
            }
        )


def test_model_reference_factory_loads_dense_sparse_and_shared_indexer_layers(tmp_path) -> None:
    names: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn(16, 8).bfloat16(),
        "model.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(16, 8).bfloat16(),
    }
    _add_tiny_bundle_layer(names, 0, dense=True, include_indexer=True)
    _add_tiny_bundle_layer(names, 1, dense=True, include_indexer=False)
    _add_tiny_bundle_layer(names, 2, dense=False, include_indexer=True)
    config: dict[str, object] = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 3,
        "hidden_size": 8,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_intermediate_size": 3,
        "index_topk": 1,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "indexer_types": ["full", "shared", "full"],
        "mlp_layer_types": ["dense", "dense", "sparse"],
        "num_attention_heads": 2,
        "q_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_theta": 10000.0},
        "max_position_embeddings": 64,
        "vocab_size": 16,
        "num_nextn_predict_layers": 1,
        "routed_scaling_factor": 2.5,
    }
    source = tmp_path / "model.safetensors"
    save_file(names, str(source))
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
    assert assemble_glm5x_expert_bundle(artifacts, bundle_path).completed is True

    model = GLM5XDecoderModelReference.from_bundle(
        bundle_path,
        config=config,
        layer_cache_capacity=3,
    )
    forward = model.forward_tokens(torch.tensor([1, 2]))
    sparse_model = GLM5XDecoderModelReference.from_bundle(
        bundle_path,
        config=config,
        layer_cache_capacity=3,
        use_sparse_topk=True,
    )
    sparse_forward = sparse_model.forward_tokens(torch.tensor([1, 2]))
    torch.testing.assert_close(sparse_forward.logits, forward.logits, rtol=1e-5, atol=1e-5)

    assert isinstance(model._load_layer(0).moe, GLM5XDenseMlpReference)
    assert isinstance(model._load_layer(1).moe, GLM5XDenseMlpReference)
    assert isinstance(model._load_layer(2).moe, GLM5XLayer10MoEReference)
    assert forward.layers[0].moe.topk_indices.shape[-1] == 0
    assert forward.layers[1].moe.topk_indices.shape[-1] == 0
    assert forward.layers[2].moe.topk_indices.shape[-1] == 1
    assert forward.layers[2].moe.expert_load_count > 0

    state = model.empty_state()
    for index, token in enumerate((1, 2)):
        step = model.forward_token(token, state)
        torch.testing.assert_close(
            step.logits,
            forward.logits[:, index : index + 1],
            rtol=1e-5,
            atol=1e-5,
        )
        state = step.state

    if torch.cuda.is_available():
        cuda_model = GLM5XDecoderModelReference.from_bundle(
            bundle_path,
            config=config,
            device="cuda",
            verify_payloads=False,
            verify_root=False,
            layer_cache_capacity=3,
        )
        cuda_forward = cuda_model.forward_tokens(torch.tensor([1, 2]))
        assert cuda_forward.logits.device.type == "cuda"
        torch.testing.assert_close(
            cuda_forward.logits.cpu(), forward.logits, rtol=2e-3, atol=2e-3
        )
