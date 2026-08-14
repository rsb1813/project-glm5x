# GLM-5.2 checkpoint manifest 검증 경계를 테스트합니다.

import pytest
import torch
from safetensors.torch import save_file

from glm5x_ref.manifest import GLM5XTensorManifest, inspect_safetensors_shard


def _config() -> dict[str, object]:
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


def _index() -> dict[str, object]:
    return {
        "metadata": {"total_size": 1506659919872},
        "weight_map": {
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.gate.weight": "model-00002-of-00002.safetensors",
        },
    }


def test_manifest_reads_index_and_descriptor() -> None:
    manifest = GLM5XTensorManifest.from_json(_config(), _index())

    assert manifest.descriptor.hidden_layers == 78
    assert manifest.tensor_count == 2
    assert manifest.shard_count == 2
    assert manifest.total_size == 1506659919872
    assert manifest.shard_names == (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    )


def test_manifest_resolves_shared_indexer_tensor_to_nearest_full_layer() -> None:
    config = _config()
    config["indexer_types"] = ["full", "shared"] + ["full"] * 76
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {
            "model.layers.0.self_attn.indexer.wk.weight": "model-00001-of-00001.safetensors",
        },
    }
    manifest = GLM5XTensorManifest.from_json(config, index)

    assert manifest.indexer_source_layer(1) == 0
    assert manifest.resolve_indexer_tensor(1, "wk.weight") == (
        "model.layers.0.self_attn.indexer.wk.weight",
        "model-00001-of-00001.safetensors",
    )


def test_safetensors_header_inspection_does_not_materialize_tensor_payloads(tmp_path) -> None:
    path = tmp_path / "model-00001-of-00001.safetensors"
    save_file(
        {
            "model.layers.0.self_attn.indexer.wk.weight": torch.zeros((4, 6), dtype=torch.bfloat16),
            "model.layers.0.self_attn.indexer.k_norm.weight": torch.ones((4,), dtype=torch.bfloat16),
        },
        str(path),
    )

    headers = inspect_safetensors_shard(path)

    assert headers[0].name == "model.layers.0.self_attn.indexer.k_norm.weight"
    assert headers[0].shape == (4,)
    assert headers[0].dtype == "BF16"
    assert headers[1].shape == (4, 6)

    manifest = GLM5XTensorManifest.from_json(
        _config(),
        {
            "metadata": {"total_size": path.stat().st_size},
            "weight_map": {
                header.name: "model-00001-of-00001.safetensors" for header in headers
            },
        },
    )
    assert manifest.validate_safetensors_shard(path, "model-00001-of-00001.safetensors") == headers


@pytest.mark.parametrize(
    "index, error",
    [
        ({"metadata": {"total_size": 1}}, "WEIGHT_MAP_REQUIRED"),
        (
            {"metadata": {"total_size": 0}, "weight_map": {"x": "a.safetensors"}},
            "INVALID_TOTAL_SIZE",
        ),
        (
            {
                "metadata": {"total_size": 1},
                "weight_map": {"x": "../outside.safetensors"},
            },
            "INVALID_SHARD_NAME",
        ),
    ],
)
def test_manifest_rejects_incomplete_index(
    index: dict[str, object], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        GLM5XTensorManifest.from_json(_config(), index)
