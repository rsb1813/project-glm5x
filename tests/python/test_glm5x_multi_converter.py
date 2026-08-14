# GLM5X 다중 shard 변환 단위의 독립 재개와 완료 건너뛰기를 검증합니다.

import torch
from safetensors.torch import save_file

from glm5x_converter.multi import convert_glm5x_shards
from glm5x_ref.manifest import GLM5XTensorManifest


def _config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_intermediate_size": 4,
        "index_topk": 2,
        "index_topk_freq": 1,
        "index_n_heads": 1,
        "index_head_dim": 2,
        "index_share_for_mtp_iteration": True,
        "indexer_types": ["full"],
        "max_position_embeddings": 1024,
        "vocab_size": 8,
        "num_nextn_predict_layers": 1,
    }


def test_multi_shard_conversion_is_independently_restartable(tmp_path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    names = {
        "model-00001-of-00002.safetensors": "model.embed_tokens.weight",
        "model-00002-of-00002.safetensors": "lm_head.weight",
    }
    total_size = 0
    for shard_name, tensor_name in names.items():
        path = source_dir / shard_name
        save_file({tensor_name: torch.ones((8, 4), dtype=torch.bfloat16)}, str(path))
        total_size += path.stat().st_size
    manifest = GLM5XTensorManifest.from_json(
        _config(),
        {
            "metadata": {"total_size": total_size},
            "weight_map": {tensor_name: shard_name for shard_name, tensor_name in names.items()},
        },
    )

    first = convert_glm5x_shards(source_dir, output_dir, manifest, chunk_bytes=17)
    second = convert_glm5x_shards(source_dir, output_dir, manifest, chunk_bytes=17)

    assert first.completed is True
    assert set(first.output_paths) == {
        output_dir / "model-00001-of-00002.k3x",
        output_dir / "model-00002-of-00002.k3x",
    }
    assert second.completed is True
    assert second.skipped_shards == manifest.shard_names
