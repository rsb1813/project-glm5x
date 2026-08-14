# GLM5X 스트리밍 checkpoint 도구의 shard 단위 manifest 경계를 검증합니다.
from __future__ import annotations

from glm5x_ref.manifest import GLM5XTensorManifest
from tools.stream_glm5x_checkpoint import has_verified_deleted_artifact, manifest_for_shard


def test_manifest_for_shard_keeps_only_the_selected_tensors() -> None:
    config = {
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
    manifest = GLM5XTensorManifest.from_json(
        config,
        {
            "metadata": {"total_size": 12},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        },
    )

    selected = manifest_for_shard(manifest, "model-00002-of-00002.safetensors")

    assert selected.shard_names == ("model-00002-of-00002.safetensors",)
    assert selected.tensor_shards == (("lm_head.weight", "model-00002-of-00002.safetensors"),)
    assert selected.total_size == manifest.total_size


def test_stream_recognizes_verified_deleted_artifact(tmp_path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact = output_dir / "model-00001-of-00002.k3x"
    artifact.write_bytes(b"artifact")
    marker = output_dir / "model-00001-of-00002.k3x.source-deleted.json"
    marker.write_text("{}", encoding="utf-8")

    assert has_verified_deleted_artifact(
        output_dir, "model-00001-of-00002.safetensors"
    )
    assert not has_verified_deleted_artifact(
        output_dir, "model-00002-of-00002.safetensors"
    )


def test_stream_parser_exposes_disjoint_worker_range() -> None:
    from tools.stream_glm5x_checkpoint import _parser

    args = _parser().parse_args(
        [
            "--source-dir",
            "source",
            "--output-dir",
            "output",
            "--bundle",
            "output/bundle.json",
            "--shard-start",
            "12",
            "--shard-end",
            "34",
            "--no-assemble",
        ]
    )

    assert args.shard_start == 12
    assert args.shard_end == 34
    assert args.no_assemble is True
