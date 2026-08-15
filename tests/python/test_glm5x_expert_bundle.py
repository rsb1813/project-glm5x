# GLM5X 여러 shard artifact의 expert tensor bundle 조립을 검증합니다.

import json

import pytest
import torch
from safetensors.torch import save_file

from glm5x_converter.bundle import GLM5XExpertBundle, assemble_glm5x_expert_bundle
from glm5x_converter.shard import convert_glm5x_shard
from glm5x_ref.manifest import GLM5XTensorManifest
from k3x_converter.format import K3XError


def _config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 1,
        "hidden_size": 4,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 1,
        "moe_intermediate_size": 2,
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


def _make_manifest(shard_names: list[str], names: list[str], sizes: int) -> GLM5XTensorManifest:
    return GLM5XTensorManifest.from_json(
        _config(),
        {
            "metadata": {"total_size": sizes},
            "weight_map": {name: shard_names[index] for index, name in enumerate(names)},
        },
    )


def test_expert_bundle_joins_roles_from_independent_artifacts(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    shard_names = ["a.safetensors", "a.safetensors", "b.safetensors"]
    save_file({name: torch.ones((2, 4), dtype=torch.bfloat16) for name in names[:2]}, str(source_dir / shard_names[0]))
    save_file({names[2]: torch.ones((2, 4), dtype=torch.bfloat16)}, str(source_dir / shard_names[2]))
    total_size = sum(path.stat().st_size for path in source_dir.glob("*.safetensors"))
    manifest = _make_manifest(shard_names, names, total_size)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    convert_glm5x_shard(source_dir / shard_names[0], artifact_dir / "a.k3x", manifest, shard_names[0])
    convert_glm5x_shard(source_dir / shard_names[2], artifact_dir / "b.k3x", manifest, shard_names[2])

    report = assemble_glm5x_expert_bundle(
        artifact_dir,
        artifact_dir / "experts.json",
        verify_payloads=False,
        verify_root=False,
    )
    bundle = json.loads((artifact_dir / "experts.json").read_text(encoding="utf-8"))

    assert report.completed is True
    assert report.artifact_count == 2
    assert report.complete_expert_count == 1
    assert report.incomplete_expert_count == 0
    expert = bundle["experts"][0]
    assert (expert["layer_id"], expert["expert_id"]) == (0, 0)
    assert set(expert["roles"]) == {"gate_proj", "up_proj", "down_proj"}
    assert expert["roles"]["gate_proj"]["ref"]["artifact"] == "a.k3x"
    assert expert["roles"]["down_proj"]["ref"]["artifact"] == "b.k3x"
    assert not (artifact_dir / "experts.json.partial").exists()
    opened_bundle = GLM5XExpertBundle.open(artifact_dir / "experts.json")
    assert all(
        record.tensor_id in opened_bundle.record_indexes[artifact]
        for artifact, reader in opened_bundle.readers.items()
        for record in reader.tensor_records
    )
    payload = opened_bundle.read_expert(0, 0)
    assert set(payload) == {"gate_proj", "up_proj", "down_proj"}
    assert all(len(value) == 16 for value in payload.values())
    lazy_bundle = GLM5XExpertBundle.open(
        artifact_dir / "experts.json", verify_payloads=False, verify_root=False
    )
    lazy_payload = lazy_bundle.read_expert(0, 0)
    assert lazy_payload == payload
    batched_payload = lazy_bundle.read_experts(0, [0])[0]
    assert batched_payload == payload

    cached_bundle = GLM5XExpertBundle.open(
        artifact_dir / "experts.json",
        verify_payloads=False,
        verify_root=False,
        expert_cache_capacity_bytes=64,
    )
    assert cached_bundle.read_expert(0, 0) == payload
    assert cached_bundle.expert_payload_cache_stats.hits == 0
    assert cached_bundle.read_expert(0, 0) == payload
    cache_stats = cached_bundle.expert_payload_cache_stats
    assert cache_stats.hits == 1
    assert cache_stats.misses == 1
    assert cache_stats.entries == 1
    assert cache_stats.resident_bytes == 48


def test_expert_bundle_rejects_reference_offset_tampering(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    save_file({name: torch.ones((2, 4), dtype=torch.bfloat16)}, str(source))
    manifest = _make_manifest([source.name], [name], source.stat().st_size)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    convert_glm5x_shard(source, artifact_dir / "a.k3x", manifest, source.name)
    # Complete the bundle with synthetic role metadata so the loader reaches the offset check.
    for role in ("up_proj", "down_proj"):
        role_source = tmp_path / f"{role}.safetensors"
        role_name = f"model.layers.0.mlp.experts.0.{role}.weight"
        save_file({role_name: torch.ones((2, 4), dtype=torch.bfloat16)}, str(role_source))
        role_manifest = _make_manifest([role_source.name], [role_name], role_source.stat().st_size)
        convert_glm5x_shard(role_source, artifact_dir / f"{role}.k3x", role_manifest, role_source.name)
    bundle_path = artifact_dir / "experts.json"
    assemble_glm5x_expert_bundle(artifact_dir, bundle_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["experts"][0]["roles"]["gate_proj"]["ref"]["data_offset"] += 4096
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(K3XError, match="EXPERT_BUNDLE_REFERENCE_MISMATCH"):
        GLM5XExpertBundle.open(bundle_path).read_expert(0, 0)


def test_expert_bundle_rejects_duplicate_role(tmp_path) -> None:
    source = tmp_path / "source.safetensors"
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    save_file({name: torch.ones((2, 4), dtype=torch.bfloat16)}, str(source))
    manifest = _make_manifest([source.name], [name], source.stat().st_size)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    convert_glm5x_shard(source, artifact_dir / "a.k3x", manifest, source.name)
    convert_glm5x_shard(source, artifact_dir / "b.k3x", manifest, source.name)

    with pytest.raises(K3XError, match="EXPERT_BUNDLE_DUPLICATE_ROLE"):
        assemble_glm5x_expert_bundle(artifact_dir, artifact_dir / "experts.json")
