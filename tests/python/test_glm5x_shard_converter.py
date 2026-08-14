# bounded GLM-5.2 safetensors shard의 streaming K3X round-trip을 검증합니다.

import json

import torch
from safetensors.torch import save_file

from glm5x_ref.manifest import GLM5XTensorManifest
from glm5x_converter.shard import convert_glm5x_shard
from k3x_converter.format import DType
from k3x_converter.reader import K3XReader
from k3x_converter.safetensors_reader import inspect_shard, iter_tensor_chunks


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


def test_bounded_glm_shard_streams_bf16_bytes_and_round_trips(tmp_path) -> None:
    shard_name = "model-00001-of-00001.safetensors"
    source = tmp_path / shard_name
    save_file(
        {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
            "model.layers.0.self_attn.indexer.wk.weight": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4),
        },
        str(source),
    )
    manifest = GLM5XTensorManifest.from_json(
        _config(),
        {
            "metadata": {"total_size": source.stat().st_size},
            "weight_map": {
                "model.embed_tokens.weight": shard_name,
                "model.layers.0.self_attn.indexer.wk.weight": shard_name,
            },
        },
    )
    output = tmp_path / "bounded.k3x"

    report = convert_glm5x_shard(
        source,
        output,
        manifest,
        shard_name,
        chunk_bytes=17,
    )
    reader = K3XReader.open(output)
    source_tensors = inspect_shard(source)

    assert report.completed is True
    assert report.tensor_count == 2
    assert report.maximum_source_read_bytes <= 17
    assert all(record.dtype == DType.BF16 for record in reader.tensor_records)
    by_id = {record.tensor_id: record for record in reader.tensor_records}
    for name, tensor in source_tensors.items():
        record = by_id[report.tensor_ids[name]]
        data, auxiliary = reader.read_tensor_extents(record)
        assert auxiliary == b""
        assert data == b"".join(iter_tensor_chunks(tensor, 17))
    sidecar = json.loads(report.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["tensor_count"] == 2
    assert {item["name"] for item in sidecar["tensors"]} == set(source_tensors)
