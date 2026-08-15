# GLM5X C++ runtime index의 고정 레코드와 무결성 계약을 검증합니다.

import hashlib
import json
import struct
import subprocess

import google_crc32c
import torch
from safetensors.torch import save_file

from conftest import cpp_binary
from glm5x_converter.bundle import GLM5XExpertBundle, assemble_glm5x_expert_bundle
from glm5x_converter.runtime_index import build_glm5x_runtime_index
from glm5x_converter.shard import convert_glm5x_shard
from glm5x_ref.manifest import GLM5XTensorManifest
from k3x_converter.reader import K3XReader


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


def _make_bundle(tmp_path, *, include_mtp: bool = False):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    names = [
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
    ]
    shard_names = ["a.safetensors", "a.safetensors", "b.safetensors"]
    if include_mtp:
        names.extend(
            [
                "model.layers.1.eh_proj.weight",
                "model.layers.1.mlp.experts.1.gate_proj.weight",
            ]
        )
        shard_names.extend(["b.safetensors", "b.safetensors"])
    save_file(
        {name: torch.ones((2, 4), dtype=torch.bfloat16) for name in names[:2]},
        str(source_dir / shard_names[0]),
    )
    second_shard = {names[2]: torch.ones((4, 2), dtype=torch.bfloat16)}
    if include_mtp:
        second_shard[names[3]] = torch.ones((4, 8), dtype=torch.bfloat16)
        second_shard[names[4]] = torch.ones((2, 4), dtype=torch.bfloat16)
    save_file(second_shard, str(source_dir / shard_names[2]))
    total_size = sum(path.stat().st_size for path in source_dir.glob("*.safetensors"))
    manifest = GLM5XTensorManifest.from_json(
        _config(),
        {
            "metadata": {"total_size": total_size},
            "weight_map": {
                name: shard_names[index] for index, name in enumerate(names)
            },
        },
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    convert_glm5x_shard(
        source_dir / shard_names[0], artifact_dir / "a.k3x", manifest, shard_names[0]
    )
    convert_glm5x_shard(
        source_dir / shard_names[2], artifact_dir / "b.k3x", manifest, shard_names[2]
    )
    bundle_path = artifact_dir / "experts.json"
    assemble_glm5x_expert_bundle(
        artifact_dir,
        bundle_path,
        verify_payloads=False,
        verify_root=False,
    )
    return artifact_dir, bundle_path


def test_runtime_index_has_deterministic_validated_tensor_owners(tmp_path) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path)
    output = artifact_dir / "model.gxi"

    report = build_glm5x_runtime_index(bundle_path, output)
    duplicate = artifact_dir / "model-copy.gxi"
    build_glm5x_runtime_index(bundle_path, duplicate)
    data = output.read_bytes()
    header = struct.unpack_from("<8sHHIIIIIQQQQQQQ32sII", data, 0)
    (
        magic,
        major,
        minor,
        header_bytes,
        artifact_record_bytes,
        tensor_record_bytes,
        artifact_count,
        reserved_count,
        tensor_count,
        artifact_offset,
        artifact_length,
        tensor_offset,
        tensor_length,
        string_offset,
        string_length,
        body_sha256,
        reserved_header,
        header_crc32c,
    ) = header

    assert (magic, major, minor) == (b"GLM5XIDX", 1, 0)
    assert (header_bytes, artifact_record_bytes, tensor_record_bytes) == (128, 64, 24)
    assert (artifact_count, tensor_count) == (2, 3)
    assert (reserved_count, reserved_header) == (0, 0)
    assert artifact_offset == 128
    assert artifact_length == 2 * 64
    assert tensor_offset == artifact_offset + artifact_length
    assert tensor_length == 3 * 24
    assert string_offset == tensor_offset + tensor_length
    assert string_offset + string_length == len(data)
    assert hashlib.sha256(data[128:]).digest() == body_sha256
    assert google_crc32c.value(data[:124]) == header_crc32c
    assert duplicate.read_bytes() == data
    assert not output.with_suffix(output.suffix + ".partial").exists()

    paths = []
    roots = []
    tensor_counts = []
    for index in range(artifact_count):
        path_offset, path_length, count, root_sha256, reserved = struct.unpack_from(
            "<QII32s16s", data, artifact_offset + index * artifact_record_bytes
        )
        paths.append(
            data[
                string_offset + path_offset : string_offset + path_offset + path_length
            ].decode("utf-8")
        )
        roots.append(root_sha256)
        tensor_counts.append(count)
        assert reserved == bytes(16)
    assert paths == ["a.k3x", "b.k3x"]
    assert tensor_counts == [2, 1]
    assert roots == [
        K3XReader.open(artifact_dir / path, verify_payloads=False, verify_root=False)
        .superblock.root_sha256
        for path in paths
    ]

    expected = []
    for artifact_index, path in enumerate(paths):
        reader = K3XReader.open(
            artifact_dir / path, verify_payloads=False, verify_root=False
        )
        expected.extend(
            (record.tensor_id, artifact_index, record_index, record.data_crc32c, 0)
            for record_index, record in enumerate(reader.tensor_records)
        )
    actual = [
        struct.unpack_from("<QIIII", data, tensor_offset + index * tensor_record_bytes)
        for index in range(tensor_count)
    ]
    assert actual == sorted(expected)
    assert report.completed is True
    assert report.artifact_count == 2
    assert report.tensor_count == 3
    assert report.file_bytes == len(data)


def test_cpp_runtime_index_reads_exact_cross_shard_expert(tmp_path) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path)
    output = artifact_dir / "model.gxi"
    build_glm5x_runtime_index(bundle_path, output)
    runner = cpp_binary("test_glm5x_runtime_index")
    assert runner.exists(), "build test_glm5x_runtime_index before the parity test"

    result = subprocess.run(
        [str(runner), str(output), "0", "0", "4", "2"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)
    payload = GLM5XExpertBundle.open(bundle_path).read_expert(0, 0)
    assert measured["artifact_count"] == 2
    assert measured["tensor_count"] == 3
    assert measured["payload_bytes"] == 48
    assert measured["role_sha256"] == [
        hashlib.sha256(payload[role]).hexdigest()
        for role in ("gate_proj", "up_proj", "down_proj")
    ]
    assert measured["reader_read_calls"] == 3
    assert measured["reader_completed_bytes"] == 48


def test_cpp_runtime_index_accepts_mtp_tensor_layer(tmp_path) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path, include_mtp=True)
    output = artifact_dir / "model.gxi"
    build_glm5x_runtime_index(bundle_path, output)
    runner = cpp_binary("test_glm5x_runtime_index")

    result = subprocess.run(
        [str(runner), str(output), "0", "0", "4", "2"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)
    assert measured["artifact_count"] == 2
    assert measured["tensor_count"] == 5


def test_cpp_runtime_index_rejects_index_and_payload_corruption(tmp_path) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path)
    output = artifact_dir / "model.gxi"
    build_glm5x_runtime_index(bundle_path, output)
    runner = cpp_binary("test_glm5x_runtime_index")
    arguments = [str(runner), str(output), "0", "0", "4", "2"]

    index_bytes = bytearray(output.read_bytes())
    index_bytes[-1] ^= 1
    output.write_bytes(index_bytes)
    corrupted_index = subprocess.run(arguments, capture_output=True, text=True)
    assert corrupted_index.returncode == 3
    assert "DIRECTORY_SHA256_MISMATCH" in corrupted_index.stderr

    build_glm5x_runtime_index(bundle_path, output)
    artifact = artifact_dir / "a.k3x"
    record = K3XReader.open(
        artifact, verify_payloads=False, verify_root=False
    ).tensor_records[0]
    with artifact.open("r+b") as stream:
        stream.seek(record.data_offset)
        value = stream.read(1)
        stream.seek(record.data_offset)
        stream.write(bytes([value[0] ^ 1]))
    corrupted_payload = subprocess.run(arguments, capture_output=True, text=True)
    assert corrupted_payload.returncode == 4
    assert "DATA_CRC_MISMATCH" in corrupted_payload.stderr
