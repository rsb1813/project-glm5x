# 실제 GLM5X CUDA benchmark의 runtime-index 입력 경계를 검증합니다.
import subprocess

import pytest
import torch

from conftest import cpp_binary
from glm5x_converter.activation import write_bf16_activation
from glm5x_converter.runtime_index import build_glm5x_runtime_index
from test_glm5x_runtime_index import _make_bundle


def test_real_glm5x_bench_opens_runtime_index_before_shape_validation(
    tmp_path,
) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path)
    runtime_index = artifact_dir / "model.gxi"
    build_glm5x_runtime_index(bundle_path, runtime_index)
    runner = cpp_binary("k3x_cuda_glm5x_real_expert_bench")
    if not runner.exists():
        pytest.skip("real GLM5X CUDA benchmark is not built")

    result = subprocess.run(
        [
            str(runner),
            "--runtime-index",
            str(runtime_index),
            "--layer",
            "0",
            "--expert",
            "0",
            "--precision",
            "bf16-rounded",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 6
    assert "invalid GLM BF16 expert tensor" in result.stderr


def test_real_glm5x_bench_accepts_full_decoder_layer_mode(tmp_path) -> None:
    artifact_dir, bundle_path = _make_bundle(tmp_path)
    runtime_index = artifact_dir / "model.gxi"
    build_glm5x_runtime_index(bundle_path, runtime_index)
    layer_input = tmp_path / "layer-input.gmlxact"
    layer_output = tmp_path / "layer-output.gmlxact"
    activation = torch.zeros((1, 6144), dtype=torch.bfloat16)
    write_bf16_activation(layer_input, activation)
    write_bf16_activation(layer_output, activation)
    runner = cpp_binary("k3x_cuda_glm5x_real_expert_bench")
    if not runner.exists():
        pytest.skip("real GLM5X CUDA benchmark is not built")

    result = subprocess.run(
        [
            str(runner),
            "--runtime-index",
            str(runtime_index),
            "--layer",
            "0",
            "--expert",
            "0",
            "--tokens",
            "1",
            "--precision",
            "bf16-rounded",
            "--output",
            "fp32",
            "--input-mode",
            "learned-decoder-layer",
            "--device-accumulate",
            "1",
            "--fuse-shared",
            "1",
            "--input-bf16",
            str(layer_input),
            "--expected-bf16",
            str(layer_output),
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 5
    assert "invalid GLM decoder layer tensor" in result.stderr
