# 실제 GLM5X CUDA benchmark의 runtime-index 입력 경계를 검증합니다.
import subprocess

import pytest

from conftest import cpp_binary
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
