# GLM-5.2 형상 합성 expert FFN CUDA 벤치마크의 JSON 계약을 검증합니다.

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary


def _runner() -> Path:
    build = Path(os.environ.get("K3X_BUILD_DIR", "build"))
    if build.name != "build-cuda":
        pytest.skip("GLM5X CUDA benchmark requires build-cuda")
    runner = cpp_binary("k3x_cuda_glm5x_moe_bench")
    if not runner.exists():
        pytest.skip("GLM5X CUDA benchmark is not built")
    return runner


def test_glm52_shaped_expert_bench_contract() -> None:
    result = subprocess.run(
        [
            str(_runner()),
            "--experts",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["artifact_kind"] == "glm5.2_shaped_expert_ffn"
    assert payload["model_family"] == "glm5"
    assert payload["hidden_size"] == 6144
    assert payload["expert_intermediate_size"] == 2048
    assert payload["experts"] == 1
    assert payload["maximum_absolute_error"] <= 1.0e-5
    assert payload["latency_nanoseconds_median"] > 0
    assert payload["peak_vram_bytes"] > 0


def test_glm52_expert_batch_mode_contract() -> None:
    result = subprocess.run(
        [
            str(_runner()),
            "--mode",
            "expert-batch",
            "--experts",
            "1",
            "--tokens",
            "2",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["mode"] == "expert-batch"
    assert payload["batched_expert_ffn_calls"] == 1
    assert payload["batched_expert_ffn_tokens"] == 2
    assert payload["weight_h2d_bytes"] == 0
    assert payload["maximum_absolute_error"] <= 1.0e-5
