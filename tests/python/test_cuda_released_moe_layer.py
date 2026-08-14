# released-dimension resident MoE-layer CUDA 벤치마크의 CLI 경계를 검증합니다.
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _require_cuda_build() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released MoE-layer benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_moe_layer_bench")


def _released_artifact(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    return artifact


@pytest.mark.parametrize("boundary", ["ffn-block", "moe-layer"])
@pytest.mark.parametrize("experts", [1, 16])
def test_released_moe_layer_bench_executes(
    boundary: str, experts: int, tmp_path: Path
) -> None:
    runner = _require_cuda_build()
    artifact = _released_artifact(tmp_path)
    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--boundary",
            boundary,
            "--experts",
            str(experts),
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
    assert payload["artifact_kind"] == "released_dimension_moe_layer"
    assert payload["routing_semantics"] is False
    assert payload["boundary"] == boundary
    assert payload["experts"] == experts
    assert payload["hidden_width"] == 7168
    assert payload["latent_width"] == 3584
    assert payload["expert_intermediate_width"] == 3072
    assert payload["expert_payload_bytes"] == 17_547_264
    assert payload["resident_capacity_bytes"] == 1 << 30
    assert payload["warmup"] == 0
    assert payload["iterations"] == 1
    assert payload["validation"] == "per-call"
    assert payload["profiler"] is True
    assert payload["maximum_absolute_error"] <= 1.0e-5
    assert payload["latency_nanoseconds_median"] > 0
    assert payload["kernel_nanoseconds"] > 0
    assert payload["activation_h2d_bytes"] > 0
    assert payload["device_to_host_bytes"] > 0
    assert payload["weight_h2d_bytes"] == 0
    assert payload["cold_weight_h2d_bytes"] > 0
    assert payload["resident_weight_bytes"] > 0
    assert payload["peak_resident_weight_bytes"] > 0
    assert payload["oracle_peak_vram_bytes"] > 0
    assert payload["peak_vram_bytes"] >= payload["oracle_peak_vram_bytes"]
    assert payload["peak_vram_bytes"] >= payload["resident_weight_bytes"]
    assert payload["weight_cache_bypasses"] == 0
    assert payload["resident_grid_calls"] == 1
    assert payload["resident_grid_kernel_launches"] == 4
    assert payload["resident_grid_fallbacks"] == 0
    assert payload["resident_moe_layer_fallbacks"] == 0
    if boundary == "ffn-block":
        assert payload["immutable_validation_scans"] == 0
        assert payload["immutable_validation_bytes"] == 0
        assert payload["stream_synchronization_count"] == 4
        assert payload["resident_moe_layer_calls"] == 0
        assert payload["resident_moe_layer_experts"] == 0
        assert payload["resident_moe_layer_kernel_launches"] == 0
        assert payload["resident_moe_layer_contribution_h2d_bytes"] == 0
    else:
        assert payload["immutable_validation_scans"] == 6
        assert payload["immutable_validation_hits"] == 0
        assert payload["immutable_validation_bytes"] == 469_776_384
        assert payload["immutable_validation_nanoseconds"] > 0
        assert payload["stream_synchronization_count"] == 1
        assert payload["resident_moe_layer_calls"] == 1
        assert payload["resident_moe_layer_experts"] == experts
        assert payload["resident_moe_layer_kernel_launches"] == 13
        assert payload["resident_moe_layer_contribution_h2d_bytes"] == experts * 4


def test_released_moe_layer_admission_and_profiler_off(
    tmp_path: Path,
) -> None:
    runner = _require_cuda_build()
    artifact = _released_artifact(tmp_path)
    result = subprocess.run(
        [
            str(runner), "--model", str(artifact),
            "--boundary", "moe-layer", "--experts", "1",
            "--warmup", "0", "--iterations", "2",
            "--validation", "admission", "--profiler", "off",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["validation"] == "admission"
    assert payload["profiler"] is False
    assert payload["kernel_nanoseconds"] is None
    assert payload["cold_immutable_validation_scans"] == 6
    assert payload["cold_immutable_validation_bytes"] == 469_776_384
    assert payload["immutable_validation_scans"] == 0
    assert payload["immutable_validation_hits"] == 12
    assert payload["immutable_validation_bytes"] == 0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--boundary", "bogus"), "unknown boundary: bogus"),
        (("--experts", "0"), "experts must be one of 1, 4, or 16"),
        (("--experts", "2"), "experts must be one of 1, 4, or 16"),
        (("--experts", "17"), "experts must be one of 1, 4, or 16"),
        ((), "model path is required"),
        (("--iterations", "0"), "iterations must be positive"),
    ],
)
def test_released_moe_layer_bench_rejects_invalid_arguments(
    arguments: tuple[str, ...], message: str
) -> None:
    runner = _require_cuda_build()
    result = subprocess.run(
        [str(runner), *arguments], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message
