# Top-16 합성 artifact에서 CUDA persistent AURORA draft CLI 경계를 검증합니다.
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint


def _top16_artifact(tmp_path: Path) -> Path:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    return artifact


def test_cpu_build_rejects_cuda_draft_without_fallback(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name == "build-cuda":
        pytest.skip("CPU unavailable behavior is exercised against build")
    artifact = _top16_artifact(tmp_path)
    output = tmp_path / "unexpected-fallback.json"
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model", str(artifact),
            "--prompt-ids", "1,7,3,9",
            "--generate", "6",
            "--mode", "incremental",
            "--speculative-mode", "aurora-persistent",
            "--speculative-block-size", "2",
            "--aurora-draft-k", "4",
            "--aurora-draft-backend", "cuda-custom",
            "--aurora-draft-resident-bytes", "8388608",
            "--json", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == (
        "BACKEND_UNAVAILABLE: CUDA backend is disabled at build time"
    )
    assert not output.exists()


def test_cpu_build_rejects_cuda_moe_layer_target_without_fallback(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name == "build-cuda":
        pytest.skip("CPU unavailable behavior is exercised against build")
    artifact = _top16_artifact(tmp_path)
    output = tmp_path / "unexpected-target-fallback.json"
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model", str(artifact),
            "--prompt-ids", "1,7,3,9",
            "--generate", "2",
            "--mode", "incremental",
            "--backend", "cuda-custom",
            "--dense-precision", "fp32",
            "--cuda-boundary", "moe-layer",
            "--cuda-allocation", "reused",
            "--cuda-weights", "resident",
            "--cuda-resident-bytes", "8388608",
            "--cuda-batching", "resident-grid",
            "--cuda-transfer", "synchronous",
            "--cuda-moe-fusion", "none",
            "--json", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == (
        "BACKEND_UNAVAILABLE: CUDA backend is disabled at build time"
    )
    assert not output.exists()


@pytest.mark.parametrize("verification", ["token-major", "expert-major"])
def test_cuda_draft_matches_cpu_persistent_target_execution(
    verification: str,
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA AURORA CLI parity is exercised only against build-cuda")
    artifact = _top16_artifact(tmp_path)
    common = [
        str(cpp_binary("k3x_run")),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--diagnostics", "true",
        "--backend", "cpu",
        "--speculative-mode", "aurora-persistent",
        "--speculative-verification", verification,
        "--speculative-block-size", "2",
        "--aurora-draft-k", "4",
    ]

    for policy in ("fixed", "adaptive"):
        results: dict[str, dict] = {}
        identities = (
            ("cpu", "cpu", []),
            ("transient", "cuda-custom", []),
            (
                "resident",
                "cuda-custom",
                ["--aurora-draft-resident-bytes", "8388608"],
            ),
            (
                "resident-grid",
                "cuda-custom",
                [
                    "--aurora-draft-resident-bytes", "8388608",
                    "--aurora-draft-batching", "resident-grid",
                ],
            ),
            (
                "resident-grid-bypass",
                "cuda-custom",
                [
                    "--aurora-draft-resident-bytes", "1",
                    "--aurora-draft-batching", "resident-grid",
                ],
            ),
            (
                "moe-layer",
                "cuda-custom",
                [
                    "--aurora-draft-resident-bytes", "8388608",
                    "--aurora-draft-batching", "resident-grid",
                    "--aurora-draft-boundary", "moe-layer",
                ],
            ),
            (
                "moe-layer-bypass",
                "cuda-custom",
                [
                    "--aurora-draft-resident-bytes", "1",
                    "--aurora-draft-batching", "resident-grid",
                    "--aurora-draft-boundary", "moe-layer",
                ],
            ),
        )
        for identity, draft_backend, extra in identities:
            output = tmp_path / f"{verification}-{policy}-{identity}.json"
            subprocess.run(
                [
                    *common,
                    "--aurora-block-policy", policy,
                    "--aurora-draft-backend", draft_backend,
                    *extra,
                    "--json", str(output),
                ],
                check=True,
            )
            results[identity] = json.loads(
                output.read_text(encoding="utf-8")
            )
        cpu = results["cpu"]
        cuda = results["transient"]
        resident = results["resident"]
        resident_grid = results["resident-grid"]
        resident_grid_bypass = results["resident-grid-bypass"]
        moe_layer = results["moe-layer"]
        moe_layer_bypass = results["moe-layer-bypass"]
        assert cpu["aurora_draft_backend"] == "cpu"
        assert cuda["aurora_draft_backend"] == "cuda-custom"
        assert cpu["draft_device"] == "CPU"
        assert cpu["draft_kernel_nanoseconds"] == 0
        assert cpu["draft_host_to_device_bytes"] == 0
        assert cpu["draft_peak_vram_bytes"] == 0
        assert cuda["draft_cuda_allocation"] == "reused"
        assert cuda["draft_cuda_weights"] == "transient"
        assert resident["draft_cuda_weights"] == "resident"
        assert cuda["draft_cuda_resident_bytes"] == 0
        assert cuda["draft_resident_weight_bytes"] == 0
        assert cuda["draft_peak_resident_weight_bytes"] == 0
        assert resident["draft_cuda_resident_bytes"] == 8388608
        assert 0 < resident["draft_resident_weight_bytes"] <= 8388608
        assert (
            resident["draft_resident_weight_bytes"]
            <= resident["draft_peak_resident_weight_bytes"]
            <= 8388608
        )
        assert resident["draft_weight_cache_hits"] > 0
        assert resident["draft_weight_cache_misses"] > 0
        assert resident["draft_weight_cache_bypasses"] == 0
        assert (
            resident["draft_weight_h2d_bytes"]
            < cuda["draft_weight_h2d_bytes"]
        )
        assert resident["cuda_resident_bytes"] == 0
        assert resident["resident_weight_bytes"] == 0
        assert resident["peak_resident_weight_bytes"] == 0
        assert cuda["draft_cuda_batching"] == "grouped"
        assert resident_grid["draft_cuda_batching"] == "resident-grid"
        assert resident_grid_bypass["draft_cuda_batching"] == "resident-grid"
        assert resident_grid["draft_resident_grid_calls"] > 0
        assert resident_grid["draft_resident_grid_experts"] > 0
        assert resident_grid["draft_resident_grid_tokens"] > 0
        assert resident_grid["draft_resident_grid_expert_tokens"] > 0
        assert resident_grid["draft_resident_grid_kernel_launches"] == (
            resident_grid["draft_resident_grid_calls"] * 4
        )
        assert resident_grid["draft_resident_grid_fallbacks"] == 0
        assert resident_grid["draft_resident_grid_descriptor_h2d_bytes"] > 0
        assert resident_grid_bypass["draft_resident_grid_calls"] == 0
        assert resident_grid_bypass["draft_resident_grid_fallbacks"] > 0
        assert resident_grid["resident_grid_calls"] == 0
        assert moe_layer["draft_cuda_boundary"] == "moe-layer"
        assert moe_layer_bypass["draft_cuda_boundary"] == "moe-layer"
        assert moe_layer["draft_resident_grid_calls"] > 0
        assert moe_layer["draft_resident_grid_fallbacks"] == 0
        assert moe_layer_bypass["draft_resident_grid_calls"] == 0
        assert moe_layer_bypass["draft_resident_grid_fallbacks"] > 0
        assert moe_layer["draft_resident_moe_layer_calls"] > 0
        assert moe_layer["draft_resident_moe_layer_kernel_launches"] == (
            moe_layer["draft_resident_moe_layer_calls"] * 13
        )
        assert moe_layer["draft_resident_moe_layer_fallbacks"] == 0
        assert moe_layer["draft_resident_moe_layer_experts"] > 0
        assert moe_layer[
            "draft_resident_moe_layer_contribution_h2d_bytes"
        ] > 0
        assert moe_layer["resident_moe_layer_calls"] == 0
        assert moe_layer_bypass["draft_resident_moe_layer_calls"] == 0
        assert moe_layer_bypass["draft_resident_moe_layer_fallbacks"] > 0
        assert (
            moe_layer["draft_stream_synchronization_count"]
            < resident_grid["draft_stream_synchronization_count"]
        )
        assert (
            moe_layer["draft_activation_h2d_bytes"]
            < resident_grid["draft_activation_h2d_bytes"]
        )
        assert (
            moe_layer["draft_device_to_host_bytes"]
            < resident_grid["draft_device_to_host_bytes"]
        )
        assert cuda["draft_cuda_boundary"] == "ffn-block"
        assert cuda["draft_cuda_transfer"] == "synchronous"
        assert cuda["draft_cuda_moe_fusion"] == "none"
        assert cuda["draft_kernel_nanoseconds"] > 0
        assert cuda["draft_host_to_device_bytes"] > 0
        assert cuda["draft_weight_h2d_bytes"] > 0
        assert cuda["draft_activation_h2d_bytes"] > 0
        assert cuda["draft_device_to_host_bytes"] > 0
        assert cuda["draft_peak_vram_bytes"] > 0
        assert cuda["draft_device_allocation_count"] > 0
        assert cuda["draft_stream_synchronization_count"] > 0
        assert cuda["kernel_nanoseconds"] == 0
        assert cuda["host_to_device_bytes"] == 0
        assert cuda["peak_vram_bytes"] == 0
        for field in (
            "token_ids",
            "final_state",
            "routed_experts",
            "routed_k",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "draft_proposal_calls",
            "draft_candidate_tokens",
            "draft_context_prefill_tokens",
            "draft_incremental_forward_calls",
            "draft_rollback_events",
            "draft_mla_positions_cropped",
            "draft_kda_checkpoint_bytes",
        ):
            assert cuda[field] == cpu[field]
            assert resident[field] == cpu[field]
            assert resident_grid[field] == cpu[field]
            assert resident_grid_bypass[field] == cpu[field]
            assert moe_layer[field] == cpu[field]
            assert moe_layer_bypass[field] == cpu[field]


def test_admission_validation_propagates_to_cuda_moe_layer_draft(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA AURORA admission propagation requires build-cuda")
    artifact = _top16_artifact(tmp_path)
    output = tmp_path / "admission-draft.json"
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model", str(artifact),
            "--prompt-ids", "1,7,3,9",
            "--generate", "6",
            "--mode", "incremental",
            "--backend", "cuda-custom",
            "--cuda-allocation", "reused",
            "--cuda-weights", "resident",
            "--cuda-batching", "resident-grid",
            "--cuda-boundary", "moe-layer",
            "--cuda-resident-bytes", "8388608",
            "--cuda-weight-validation", "admission",
            "--speculative-mode", "aurora-persistent",
            "--speculative-verification", "token-major",
            "--speculative-block-size", "2",
            "--aurora-draft-k", "4",
            "--aurora-draft-backend", "cuda-custom",
            "--aurora-draft-resident-bytes", "8388608",
            "--aurora-draft-batching", "resident-grid",
            "--aurora-draft-boundary", "moe-layer",
            "--json", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cuda_weight_validation"] == "admission"
    assert payload["draft_cuda_weight_validation"] == "admission"
    assert payload["immutable_validation_scans"] > 0
    assert payload["draft_immutable_validation_scans"] > 0
    assert payload["immutable_validation_hits"] > 0
    assert payload["draft_immutable_validation_hits"] > 0
