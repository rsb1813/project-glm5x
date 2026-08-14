# released-dimension repeated-view CUDA expert benchmark의 경계와 수치를 검증합니다.
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source
from tools.ablate_cuda_expert_fusion import run_expert_fusion_ablation


def test_cuda_expert_bench_reduces_top16_d2h(tmp_path: Path) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released-dimension CUDA bench runs only against build-cuda")
    runner = cpp_binary("k3x_cuda_expert_bench")
    assert runner.is_file(), "build k3x_cuda_expert_bench before running test"
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)

    records: dict[str, dict] = {}
    for fusion in ("none", "routed-accumulate"):
        result = subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--fusion", fusion,
                "--slots", "16",
                "--warmup", "0",
                "--iterations", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        records[fusion] = json.loads(result.stdout)

    unfused = records["none"]
    fused = records["routed-accumulate"]
    for record in records.values():
        assert record["artifact_kind"] == "released_dimension_repeated_view"
        assert record["routing_semantics"] is False
        assert record["expert_payload_bytes"] == 17_547_264
        assert record["slots"] == 16
        assert record["iterations"] == 1
        assert record["maximum_absolute_error"] <= 1.0e-3
        assert record["kernel_nanoseconds"] > 0
    assert unfused["device_to_host_bytes"] == 16 * 3584 * 4
    assert fused["device_to_host_bytes"] == 3584 * 4
    assert unfused["fused_moe_calls"] == 0
    assert fused["fused_moe_calls"] == 1
    assert fused["fused_moe_experts"] == 16


def test_cuda_expert_bench_rejects_invalid_slots() -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released-dimension CUDA bench runs only against build-cuda")
    result = subprocess.run(
        [str(cpp_binary("k3x_cuda_expert_bench")), "--slots", "0"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "slots must be between 1 and 16"


def test_cuda_expert_ablation_writes_cross_checked_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run_case(
        artifact: Path,
        runner: Path,
        fusion: str,
        slots: int,
        warmup: int,
        iterations: int,
    ) -> dict:
        fused = fusion == "routed-accumulate"
        return {
            "artifact_kind": "released_dimension_repeated_view",
            "routing_semantics": False,
            "fusion": fusion,
            "expert_payload_bytes": 17_547_264,
            "slots": slots,
            "warmup": warmup,
            "iterations": iterations,
            "latency_nanoseconds_median": 80 if fused else 100,
            "maximum_absolute_error": 1.0e-5 if fused else 0.0,
            "kernel_nanoseconds": 70 if fused else 90,
            "device_to_host_bytes": 14_336 if fused else 229_376,
            "weight_h2d_bytes": 0,
            "activation_h2d_bytes": 14_336,
            "fused_moe_calls": iterations if fused else 0,
            "fused_moe_experts": slots * iterations if fused else 0,
            "peak_vram_bytes": 20_000_000,
        }

    monkeypatch.setattr(
        "tools.ablate_cuda_expert_fusion._run_case", fake_run_case
    )
    summary = run_expert_fusion_ablation(
        tmp_path / "bounded.k3x",
        tmp_path / "k3x_cuda_expert_bench",
        slots=16,
        warmup=3,
        iterations=20,
        output_dir=tmp_path / "results",
    )
    assert summary["d2h_reduction_bytes"] == 215_040
    assert summary["latency_nanoseconds_delta"] == -20
    assert (tmp_path / "results" / "summary.json").is_file()
    assert (tmp_path / "results" / "none.csv").is_file()
    assert (tmp_path / "results" / "routed-accumulate.csv").is_file()
