# CUDA routed accumulation ablation의 축과 correctness gate를 검증합니다.
from pathlib import Path
import os

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint
from tools.ablate_cuda_fusion import fusion_matrix, run_fusion_ablation
from tools.benchmark_synthetic import BenchmarkRecord
from tests.python.test_benchmark_schema import _record


def test_fusion_matrix_crosses_transfer_and_fusion_only() -> None:
    assert fusion_matrix(4096) == (
        {
            "name": "synchronous-none",
            "cuda_transfer": "synchronous",
            "cuda_moe_fusion": "none",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "synchronous-routed-accumulate",
            "cuda_transfer": "synchronous",
            "cuda_moe_fusion": "routed-accumulate",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "prefetch-none",
            "cuda_transfer": "prefetch",
            "cuda_moe_fusion": "none",
            "cuda_pinned_bytes": 4096,
        },
        {
            "name": "prefetch-routed-accumulate",
            "cuda_transfer": "prefetch",
            "cuda_moe_fusion": "routed-accumulate",
            "cuda_pinned_bytes": 4096,
        },
    )


def test_fusion_ablation_requires_parity_and_reduced_d2h(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_benchmark_once(*args: object, **kwargs: object) -> BenchmarkRecord:
        fusion = str(kwargs["cuda_moe_fusion"])
        fused = fusion == "routed-accumulate"
        return BenchmarkRecord(
            **{
                **_record().__dict__,
                "scope": "synthetic-milestone-one",
                "backend": "cuda-custom",
                "device": "NVIDIA GeForce RTX 5080",
                "cuda_allocation": "reused",
                "cuda_weights": "transient",
                "cuda_boundary": "ffn-block",
                "cuda_transfer": str(kwargs["cuda_transfer"]),
                "cuda_moe_fusion": fusion,
                "cuda_pinned_bytes": int(kwargs["cuda_pinned_bytes"]),
                "device_to_host_bytes": 40 if fused else 80,
                "ffn_block_calls": 10,
                "ffn_block_experts": 16,
                "fused_moe_calls": 4 if fused else 0,
                "fused_moe_experts": 16 if fused else 0,
                "token_ids": (1, 2, 3),
                "routed_experts": (4, 5, 6, 7),
                "routed_k": (2, 2),
                "max_absolute_error": 1.0e-7,
                "max_relative_error": 2.0e-7,
            }
        )

    monkeypatch.setattr(
        "tools.ablate_cuda_fusion.benchmark_once", fake_benchmark_once
    )
    summary = run_fusion_ablation(
        tmp_path / "model.k3x",
        tmp_path / "k3x_run",
        warmup=0,
        iterations=1,
        pinned_bytes=4096,
        output_dir=tmp_path / "fusion",
    )
    assert len(summary["records"]) == 4
    assert all(item["parity_status"] == "exact" for item in summary["records"])
    assert summary["deltas"][0]["d2h_reduction_bytes"] == 40
    assert summary["deltas"][1]["d2h_reduction_bytes"] == 40
    assert (tmp_path / "fusion" / "summary.json").is_file()


def test_fusion_ablation_rejects_nonpositive_pinned_capacity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_fusion_ablation(
            tmp_path / "model.k3x",
            tmp_path / "k3x_run",
            warmup=0,
            iterations=1,
            pinned_bytes=0,
            output_dir=tmp_path / "fusion",
        )


def test_fusion_ablation_runs_on_top16_cuda_fixture(tmp_path: Path) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("fusion ablation smoke is exercised only against build-cuda")
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    summary = run_fusion_ablation(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        pinned_bytes=1024 * 1024,
        output_dir=tmp_path / "fusion",
    )
    assert all(item["parity_status"] == "exact" for item in summary["records"])
    assert all(item["d2h_reduction_bytes"] > 0 for item in summary["deltas"])
