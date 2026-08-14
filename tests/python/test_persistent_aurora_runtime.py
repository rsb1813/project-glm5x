# Persistent AURORA C++ cursor를 Top-16 합성 K3X artifact로 검증합니다.
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint


def test_incremental_cursor_matches_reduced_top_k_oracle(
    tmp_path: Path,
) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(cpp_binary("test_incremental_cursor")), str(artifact)],
        check=True,
    )


def test_cuda_persistent_provider_matches_cpu_proposals(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA AURORA parity is exercised only against build-cuda")
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-cuda-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "cuda-top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(cpp_binary("test_cuda_aurora_draft")), str(artifact)],
        check=True,
    )
