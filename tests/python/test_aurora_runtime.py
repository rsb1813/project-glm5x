# AURORA replay C++ provider를 Top-16 합성 K3X artifact로 검증합니다.
import subprocess
from pathlib import Path

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint


def test_aurora_replay_provider_lifecycle(tmp_path: Path) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    subprocess.run([str(cpp_binary("test_aurora")), str(artifact)], check=True)
