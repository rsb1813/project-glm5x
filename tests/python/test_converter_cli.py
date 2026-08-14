# K3X 변환 CLI의 dry-run과 검증 경로를 검사합니다.
from pathlib import Path

from k3x_converter.cli import main
from k3x_ref.storage_fixture import write_bounded_expert_source


def test_dry_run_does_not_create_artifact(
    synthetic_source: Path, tmp_path: Path, capsys
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    assert main(["convert", str(synthetic_source), str(artifact), "--dry-run"]) == 0
    assert not artifact.exists()
    assert '"completed":false' in capsys.readouterr().out


def test_convert_then_validate_from_cli(
    synthetic_source: Path, tmp_path: Path, capsys
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    assert main(["convert", str(synthetic_source), str(artifact), "--chunk-bytes", "257"]) == 0
    assert main(["validate", str(artifact)]) == 0
    assert '"valid":true' in capsys.readouterr().out


def test_bounded_storage_fixture_converts_and_validates_from_cli(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    assert main(
        ["convert", str(source), str(artifact), "--chunk-bytes", str(193 * 1024)]
    ) == 0
    assert main(["validate", str(artifact)]) == 0
    output = capsys.readouterr().out
    assert '"completed":true' in output
    assert '"valid":true' in output
