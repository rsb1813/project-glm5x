# GLM5X 변환기 CLI의 외부 이름과 parser 계약을 검증합니다.

from glm5x_converter.cli import _parser


def test_cli_uses_glm5x_program_name() -> None:
    assert _parser().prog == "glm5x-convert"


def test_cli_exposes_multi_shard_conversion_command() -> None:
    args = _parser().parse_args(
        [
            "convert-shards",
            "source",
            "output",
            "--config",
            "config.json",
            "--index",
            "index.json",
        ]
    )
    assert args.command == "convert-shards"
    assert args.chunk_bytes == 8 * 1024 * 1024


def test_cli_exposes_safe_source_deletion_switch() -> None:
    args = _parser().parse_args(
        [
            "convert-shards",
            "source",
            "output",
            "--config",
            "config.json",
            "--index",
            "index.json",
            "--delete-source",
        ]
    )
    assert args.delete_source is True


def test_cli_exposes_cross_shard_expert_bundle_command() -> None:
    args = _parser().parse_args(["assemble-experts", "artifacts", "experts.json"])
    assert args.command == "assemble-experts"
    assert args.dry_run is False


def test_cli_exposes_cpp_runtime_index_command() -> None:
    args = _parser().parse_args(
        ["build-runtime-index", "experts.json", "model.gxi"]
    )
    assert args.command == "build-runtime-index"
    assert args.bundle.name == "experts.json"
    assert args.output.name == "model.gxi"
