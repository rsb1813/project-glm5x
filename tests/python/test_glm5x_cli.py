# GLM5X 변환기 CLI의 외부 이름과 parser 계약을 검증합니다.

from glm5x_converter.cli import _parser


def test_cli_uses_glm5x_program_name() -> None:
    assert _parser().prog == "glm5x-convert"

