# 합성 source checkpoint를 재사용 가능한 pytest fixture로 제공합니다.
import os
from pathlib import Path

import pytest

from k3x_ref.fixtures import write_source_checkpoint


def cpp_binary(name: str) -> Path:
    build = Path(os.environ.get("K3X_BUILD_DIR", "build")).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    return build / f"{name}{suffix}"


@pytest.fixture
def synthetic_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    write_source_checkpoint(source)
    return source

