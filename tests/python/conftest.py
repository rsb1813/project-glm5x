# 합성 source checkpoint를 재사용 가능한 pytest fixture로 제공합니다.
import os
import re
from pathlib import Path

import pytest

from k3x_ref.fixtures import write_source_checkpoint


_COMMITTED_EVIDENCE = {
    "0007": "b0007-l2-reader-wsl",
    "0009": "b0009-deadline-loader-wsl",
    "0010": "b0010-expert-cache-policies-wsl",
    "0011": "b0011-task-session-profiles-wsl",
    "0012": "b0012-adaptive-routing-wsl",
    "0018": "b0018-persistent-aurora-wsl",
    "0019": "b0019-cuda-aurora-draft-wsl",
    "0020": "b0020-cuda-aurora-residency-wsl",
    "0021": "b0021-cuda-aurora-grid-wsl",
    "0022": "b0022-cuda-aurora-moe-layer-wsl",
    "0023": "b0023-cuda-released-moe-layer-wsl",
    "0024": "b0024-cuda-admission-validation-wsl",
}


def _evidence_path(root: Path, benchmark_id: str) -> Path | None:
    if benchmark_id == "0006":
        return root / "results" / "b0006-l1-cache.json"
    directory = _COMMITTED_EVIDENCE.get(benchmark_id)
    return None if directory is None else root / "results" / directory / "summary.json"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """커밋하지 않은 과거 측정 산출물 검증은 산출물이 있을 때만 실행합니다."""
    del config
    root = Path(__file__).resolve().parents[2]
    for item in items:
        name = item.name
        benchmark_id: str | None = None
        if name == "test_b0006_compact_manifest_matches_all_raw_records":
            benchmark_id = "0006"
        elif name.startswith("test_committed_") or "manifest_matches" in name:
            match = re.search(r"b(\d{4})", name)
            if match:
                benchmark_id = match.group(1)
        if benchmark_id is None:
            continue
        evidence = _evidence_path(root, benchmark_id)
        if evidence is not None and not evidence.is_file():
            item.add_marker(
                pytest.mark.skip(
                    reason=f"historical benchmark evidence is not present: {evidence}"
                )
            )


def cpp_binary(name: str) -> Path:
    build = Path(os.environ.get("K3X_BUILD_DIR", "build")).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    return build / f"{name}{suffix}"


@pytest.fixture
def synthetic_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    write_source_checkpoint(source)
    return source

