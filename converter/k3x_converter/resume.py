# 중단 가능한 변환의 검증 완료 extent ledger를 원자적으로 관리합니다.
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompletedExtent:
    extent_id: str
    offset: int
    length: int
    crc32c: int


@dataclass(frozen=True)
class ResumeManifest:
    source_fingerprint: str
    converter_version: str
    configuration_fingerprint: str
    file_uuid: str
    completed: tuple[CompletedExtent, ...]


def read_resume_manifest(path: Path) -> ResumeManifest:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ResumeManifest(
        source_fingerprint=data["source_fingerprint"],
        converter_version=data["converter_version"],
        configuration_fingerprint=data["configuration_fingerprint"],
        file_uuid=data["file_uuid"],
        completed=tuple(CompletedExtent(**item) for item in data["completed"]),
    )


def write_resume_manifest(path: Path, manifest: ResumeManifest) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data = asdict(manifest)
    data["completed"] = [asdict(item) for item in manifest.completed]
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
