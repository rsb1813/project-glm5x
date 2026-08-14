# GLM5X 샤드 artifact에 흩어진 expert tensor를 복사 없이 실행 인덱스로 묶습니다.

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from k3x_converter.format import K3XError, TensorRecord
from k3x_converter.reader import K3XReader


_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
_ROLES = ("gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class GLM5XExpertBundleReport:
    completed: bool
    output_path: Path
    artifact_count: int
    tensor_count: int
    complete_expert_count: int
    incomplete_expert_count: int


def _relative_artifact(path: Path, output: Path) -> str:
    return os.path.relpath(path, output.parent).replace(os.sep, "/")


def _record_ref(artifact: Path, output: Path, record: TensorRecord) -> dict[str, object]:
    return {
        "artifact": _relative_artifact(artifact, output),
        "tensor_id": record.tensor_id,
        "dtype": record.dtype.name,
        "quantization": record.quantization.name,
        "shape": list(record.dimensions),
        "data_offset": record.data_offset,
        "data_length": record.data_length,
        "logical_length": record.logical_length,
        "data_crc32c": record.data_crc32c,
    }


def _load_artifact(
    artifact: Path,
    output: Path,
    groups: dict[tuple[int, int], dict[str, dict[str, object]]],
) -> tuple[dict[str, object], int]:
    sidecar = artifact.with_suffix(artifact.suffix + ".manifest.json")
    if not sidecar.is_file():
        raise K3XError("EXPERT_BUNDLE_SIDECAR_MISSING", str(artifact))
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise K3XError("EXPERT_BUNDLE_SIDECAR_INVALID", str(sidecar)) from exc
    if metadata.get("format") != "glm5x-bounded-shard-v1":
        raise K3XError("EXPERT_BUNDLE_SIDECAR_FORMAT", str(sidecar))
    reader = K3XReader.open(artifact)
    if metadata.get("source_sha256") != reader.superblock.source_sha256.hex():
        raise K3XError("EXPERT_BUNDLE_SOURCE_MISMATCH", str(artifact))
    records = {record.tensor_id: record for record in reader.tensor_records}
    tensor_items = metadata.get("tensors")
    if not isinstance(tensor_items, list) or len(tensor_items) != len(reader.tensor_records):
        raise K3XError("EXPERT_BUNDLE_TENSOR_COUNT", str(artifact))
    seen_names: set[str] = set()
    seen_ids: set[int] = set()
    for item in tensor_items:
        if not isinstance(item, dict):
            raise K3XError("EXPERT_BUNDLE_TENSOR_METADATA", str(artifact))
        name = item.get("name")
        tensor_id = item.get("tensor_id")
        if not isinstance(name, str) or not isinstance(tensor_id, int):
            raise K3XError("EXPERT_BUNDLE_TENSOR_METADATA", str(artifact))
        if name in seen_names:
            raise K3XError("EXPERT_BUNDLE_DUPLICATE_TENSOR", str(artifact))
        if tensor_id in seen_ids:
            raise K3XError("EXPERT_BUNDLE_DUPLICATE_TENSOR", str(artifact))
        seen_names.add(name)
        seen_ids.add(tensor_id)
        record = records.get(tensor_id)
        if record is None:
            raise K3XError("EXPERT_BUNDLE_TENSOR_ID_MISMATCH", str(artifact))
        match = _EXPERT_RE.match(name)
        if match is None:
            continue
        layer_id, expert_id, role = int(match.group(1)), int(match.group(2)), match.group(3)
        group = groups.setdefault((layer_id, expert_id), {})
        if role in group:
            raise K3XError("EXPERT_BUNDLE_DUPLICATE_ROLE", f"{layer_id}:{expert_id}:{role}")
        group[role] = {
            "name": name,
            "tensor_id": tensor_id,
            "ref": _record_ref(artifact, output, record),
        }
    artifact_metadata = {
        "path": _relative_artifact(artifact, output),
        "file_uuid": reader.superblock.file_uuid.hex(),
        "root_sha256": reader.superblock.root_sha256.hex(),
        "source_sha256": reader.superblock.source_sha256.hex(),
        "tensor_count": len(reader.tensor_records),
    }
    return artifact_metadata, len(reader.tensor_records)


def assemble_glm5x_expert_bundle(
    artifact_dir: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
) -> GLM5XExpertBundleReport:
    artifact_dir, output = Path(artifact_dir), Path(output)
    if not artifact_dir.is_dir():
        raise K3XError("EXPERT_BUNDLE_ARTIFACT_DIRECTORY_NOT_FOUND", str(artifact_dir))
    artifacts = tuple(sorted(artifact_dir.glob("*.k3x")))
    if not artifacts:
        raise K3XError("EXPERT_BUNDLE_NO_ARTIFACTS", str(artifact_dir))
    if output in artifacts:
        artifacts = tuple(item for item in artifacts if item != output)
    groups: dict[tuple[int, int], dict[str, dict[str, object]]] = {}
    artifact_metadata: list[dict[str, object]] = []
    tensor_count = 0
    for artifact in artifacts:
        metadata, count = _load_artifact(artifact, output, groups)
        artifact_metadata.append(metadata)
        tensor_count += count
    complete = []
    incomplete = []
    for (layer_id, expert_id), roles in sorted(groups.items()):
        missing = [role for role in _ROLES if role not in roles]
        item = {
            "layer_id": layer_id,
            "expert_id": expert_id,
            "roles": {role: roles[role] for role in _ROLES if role in roles},
        }
        if missing:
            item["missing_roles"] = missing
            incomplete.append(item)
        else:
            complete.append(item)
    payload = {
        "format": "glm5x-expert-bundle-v1",
        "artifact_count": len(artifact_metadata),
        "tensor_count": tensor_count,
        "complete_expert_count": len(complete),
        "incomplete_expert_count": len(incomplete),
        "artifacts": artifact_metadata,
        "experts": complete,
        "incomplete": incomplete,
    }
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        partial.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        with partial.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, output)
    return GLM5XExpertBundleReport(
        not dry_run,
        output,
        len(artifact_metadata),
        tensor_count,
        len(complete),
        len(incomplete),
    )
