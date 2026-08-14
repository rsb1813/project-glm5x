# GLM5X 샤드 artifact에 흩어진 expert tensor를 복사 없이 실행 인덱스로 묶습니다.

from __future__ import annotations

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Mapping

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


@dataclass(frozen=True)
class GLM5XExpertPayloadCacheStats:
    capacity_bytes: int
    resident_bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int


class GLM5XExpertPayloadCache:
    """Bounded exact raw-BF16 payload cache shared across layer loads."""

    def __init__(self, capacity_bytes: int) -> None:
        if (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes <= 0
        ):
            raise ValueError("GLM5X_EXPERT_CACHE_CAPACITY")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[
            tuple[int, int], tuple[dict[str, bytes], int]
        ] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = Lock()

    def get(self, key: tuple[int, int]) -> dict[str, bytes] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return dict(entry[0])

    def put(self, key: tuple[int, int], payload: Mapping[str, bytes]) -> None:
        value = {role: bytes(data) for role, data in payload.items()}
        size = sum(len(data) for data in value.values())
        if size > self.capacity_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._resident_bytes -= previous[1]
            while self._entries and self._resident_bytes + size > self.capacity_bytes:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._resident_bytes -= evicted_size
                self._evictions += 1
            self._entries[key] = (value, size)
            self._resident_bytes += size

    @property
    def stats(self) -> GLM5XExpertPayloadCacheStats:
        with self._lock:
            return GLM5XExpertPayloadCacheStats(
                capacity_bytes=self.capacity_bytes,
                resident_bytes=self._resident_bytes,
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )


@dataclass(frozen=True)
class GLM5XExpertBundle:
    """검증된 cross-shard expert 인덱스와 payload reader 묶음입니다."""

    path: Path
    metadata: Mapping[str, object]
    artifact_paths: Mapping[str, Path]
    readers: Mapping[str, K3XReader]
    experts: Mapping[tuple[int, int], Mapping[str, Mapping[str, object]]]
    payload_cache: GLM5XExpertPayloadCache | None = None

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        verify_payloads: bool = True,
        verify_root: bool = True,
        expert_cache_capacity_bytes: int = 0,
    ) -> "GLM5XExpertBundle":
        path = Path(path)
        if (
            not isinstance(expert_cache_capacity_bytes, int)
            or isinstance(expert_cache_capacity_bytes, bool)
            or expert_cache_capacity_bytes < 0
        ):
            raise ValueError("GLM5X_EXPERT_CACHE_CAPACITY")
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise K3XError("EXPERT_BUNDLE_INVALID", str(path)) from exc
        if metadata.get("format") != "glm5x-expert-bundle-v1":
            raise K3XError("EXPERT_BUNDLE_FORMAT", str(path))
        artifacts = metadata.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise K3XError("EXPERT_BUNDLE_ARTIFACTS", str(path))
        artifact_paths: dict[str, Path] = {}
        readers: dict[str, K3XReader] = {}
        for item in artifacts:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise K3XError("EXPERT_BUNDLE_ARTIFACT_METADATA", str(path))
            relative = item["path"]
            if relative in artifact_paths:
                raise K3XError("EXPERT_BUNDLE_DUPLICATE_ARTIFACT", relative)
            artifact = (path.parent / relative).resolve()
            if not artifact.is_file():
                raise K3XError("EXPERT_BUNDLE_ARTIFACT_MISSING", str(artifact))
            reader = K3XReader.open(
                artifact,
                verify_payloads=verify_payloads,
                verify_root=verify_root,
            )
            if (
                item.get("file_uuid") != reader.superblock.file_uuid.hex()
                or item.get("root_sha256") != reader.superblock.root_sha256.hex()
                or item.get("source_sha256") != reader.superblock.source_sha256.hex()
                or item.get("tensor_count") != len(reader.tensor_records)
            ):
                raise K3XError("EXPERT_BUNDLE_ARTIFACT_MISMATCH", relative)
            artifact_paths[relative] = artifact
            readers[relative] = reader
        raw_experts = metadata.get("experts")
        if not isinstance(raw_experts, list):
            raise K3XError("EXPERT_BUNDLE_EXPERTS", str(path))
        experts: dict[tuple[int, int], Mapping[str, Mapping[str, object]]] = {}
        for item in raw_experts:
            if not isinstance(item, dict) or not isinstance(item.get("roles"), dict):
                raise K3XError("EXPERT_BUNDLE_EXPERT_METADATA", str(path))
            layer_id, expert_id = item.get("layer_id"), item.get("expert_id")
            if not isinstance(layer_id, int) or not isinstance(expert_id, int):
                raise K3XError("EXPERT_BUNDLE_EXPERT_METADATA", str(path))
            key = (layer_id, expert_id)
            if key in experts:
                raise K3XError("EXPERT_BUNDLE_DUPLICATE_EXPERT", f"{layer_id}:{expert_id}")
            roles = item["roles"]
            if set(roles) != set(_ROLES):
                raise K3XError("EXPERT_BUNDLE_INCOMPLETE_EXPERT", f"{layer_id}:{expert_id}")
            experts[key] = roles
        payload_cache = (
            GLM5XExpertPayloadCache(expert_cache_capacity_bytes)
            if expert_cache_capacity_bytes
            else None
        )
        return cls(path, metadata, artifact_paths, readers, experts, payload_cache)

    def read_expert(self, layer_id: int, expert_id: int) -> dict[str, bytes]:
        roles = self.experts.get((layer_id, expert_id))
        if roles is None:
            raise K3XError("EXPERT_BUNDLE_EXPERT_NOT_FOUND", f"{layer_id}:{expert_id}")
        if self.payload_cache is not None:
            cached = self.payload_cache.get((layer_id, expert_id))
            if cached is not None:
                return cached
        result: dict[str, bytes] = {}
        for role in _ROLES:
            item = roles[role]
            ref = item.get("ref")
            if not isinstance(ref, dict) or not isinstance(ref.get("artifact"), str):
                raise K3XError("EXPERT_BUNDLE_REFERENCE_METADATA", role)
            artifact_key = ref["artifact"]
            reader = self.readers.get(artifact_key)
            if reader is None or not isinstance(ref.get("tensor_id"), int):
                raise K3XError("EXPERT_BUNDLE_REFERENCE_ARTIFACT", artifact_key)
            record = next(
                (record for record in reader.tensor_records if record.tensor_id == ref["tensor_id"]),
                None,
            )
            if record is None:
                raise K3XError("EXPERT_BUNDLE_REFERENCE_TENSOR", role)
            expected = {
                "dtype": record.dtype.name,
                "quantization": record.quantization.name,
                "shape": list(record.dimensions),
                "data_offset": record.data_offset,
                "data_length": record.data_length,
                "logical_length": record.logical_length,
                "data_crc32c": record.data_crc32c,
            }
            if any(ref.get(key) != value for key, value in expected.items()):
                raise K3XError("EXPERT_BUNDLE_REFERENCE_MISMATCH", role)
            if (
                record.dtype.name != "BF16"
                or record.quantization.name != "NONE"
                or record.auxiliary_length != 0
            ):
                raise K3XError("EXPERT_BUNDLE_UNSUPPORTED_PAYLOAD", role)
            data, auxiliary = reader.read_tensor_extents(record)
            if auxiliary:
                raise K3XError("EXPERT_BUNDLE_AUXILIARY_PAYLOAD", role)
            result[role] = data
        if self.payload_cache is not None:
            self.payload_cache.put((layer_id, expert_id), result)
        return result

    @property
    def expert_payload_cache_stats(self) -> GLM5XExpertPayloadCacheStats:
        if self.payload_cache is None:
            return GLM5XExpertPayloadCacheStats(0, 0, 0, 0, 0, 0)
        return self.payload_cache.stats


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
    *,
    verify_payloads: bool = True,
    verify_root: bool = True,
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
    reader = K3XReader.open(
        artifact,
        verify_payloads=verify_payloads,
        verify_root=verify_root,
    )
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
    verify_payloads: bool = True,
    verify_root: bool = True,
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
        metadata, count = _load_artifact(
            artifact,
            output,
            groups,
            verify_payloads=verify_payloads,
            verify_root=verify_root,
        )
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
