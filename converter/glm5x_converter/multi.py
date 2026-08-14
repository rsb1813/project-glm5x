# GLM5X 체크포인트 shard를 독립 산출물로 순차 변환하고 완료 shard를 검증합니다.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from k3x_converter.format import K3XError
from k3x_converter.reader import K3XReader

from .shard import (
    GLM5XShardConversionReport,
    _source_sha256,
    convert_glm5x_shard,
)
from glm5x_ref.manifest import GLM5XTensorManifest


@dataclass(frozen=True)
class GLM5XMultiShardConversionReport:
    completed: bool
    output_paths: tuple[Path, ...]
    skipped_shards: tuple[str, ...]
    shard_reports: tuple[GLM5XShardConversionReport, ...]
    maximum_source_read_bytes: int
    deleted_shards: tuple[str, ...] = ()


def _source_deleted_marker(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".source-deleted.json")


def _artifact_report(
    output: Path,
    shard_name: str,
    source_sha256: str,
    maximum_source_read_bytes: int,
) -> GLM5XShardConversionReport:
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        reader = K3XReader.open(output)
        tensor_items = metadata["tensors"]
        tensor_ids = {item["name"]: int(item["tensor_id"]) for item in tensor_items}
        if (
            metadata.get("format") != "glm5x-bounded-shard-v1"
            or metadata.get("source_shard") != shard_name
            or metadata.get("source_sha256") != source_sha256
            or metadata.get("tensor_count") != len(reader.tensor_records)
            or reader.superblock.source_sha256.hex() != source_sha256
        ):
            raise K3XError("EXISTING_ARTIFACT_SOURCE_MISMATCH", str(output))
    except K3XError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise K3XError("INVALID_EXISTING_ARTIFACT", str(output)) from exc
    return GLM5XShardConversionReport(
        True,
        output,
        sidecar,
        len(reader.tensor_records),
        tensor_ids,
        maximum_source_read_bytes,
        source_sha256,
        tuple(f"{record.tensor_id:016x}:data" for record in reader.tensor_records),
    )


def _write_source_deleted_marker(
    output: Path, report: GLM5XShardConversionReport, shard_name: str
) -> None:
    marker = _source_deleted_marker(output)
    partial = marker.with_suffix(marker.suffix + ".partial")
    partial.write_text(
        json.dumps(
            {
                "format": "glm5x-source-deleted-v1",
                "source_shard": shard_name,
                "source_sha256": report.source_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="\n",
    )
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, marker)


def _existing_report(
    source: Path,
    output: Path,
    shard_name: str,
    chunk_bytes: int,
) -> tuple[GLM5XShardConversionReport, int]:
    source_sha256, maximum = _source_sha256(source, chunk_bytes)
    return _artifact_report(output, shard_name, source_sha256, maximum), maximum


def _deleted_report(
    output: Path, shard_name: str
) -> tuple[GLM5XShardConversionReport, int]:
    marker = _source_deleted_marker(output)
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        source_sha256 = metadata["source_sha256"]
        if (
            metadata.get("format") != "glm5x-source-deleted-v1"
            or metadata.get("source_shard") != shard_name
        ):
            raise K3XError("INVALID_SOURCE_DELETED_MARKER", str(marker))
    except K3XError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise K3XError("INVALID_SOURCE_DELETED_MARKER", str(marker)) from exc
    return _artifact_report(output, shard_name, source_sha256, 0), 0


def convert_glm5x_shards(
    source_dir: str | Path,
    output_dir: str | Path,
    manifest: GLM5XTensorManifest,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    dry_run: bool = False,
    delete_source: bool = False,
) -> GLM5XMultiShardConversionReport:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise K3XError("SOURCE_DIRECTORY_NOT_FOUND", str(source_dir))
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    if delete_source and not dry_run:
        source_root = source_dir.resolve()
        output_root = output_dir.resolve()
        if (
            source_root == output_root
            or source_root.is_relative_to(output_root)
            or output_root.is_relative_to(source_root)
        ):
            raise K3XError("DELETE_SOURCE_OUTPUT_OVERLAP")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    skipped: list[str] = []
    reports: list[GLM5XShardConversionReport] = []
    deleted: list[str] = []
    maximum_read = 0
    for shard_name in manifest.shard_names:
        source = source_dir / shard_name
        output = output_dir / f"{Path(shard_name).stem}.k3x"
        output_paths.append(output)
        resume_path = output.with_suffix(output.suffix + ".resume.json")
        if output.exists() and not resume_path.exists() and not dry_run:
            if source.is_file():
                report, observed = _existing_report(source, output, shard_name, chunk_bytes)
            elif delete_source and _source_deleted_marker(output).is_file():
                report, observed = _deleted_report(output, shard_name)
            else:
                raise K3XError("SOURCE_SHARD_NOT_FOUND", str(source))
            skipped.append(shard_name)
            maximum_read = max(maximum_read, observed)
        else:
            report = convert_glm5x_shard(
                source,
                output,
                manifest,
                shard_name,
                chunk_bytes=chunk_bytes,
                dry_run=dry_run,
            )
        reports.append(report)
        maximum_read = max(maximum_read, report.maximum_source_read_bytes)
        if delete_source and report.completed and not dry_run:
            if source.is_file():
                K3XReader.open(output)
                _write_source_deleted_marker(output, report, shard_name)
                source.unlink()
            elif not _source_deleted_marker(output).is_file():
                raise K3XError("SOURCE_SHARD_NOT_FOUND", str(source))
            deleted.append(shard_name)

    return GLM5XMultiShardConversionReport(
        all(report.completed for report in reports) and not dry_run,
        tuple(output_paths),
        tuple(skipped),
        tuple(reports),
        maximum_read,
        tuple(deleted),
    )
