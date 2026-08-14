# GLM5X 체크포인트 shard를 독립 산출물로 순차 변환하고 완료 shard를 검증합니다.

from __future__ import annotations

import json
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


def _existing_report(
    source: Path,
    output: Path,
    shard_name: str,
    chunk_bytes: int,
) -> tuple[GLM5XShardConversionReport, int]:
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    if not sidecar.is_file():
        raise K3XError("EXISTING_ARTIFACT_METADATA_MISSING", str(output))
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        source_sha256, maximum = _source_sha256(source, chunk_bytes)
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
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise K3XError("INVALID_EXISTING_ARTIFACT", str(output)) from exc
    report = GLM5XShardConversionReport(
        True,
        output,
        sidecar,
        len(reader.tensor_records),
        tensor_ids,
        maximum,
        source_sha256,
        tuple(f"{record.tensor_id:016x}:data" for record in reader.tensor_records),
    )
    return report, maximum


def convert_glm5x_shards(
    source_dir: str | Path,
    output_dir: str | Path,
    manifest: GLM5XTensorManifest,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    dry_run: bool = False,
) -> GLM5XMultiShardConversionReport:
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise K3XError("SOURCE_DIRECTORY_NOT_FOUND", str(source_dir))
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    skipped: list[str] = []
    reports: list[GLM5XShardConversionReport] = []
    maximum_read = 0
    for shard_name in manifest.shard_names:
        source = source_dir / shard_name
        output = output_dir / f"{Path(shard_name).stem}.k3x"
        output_paths.append(output)
        resume_path = output.with_suffix(output.suffix + ".resume.json")
        if output.exists() and not resume_path.exists() and not dry_run:
            report, observed = _existing_report(source, output, shard_name, chunk_bytes)
            reports.append(report)
            skipped.append(shard_name)
            maximum_read = max(maximum_read, observed)
            continue
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

    return GLM5XMultiShardConversionReport(
        all(report.completed for report in reports) and not dry_run,
        tuple(output_paths),
        tuple(skipped),
        tuple(reports),
        maximum_read,
    )
