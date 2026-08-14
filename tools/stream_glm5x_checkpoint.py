# HF GLM5X checkpoint를 shard 단위로 받아 변환하고 검증된 원본을 정리합니다.
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import quote

from glm5x_converter.bundle import assemble_glm5x_expert_bundle
from glm5x_converter.multi import convert_glm5x_shards
from glm5x_ref.manifest import GLM5XTensorManifest


@dataclass(frozen=True)
class RepositoryFile:
    path: str
    size: int


def manifest_for_shard(manifest: GLM5XTensorManifest, shard_name: str) -> GLM5XTensorManifest:
    """Return a validated view containing only one independently convertible shard."""
    selected = tuple(
        (name, shard)
        for name, shard in manifest.tensor_shards
        if shard == shard_name
    )
    if not selected or shard_name not in manifest.shard_names:
        raise ValueError("GLM5X_STREAM_SHARD_NOT_IN_MANIFEST")
    return replace(manifest, tensor_shards=selected, shard_names=(shard_name,))


def _request_json(url: str, token: str | None) -> object:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"GLM5X_STREAM_METADATA_FETCH_FAILED: {url}") from exc


def list_repository_files(
    repo: str = "zai-org/GLM-5.2", *, revision: str = "main", token: str | None = None
) -> dict[str, RepositoryFile]:
    encoded_repo = quote(repo, safe="/")
    encoded_revision = quote(revision, safe="")
    url = (
        f"https://huggingface.co/api/models/{encoded_repo}/tree/{encoded_revision}"
        "?recursive=false&expand=false"
    )
    payload = _request_json(url, token)
    if not isinstance(payload, list):
        raise RuntimeError("GLM5X_STREAM_METADATA_SHAPE")
    files: dict[str, RepositoryFile] = {}
    for item in payload:
        if not isinstance(item, Mapping) or item.get("type") != "file":
            continue
        path = item.get("path")
        size = item.get("size")
        if not isinstance(path, str) or not isinstance(size, int) or size < 0:
            continue
        files[path] = RepositoryFile(path, size)
    if "config.json" not in files or "model.safetensors.index.json" not in files:
        raise RuntimeError("GLM5X_STREAM_REQUIRED_METADATA_MISSING")
    return files


def download_resumable(
    url: str,
    destination: str | Path,
    expected_size: int,
    *,
    token: str | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> Path:
    """Download one file with a local .part file and HTTP Range resume."""
    if expected_size < 0 or chunk_bytes <= 0:
        raise ValueError("GLM5X_STREAM_DOWNLOAD_ARGUMENTS")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == expected_size:
            return destination
        raise RuntimeError(f"GLM5X_STREAM_SIZE_MISMATCH: {destination}")
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_size:
        partial.unlink()
        existing = 0

    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = response.getcode() or 200
            append = existing > 0 and status == 206
            if not append:
                existing = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                while True:
                    chunk = response.read(chunk_bytes)
                    if not chunk:
                        break
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(f"GLM5X_STREAM_DOWNLOAD_FAILED: {url}") from exc
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"GLM5X_STREAM_SIZE_MISMATCH: {destination} "
            f"{partial.stat().st_size}!={expected_size}"
        )
    os.replace(partial, destination)
    return destination


def _download_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{quote(revision, safe='')}/{quote(path, safe='')}?download=true"


def has_verified_deleted_artifact(output_dir: str | Path, shard_name: str) -> bool:
    artifact = Path(output_dir) / (Path(shard_name).with_suffix(".k3x").name)
    marker = artifact.with_suffix(artifact.suffix + ".source-deleted.json")
    return artifact.is_file() and marker.is_file()


def stream_checkpoint(
    *,
    repo: str,
    revision: str,
    source_dir: str | Path,
    output_dir: str | Path,
    bundle_path: str | Path,
    token: str | None = None,
    chunk_bytes: int = 8 * 1024 * 1024,
    max_shards: int | None = None,
    shard_start: int = 0,
    shard_end: int | None = None,
    assemble: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    source_dir, output_dir, bundle_path = map(Path, (source_dir, output_dir, bundle_path))
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("GLM5X_STREAM_SOURCE_OUTPUT_OVERLAP")
    files = list_repository_files(repo, revision=revision, token=token)
    shard_names = sorted(path for path in files if path.endswith(".safetensors"))
    if not shard_names:
        raise RuntimeError("GLM5X_STREAM_SHARDS_MISSING")
    if max_shards is not None and (max_shards <= 0 or max_shards > len(shard_names)):
        raise ValueError("GLM5X_STREAM_MAX_SHARDS")
    if shard_start < 0 or shard_start > len(shard_names):
        raise ValueError("GLM5X_STREAM_SHARD_START")
    if shard_end is None:
        shard_end = len(shard_names)
    if shard_end < shard_start or shard_end > len(shard_names):
        raise ValueError("GLM5X_STREAM_SHARD_END")
    selected_shards = shard_names[:max_shards] if max_shards is not None else shard_names
    selected_shards = selected_shards[shard_start:shard_end]
    metadata_names = [
        name
        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
        if name in files
    ]
    if not dry_run:
        for name in metadata_names:
            download_resumable(
                _download_url(repo, revision, name),
                source_dir / name,
                files[name].size,
                token=token,
                chunk_bytes=chunk_bytes,
            )
        config = json.loads((source_dir / "config.json").read_text(encoding="utf-8"))
        index = json.loads((source_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        manifest = GLM5XTensorManifest.from_json(config, index)
    else:
        config = None
        index = None
        manifest = None

    converted: list[str] = []
    for shard_name in selected_shards:
        if dry_run:
            continue
        assert manifest is not None
        if shard_name not in manifest.shard_names:
            raise RuntimeError(f"GLM5X_STREAM_INDEX_SHARD_MISSING: {shard_name}")
        if not has_verified_deleted_artifact(output_dir, shard_name):
            download_resumable(
                _download_url(repo, revision, shard_name),
                source_dir / shard_name,
                files[shard_name].size,
                token=token,
                chunk_bytes=chunk_bytes,
            )
        report = convert_glm5x_shards(
            source_dir,
            output_dir,
            manifest_for_shard(manifest, shard_name),
            chunk_bytes=chunk_bytes,
            delete_source=True,
        )
        if not report.completed:
            raise RuntimeError(f"GLM5X_STREAM_CONVERSION_INCOMPLETE: {shard_name}")
        converted.append(shard_name)
        print(json.dumps({"converted": shard_name, "remaining": len(selected_shards) - len(converted)}), flush=True)

    assembled = False
    if not dry_run and assemble and max_shards is None:
        # Every shard was strict-reader verified before its source deletion marker.
        # Reuse that gate and avoid a second full payload scan for the final index.
        bundle_report = assemble_glm5x_expert_bundle(
            output_dir,
            bundle_path,
            verify_payloads=False,
            verify_root=False,
        )
        assembled = bundle_report.completed
    return {
        "repo": repo,
        "revision": revision,
        "shard_count": len(shard_names),
        "selected_shards": len(selected_shards),
        "shard_start": shard_start,
        "shard_end": shard_end,
        "assemble": assemble,
        "converted_shards": len(converted),
        "assembled": assembled,
        "dry_run": dry_run,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stream-glm5x-checkpoint")
    parser.add_argument("--repo", default="zai-org/GLM-5.2")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument(
        "--no-assemble",
        action="store_true",
        help="convert only the selected shard range without assembling the final bundle",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    token = os.environ.get("HF_TOKEN")
    try:
        result = stream_checkpoint(
            repo=args.repo,
            revision=args.revision,
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            bundle_path=args.bundle,
            token=token,
            chunk_bytes=args.chunk_bytes,
            max_shards=args.max_shards,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
            assemble=not args.no_assemble,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
