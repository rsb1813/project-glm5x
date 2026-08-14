# GLM5X artifact 변환과 무결성 검증 CLI를 제공합니다.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from glm5x_ref.manifest import GLM5XTensorManifest
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert

from .shard import convert_glm5x_shard


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glm5x-convert")
    subcommands = parser.add_subparsers(dest="command", required=True)
    conversion = subcommands.add_parser("convert")
    conversion.add_argument("source", type=Path)
    conversion.add_argument("output", type=Path)
    conversion.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    conversion.add_argument("--dry-run", action="store_true")
    shard = subcommands.add_parser("convert-shard")
    shard.add_argument("source", type=Path)
    shard.add_argument("output", type=Path)
    shard.add_argument("--config", type=Path, required=True)
    shard.add_argument("--index", type=Path, required=True)
    shard.add_argument("--shard-name", required=True)
    shard.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    shard.add_argument("--dry-run", action="store_true")
    validation = subcommands.add_parser("validate")
    validation.add_argument("artifact", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.command == "convert":
        report = convert(
            args.source,
            args.output,
            chunk_bytes=args.chunk_bytes,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    "completed": report.completed,
                    "dry_run": args.dry_run,
                    "maximum_source_read_bytes": report.maximum_source_read_bytes,
                    "output": str(report.output_path),
                    "reused_extent_count": len(report.reused_extent_ids),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "convert-shard":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        index = json.loads(args.index.read_text(encoding="utf-8"))
        manifest = GLM5XTensorManifest.from_json(config, index)
        report = convert_glm5x_shard(
            args.source,
            args.output,
            manifest,
            args.shard_name,
            chunk_bytes=args.chunk_bytes,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    "completed": report.completed,
                    "dry_run": args.dry_run,
                    "maximum_source_read_bytes": report.maximum_source_read_bytes,
                    "output": str(report.output_path),
                    "sidecar": str(report.sidecar_path),
                    "source_sha256": report.source_sha256,
                    "tensor_count": report.tensor_count,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    reader = K3XReader.open(args.artifact)
    print(
        json.dumps(
            {
                "experts": len(reader.expert_records),
                "layers": len(reader.layer_records),
                "tensors": len(reader.tensor_records),
                "valid": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
