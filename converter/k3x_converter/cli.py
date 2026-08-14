# K3X 변환과 완전 무결성 검증 명령줄 인터페이스를 제공합니다.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .reader import K3XReader
from .writer import convert


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="k3x-convert")
    subcommands = parser.add_subparsers(dest="command", required=True)
    conversion = subcommands.add_parser("convert")
    conversion.add_argument("source", type=Path)
    conversion.add_argument("output", type=Path)
    conversion.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    conversion.add_argument("--dry-run", action="store_true")
    validation = subcommands.add_parser("validate")
    validation.add_argument("artifact", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.command == "convert":
        report = convert(args.source, args.output, chunk_bytes=args.chunk_bytes,
                         dry_run=args.dry_run)
        print(json.dumps({
            "completed": report.completed,
            "dry_run": args.dry_run,
            "maximum_source_read_bytes": report.maximum_source_read_bytes,
            "output": str(report.output_path),
            "reused_extent_count": len(report.reused_extent_ids),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    reader = K3XReader.open(args.artifact)
    print(json.dumps({
        "experts": len(reader.expert_records),
        "layers": len(reader.layer_records),
        "tensors": len(reader.tensor_records),
        "valid": True,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
