# 실제 K3 expert 크기의 bounded source fixture 생성 CLI를 제공합니다.
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from k3x_ref.storage_fixture import write_bounded_expert_source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--chunk-bytes", type=int, default=1 << 20)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()
    report = write_bounded_expert_source(
        args.output,
        seed=args.seed,
        chunk_bytes=args.chunk_bytes,
        layer_id=args.layer,
        expert_id=args.expert,
    )
    payload = asdict(report)
    payload["shard_path"] = str(report.shard_path)
    payload["manifest_path"] = str(report.manifest_path)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
