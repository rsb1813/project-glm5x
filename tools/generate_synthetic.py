# 합성 K3 source shard와 golden correctness fixture를 생성합니다.
from __future__ import annotations

import argparse
from pathlib import Path

from k3x_ref.fixtures import (
    build_synthetic_model,
    write_digest_manifest,
    write_golden,
    write_source_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    write_source_checkpoint(args.output / "source", args.seed)
    write_golden(
        args.output / "golden.npz",
        build_synthetic_model(args.seed),
        [1, 7, 3, 9],
    )
    write_digest_manifest(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
