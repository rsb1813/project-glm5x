# 실제 GLM-5.2 layer-10에서 reduced Top-K의 출력 오차와 payload 비용을 측정합니다.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from glm5x_ref.layer10_moe import GLM5XLayer10MoEReference


def _read_activation(path: Path) -> torch.Tensor:
    payload = path.read_bytes()[40:]
    values = torch.frombuffer(bytearray(payload), dtype=torch.int16).view(torch.bfloat16)
    return values.reshape(1, -1, 6144).to(torch.float32)


def _run(
    bundle: Path,
    hidden: torch.Tensor,
    top_k: int,
    expert_precision: str,
    proxy_mode: str,
    proxy_top_k: int | None,
) -> tuple[torch.Tensor, dict[str, object]]:
    layer = GLM5XLayer10MoEReference.from_bundle(
        bundle,
        layer_id=10,
        top_k=top_k,
        device="cuda",
        verify_payloads=False,
        verify_root=False,
        expert_load_workers=16,
        expert_cache_capacity_bytes=8 * 1024 * 1024 * 1024,
        expert_precision=expert_precision,
        proxy_mode=proxy_mode,
        proxy_top_k=proxy_top_k,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        result = layer(hidden.to(device="cuda"))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return result.output.detach().cpu(), {
        "top_k": top_k,
        "expert_precision": expert_precision,
        "proxy_mode": proxy_mode,
        "proxy_top_k": proxy_top_k,
        "seconds": elapsed,
        "unique_experts": len(result.loaded_experts),
        "loaded_experts": list(result.loaded_experts),
        "output": result.output.detach().cpu(),
        "topk_indices": result.topk_indices.detach().cpu(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[8, 6, 4, 2])
    parser.add_argument("--expert-precision", choices=("bf16", "int4"), default="bf16")
    parser.add_argument("--proxy-mode", choices=("none", "shared"), default="none")
    parser.add_argument("--proxy-top-k", type=int, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    hidden = _read_activation(args.activation)
    rows: list[dict[str, object]] = []
    reference = None
    if args.proxy_mode != "none":
        reference, _ = _run(
            args.bundle,
            hidden,
            args.top_k[0],
            args.expert_precision,
            "none",
            None,
        )
    for top_k in args.top_k:
        output, row = _run(
            args.bundle,
            hidden,
            top_k,
            args.expert_precision,
            args.proxy_mode,
            args.proxy_top_k,
        )
        if reference is None:
            reference = output
        diff = output.float() - reference.float()
        row["max_abs_error"] = float(diff.abs().max())
        row["relative_l2_error"] = float(diff.norm() / reference.float().norm().clamp_min(1e-12))
        row.pop("output")
        row["route_ids"] = row.pop("topk_indices").tolist()
        rows.append(row)
    print(json.dumps(rows, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
