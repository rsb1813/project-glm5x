# GLM5X bundle-backed reference의 prefill, TTFT, decode 측정 CLI입니다.
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import torch

from glm5x_ref import GLM5XDecoderModelReference


def _parse_tokens(value: str) -> list[int]:
    try:
        tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("prompt must be comma-separated integers") from exc
    if not tokens or any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError("prompt must contain non-negative token IDs")
    return tokens


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure a bundle-backed GLM5X reference prefill/decode path."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=_parse_tokens, required=True)
    parser.add_argument("--new-tokens", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cache-experts", action="store_true")
    parser.add_argument("--layer-cache-capacity", type=int, default=0)
    parser.add_argument(
        "--execution-mode", choices=("loop", "expert_major"), default="loop"
    )
    parser.add_argument(
        "--sparse-topk-attention",
        action="store_true",
        help="use only DSA-selected key/value positions in the MLA attention path",
    )
    parser.add_argument(
        "--expert-load-workers",
        type=int,
        default=1,
        help="parallel K3X expert payload readers; 1 keeps the serial reference path",
    )
    parser.add_argument(
        "--expert-cache-bytes",
        type=int,
        default=0,
        help="bounded exact host payload cache capacity; 0 disables it",
    )
    parser.add_argument(
        "--expert-device-cache-bytes",
        type=int,
        default=0,
        help="bounded exact decoded CUDA expert cache capacity; 0 disables it",
    )
    parser.add_argument(
        "--expert-precision",
        choices=("bf16", "fp8"),
        default="bf16",
        help="expert projection precision; fp8 is experimental and default-off",
    )
    parser.add_argument(
        "--lazy-bundle",
        action="store_true",
        help="skip whole-artifact payload/root scans and CRC-check selected tensors on read",
    )
    return parser


def measure(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.new_tokens <= 0:
        raise ValueError("new-tokens must be positive")
    if arguments.layer_cache_capacity < 0:
        raise ValueError("layer-cache-capacity must be non-negative")
    if arguments.expert_load_workers <= 0:
        raise ValueError("expert-load-workers must be positive")
    if arguments.expert_cache_bytes < 0:
        raise ValueError("expert-cache-bytes must be non-negative")
    if arguments.expert_device_cache_bytes < 0:
        raise ValueError("expert-device-cache-bytes must be non-negative")
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    device = torch.device(arguments.device)
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config must contain a JSON object")

    model = GLM5XDecoderModelReference.from_bundle(
        arguments.bundle,
        config=config,
        cache_experts=arguments.cache_experts,
        verify_payloads=not arguments.lazy_bundle,
        verify_root=not arguments.lazy_bundle,
        layer_cache_capacity=arguments.layer_cache_capacity,
        device=device,
        execution_mode=arguments.execution_mode,
        use_sparse_topk=arguments.sparse_topk_attention,
        expert_load_workers=arguments.expert_load_workers,
        expert_cache_capacity_bytes=arguments.expert_cache_bytes,
        expert_device_cache_capacity_bytes=arguments.expert_device_cache_bytes,
        expert_precision=arguments.expert_precision,
    )
    storage_before = model.bundle_read_stats
    prompt = torch.tensor(arguments.prompt, dtype=torch.long, device=device)
    _synchronize(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        prefill = model.forward_tokens(prompt)
    _synchronize(device)
    prefill_seconds = time.perf_counter() - prefill_start
    storage_after_prefill = model.bundle_read_stats

    state = prefill.state
    generated: list[int] = []
    decode_seconds = 0.0
    first_decode_seconds: float | None = None
    decode_step_seconds: list[float] = []
    decode_step_cache_hits: list[int] = []
    decode_step_device_cache_hits: list[int] = []
    with torch.inference_mode():
        for index in range(arguments.new_tokens):
            token = int(torch.argmax(prefill.logits[:, -1, :], dim=-1).item())
            start = time.perf_counter()
            step = model.forward_token(token, state)
            _synchronize(device)
            elapsed = time.perf_counter() - start
            if first_decode_seconds is None:
                first_decode_seconds = elapsed
            decode_seconds += elapsed
            decode_step_seconds.append(elapsed)
            decode_step_cache_hits.append(model.expert_payload_cache_stats.hits)
            decode_step_device_cache_hits.append(model.expert_device_cache_stats.hits)
            generated.append(token)
            state = step.state
            prefill = step
    storage_after_decode = model.bundle_read_stats

    def storage_delta(after, before) -> dict[str, int]:
        if after is None or before is None:
            return {"calls": 0, "bytes": 0}
        return {
            "calls": after.calls - before.calls,
            "bytes": after.bytes - before.bytes,
        }

    prefill_storage = storage_delta(storage_after_prefill, storage_before)
    decode_storage = storage_delta(storage_after_decode, storage_after_prefill)

    payload: dict[str, object] = {
        "measured": True,
        "model": "GLM5XDecoderModelReference",
        "bundle": str(arguments.bundle),
        "device": str(device),
        "layers": model.layer_count,
        "context_length": len(arguments.prompt),
        "prefill_tokens": len(arguments.prompt),
        "decode_tokens": arguments.new_tokens,
        "prefill_seconds": prefill_seconds,
        "prefill_tok_s": len(arguments.prompt) / prefill_seconds,
        "prefill_storage_read_calls": prefill_storage["calls"],
        "prefill_storage_read_bytes": prefill_storage["bytes"],
        "prefill_storage_read_bytes_per_token": prefill_storage["bytes"] / len(arguments.prompt),
        "ttft_seconds": prefill_seconds + (first_decode_seconds or 0.0),
        "decode_seconds": decode_seconds,
        "decode_tok_s": arguments.new_tokens / decode_seconds,
        "decode_storage_read_calls": decode_storage["calls"],
        "decode_storage_read_bytes": decode_storage["bytes"],
        "decode_storage_read_bytes_per_token": decode_storage["bytes"] / arguments.new_tokens,
        "decode_step_seconds": decode_step_seconds,
        "decode_step_tok_s": [1.0 / value for value in decode_step_seconds],
        "decode_step_expert_cache_hits": decode_step_cache_hits,
        "decode_step_expert_device_cache_hits": decode_step_device_cache_hits,
        "prompt": arguments.prompt,
        "generated_tokens": generated,
        "cache_experts": arguments.cache_experts,
        "layer_cache_capacity": arguments.layer_cache_capacity,
        "execution_mode": arguments.execution_mode,
        "sparse_topk_attention": arguments.sparse_topk_attention,
        "expert_load_workers": arguments.expert_load_workers,
        "expert_cache_bytes": arguments.expert_cache_bytes,
        "expert_device_cache_bytes": arguments.expert_device_cache_bytes,
        "expert_precision": arguments.expert_precision,
        "lazy_bundle": arguments.lazy_bundle,
    }
    cache_stats = model.expert_payload_cache_stats
    device_cache_stats = model.expert_device_cache_stats
    cache_lookups = cache_stats.hits + cache_stats.misses
    device_cache_lookups = device_cache_stats.hits + device_cache_stats.misses
    payload.update(
        {
            "expert_cache_resident_bytes": cache_stats.resident_bytes,
            "expert_cache_entries": cache_stats.entries,
            "expert_cache_hits": cache_stats.hits,
            "expert_cache_misses": cache_stats.misses,
            "expert_cache_evictions": cache_stats.evictions,
            "expert_cache_hit_rate": (
                cache_stats.hits / cache_lookups if cache_lookups else 0.0
            ),
            "expert_device_cache_resident_bytes": device_cache_stats.resident_bytes,
            "expert_device_cache_entries": device_cache_stats.entries,
            "expert_device_cache_hits": device_cache_stats.hits,
            "expert_device_cache_misses": device_cache_stats.misses,
            "expert_device_cache_evictions": device_cache_stats.evictions,
            "expert_device_cache_hit_rate": (
                device_cache_stats.hits / device_cache_lookups
                if device_cache_lookups
                else 0.0
            ),
        }
    )
    if device.type == "cuda":
        payload.update(
            {
                "vram_peak_bytes": torch.cuda.max_memory_allocated(device),
                "vram_reserved_peak_bytes": torch.cuda.max_memory_reserved(device),
            }
        )
    return payload


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _build_parser().parse_args(arguments)
    try:
        result = measure(parsed)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
