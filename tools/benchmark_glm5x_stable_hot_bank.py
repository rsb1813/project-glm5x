# 실제 GLM-5.2 sidecar로 안정 hot-bank의 반복 H2D 절감을 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
from dataclasses import asdict
from pathlib import Path

import torch

from glm5x_converter.bundle import GLM5XExpertBundle
from glm5x_ref.layer10_moe import GLM5XExpertTensorCache
from glm5x_ref.packed_cache import GLM5XPackedExpertCache


_SIDECAR_RE = re.compile(r"^layer-(\d+)-expert-(\d+)\.pgu$")
_SIDECAR_HEADER = struct.Struct("<8sII")


def _counter_delta(before: object, after: object) -> dict[str, int]:
    return {
        name: int(getattr(after, name)) - int(getattr(before, name))
        for name in asdict(after)
        if name not in {"capacity_bytes", "resident_bytes", "entries"}
    }


def _select_trace(
    root: Path,
    bundle: GLM5XExpertBundle,
    *,
    start_layer: int,
    layer_count: int,
    experts_per_layer: int,
) -> tuple[tuple[int, int], ...]:
    by_layer: dict[int, list[int]] = {}
    for path in root.glob("*.pgu"):
        match = _SIDECAR_RE.match(path.name)
        if match is None:
            continue
        layer_id, expert_id = (int(value) for value in match.groups())
        with path.open("rb") as source:
            header = source.read(_SIDECAR_HEADER.size)
            if len(header) != _SIDECAR_HEADER.size:
                continue
            _, _, metadata_length = _SIDECAR_HEADER.unpack(header)
            metadata = json.loads(source.read(metadata_length).decode("utf-8"))
        if metadata.get("source_digest") != bundle.expert_source_digest(
            layer_id, expert_id
        ):
            continue
        by_layer.setdefault(layer_id, []).append(expert_id)

    trace: list[tuple[int, int]] = []
    for layer_id in range(start_layer, start_layer + layer_count):
        experts = sorted(by_layer.get(layer_id, ()))
        if len(experts) < experts_per_layer:
            raise ValueError(f"GLM5X_HOT_BANK_TRACE_LAYER:{layer_id}")
        trace.extend((layer_id, expert_id) for expert_id in experts[:experts_per_layer])
    return tuple(trace)


def benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("GLM5X_HOT_BANK_CUDA_REQUIRED")
    if arguments.policy == "stable_hot_bank" and arguments.protected_entries <= 0:
        raise ValueError("GLM5X_HOT_BANK_PROTECTED_ENTRIES")

    bundle = GLM5XExpertBundle.open(
        arguments.bundle, verify_payloads=False, verify_root=False
    )
    trace = _select_trace(
        arguments.sidecar_dir,
        bundle,
        start_layer=arguments.start_layer,
        layer_count=arguments.layers,
        experts_per_layer=arguments.experts_per_layer,
    )
    packed_cache = GLM5XPackedExpertCache(
        arguments.sidecar_dir,
        host_cache_capacity_bytes=arguments.host_cache_bytes,
        telemetry_enabled=True,
    )
    device_cache = GLM5XExpertTensorCache(
        arguments.device_cache_bytes,
        policy=arguments.policy,
        protected_entries_per_layer=(
            arguments.protected_entries
            if arguments.policy == "stable_hot_bank"
            else 0
        ),
    )

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    passes: list[dict[str, object]] = []
    for pass_index in range(arguments.passes):
        before_device = device_cache.stats
        before_packed = packed_cache.stats
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        for key in trace:
            expert = device_cache.get(key)
            if expert is not None:
                continue
            expert = packed_cache.get(
                key,
                bundle.expert_source_digest(*key),
                device=device,
                precision="nvfp4_gate_up",
            )
            if expert is None:
                raise RuntimeError(f"GLM5X_HOT_BANK_SIDECAR_MISS:{key[0]}:{key[1]}")
            device_cache.put(key, expert)
        torch.cuda.synchronize(device)
        elapsed_ns = time.perf_counter_ns() - started
        after_device = device_cache.stats
        after_packed = packed_cache.stats
        passes.append(
            {
                "pass": pass_index + 1,
                "elapsed_nanoseconds": elapsed_ns,
                "device_cache": _counter_delta(before_device, after_device),
                "packed_cache": _counter_delta(before_packed, after_packed),
                "resident_bytes": after_device.resident_bytes,
                "resident_entries": after_device.entries,
            }
        )

    return {
        "schema_version": 1,
        "benchmark_id": "B-0002",
        "boundary": "structured-real-sidecar-residency",
        "full_model_tps": None,
        "quality_claim": "cache-only exact payload identity; no model-logit claim",
        "policy": arguments.policy,
        "protected_entries_per_layer": (
            arguments.protected_entries
            if arguments.policy == "stable_hot_bank"
            else 0
        ),
        "device_cache_capacity_bytes": arguments.device_cache_bytes,
        "host_cache_capacity_bytes": arguments.host_cache_bytes,
        "bundle": str(arguments.bundle),
        "bundle_sha256": hashlib.sha256(arguments.bundle.read_bytes()).hexdigest(),
        "sidecar_dir": str(arguments.sidecar_dir),
        "precision": "nvfp4_gate_up",
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "trace": [list(key) for key in trace],
        "trace_entries": len(trace),
        "layers": arguments.layers,
        "experts_per_layer": arguments.experts_per_layer,
        "passes": passes,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("lru", "stable_hot_bank"), required=True
    )
    parser.add_argument("--device-cache-bytes", type=int, default=805_306_368)
    parser.add_argument("--host-cache-bytes", type=int, default=8_589_934_592)
    parser.add_argument("--protected-entries", type=int, default=1)
    parser.add_argument("--start-layer", type=int, default=3)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--experts-per-layer", type=int, default=8)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    result = benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
