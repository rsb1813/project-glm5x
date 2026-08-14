# B-0023 released-dimension split/layer CUDA 경계를 동일 조건에서 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


CASES = (
    ("split-1", "ffn-block", 1),
    ("layer-1", "moe-layer", 1),
    ("split-4", "ffn-block", 4),
    ("layer-4", "moe-layer", 4),
    ("split-16", "ffn-block", 16),
    ("layer-16", "moe-layer", 16),
)
PAIRS = (
    ("experts-1", "split-1", "layer-1"),
    ("experts-4", "split-4", "layer-4"),
    ("experts-16", "split-16", "layer-16"),
)

_HIDDEN = 7168
_LATENT = 3584
_INTERMEDIATE = 3072
_EXPERT_PAYLOAD_BYTES = 17_547_264
_RESIDENT_CAPACITY_BYTES = 1 << 30
_ROUTED_NORM_BYTES = _LATENT * 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_case(
    artifact: Path,
    runner: Path,
    boundary: str,
    experts: int,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--boundary",
            boundary,
            "--experts",
            str(experts),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
            or f"{boundary}/{experts} benchmark failed"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"{boundary}/{experts} emitted invalid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{boundary}/{experts} did not emit an object")
    return payload


def _validate_record(
    record: dict,
    *,
    name: str,
    boundary: str,
    experts: int,
    warmup: int,
    iterations: int,
) -> None:
    identity = {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": boundary,
        "experts": experts,
        "hidden_width": _HIDDEN,
        "latent_width": _LATENT,
        "expert_intermediate_width": _INTERMEDIATE,
        "expert_payload_bytes": _EXPERT_PAYLOAD_BYTES,
        "resident_capacity_bytes": _RESIDENT_CAPACITY_BYTES,
        "warmup": warmup,
        "iterations": iterations,
    }
    for field, expected in identity.items():
        if record.get(field) != expected:
            raise RuntimeError(f"{name} identity field {field} diverged")
    if record.get("maximum_absolute_error", float("inf")) > 1.0e-5:
        raise RuntimeError(f"{name} numerical divergence")
    if record.get("latency_nanoseconds_median", 0) <= 0:
        raise RuntimeError(f"{name} latency is not positive")
    if record.get("kernel_nanoseconds", 0) <= 0:
        raise RuntimeError(f"{name} kernel time is not positive")
    if record.get("weight_h2d_bytes") != 0:
        raise RuntimeError(f"{name} warm weight H2D is nonzero")
    if record.get("cold_weight_h2d_bytes", 0) <= 0:
        raise RuntimeError(f"{name} cold weight admission is empty")
    if (
        record.get("resident_weight_bytes", 0) <= 0
        or record.get("peak_resident_weight_bytes", 0) <= 0
        or record.get("peak_vram_bytes", 0)
        < record.get("resident_weight_bytes", 0)
    ):
        raise RuntimeError(f"{name} residency telemetry is invalid")
    if (
        record.get("oracle_peak_vram_bytes", 0) <= 0
        or record.get("peak_vram_bytes", 0)
        < record.get("oracle_peak_vram_bytes", 0)
    ):
        raise RuntimeError(f"{name} oracle residency telemetry is invalid")
    if record.get("weight_cache_bypasses") != 0:
        raise RuntimeError(f"{name} capacity bypass occurred")
    if (
        record.get("resident_grid_fallbacks") != 0
        or record.get("resident_moe_layer_fallbacks") != 0
    ):
        raise RuntimeError(f"{name} fallback occurred")
    if (
        record.get("activation_h2d_bytes", 0) <= 0
        or record.get("device_to_host_bytes", 0) <= 0
    ):
        raise RuntimeError(f"{name} measured traffic is empty")
    if (
        record.get("resident_grid_calls") != iterations
        or record.get("resident_grid_kernel_launches") != iterations * 4
    ):
        raise RuntimeError(f"{name} resident grid accounting diverged")

    if boundary == "ffn-block":
        if record.get("stream_synchronization_count") != iterations * 4:
            raise RuntimeError(f"{name} synchronization count diverged")
        for field in (
            "resident_moe_layer_calls",
            "resident_moe_layer_experts",
            "resident_moe_layer_kernel_launches",
            "resident_moe_layer_fallbacks",
            "resident_moe_layer_contribution_h2d_bytes",
        ):
            if record.get(field) != 0:
                raise RuntimeError(f"{name} leaked MoE-layer accounting")
        return

    if record.get("stream_synchronization_count") != iterations:
        raise RuntimeError(f"{name} synchronization count diverged")
    if (
        record.get("resident_moe_layer_calls") != iterations
        or record.get("resident_moe_layer_experts") != experts * iterations
        or record.get("resident_moe_layer_kernel_launches") != iterations * 13
        or record.get("resident_moe_layer_contribution_h2d_bytes")
        != experts * 4 * iterations
    ):
        raise RuntimeError(f"{name} MoE-layer accounting diverged")


def _validate_pair(pair_name: str, split: dict, layer: dict) -> None:
    cold_delta = (
        layer["cold_weight_h2d_bytes"] - split["cold_weight_h2d_bytes"]
    )
    resident_delta = (
        layer["resident_weight_bytes"] - split["resident_weight_bytes"]
    )
    if cold_delta != _ROUTED_NORM_BYTES or resident_delta != _ROUTED_NORM_BYTES:
        raise RuntimeError(f"{pair_name} norm residency delta diverged")
    activation_reduction = (
        split["activation_h2d_bytes"] - layer["activation_h2d_bytes"]
    )
    d2h_reduction = (
        split["device_to_host_bytes"] - layer["device_to_host_bytes"]
    )
    if activation_reduction <= 0 or d2h_reduction <= 0:
        raise RuntimeError(f"{pair_name} traffic did not decrease")
    sync_reduction = (
        split["stream_synchronization_count"]
        - layer["stream_synchronization_count"]
    )
    if sync_reduction != layer["iterations"] * 3:
        raise RuntimeError(f"{pair_name} synchronization reduction diverged")

    split.update(
        paired_latency_delta_percent=0.0,
        paired_sync_reduction=0,
        paired_activation_h2d_reduction_bytes=0,
        paired_d2h_reduction_bytes=0,
        paired_cold_weight_delta_bytes=0,
        paired_resident_weight_delta_bytes=0,
    )
    layer.update(
        paired_latency_delta_percent=(
            layer["latency_nanoseconds_median"]
            / split["latency_nanoseconds_median"]
            - 1.0
        )
        * 100.0,
        paired_sync_reduction=sync_reduction,
        paired_activation_h2d_reduction_bytes=activation_reduction,
        paired_d2h_reduction_bytes=d2h_reduction,
        paired_cold_weight_delta_bytes=cold_delta,
        paired_resident_weight_delta_bytes=resident_delta,
    )


def run_ablation(
    artifact: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmup: int,
    iterations: int,
) -> dict:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    artifact = Path(artifact).resolve()
    runner = Path(runner).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_names = {
        case_name: pair_name
        for pair_name, split_name, layer_name in PAIRS
        for case_name in (split_name, layer_name)
    }
    records: list[dict] = []
    for name, boundary, experts in CASES:
        raw = _run_case(
            artifact, runner, boundary, experts, warmup, iterations
        )
        _validate_record(
            raw,
            name=name,
            boundary=boundary,
            experts=experts,
            warmup=warmup,
            iterations=iterations,
        )
        raw_path = output_dir / f"{name}.json"
        _write_json(raw_path, raw)
        record = dict(raw)
        record.update(
            name=name,
            pair_name=pair_names[name],
            raw_json_sha256=_sha256(raw_path),
        )
        records.append(record)

    by_name = {record["name"]: record for record in records}
    for pair_name, split_name, layer_name in PAIRS:
        _validate_pair(pair_name, by_name[split_name], by_name[layer_name])

    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "released-dimension-repeated-view",
        "evidence": "measured",
        "benchmark": "B-0023",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    fieldnames = list(
        dict.fromkeys(field for record in records for field in record)
    )
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    summary["summary_csv_sha256"] = _sha256(summary_csv)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    arguments = parser.parse_args()
    summary = run_ablation(
        arguments.artifact,
        arguments.runner,
        output_dir=arguments.output_dir,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
