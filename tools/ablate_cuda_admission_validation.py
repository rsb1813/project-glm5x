# CUDA immutable-weight admission validation의 18행 ablation 증거를 생성합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


CASES = tuple(
    (
        f"{kind}-{experts}-{validation}-profiler-{profiler_name}",
        "ffn-block" if kind == "split" else "moe-layer",
        experts,
        validation,
        profiler_name == "on",
    )
    for kind, validation in (
        ("split", "per-call"),
        ("layer", "per-call"),
        ("layer", "admission"),
    )
    for experts in (1, 4, 16)
    for profiler_name in ("off", "on")
)
GROUPS = tuple(
    (
        f"{kind}-{experts}-{validation}",
        f"{kind}-{experts}-{validation}-profiler-off",
        f"{kind}-{experts}-{validation}-profiler-on",
    )
    for kind, validation in (
        ("split", "per-call"),
        ("layer", "per-call"),
        ("layer", "admission"),
    )
    for experts in (1, 4, 16)
)

_IMMUTABLE_BYTES = 469_776_384


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
    *,
    boundary: str,
    experts: int,
    validation: str,
    profiler: bool,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner), "--model", str(artifact),
            "--boundary", boundary, "--experts", str(experts),
            "--validation", validation,
            "--profiler", "on" if profiler else "off",
            "--warmup", str(warmup), "--iterations", str(iterations),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CUDA benchmark failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA benchmark did not emit an object")
    return payload


def _validate_record(
    record: dict,
    *,
    name: str,
    boundary: str,
    experts: int,
    validation: str,
    profiler: bool,
    warmup: int,
    iterations: int,
) -> None:
    expected = {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": boundary,
        "experts": experts,
        "hidden_width": 7168,
        "latent_width": 3584,
        "expert_intermediate_width": 3072,
        "expert_payload_bytes": 17_547_264,
        "resident_capacity_bytes": 1 << 30,
        "warmup": warmup,
        "iterations": iterations,
        "validation": validation,
        "profiler": profiler,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(f"{name} identity field {field} diverged")
    if record.get("maximum_absolute_error", float("inf")) > 1.0e-5:
        raise RuntimeError(f"{name} numerical divergence")
    if record.get("latency_nanoseconds_median", 0) <= 0:
        raise RuntimeError(f"{name} latency is not positive")
    if profiler != (record.get("kernel_nanoseconds") is not None):
        raise RuntimeError(f"{name} profiler attribution diverged")
    for field in (
        "weight_h2d_bytes", "weight_cache_bypasses",
        "resident_grid_fallbacks", "resident_moe_layer_fallbacks",
    ):
        if record.get(field) != 0:
            raise RuntimeError(f"{name} {field} is nonzero")
    if any(
        marker in key.lower()
        for key in record
        for marker in ("token", "prefill", "ttft")
    ):
        raise RuntimeError(f"{name} contains token-like telemetry")

    split = boundary == "ffn-block"
    expected_activation = iterations * (
        86_016 if split else 28_672 + experts * 52
    )
    expected_d2h = iterations * (
        71_680 + experts * 14_336 if split else 28_672
    )
    expected_sync = iterations * (4 if split else 1)
    if (
        record.get("activation_h2d_bytes") != expected_activation
        or record.get("device_to_host_bytes") != expected_d2h
        or record.get("stream_synchronization_count") != expected_sync
        or record.get("resident_grid_calls") != iterations
        or record.get("resident_grid_kernel_launches") != iterations * 4
    ):
        raise RuntimeError(f"{name} physical traffic accounting diverged")
    if split:
        expected_layer = (0, 0, 0, 0)
    else:
        expected_layer = (
            iterations, experts * iterations, iterations * 13,
            experts * 4 * iterations,
        )
    observed_layer = (
        record.get("resident_moe_layer_calls"),
        record.get("resident_moe_layer_experts"),
        record.get("resident_moe_layer_kernel_launches"),
        record.get("resident_moe_layer_contribution_h2d_bytes"),
    )
    if observed_layer != expected_layer:
        raise RuntimeError(f"{name} MoE-layer accounting diverged")

    layer = boundary == "moe-layer"
    cold_scans = 6 if layer else 0
    cold_bytes = _IMMUTABLE_BYTES if layer else 0
    if (
        record.get("cold_immutable_validation_scans") != cold_scans
        or record.get("cold_immutable_validation_bytes") != cold_bytes
    ):
        raise RuntimeError(f"{name} cold validation accounting diverged")
    if not layer:
        expected_scans = expected_hits = expected_bytes = 0
    elif validation == "per-call":
        expected_scans = iterations * 6
        expected_hits = 0
        expected_bytes = iterations * _IMMUTABLE_BYTES
    else:
        expected_scans = 0
        expected_hits = iterations * 6
        expected_bytes = 0
    if (
        record.get("immutable_validation_scans") != expected_scans
        or record.get("immutable_validation_hits") != expected_hits
        or record.get("immutable_validation_bytes") != expected_bytes
    ):
        raise RuntimeError(f"{name} warm validation accounting diverged")


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
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, boundary, experts, validation, profiler in CASES:
        raw = _run_case(
            artifact, runner, boundary=boundary, experts=experts,
            validation=validation, profiler=profiler,
            warmup=warmup, iterations=iterations,
        )
        _validate_record(
            raw, name=name, boundary=boundary, experts=experts,
            validation=validation, profiler=profiler,
            warmup=warmup, iterations=iterations,
        )
        json_path = output_dir / f"{name}.json"
        _write_json(json_path, raw)
        records.append({
            **raw,
            "name": name,
            "raw_json_sha256": _sha256(json_path),
        })
    by_name = {record["name"]: record for record in records}
    parity_fields = (
        "maximum_absolute_error", "activation_h2d_bytes",
        "device_to_host_bytes", "weight_h2d_bytes",
        "cold_weight_h2d_bytes", "resident_weight_bytes",
        "peak_resident_weight_bytes", "stream_synchronization_count",
        "resident_grid_calls", "resident_grid_kernel_launches",
        "resident_moe_layer_calls", "resident_moe_layer_experts",
        "resident_moe_layer_kernel_launches",
        "resident_moe_layer_contribution_h2d_bytes",
        "cold_immutable_validation_scans",
        "cold_immutable_validation_bytes", "immutable_validation_scans",
        "immutable_validation_hits", "immutable_validation_bytes",
    )
    for group, off_name, on_name in GROUPS:
        off, on = by_name[off_name], by_name[on_name]
        if any(off.get(field) != on.get(field) for field in parity_fields):
            raise RuntimeError(f"{group} profiler physical parity diverged")
        off["paired_profiler_latency_delta_percent"] = 0.0
        on["paired_profiler_latency_delta_percent"] = (
            on["latency_nanoseconds_median"]
            / off["latency_nanoseconds_median"] - 1.0
        ) * 100.0
    for experts in (1, 4, 16):
        for profiler_name in ("off", "on"):
            per_call = by_name[
                f"layer-{experts}-per-call-profiler-{profiler_name}"
            ]
            admission = by_name[
                f"layer-{experts}-admission-profiler-{profiler_name}"
            ]
            admission["paired_admission_latency_delta_percent"] = (
                admission["latency_nanoseconds_median"]
                / per_call["latency_nanoseconds_median"] - 1.0
            ) * 100.0
            per_call["paired_admission_latency_delta_percent"] = 0.0
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "released-dimension-validation-attribution",
        "evidence": "measured",
        "benchmark": "B-0024",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    fieldnames = list(dict.fromkeys(
        field for record in records for field in record
    ))
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    summary["summary_csv_sha256"] = _sha256(output_dir / "summary.csv")
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    run_ablation(
        args.artifact, args.runner, output_dir=args.output_dir,
        warmup=args.warmup, iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
