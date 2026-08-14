# released-dimension repeated-view expert에서 CUDA fusion을 순차 비교합니다.
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


def _run_case(
    artifact: Path,
    runner: Path,
    fusion: str,
    slots: int,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner),
            "--model", str(artifact),
            "--fusion", fusion,
            "--slots", str(slots),
            "--warmup", str(warmup),
            "--iterations", str(iterations),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CUDA expert benchmark failed")
    return json.loads(result.stdout)


def _write_record(record: dict, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=record.keys())
        writer.writeheader()
        writer.writerow(record)


def run_expert_fusion_ablation(
    artifact: Path,
    runner: Path,
    *,
    slots: int,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict[str, object]:
    if slots <= 0 or slots > 16:
        raise ValueError("slots must be between 1 and 16")
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for fusion in ("none", "routed-accumulate"):
        record = _run_case(
            artifact, runner, fusion, slots, warmup, iterations
        )
        _write_record(
            record,
            output_dir / f"{fusion}.json",
            output_dir / f"{fusion}.csv",
        )
        records.append(record)

    unfused, fused = records
    invariant_fields = (
        "artifact_kind",
        "routing_semantics",
        "expert_payload_bytes",
        "slots",
        "warmup",
        "iterations",
        "weight_h2d_bytes",
        "activation_h2d_bytes",
    )
    if any(fused[field] != unfused[field] for field in invariant_fields):
        raise RuntimeError("released-dimension fusion provenance changed")
    if (
        unfused["artifact_kind"] != "released_dimension_repeated_view"
        or unfused["routing_semantics"] is not False
        or int(unfused["expert_payload_bytes"]) != 17_547_264
    ):
        raise RuntimeError("invalid released-dimension benchmark identity")
    if int(unfused["fused_moe_calls"]) != 0:
        raise RuntimeError("unfused released-dimension case reported fusion")
    if (
        int(fused["fused_moe_calls"]) != iterations
        or int(fused["fused_moe_experts"]) != slots * iterations
    ):
        raise RuntimeError("fused released-dimension counters are inconsistent")
    error = float(fused["maximum_absolute_error"])
    if not math.isfinite(error) or error > 1.0e-3:
        raise RuntimeError("released-dimension fused numerical parity failed")
    d2h_reduction = int(unfused["device_to_host_bytes"]) - int(
        fused["device_to_host_bytes"]
    )
    if d2h_reduction <= 0:
        raise RuntimeError("released-dimension fusion did not reduce D2H")

    summary: dict[str, object] = {
        "records": records,
        "d2h_reduction_bytes": d2h_reduction,
        "latency_nanoseconds_delta": int(
            fused["latency_nanoseconds_median"]
        )
        - int(unfused["latency_nanoseconds_median"]),
        "kernel_nanoseconds_delta": int(fused["kernel_nanoseconds"])
        - int(unfused["kernel_nanoseconds"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_expert_fusion_ablation(
        args.artifact,
        args.runner,
        slots=args.slots,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
