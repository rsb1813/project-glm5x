# CUDA expert-major 그래프 실행과 released-dimension 배치 경계를 함께 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    for import_root in (repository, repository / "converter", repository / "reference"):
        sys.path.insert(0, str(import_root))

from tools.ablate_expert_major_verification import expert_major_matrix
from tools.benchmark_synthetic import benchmark_once, write_results


def released_batch_matrix() -> tuple[tuple[str, str, int], ...]:
    return (
        ("scalar-2", "scalar", 2),
        ("batch-2", "batch", 2),
        ("scalar-4", "scalar", 4),
        ("batch-4", "batch", 4),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cuda_diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    *,
    mode: str,
    verification: str,
    block_size: int,
    script: str,
) -> dict:
    command = [
        str(runner),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--backend", "cuda-custom",
        "--cuda-allocation", "reused",
        "--cuda-weights", "transient",
        "--cuda-batching", "scalar",
        "--cuda-boundary", "ffn-block",
        "--cuda-transfer", "synchronous",
        "--cuda-moe-fusion", "none",
        "--l1-expert-cache", "disabled",
        "--l2-schedule", "blocking",
        "--diagnostics", "true",
        "--speculative-mode", mode,
        "--speculative-verification", verification,
        "--speculative-block-size", str(block_size),
        "--speculative-script", script,
        "--json", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CUDA graph benchmark failed")
    return json.loads(output.read_text(encoding="utf-8"))


def _run_graph_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _cuda_diagnostic(
        artifact,
        runner,
        output_dir / "greedy-diagnostic.json",
        mode="none",
        verification="token-major",
        block_size=0,
        script="",
    )
    matrix = expert_major_matrix(baseline["token_ids"])
    parity_fields = ("token_ids", "final_state", "routed_experts", "routed_k")
    records: list[dict] = []
    for case in matrix:
        name = str(case["name"])
        mode = str(case["mode"])
        verification = str(case["verification"])
        block_size = int(case["block_size"])
        script = str(case["script"])
        benchmark = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cuda-custom",
            dense_precision="fp32",
            cuda_allocation="reused",
            cuda_weights="transient",
            cuda_batching="scalar",
            cuda_boundary="ffn-block",
            cuda_transfer="synchronous",
            cuda_moe_fusion="none",
            l1_expert_cache="disabled",
            l2_expert_schedule="blocking",
            speculative_mode=mode,
            speculative_verification=verification,
            speculative_block_size=block_size,
            speculative_script=script,
        )
        raw_json = output_dir / f"{name}.json"
        raw_csv = output_dir / f"{name}.csv"
        write_results(benchmark, raw_json, raw_csv)
        diagnostic = baseline if name == "greedy" else _cuda_diagnostic(
            artifact,
            runner,
            output_dir / f"{name}-diagnostic.json",
            mode=mode,
            verification=verification,
            block_size=block_size,
            script=script,
        )
        if any(diagnostic[field] != baseline[field] for field in parity_fields):
            raise RuntimeError(f"{name} diverged from greedy CUDA execution")
        expected = {
            "speculative_verification_blocks": "expected_blocks",
            "speculative_proposed_draft_tokens": "expected_proposed",
            "speculative_accepted_draft_tokens": "expected_accepted",
            "speculative_committed_tokens": "expected_committed",
            "target_positions_evaluated": "expected_evaluated",
            "target_positions_discarded": "expected_discarded",
        }
        for field, case_field in expected.items():
            if diagnostic[field] != int(case[case_field]):
                raise RuntimeError(f"{name} {field} diverged")
        if verification == "expert-major":
            if diagnostic["batched_expert_ffn_calls"] != diagnostic[
                "expert_major_payload_loads"
            ]:
                raise RuntimeError(f"{name} batch-call accounting diverged")
            if diagnostic["batched_expert_ffn_tokens"] != diagnostic[
                "expert_major_assignments"
            ]:
                raise RuntimeError(f"{name} batch-token accounting diverged")
        elif diagnostic["batched_expert_ffn_calls"] != 0:
            raise RuntimeError(f"{name} unexpectedly used batched expert FFN")

        payload = asdict(benchmark)
        payload.update(
            name=name,
            script=script,
            parity_status="exact",
            raw_json_sha256=_sha256(raw_json),
            raw_csv_sha256=_sha256(raw_csv),
        )
        records.append(payload)
    return {"benchmark": "B-0016-graph", "records": records}


def _run_released_case(
    artifact: Path,
    runner: Path,
    execution: str,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner),
            "--model", str(artifact),
            "--execution", execution,
            "--batch-size", str(batch_size),
            "--warmup", str(warmup),
            "--iterations", str(iterations),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CUDA batch benchmark failed")
    return json.loads(result.stdout)


def _write_record(record: dict, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=record.keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(record)


def _validate_released_pair(scalar: dict, batch: dict, iterations: int) -> None:
    invariant_fields = (
        "artifact_kind",
        "routing_semantics",
        "batch_size",
        "expert_payload_bytes",
        "warmup",
        "iterations",
        "device_to_host_bytes",
        "activation_h2d_bytes",
    )
    if any(scalar[field] != batch[field] for field in invariant_fields):
        raise RuntimeError("released scalar/batch provenance diverged")
    if (
        scalar["artifact_kind"] != "released_dimension_single_expert"
        or scalar["routing_semantics"] is not False
        or int(scalar["expert_payload_bytes"]) != 17_547_264
    ):
        raise RuntimeError("invalid released-dimension benchmark identity")
    batch_size = int(batch["batch_size"])
    payload = int(batch["expert_payload_bytes"])
    if int(scalar["weight_h2d_bytes"]) != payload * batch_size * iterations:
        raise RuntimeError("scalar weight traffic accounting diverged")
    if int(batch["weight_h2d_bytes"]) != payload * iterations:
        raise RuntimeError("batch weight traffic accounting diverged")
    if int(scalar["batched_expert_ffn_calls"]) != 0:
        raise RuntimeError("scalar case reported batched calls")
    if (
        int(batch["batched_expert_ffn_calls"]) != iterations
        or int(batch["batched_expert_ffn_tokens"]) != batch_size * iterations
    ):
        raise RuntimeError("batch counters are inconsistent")
    error = float(batch["maximum_absolute_error"])
    if not math.isfinite(error) or error > 1.0e-3:
        raise RuntimeError("released batch numerical parity failed")


def run_cuda_expert_major_ablation(
    artifact: Path,
    runner: Path,
    released_artifact: Path,
    released_runner: Path,
    *,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    artifact = Path(artifact).resolve()
    released_artifact = Path(released_artifact).resolve()
    output_dir = Path(output_dir)
    graph = _run_graph_ablation(
        artifact,
        Path(runner).resolve(),
        warmup=warmup,
        iterations=iterations,
        output_dir=output_dir / "graph",
    )
    released_records: list[dict] = []
    for name, execution, batch_size in released_batch_matrix():
        record = _run_released_case(
            released_artifact,
            Path(released_runner).resolve(),
            execution,
            batch_size,
            warmup,
            iterations,
        )
        raw_json = output_dir / f"{name}.json"
        raw_csv = output_dir / f"{name}.csv"
        _write_record(record, raw_json, raw_csv)
        record = {
            **record,
            "name": name,
            "parity_status": "exact",
            "raw_json_sha256": _sha256(raw_json),
            "raw_csv_sha256": _sha256(raw_csv),
        }
        released_records.append(record)
    for batch_size in (2, 4):
        scalar = next(
            record for record in released_records
            if record["execution"] == "scalar" and record["batch_size"] == batch_size
        )
        batch = next(
            record for record in released_records
            if record["execution"] == "batch" and record["batch_size"] == batch_size
        )
        _validate_released_pair(scalar, batch, iterations)

    aggregate_source = {
        "graph": graph["records"],
        "released": released_records,
    }
    aggregate = json.dumps(
        aggregate_source, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0016",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "released_artifact_sha256": _sha256(released_artifact),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "graph": graph,
        "released": {"benchmark": "B-0016-released", "records": released_records},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        {"evidence_kind": kind, **record}
        for kind, records in (
            ("graph", graph["records"]),
            ("released", released_records),
        )
        for record in records
    ]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--released-artifact", type=Path, required=True)
    parser.add_argument("--released-runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_cuda_expert_major_ablation(
        args.artifact,
        args.runner,
        args.released_artifact,
        args.released_runner,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
