# CUDA expert-major ablation의 실행 행렬과 증거 무결성을 검증합니다.
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source
from tools.ablate_cuda_expert_major import (
    released_batch_matrix,
    run_cuda_expert_major_ablation,
)


def _released_record(execution: str, batch_size: int, iterations: int) -> dict:
    batched = execution == "batch"
    payload = 17_547_264
    return {
        "artifact_kind": "released_dimension_single_expert",
        "routing_semantics": False,
        "execution": execution,
        "batch_size": batch_size,
        "expert_payload_bytes": payload,
        "warmup": 3,
        "iterations": iterations,
        "latency_nanoseconds_median": 80 if batched else 100,
        "maximum_absolute_error": 1.0e-5 if batched else 0.0,
        "kernel_nanoseconds": 70 if batched else 90,
        "device_to_host_bytes": batch_size * 3584 * 4 * iterations,
        "weight_h2d_bytes": payload * iterations * (1 if batched else batch_size),
        "activation_h2d_bytes": batch_size * 3584 * 4 * iterations,
        "batched_expert_ffn_calls": iterations if batched else 0,
        "batched_expert_ffn_tokens": batch_size * iterations if batched else 0,
        "peak_vram_bytes": 20_000_000,
    }


def test_released_batch_matrix_has_exact_scalar_batch_pairs() -> None:
    assert released_batch_matrix() == (
        ("scalar-2", "scalar", 2),
        ("batch-2", "batch", 2),
        ("scalar-4", "scalar", 4),
        ("batch-4", "batch", 4),
    )


def test_cuda_expert_major_tool_supports_direct_cli_execution() -> None:
    result = subprocess.run(
        [sys.executable, "tools/ablate_cuda_expert_major.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--released-artifact" in result.stdout


def test_released_cuda_batch_bench_reuses_one_expert_payload(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released-dimension CUDA batch bench requires build-cuda")
    runner = cpp_binary("k3x_cuda_expert_batch_bench")
    assert runner.is_file(), "build k3x_cuda_expert_batch_bench before running test"
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)

    records: dict[str, dict] = {}
    for execution in ("scalar", "batch"):
        result = subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--execution", execution,
                "--batch-size", "2",
                "--warmup", "0",
                "--iterations", "1",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        records[execution] = json.loads(result.stdout)

    scalar, batch = records["scalar"], records["batch"]
    assert scalar["artifact_kind"] == "released_dimension_single_expert"
    assert scalar["routing_semantics"] is False
    assert scalar["expert_payload_bytes"] == 17_547_264
    assert scalar["weight_h2d_bytes"] == 2 * scalar["expert_payload_bytes"]
    assert batch["weight_h2d_bytes"] == batch["expert_payload_bytes"]
    assert scalar["activation_h2d_bytes"] == batch["activation_h2d_bytes"]
    assert scalar["device_to_host_bytes"] == batch["device_to_host_bytes"]
    assert scalar["batched_expert_ffn_calls"] == 0
    assert batch["batched_expert_ffn_calls"] == 1
    assert batch["batched_expert_ffn_tokens"] == 2
    assert batch["maximum_absolute_error"] <= 1.0e-3


def test_released_cuda_batch_bench_rejects_invalid_batch_size() -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released-dimension CUDA batch bench requires build-cuda")
    result = subprocess.run(
        [
            str(cpp_binary("k3x_cuda_expert_batch_bench")),
            "--batch-size", "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "batch size must be between 1 and 4"


def test_cuda_expert_major_ablation_cross_checks_released_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    released_artifact = tmp_path / "bounded.k3x"
    artifact.write_bytes(b"synthetic")
    released_artifact.write_bytes(b"released")

    def fake_graph(*args: object, **kwargs: object) -> dict:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for name in (
            "greedy",
            "token-major-perfect-2",
            "expert-major-perfect-2",
            "token-major-mixed-2",
            "expert-major-mixed-2",
        ):
            record = {
                "name": name,
                "parity_status": "exact",
                "token_ids": [43, 32, 28, 49, 9, 28],
                "routed_experts": [0, 1],
                "evaluated_routed_experts": [0, 1],
                "weight_h2d_bytes": 10 if name.startswith("expert") else 20,
                "batched_expert_ffn_calls": 2 if name.startswith("expert") else 0,
                "batched_expert_ffn_tokens": 4 if name.startswith("expert") else 0,
            }
            raw_json = output_dir / f"{name}.json"
            raw_csv = output_dir / f"{name}.csv"
            raw_json.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with raw_csv.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=record.keys(), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerow(record)
            record["raw_json_sha256"] = hashlib.sha256(raw_json.read_bytes()).hexdigest()
            record["raw_csv_sha256"] = hashlib.sha256(raw_csv.read_bytes()).hexdigest()
            records.append(record)
        return {"benchmark": "B-0016-graph", "records": records}

    monkeypatch.setattr("tools.ablate_cuda_expert_major._run_graph_ablation", fake_graph)
    monkeypatch.setattr(
        "tools.ablate_cuda_expert_major._run_released_case",
        lambda artifact, runner, execution, batch_size, warmup, iterations: (
            _released_record(execution, batch_size, iterations)
        ),
    )
    summary = run_cuda_expert_major_ablation(
        artifact,
        tmp_path / "k3x_run",
        released_artifact,
        tmp_path / "k3x_cuda_expert_batch_bench",
        warmup=3,
        iterations=20,
        output_dir=tmp_path / "b0016",
    )

    assert summary["benchmark"] == "B-0016"
    assert len(summary["graph"]["records"]) == 5
    assert len(summary["released"]["records"]) == 4
    for batch_size in (2, 4):
        scalar = next(
            record for record in summary["released"]["records"]
            if record["execution"] == "scalar" and record["batch_size"] == batch_size
        )
        batch = next(
            record for record in summary["released"]["records"]
            if record["execution"] == "batch" and record["batch_size"] == batch_size
        )
        assert batch["weight_h2d_bytes"] * batch_size == scalar["weight_h2d_bytes"]
        assert batch["batched_expert_ffn_calls"] == 20
        assert batch["batched_expert_ffn_tokens"] == batch_size * 20

    aggregate = json.dumps(
        {
            "graph": summary["graph"]["records"],
            "released": summary["released"]["records"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert summary["aggregate_sha256"] == hashlib.sha256(aggregate).hexdigest()
    assert (tmp_path / "b0016" / "summary.json").is_file()
    assert (tmp_path / "b0016" / "summary.csv").is_file()
    for csv_path in (tmp_path / "b0016").rglob("*.csv"):
        assert b"\r\n" not in csv_path.read_bytes()
