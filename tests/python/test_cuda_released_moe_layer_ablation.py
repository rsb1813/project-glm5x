# B-0023 released-dimension MoE-layer ablation 행렬과 물리 gate를 검증합니다.
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source
from tools.ablate_cuda_released_moe_layer import CASES, PAIRS, run_ablation


def _record(boundary: str, experts: int) -> dict:
    layer = boundary == "moe-layer"
    cold = 1_000_000 + experts * 100_000 + (14_336 if layer else 0)
    return {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": boundary,
        "experts": experts,
        "hidden_width": 7168,
        "latent_width": 3584,
        "expert_intermediate_width": 3072,
        "expert_payload_bytes": 17_547_264,
        "resident_capacity_bytes": 1 << 30,
        "warmup": 3,
        "iterations": 20,
        "maximum_absolute_error": 1.0e-6,
        "latency_nanoseconds_median": 90 if layer else 100,
        "kernel_nanoseconds": 1_000,
        "activation_h2d_bytes": 18_000 if layer else 20_000,
        "device_to_host_bytes": 14_000 if layer else 16_000,
        "weight_h2d_bytes": 0,
        "stream_synchronization_count": 20 if layer else 80,
        "cold_weight_h2d_bytes": cold,
        "resident_weight_bytes": cold,
        "peak_resident_weight_bytes": cold,
        "oracle_peak_vram_bytes": cold + 40_000,
        "peak_vram_bytes": cold + 50_000,
        "weight_cache_bypasses": 0,
        "resident_grid_calls": 20,
        "resident_grid_kernel_launches": 80,
        "resident_grid_fallbacks": 0,
        "resident_moe_layer_calls": 20 if layer else 0,
        "resident_moe_layer_experts": experts * 20 if layer else 0,
        "resident_moe_layer_kernel_launches": 260 if layer else 0,
        "resident_moe_layer_fallbacks": 0,
        "resident_moe_layer_contribution_h2d_bytes": (
            experts * 4 * 20 if layer else 0
        ),
    }


def _fake_runner(
    mutator: Callable[[dict, str, int], None] | None = None,
) -> Callable[..., dict]:
    def run(
        artifact: Path,
        runner: Path,
        boundary: str,
        experts: int,
        warmup: int,
        iterations: int,
    ) -> dict:
        del artifact, runner
        record = _record(boundary, experts)
        record["warmup"] = warmup
        record["iterations"] = iterations
        if mutator is not None:
            mutator(record, boundary, experts)
        return record

    return run


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "bounded.k3x"
    runner = tmp_path / "k3x_cuda_moe_layer_bench"
    artifact.write_bytes(b"artifact")
    runner.write_bytes(b"runner")
    return artifact, runner, tmp_path / "results"


def test_released_moe_layer_matrix_is_canonical() -> None:
    assert CASES == (
        ("split-1", "ffn-block", 1),
        ("layer-1", "moe-layer", 1),
        ("split-4", "ffn-block", 4),
        ("layer-4", "moe-layer", 4),
        ("split-16", "ffn-block", 16),
        ("layer-16", "moe-layer", 16),
    )
    assert PAIRS == (
        ("experts-1", "split-1", "layer-1"),
        ("experts-4", "split-4", "layer-4"),
        ("experts-16", "split-16", "layer-16"),
    )


def test_released_moe_layer_ablation_writes_cross_checked_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, runner, output = _inputs(tmp_path)
    monkeypatch.setattr(
        "tools.ablate_cuda_released_moe_layer._run_case", _fake_runner()
    )
    summary = run_ablation(
        artifact,
        runner,
        output_dir=output,
        warmup=3,
        iterations=20,
    )
    assert summary["benchmark"] == "B-0023"
    assert len(summary["records"]) == 6
    assert summary["artifact_sha256"] == hashlib.sha256(b"artifact").hexdigest()
    assert summary["runner_sha256"] == hashlib.sha256(b"runner").hexdigest()
    records = {record["name"]: record for record in summary["records"]}
    for _, split_name, layer_name in PAIRS:
        split = records[split_name]
        layer = records[layer_name]
        assert split["paired_latency_delta_percent"] == 0.0
        assert layer["paired_latency_delta_percent"] == pytest.approx(-10.0)
        assert layer["paired_activation_h2d_reduction_bytes"] == 2_000
        assert layer["paired_d2h_reduction_bytes"] == 2_000
        assert layer["paired_cold_weight_delta_bytes"] == 14_336
        assert layer["paired_resident_weight_delta_bytes"] == 14_336
    for record in summary["records"]:
        raw = output / f"{record['name']}.json"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[
            "raw_json_sha256"
        ]
    summary_csv = output / "summary.csv"
    assert b"\r\n" not in summary_csv.read_bytes()
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda r, b, e: r.update(maximum_absolute_error=1.0e-4)
            if b == "moe-layer" and e == 1
            else None,
            "numerical",
        ),
        (
            lambda r, b, e: r.update(resident_grid_fallbacks=1)
            if b == "moe-layer" and e == 1
            else None,
            "fallback",
        ),
        (
            lambda r, b, e: r.update(weight_h2d_bytes=4)
            if b == "moe-layer" and e == 1
            else None,
            "warm weight",
        ),
        (
            lambda r, b, e: r.update(oracle_peak_vram_bytes=0)
            if b == "moe-layer" and e == 1
            else None,
            "oracle residency",
        ),
        (
            lambda r, b, e: r.update(stream_synchronization_count=21)
            if b == "moe-layer" and e == 1
            else None,
            "synchronization",
        ),
        (
            lambda r, b, e: r.update(cold_weight_h2d_bytes=r["cold_weight_h2d_bytes"] - 1)
            if b == "moe-layer" and e == 1
            else None,
            "norm residency",
        ),
        (
            lambda r, b, e: r.update(activation_h2d_bytes=20_000)
            if b == "moe-layer" and e == 1
            else None,
            "traffic",
        ),
        (
            lambda r, b, e: r.update(device_to_host_bytes=16_000)
            if b == "moe-layer" and e == 1
            else None,
            "traffic",
        ),
    ],
)
def test_released_moe_layer_ablation_rejects_invalid_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Callable[[dict, str, int], None],
    message: str,
) -> None:
    artifact, runner, output = _inputs(tmp_path)
    monkeypatch.setattr(
        "tools.ablate_cuda_released_moe_layer._run_case",
        _fake_runner(mutator),
    )
    with pytest.raises(RuntimeError, match=message):
        run_ablation(
            artifact,
            runner,
            output_dir=output,
            warmup=3,
            iterations=20,
        )


def test_live_cuda_released_moe_layer_ablation(tmp_path: Path) -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA ablation requires K3X_TEST_CUDA=1")
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    summary = run_ablation(
        artifact,
        cpp_binary("k3x_cuda_moe_layer_bench"),
        output_dir=tmp_path / "b0023",
        warmup=0,
        iterations=1,
    )
    assert len(summary["records"]) == 6


def test_committed_b0023_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0023-cuda-released-moe-layer-wsl"
    if not output.exists():
        pytest.skip("B-0023 evidence is committed in the measurement task")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["benchmark"] == "B-0023"
    assert summary["scope"] == "released-dimension-repeated-view"
    assert summary["evidence"] == "measured"
    assert summary["warmup"] == 3
    assert summary["iterations"] == 20
    assert len(summary["records"]) == 6
    assert len(tuple(output.glob("*.json"))) == 7
    assert len(tuple(output.glob("*.csv"))) == 1

    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary[
        "aggregate_sha256"
    ]
    records = {record["name"]: record for record in summary["records"]}
    assert tuple(record["name"] for record in summary["records"]) == tuple(
        case[0] for case in CASES
    )
    for name, boundary, experts in CASES:
        record = records[name]
        raw_path = output / f"{name}.json"
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == record[
            "raw_json_sha256"
        ]
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        for field, value in raw.items():
            assert record[field] == value
        assert record["boundary"] == boundary
        assert record["experts"] == experts
        assert record["routing_semantics"] is False
        assert record["maximum_absolute_error"] <= 1.0e-5
        assert record["weight_h2d_bytes"] == 0
        assert record["weight_cache_bypasses"] == 0
        assert record["resident_grid_fallbacks"] == 0
        assert record["resident_moe_layer_fallbacks"] == 0

    for _, split_name, layer_name in PAIRS:
        split = records[split_name]
        layer = records[layer_name]
        assert split["stream_synchronization_count"] == 80
        assert layer["stream_synchronization_count"] == 20
        assert layer["paired_sync_reduction"] == 60
        assert layer["resident_moe_layer_calls"] == 20
        assert layer["resident_moe_layer_kernel_launches"] == 260
        assert layer["paired_cold_weight_delta_bytes"] == 14_336
        assert layer["paired_resident_weight_delta_bytes"] == 14_336
        assert layer["activation_h2d_bytes"] < split["activation_h2d_bytes"]
        assert layer["device_to_host_bytes"] < split["device_to_host_bytes"]
        assert layer["paired_latency_delta_percent"] == (
            layer["latency_nanoseconds_median"]
            / split["latency_nanoseconds_median"]
            - 1.0
        ) * 100.0

    forbidden_metric_fragments = ("tokens_per_second", "tok/s", "ttft")
    assert not any(
        fragment in field.lower()
        for record in summary["records"]
        for field in record
        for fragment in forbidden_metric_fragments
    )
    summary_csv = output / "summary.csv"
    assert b"\r\n" not in summary_csv.read_bytes()
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
