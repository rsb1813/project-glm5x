# CUDA admission validation 18행 ablation의 행렬과 증거 계약을 검증합니다.
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.ablate_cuda_admission_validation import CASES, run_ablation


def _payload(boundary: str, experts: int, validation: str, profiler: bool) -> dict:
    layer = boundary == "moe-layer"
    admission = validation == "admission"
    iterations = 2
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
        "warmup": 1,
        "iterations": iterations,
        "validation": validation,
        "profiler": profiler,
        "maximum_absolute_error": 0.0,
        "latency_nanoseconds_median": 1000,
        "kernel_nanoseconds": 500 if profiler else None,
        "weight_h2d_bytes": 0,
        "cold_weight_h2d_bytes": 1,
        "activation_h2d_bytes": iterations * (
            86_016 if not layer else 28_672 + experts * 52
        ),
        "device_to_host_bytes": iterations * (
            71_680 + experts * 14_336 if not layer else 28_672
        ),
        "stream_synchronization_count": iterations * (1 if layer else 4),
        "resident_weight_bytes": 100,
        "peak_resident_weight_bytes": 100,
        "weight_cache_bypasses": 0,
        "resident_grid_calls": iterations,
        "resident_grid_kernel_launches": iterations * 4,
        "resident_grid_fallbacks": 0,
        "resident_moe_layer_calls": iterations if layer else 0,
        "resident_moe_layer_experts": experts * iterations if layer else 0,
        "resident_moe_layer_kernel_launches": iterations * 13 if layer else 0,
        "resident_moe_layer_fallbacks": 0,
        "resident_moe_layer_contribution_h2d_bytes": (
            experts * 4 * iterations if layer else 0
        ),
        "cold_immutable_validation_scans": 6 if layer else 0,
        "cold_immutable_validation_bytes": 469_776_384 if layer else 0,
        "immutable_validation_scans": iterations * 6 if layer and not admission else 0,
        "immutable_validation_hits": iterations * 6 if layer and admission else 0,
        "immutable_validation_bytes": iterations * 469_776_384 if layer and not admission else 0,
        "immutable_validation_nanoseconds": 10 if layer and not admission else 0,
    }


def test_admission_validation_matrix_is_canonical() -> None:
    assert len(CASES) == 18
    assert len({case[0] for case in CASES}) == 18
    assert sum(case[1] == "ffn-block" for case in CASES) == 6
    assert sum(case[3] == "admission" for case in CASES) == 6


def test_admission_validation_ablation_cross_checks_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "model.k3x"
    runner = tmp_path / "bench"
    artifact.write_bytes(b"model")
    runner.write_bytes(b"runner")

    def fake_run(*args: object, **kwargs: object) -> dict:
        return _payload(
            str(kwargs["boundary"]), int(kwargs["experts"]),
            str(kwargs["validation"]), bool(kwargs["profiler"]),
        )

    monkeypatch.setattr(
        "tools.ablate_cuda_admission_validation._run_case", fake_run
    )
    output = tmp_path / "results"
    summary = run_ablation(
        artifact, runner, output_dir=output, warmup=1, iterations=2
    )
    assert summary["benchmark"] == "B-0024"
    assert len(summary["records"]) == 18
    assert (output / "summary.json").is_file()
    assert (output / "summary.csv").is_file()
    written = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert written["aggregate_sha256"] == summary["aggregate_sha256"]
    assert written["summary_csv_sha256"] == summary["summary_csv_sha256"]
    for record in summary["records"]:
        assert len(record["raw_json_sha256"]) == 64
    assert len(list(output.glob("*.json"))) == 19
    assert [path.name for path in output.glob("*.csv")] == ["summary.csv"]


def test_committed_b0024_evidence_is_self_consistent() -> None:
    root = Path(__file__).parents[2]
    evidence = root / "results" / "b0024-cuda-admission-validation-wsl"
    summary = json.loads(
        (evidence / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["benchmark"] == "B-0024"
    assert summary["warmup"] == 3
    assert summary["iterations"] == 20
    assert len(list(evidence.glob("*.json"))) == 19
    assert [path.name for path in evidence.glob("*.csv")] == ["summary.csv"]
    records = summary["records"]
    assert [record["name"] for record in records] == [case[0] for case in CASES]
    for record, case in zip(records, CASES, strict=True):
        name, boundary, experts, validation, profiler = case
        raw_path = evidence / f"{name}.json"
        digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        assert record["raw_json_sha256"] == digest
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        assert all(record[key] == value for key, value in raw.items())
        layer = boundary == "moe-layer"
        assert raw["immutable_validation_scans"] == (
            120 if layer and validation == "per-call" else 0
        )
        assert raw["immutable_validation_hits"] == (
            120 if layer and validation == "admission" else 0
        )
        assert raw["immutable_validation_bytes"] == (
            20 * 469_776_384
            if layer and validation == "per-call" else 0
        )
        if profiler:
            assert isinstance(raw["kernel_nanoseconds"], int)
            assert raw["kernel_nanoseconds"] > 0
        else:
            assert raw["kernel_nanoseconds"] is None
        assert raw["maximum_absolute_error"] <= 1.0e-5
        assert raw["weight_h2d_bytes"] == 0
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    assert summary["aggregate_sha256"] == hashlib.sha256(canonical).hexdigest()
    csv_bytes = (evidence / "summary.csv").read_bytes()
    assert b"\r\n" not in csv_bytes
    assert summary["summary_csv_sha256"] == hashlib.sha256(csv_bytes).hexdigest()
    with (evidence / "summary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["name"] for row in csv_rows] == [case[0] for case in CASES]
    for experts in (1, 4, 16):
        for profiler_name in ("off", "on"):
            by_name = {record["name"]: record for record in records}
            reference = by_name[
                f"layer-{experts}-per-call-profiler-{profiler_name}"
            ]
            admission = by_name[
                f"layer-{experts}-admission-profiler-{profiler_name}"
            ]
            expected = (
                admission["latency_nanoseconds_median"]
                / reference["latency_nanoseconds_median"] - 1.0
            ) * 100.0
            assert admission["paired_admission_latency_delta_percent"] == expected
