# B-0021 resident expert-grid ablation의 행렬, 실행 증거, digest를 검증합니다.
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from tools.ablate_cuda_aurora_grid import CASES, PAIRS, run_ablation


def test_cuda_aurora_grid_matrix_is_canonical() -> None:
    assert tuple(case[0] for case in CASES) == (
        "natural-greedy",
        "grouped-fixed-2-token",
        "grid-fixed-2-token",
        "grouped-adaptive-token",
        "grid-adaptive-token",
        "grouped-fixed-2-expert",
        "grid-fixed-2-expert",
        "grouped-adaptive-expert",
        "grid-adaptive-expert",
    )
    assert len(PAIRS) == 4


def test_live_cuda_grid_benchmark_contract() -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA grid benchmark requires K3X_TEST_CUDA=1")
    for experts, tokens in ((1, 1), (1, 4), (2, 2), (4, 4)):
        result = subprocess.run(
            [
                str(cpp_binary("k3x_cuda_expert_grid_bench")),
                "--experts", str(experts), "--tokens", str(tokens),
                "--warmup", "0", "--iterations", "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        assert payload["maximum_absolute_error"] <= 1.0e-6
        assert payload["resident_grid_calls"] == 1
        assert payload["resident_grid_kernel_launches"] == 4
        assert payload["resident_grid_fallbacks"] == 0


def test_live_cuda_aurora_grid_ablation(tmp_path: Path) -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA ablation requires K3X_TEST_CUDA=1")
    output = tmp_path / "b0021"
    summary = run_ablation(
        cpp_binary("k3x_run"), output_dir=output, warmups=0, samples=1
    )
    assert summary["benchmark"] == "B-0021"
    assert len(summary["records"]) == 9
    records = {record["name"]: record for record in summary["records"]}
    for _, grouped_name, grid_name in PAIRS:
        grouped = records[grouped_name]
        grid = records[grid_name]
        assert grouped["draft_cuda_batching"] == "grouped"
        assert grid["draft_cuda_batching"] == "resident-grid"
        assert grid["draft_resident_grid_fallbacks"] == 0
        assert grid["draft_moe_kernel_launches"] < grouped[
            "draft_moe_kernel_launches"
        ]
    for record in summary["records"]:
        for suffix, digest_key in (
            ("json", "raw_json_sha256"),
            ("csv", "raw_csv_sha256"),
        ):
            raw = output / f"{record['name']}.{suffix}"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[
                digest_key
            ]
        assert b"\r\n" not in (output / f"{record['name']}.csv").read_bytes()
    summary_csv = output / "summary.csv"
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]


def test_committed_b0021_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0021-cuda-aurora-grid-wsl"
    if not output.exists():
        pytest.skip("B-0021 evidence is committed in the measurement task")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark"] == "B-0021"
    assert summary["warmups"] == 3
    assert summary["samples"] == 20
    assert len(summary["records"]) == 9
    assert len(tuple(output.glob("*.json"))) == 10
    assert len(tuple(output.glob("*.csv"))) == 10
    for record in summary["records"]:
        for suffix, digest_key in (
            ("json", "raw_json_sha256"),
            ("csv", "raw_csv_sha256"),
        ):
            raw = output / f"{record['name']}.{suffix}"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[
                digest_key
            ]
        assert b"\r\n" not in (output / f"{record['name']}.csv").read_bytes()
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    summary_csv = output / "summary.csv"
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
    assert b"\r\n" not in summary_csv.read_bytes()
    baseline_decode = summary["records"][0]["decode_tokens_per_second"]
    records = {record["name"]: record for record in summary["records"]}
    for record in summary["records"]:
        assert record["token_parity"] is True
        assert record["committed_route_parity"] is True
        assert record["final_state_max_abs_error"] <= 1.0e-6
        assert record["decode_delta_percent"] == (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
    for _, grouped_name, grid_name in PAIRS:
        grouped = records[grouped_name]
        grid = records[grid_name]
        assert grouped["draft_resident_grid_calls"] == 0
        assert grid["draft_resident_grid_calls"] > 0
        assert grid["draft_resident_grid_kernel_launches"] == (
            grid["draft_resident_grid_calls"] * 4
        )
        assert grid["draft_resident_grid_fallbacks"] == 0
        assert grid["draft_moe_kernel_launches"] < grouped[
            "draft_moe_kernel_launches"
        ]
        assert grid["paired_decode_delta_percent"] == (
            grid["decode_tokens_per_second"]
            / grouped["decode_tokens_per_second"] - 1.0
        ) * 100.0
        assert grid["paired_moe_launch_reduction_percent"] == (
            1.0
            - grid["draft_moe_kernel_launches"]
            / grouped["draft_moe_kernel_launches"]
        ) * 100.0
