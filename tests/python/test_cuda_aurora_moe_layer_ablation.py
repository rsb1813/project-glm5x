# B-0022 resident MoE-layer ablation 행렬과 실행 증거를 검증합니다.
import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from tools.ablate_cuda_aurora_moe_layer import CASES, PAIRS, run_ablation


def test_cuda_aurora_moe_layer_matrix_is_canonical() -> None:
    assert tuple(case[0] for case in CASES) == (
        "natural-greedy",
        "grid-fixed-2-token",
        "layer-fixed-2-token",
        "grid-adaptive-token",
        "layer-adaptive-token",
        "grid-fixed-2-expert",
        "layer-fixed-2-expert",
        "grid-adaptive-expert",
        "layer-adaptive-expert",
    )
    assert tuple(pair[0] for pair in PAIRS) == (
        "fixed-2-token",
        "adaptive-token",
        "fixed-2-expert",
        "adaptive-expert",
    )


def test_live_cuda_aurora_moe_layer_ablation(tmp_path: Path) -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA ablation requires K3X_TEST_CUDA=1")
    output = tmp_path / "b0022"
    summary = run_ablation(
        cpp_binary("k3x_run"), output_dir=output, warmups=0, samples=1
    )
    assert summary["benchmark"] == "B-0022"
    assert len(summary["records"]) == 9
    records = {record["name"]: record for record in summary["records"]}
    for _, grid_name, layer_name in PAIRS:
        grid = records[grid_name]
        layer = records[layer_name]
        assert grid["draft_cuda_boundary"] == "ffn-block"
        assert layer["draft_cuda_boundary"] == "moe-layer"
        assert grid["draft_resident_moe_layer_calls"] == 0
        assert layer["draft_resident_moe_layer_calls"] > 0
        assert layer["draft_resident_moe_layer_fallbacks"] == 0
        assert layer["draft_resident_moe_layer_kernel_launches"] == (
            layer["draft_resident_moe_layer_calls"] * 13
        )
        assert layer["draft_stream_synchronization_count"] < grid[
            "draft_stream_synchronization_count"
        ]
        assert layer["draft_activation_h2d_bytes"] < grid[
            "draft_activation_h2d_bytes"
        ]
        assert layer["draft_device_to_host_bytes"] < grid[
            "draft_device_to_host_bytes"
        ]
        assert layer["draft_host_to_device_bytes"] < grid[
            "draft_host_to_device_bytes"
        ]
        assert layer["paired_weight_h2d_delta_bytes"] == (
            layer["draft_resident_weight_bytes"]
            - grid["draft_resident_weight_bytes"]
        )
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


def test_committed_b0022_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0022-cuda-aurora-moe-layer-wsl"
    if not output.exists():
        pytest.skip("B-0022 evidence is committed in the measurement task")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark"] == "B-0022"
    assert summary["warmups"] == 3
    assert summary["samples"] == 20
    assert len(summary["records"]) == 9
    assert len(tuple(output.glob("*.json"))) == 10
    assert len(tuple(output.glob("*.csv"))) == 10
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    for record in summary["records"]:
        assert record["token_parity"] is True
        assert record["committed_route_parity"] is True
        assert record["final_state_max_abs_error"] <= 1.0e-6
        assert record["decode_delta_percent"] == (
            record["decode_tokens_per_second"]
            / summary["records"][0]["decode_tokens_per_second"] - 1.0
        ) * 100.0
        for suffix, digest_key in (
            ("json", "raw_json_sha256"),
            ("csv", "raw_csv_sha256"),
        ):
            raw = output / f"{record['name']}.{suffix}"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[
                digest_key
            ]
        assert b"\r\n" not in (output / f"{record['name']}.csv").read_bytes()
    records = {record["name"]: record for record in summary["records"]}
    for _, grid_name, layer_name in PAIRS:
        grid = records[grid_name]
        layer = records[layer_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
            "draft_reader_completed_bytes",
            "draft_routing_decisions",
            "draft_routing_selected_experts",
        ):
            assert grid[field] == layer[field]
        assert grid["draft_cuda_boundary"] == "ffn-block"
        assert layer["draft_cuda_boundary"] == "moe-layer"
        assert grid["draft_resident_moe_layer_calls"] == 0
        assert layer["draft_resident_moe_layer_calls"] > 0
        assert layer["draft_resident_moe_layer_fallbacks"] == 0
        assert layer["draft_resident_moe_layer_kernel_launches"] == (
            layer["draft_resident_moe_layer_calls"] * 13
        )
        sync_reduction = (
            grid["draft_stream_synchronization_count"]
            - layer["draft_stream_synchronization_count"]
        )
        assert sync_reduction == layer["draft_resident_moe_layer_calls"] * 3
        assert layer["paired_sync_reduction"] == sync_reduction
        assert layer["draft_activation_h2d_bytes"] < grid[
            "draft_activation_h2d_bytes"
        ]
        assert layer["draft_host_to_device_bytes"] < grid[
            "draft_host_to_device_bytes"
        ]
        assert layer["draft_device_to_host_bytes"] < grid[
            "draft_device_to_host_bytes"
        ]
        weight_delta = (
            layer["draft_weight_h2d_bytes"] - grid["draft_weight_h2d_bytes"]
        )
        resident_delta = (
            layer["draft_resident_weight_bytes"]
            - grid["draft_resident_weight_bytes"]
        )
        assert weight_delta > 0
        assert weight_delta == resident_delta
        assert layer["paired_weight_h2d_delta_bytes"] == weight_delta
        assert layer["paired_decode_delta_percent"] == (
            layer["decode_tokens_per_second"]
            / grid["decode_tokens_per_second"] - 1.0
        ) * 100.0
    summary_csv = output / "summary.csv"
    assert b"\r\n" not in summary_csv.read_bytes()
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
