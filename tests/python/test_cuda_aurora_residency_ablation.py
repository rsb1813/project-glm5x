# B-0020 transient와 resident CUDA AURORA draft의 동등성과 증거를 검증합니다.
import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from tools.ablate_cuda_aurora_residency import CASES, PAIRS, run_ablation


def test_cuda_aurora_residency_matrix_is_canonical() -> None:
    assert tuple(item[0] for item in CASES) == (
        "natural-greedy",
        "transient-fixed-2-token",
        "resident-fixed-2-token",
        "transient-adaptive-token",
        "resident-adaptive-token",
        "transient-fixed-2-expert",
        "resident-fixed-2-expert",
        "transient-adaptive-expert",
        "resident-adaptive-expert",
    )
    assert tuple(item[0] for item in PAIRS) == (
        "fixed-2-token",
        "adaptive-token",
        "fixed-2-expert",
        "adaptive-expert",
    )


def test_cuda_aurora_residency_preserves_pairs_and_raw_evidence(
    tmp_path: Path,
) -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA ablation requires K3X_TEST_CUDA=1")
    output = tmp_path / "b0020"
    summary = run_ablation(
        cpp_binary("k3x_run"), output_dir=output, warmups=0, samples=1
    )
    assert summary["benchmark"] == "B-0020"
    assert len(summary["records"]) == 9
    natural = summary["records"][0]
    records = {record["name"]: record for record in summary["records"]}
    assert natural["aurora_draft_backend"] == "none"
    for record in summary["records"]:
        assert record["token_ids"] == natural["token_ids"]
        assert record["token_parity"] is True
        assert record["final_state_max_abs_error"] <= 1.0e-6
        assert record["committed_route_parity"] is True
        for suffix, digest_key in (
            ("json", "raw_json_sha256"),
            ("csv", "raw_csv_sha256"),
        ):
            raw = output / f"{record['name']}.{suffix}"
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == record[
                digest_key
            ]
        assert b"\r\n" not in (output / f"{record['name']}.csv").read_bytes()
    for _, transient_name, resident_name in PAIRS:
        transient = records[transient_name]
        resident = records[resident_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            assert transient[field] == resident[field]
        assert transient["draft_cuda_weights"] == "transient"
        assert transient["draft_cuda_resident_bytes"] == 0
        assert transient["draft_resident_weight_bytes"] == 0
        assert transient["draft_peak_resident_weight_bytes"] == 0
        assert resident["draft_cuda_weights"] == "resident"
        assert resident["draft_cuda_resident_bytes"] == 8 * 1024 * 1024
        assert 0 < resident["draft_resident_weight_bytes"] <= 8 * 1024 * 1024
        assert (
            resident["draft_resident_weight_bytes"]
            <= resident["draft_peak_resident_weight_bytes"]
            <= 8 * 1024 * 1024
        )
        assert resident["draft_weight_cache_hits"] > 0
        assert resident["draft_weight_cache_misses"] > 0
        assert resident["draft_weight_cache_bypasses"] == 0
        assert (
            resident["draft_weight_h2d_bytes"]
            < transient["draft_weight_h2d_bytes"]
        )
        assert resident["paired_decode_delta_percent"] == (
            resident["decode_tokens_per_second"]
            / transient["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        assert resident["draft_weight_h2d_reduction_percent"] == (
            1.0
            - resident["draft_weight_h2d_bytes"]
            / transient["draft_weight_h2d_bytes"]
        ) * 100.0
        assert resident["draft_resident_hit_rate"] == (
            resident["draft_weight_cache_hits"]
            / (
                resident["draft_weight_cache_hits"]
                + resident["draft_weight_cache_misses"]
            )
        )
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    summary_csv = output / "summary.csv"
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
    assert b"\r\n" not in summary_csv.read_bytes()


def test_committed_b0020_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0020-cuda-aurora-residency-wsl"
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["benchmark"] == "B-0020"
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
    summary_csv = output / "summary.csv"
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
    assert b"\r\n" not in summary_csv.read_bytes()
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]

    records = {record["name"]: record for record in summary["records"]}
    baseline_decode = summary["records"][0]["decode_tokens_per_second"]
    for record in summary["records"]:
        assert record["decode_delta_percent"] == (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
    for _, transient_name, resident_name in PAIRS:
        transient = records[transient_name]
        resident = records[resident_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            assert transient[field] == resident[field]
        assert transient["draft_cuda_resident_bytes"] == 0
        assert transient["draft_resident_weight_bytes"] == 0
        assert transient["draft_peak_resident_weight_bytes"] == 0
        assert resident["draft_cuda_resident_bytes"] == 8 * 1024 * 1024
        assert 0 < resident["draft_resident_weight_bytes"] <= 8 * 1024 * 1024
        assert (
            resident["draft_resident_weight_bytes"]
            <= resident["draft_peak_resident_weight_bytes"]
            <= 8 * 1024 * 1024
        )
        assert resident["draft_weight_cache_hits"] > 0
        assert resident["draft_weight_cache_misses"] > 0
        assert resident["draft_weight_cache_bypasses"] == 0
        assert resident["draft_weight_h2d_bytes"] < transient[
            "draft_weight_h2d_bytes"
        ]
        assert resident["paired_decode_delta_percent"] == (
            resident["decode_tokens_per_second"]
            / transient["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        assert resident["draft_weight_h2d_reduction_percent"] == (
            1.0
            - resident["draft_weight_h2d_bytes"]
            / transient["draft_weight_h2d_bytes"]
        ) * 100.0
        assert resident["draft_resident_hit_rate"] == (
            resident["draft_weight_cache_hits"]
            / (
                resident["draft_weight_cache_hits"]
                + resident["draft_weight_cache_misses"]
            )
        )
