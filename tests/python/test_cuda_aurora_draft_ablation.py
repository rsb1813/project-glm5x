# B-0019 CPU와 CUDA persistent AURORA draft 배치의 동등성과 증거를 검증합니다.
import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from tools.ablate_cuda_aurora_draft import CASES, PAIRS, run_ablation


def test_cuda_aurora_draft_matrix_is_canonical() -> None:
    assert tuple(item[0] for item in CASES) == (
        "natural-greedy",
        "cpu-fixed-2-token",
        "cuda-fixed-2-token",
        "cpu-adaptive-token",
        "cuda-adaptive-token",
        "cpu-fixed-2-expert",
        "cuda-fixed-2-expert",
        "cpu-adaptive-expert",
        "cuda-adaptive-expert",
    )


def test_cuda_aurora_draft_ablation_preserves_pairs_and_raw_evidence(
    tmp_path: Path,
) -> None:
    if os.environ.get("K3X_TEST_CUDA") != "1":
        pytest.skip("live CUDA ablation requires K3X_TEST_CUDA=1")
    output = tmp_path / "b0019"
    summary = run_ablation(
        cpp_binary("k3x_run"), output_dir=output, warmups=0, samples=1
    )
    assert summary["benchmark"] == "B-0019"
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
    for _, cpu_name, cuda_name in PAIRS:
        cpu = records[cpu_name]
        cuda = records[cuda_name]
        for field in (
            "token_ids",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            assert cpu[field] == cuda[field]
        assert cpu["aurora_draft_backend"] == "cpu"
        assert cpu["draft_kernel_nanoseconds"] == 0
        assert cpu["draft_host_to_device_bytes"] == 0
        assert cpu["draft_peak_vram_bytes"] == 0
        assert cuda["aurora_draft_backend"] == "cuda-custom"
        assert cuda["draft_kernel_nanoseconds"] > 0
        assert cuda["draft_host_to_device_bytes"] > 0
        assert cuda["draft_peak_vram_bytes"] > 0
        assert cuda["kernel_nanoseconds"] == 0
        assert cuda["host_to_device_bytes"] == 0
        assert cuda["peak_vram_bytes"] == 0
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    summary_csv = output / "summary.csv"
    assert hashlib.sha256(summary_csv.read_bytes()).hexdigest() == summary[
        "summary_csv_sha256"
    ]
    assert b"\r\n" not in summary_csv.read_bytes()


def test_committed_b0019_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0019-cuda-aurora-draft-wsl"
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["benchmark"] == "B-0019"
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
    for _, cpu_name, cuda_name in PAIRS:
        cpu = records[cpu_name]
        cuda = records[cuda_name]
        assert cuda["paired_decode_delta_percent"] == (
            cuda["decode_tokens_per_second"] / cpu["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        assert cuda["paired_draft_kernel_nanoseconds"] == cuda[
            "draft_kernel_nanoseconds"
        ]
        assert cuda["paired_draft_h2d_bytes"] == cuda[
            "draft_host_to_device_bytes"
        ]
        assert cuda["paired_peak_draft_vram_bytes"] == cuda[
            "draft_peak_vram_bytes"
        ]
