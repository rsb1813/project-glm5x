# B-0017 AURORA replay 행렬의 exact parity와 증거 digest를 검증합니다.
import hashlib
import json
from pathlib import Path

from conftest import cpp_binary
from tools.ablate_aurora_replay import aurora_matrix, run_ablation


def test_aurora_matrix_covers_fixed_adaptive_and_expert_major() -> None:
    assert [case["name"] for case in aurora_matrix()] == [
        "natural-greedy",
        "aurora-k4-fixed-1",
        "aurora-k4-fixed-2",
        "aurora-k4-fixed-4",
        "aurora-k4-adaptive-token",
        "aurora-k4-fixed-2-expert",
        "aurora-k4-adaptive-expert",
    ]


def test_aurora_ablation_preserves_target_and_checksums_raw_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "b0017"
    summary = run_ablation(
        cpp_binary("k3x_run"),
        output_dir=output,
        warmups=0,
        samples=1,
    )
    assert summary["benchmark"] == "B-0017"
    assert summary["warmups"] == 0
    assert summary["samples"] == 1
    assert len(summary["records"]) == 7
    natural = summary["records"][0]
    assert natural["name"] == "natural-greedy"
    assert natural["draft_reader_completed_bytes"] == 0
    for record in summary["records"]:
        assert record["token_ids"] == natural["token_ids"]
        assert record["token_parity"] is True
        assert record["final_state_max_abs_error"] <= 1.0e-6
        assert record["committed_route_parity"] is True
        assert record["reader_completed_bytes"] > 0
        if record["name"] != "natural-greedy":
            assert record["draft_reader_completed_bytes"] > 0
            assert record["draft_reader_read_calls"] > 0
            assert record["draft_routing_decisions"] > 0
        raw_json = output / f"{record['name']}.json"
        raw_csv = output / f"{record['name']}.csv"
        assert hashlib.sha256(raw_json.read_bytes()).hexdigest() == record[
            "raw_json_sha256"
        ]
        assert hashlib.sha256(raw_csv.read_bytes()).hexdigest() == record[
            "raw_csv_sha256"
        ]
        assert b"\r\n" not in raw_csv.read_bytes()
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    assert (output / "summary.json").is_file()
    assert (output / "summary.csv").is_file()
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
