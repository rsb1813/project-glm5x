# B-0018 replay와 persistent AURORA의 쌍대 증거와 digest를 검증합니다.
import hashlib
import json
from pathlib import Path

from conftest import cpp_binary
from tools.ablate_persistent_aurora import CASES, run_ablation


def test_persistent_aurora_matrix_is_canonical() -> None:
    assert tuple(item[0] for item in CASES) == (
        "natural-greedy",
        "replay-fixed-2-token",
        "persistent-fixed-2-token",
        "replay-adaptive-token",
        "persistent-adaptive-token",
        "replay-fixed-2-expert",
        "persistent-fixed-2-expert",
        "replay-adaptive-expert",
        "persistent-adaptive-expert",
    )


def test_persistent_aurora_ablation_preserves_pairs_and_raw_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "b0018"
    summary = run_ablation(
        cpp_binary("k3x_run"), output_dir=output, warmups=0, samples=1
    )
    assert summary["benchmark"] == "B-0018"
    assert len(summary["records"]) == 9
    natural = summary["records"][0]
    assert natural["draft_reader_completed_bytes"] == 0
    records = {record["name"]: record for record in summary["records"]}
    for record in summary["records"]:
        assert record["token_ids"] == natural["token_ids"]
        assert record["token_parity"] is True
        assert record["final_state_max_abs_error"] <= 1.0e-6
        assert record["committed_route_parity"] is True
        raw_json = output / f"{record['name']}.json"
        raw_csv = output / f"{record['name']}.csv"
        assert hashlib.sha256(raw_json.read_bytes()).hexdigest() == record[
            "raw_json_sha256"
        ]
        assert hashlib.sha256(raw_csv.read_bytes()).hexdigest() == record[
            "raw_csv_sha256"
        ]
        assert b"\r\n" not in raw_csv.read_bytes()
    for replay_name, persistent_name in (
        ("replay-fixed-2-token", "persistent-fixed-2-token"),
        ("replay-adaptive-token", "persistent-adaptive-token"),
        ("replay-fixed-2-expert", "persistent-fixed-2-expert"),
        ("replay-adaptive-expert", "persistent-adaptive-expert"),
    ):
        replay = records[replay_name]
        persistent = records[persistent_name]
        assert replay["pair_name"] == persistent["pair_name"]
        assert replay["speculative_proposed_draft_tokens"] == persistent[
            "speculative_proposed_draft_tokens"
        ]
        assert replay["speculative_accepted_draft_tokens"] == persistent[
            "speculative_accepted_draft_tokens"
        ]
        assert persistent["draft_replayed_context_tokens"] == 0
        assert persistent["draft_context_prefill_tokens"] == 5
        assert persistent["draft_incremental_forward_calls"] > 0
        assert persistent["draft_reader_completed_bytes"] < replay[
            "draft_reader_completed_bytes"
        ]
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(aggregate).hexdigest() == summary["aggregate_sha256"]
    assert b"\r\n" not in (output / "summary.csv").read_bytes()


def test_committed_b0018_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0018-persistent-aurora-wsl"
    summary = json.loads(
        (output / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["benchmark"] == "B-0018"
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
    baseline_decode = summary["records"][0]["decode_tokens_per_second"]
    records = {record["name"]: record for record in summary["records"]}
    for record in summary["records"]:
        expected = (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
        assert record["decode_delta_percent"] == expected
    for _, replay_name, persistent_name in (
        ("fixed-token", "replay-fixed-2-token", "persistent-fixed-2-token"),
        ("adaptive-token", "replay-adaptive-token", "persistent-adaptive-token"),
        ("fixed-expert", "replay-fixed-2-expert", "persistent-fixed-2-expert"),
        ("adaptive-expert", "replay-adaptive-expert", "persistent-adaptive-expert"),
    ):
        replay = records[replay_name]
        persistent = records[persistent_name]
        expected_decode = (
            persistent["decode_tokens_per_second"]
            / replay["decode_tokens_per_second"]
            - 1.0
        ) * 100.0
        expected_bytes = (
            1.0
            - persistent["draft_reader_completed_bytes"]
            / replay["draft_reader_completed_bytes"]
        ) * 100.0
        assert persistent["paired_decode_delta_percent"] == expected_decode
        assert persistent[
            "draft_reader_byte_reduction_percent"
        ] == expected_bytes
