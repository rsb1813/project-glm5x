# expert-major speculative verification ablation의 matrix와 증거 일치를 검증합니다.
import csv
import hashlib
import json
from pathlib import Path

from conftest import cpp_binary
from k3x_converter.writer import convert
from tools.ablate_expert_major_verification import expert_major_matrix, run_ablation


def test_expert_major_matrix_contains_reference_and_exact_pairs() -> None:
    matrix = expert_major_matrix([43, 32, 28, 49, 9, 28])
    assert tuple(case["name"] for case in matrix) == (
        "greedy",
        "token-major-perfect-2",
        "expert-major-perfect-2",
        "token-major-mixed-2",
        "expert-major-mixed-2",
    )
    assert matrix[2]["verification"] == "expert-major"
    assert matrix[4]["expected_evaluated"] == 8
    assert matrix[4]["expected_discarded"] == 3


def test_expert_major_ablation_preserves_exact_state_and_artifact_parity(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    output_dir = tmp_path / "b0015"
    convert(synthetic_source, artifact, chunk_bytes=257)
    summary = run_ablation(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        output_dir=output_dir,
    )

    assert summary["benchmark"] == "B-0015"
    assert summary["warmup"] == 0
    assert summary["iterations"] == 1
    records = summary["records"]
    assert len(records) == 5
    assert all(record["parity_status"] == "exact" for record in records)

    token_perfect, expert_perfect = records[1], records[2]
    token_mixed, expert_mixed = records[3], records[4]
    for token_record, expert_record in (
        (token_perfect, expert_perfect),
        (token_mixed, expert_mixed),
    ):
        for field in (
            "speculative_verification_blocks",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "speculative_acceptance_rate",
        ):
            assert expert_record[field] == token_record[field]

    assert expert_perfect["target_positions_evaluated"] == 5
    assert expert_perfect["target_positions_discarded"] == 0
    assert expert_mixed["target_positions_evaluated"] == 8
    assert expert_mixed["target_positions_discarded"] == 3
    assert expert_perfect["expert_major_payload_loads"] < expert_perfect[
        "expert_major_assignments"
    ]
    assert expert_perfect["reader_requested_bytes"] < token_perfect[
        "reader_requested_bytes"
    ]

    aggregate_payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    assert summary["aggregate_sha256"] == hashlib.sha256(
        aggregate_payload
    ).hexdigest()

    for record in records:
        name = record["name"]
        raw_json = output_dir / f"{name}.json"
        raw_csv = output_dir / f"{name}.csv"
        assert record["raw_json_sha256"] == hashlib.sha256(
            raw_json.read_bytes()
        ).hexdigest()
        assert record["raw_csv_sha256"] == hashlib.sha256(
            raw_csv.read_bytes()
        ).hexdigest()
        raw_payload = json.loads(raw_json.read_text(encoding="utf-8"))
        with raw_csv.open(newline="", encoding="utf-8") as stream:
            csv_payload = next(csv.DictReader(stream))
        for field in (
            "speculative_verification",
            "target_positions_evaluated",
            "target_positions_discarded",
            "expert_major_payload_loads",
            "reader_requested_bytes",
        ):
            assert record[field] == raw_payload[field]
            assert str(record[field]) == csv_payload[field]

    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
