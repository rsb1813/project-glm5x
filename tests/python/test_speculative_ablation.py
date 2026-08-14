# speculative verification ablation의 matrix와 exact parity gate를 검증합니다.
from pathlib import Path

from conftest import cpp_binary
from k3x_converter.writer import convert
from tools.ablate_speculative_verification import run_ablation, speculative_matrix


def test_speculative_matrix_contains_exact_and_mismatch_paths() -> None:
    matrix = speculative_matrix([43, 32, 28, 49, 9, 28])
    assert tuple(case["name"] for case in matrix) == (
        "greedy",
        "perfect-block2",
        "mixed-block2",
    )
    assert matrix[1]["script"] == "43:32,28;49:9"
    assert matrix[2]["expected_blocks"] == 4
    assert matrix[2]["expected_accepted"] == 1


def test_speculative_ablation_preserves_exact_target_execution(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    output_dir = tmp_path / "b0014"
    convert(synthetic_source, artifact, chunk_bytes=257)
    summary = run_ablation(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        output_dir=output_dir,
    )
    assert summary["benchmark"] == "B-0014"
    records = summary["records"]
    assert all(record["parity_status"] == "exact" for record in records)
    assert all(record["target_decode_forward_calls"] == 5 for record in records)
    assert records[0]["speculative_mode"] == "none"
    assert records[1]["speculative_acceptance_rate"] == 1.0
    assert records[2]["speculative_verification_blocks"] == 4
    assert records[2]["speculative_proposed_draft_tokens"] == 4
    assert records[2]["speculative_accepted_draft_tokens"] == 1
    for name in ("greedy", "perfect-block2", "mixed-block2"):
        assert (output_dir / f"{name}.json").is_file()
        assert (output_dir / f"{name}.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "summary.csv").is_file()
