# expert-major speculative verification의 정확성·traffic·overhead를 교차 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


def expert_major_matrix(tokens: list[int]) -> tuple[dict[str, object], ...]:
    if len(tokens) != 6:
        raise ValueError("expert-major matrix requires exactly six target tokens")
    perfect = f"{tokens[0]}:{tokens[1]},{tokens[2]};{tokens[3]}:{tokens[4]}"
    wrong_first = tokens[1] ^ 1
    wrong_late = tokens[4] ^ 1
    mixed = (
        f"{tokens[0]}:{wrong_first},{tokens[2]};"
        f"{tokens[1]}:;"
        f"{tokens[2]}:{tokens[3]},{wrong_late};"
        f"{tokens[4]}:"
    )
    return (
        {
            "name": "greedy",
            "mode": "none",
            "verification": "token-major",
            "block_size": 0,
            "script": "",
            "expected_blocks": 0,
            "expected_proposed": 0,
            "expected_accepted": 0,
            "expected_committed": 0,
            "expected_evaluated": 0,
            "expected_discarded": 0,
        },
        {
            "name": "token-major-perfect-2",
            "mode": "scripted-reference",
            "verification": "token-major",
            "block_size": 2,
            "script": perfect,
            "expected_blocks": 2,
            "expected_proposed": 3,
            "expected_accepted": 3,
            "expected_committed": 5,
            "expected_evaluated": 0,
            "expected_discarded": 0,
        },
        {
            "name": "expert-major-perfect-2",
            "mode": "scripted-reference",
            "verification": "expert-major",
            "block_size": 2,
            "script": perfect,
            "expected_blocks": 2,
            "expected_proposed": 3,
            "expected_accepted": 3,
            "expected_committed": 5,
            "expected_evaluated": 5,
            "expected_discarded": 0,
        },
        {
            "name": "token-major-mixed-2",
            "mode": "scripted-reference",
            "verification": "token-major",
            "block_size": 2,
            "script": mixed,
            "expected_blocks": 4,
            "expected_proposed": 4,
            "expected_accepted": 1,
            "expected_committed": 5,
            "expected_evaluated": 0,
            "expected_discarded": 0,
        },
        {
            "name": "expert-major-mixed-2",
            "mode": "scripted-reference",
            "verification": "expert-major",
            "block_size": 2,
            "script": mixed,
            "expected_blocks": 4,
            "expected_proposed": 4,
            "expected_accepted": 1,
            "expected_committed": 5,
            "expected_evaluated": 8,
            "expected_discarded": 3,
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    *,
    mode: str,
    verification: str,
    block_size: int,
    script: str,
) -> dict:
    subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--prompt-ids",
            "1,7,3,9",
            "--generate",
            "6",
            "--mode",
            "incremental",
            "--backend",
            "cpu",
            "--diagnostics",
            "true",
            "--speculative-mode",
            mode,
            "--speculative-verification",
            verification,
            "--speculative-block-size",
            str(block_size),
            "--speculative-script",
            script,
            "--json",
            str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def run_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    output_dir: Path,
) -> dict:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    artifact = Path(artifact).resolve()
    runner = Path(runner).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _run_diagnostic(
        artifact,
        runner,
        output_dir / "greedy-diagnostic.json",
        mode="none",
        verification="token-major",
        block_size=0,
        script="",
    )
    matrix = expert_major_matrix(baseline["token_ids"])
    parity_fields = (
        "token_ids",
        "final_state",
        "routed_experts",
        "routed_k",
    )
    records: list[dict] = []
    for case in matrix:
        name = str(case["name"])
        mode = str(case["mode"])
        verification = str(case["verification"])
        block_size = int(case["block_size"])
        script = str(case["script"])
        record = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cpu",
            dense_precision="fp32",
            speculative_mode=mode,
            speculative_verification=verification,
            speculative_block_size=block_size,
            speculative_script=script,
        )
        raw_json = output_dir / f"{name}.json"
        raw_csv = output_dir / f"{name}.csv"
        write_results(record, raw_json, raw_csv)
        diagnostic_path = output_dir / f"{name}-diagnostic.json"
        diagnostic = baseline if name == "greedy" else _run_diagnostic(
            artifact,
            runner,
            diagnostic_path,
            mode=mode,
            verification=verification,
            block_size=block_size,
            script=script,
        )
        if not all(diagnostic[field] == baseline[field] for field in parity_fields):
            raise RuntimeError(f"{name} diverged from greedy target execution")
        expected = {
            "speculative_verification_blocks": "expected_blocks",
            "speculative_proposed_draft_tokens": "expected_proposed",
            "speculative_accepted_draft_tokens": "expected_accepted",
            "speculative_committed_tokens": "expected_committed",
            "target_positions_evaluated": "expected_evaluated",
            "target_positions_discarded": "expected_discarded",
        }
        for field, case_field in expected.items():
            if diagnostic[field] != int(case[case_field]):
                raise RuntimeError(f"{name} {field} diverged")
        if verification == "expert-major":
            if diagnostic["expert_major_payload_loads"] != diagnostic[
                "expert_major_unique_experts_sum"
            ]:
                raise RuntimeError(f"{name} payload-load accounting diverged")
            if diagnostic["expert_major_reused_assignments"] != (
                diagnostic["expert_major_assignments"]
                - diagnostic["expert_major_payload_loads"]
            ):
                raise RuntimeError(f"{name} assignment reuse accounting diverged")

        payload = asdict(record)
        payload.update(
            name=name,
            script=script,
            parity_status="exact",
            decode_delta_percent=(
                0.0
                if not records
                else (
                    record.decode_tokens_per_second
                    / records[0]["decode_tokens_per_second"]
                    - 1.0
                )
                * 100.0
            ),
            raw_json_sha256=_sha256(raw_json),
            raw_csv_sha256=_sha256(raw_csv),
        )
        records.append(payload)

    aggregate_payload = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0015",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "aggregate_sha256": hashlib.sha256(aggregate_payload).hexdigest(),
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
