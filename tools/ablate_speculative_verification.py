# synthetic speculative verifier의 correctness와 overhead를 교차 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


def speculative_matrix(tokens: list[int]) -> tuple[dict[str, object], ...]:
    if len(tokens) != 6:
        raise ValueError("speculative matrix requires exactly six target tokens")
    wrong_first = tokens[1] ^ 1
    wrong_late = tokens[4] ^ 1
    return (
        {
            "name": "greedy",
            "mode": "none",
            "block_size": 0,
            "script": "",
            "expected_blocks": 0,
            "expected_proposed": 0,
            "expected_accepted": 0,
        },
        {
            "name": "perfect-block2",
            "mode": "scripted-reference",
            "block_size": 2,
            "script": (
                f"{tokens[0]}:{tokens[1]},{tokens[2]};"
                f"{tokens[3]}:{tokens[4]}"
            ),
            "expected_blocks": 2,
            "expected_proposed": 3,
            "expected_accepted": 3,
        },
        {
            "name": "mixed-block2",
            "mode": "scripted-reference",
            "block_size": 2,
            "script": (
                f"{tokens[0]}:{wrong_first},{tokens[2]};"
                f"{tokens[1]}:;"
                f"{tokens[2]}:{tokens[3]},{wrong_late};"
                f"{tokens[4]}:"
            ),
            "expected_blocks": 4,
            "expected_proposed": 4,
            "expected_accepted": 1,
        },
    )


def _run_diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    *,
    mode: str,
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
        block_size=0,
        script="",
    )
    matrix = speculative_matrix(baseline["token_ids"])
    parity_fields = (
        "token_ids",
        "final_state",
        "routed_experts",
        "routed_k",
        "reader_read_calls",
        "reader_completed_bytes",
        "l1_expert_cache_hits",
        "l1_expert_cache_misses",
        "target_decode_forward_calls",
    )
    records: list[dict] = []
    for case in matrix:
        name = str(case["name"])
        mode = str(case["mode"])
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
            speculative_block_size=block_size,
            speculative_script=script,
        )
        write_results(
            record,
            output_dir / f"{name}.json",
            output_dir / f"{name}.csv",
        )
        diagnostic = baseline if name == "greedy" else _run_diagnostic(
            artifact,
            runner,
            output_dir / f"{name}-diagnostic.json",
            mode=mode,
            block_size=block_size,
            script=script,
        )
        parity = all(diagnostic[field] == baseline[field] for field in parity_fields)
        if not parity:
            raise RuntimeError(f"{name} diverged from greedy target execution")
        if diagnostic["speculative_verification_blocks"] != int(
            case["expected_blocks"]
        ):
            raise RuntimeError(f"{name} verification block count diverged")
        if diagnostic["speculative_proposed_draft_tokens"] != int(
            case["expected_proposed"]
        ):
            raise RuntimeError(f"{name} proposed-token count diverged")
        if diagnostic["speculative_accepted_draft_tokens"] != int(
            case["expected_accepted"]
        ):
            raise RuntimeError(f"{name} accepted-token count diverged")
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
        )
        records.append(payload)

    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0014",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
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
