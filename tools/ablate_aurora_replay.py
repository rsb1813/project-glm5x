# B-0017 AURORA replay scheduling의 target parity와 분리 traffic을 측정합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    repository = Path(__file__).resolve().parents[1]
    for import_root in (repository, repository / "converter", repository / "reference"):
        sys.path.insert(0, str(import_root))

from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint
from tools.benchmark_synthetic import benchmark_once, write_results


def aurora_matrix() -> tuple[dict[str, object], ...]:
    return (
        {
            "name": "natural-greedy",
            "mode": "none",
            "verification": "token-major",
            "block_size": 0,
            "policy": "none",
        },
        {
            "name": "aurora-k4-fixed-1",
            "mode": "aurora-replay",
            "verification": "token-major",
            "block_size": 1,
            "policy": "fixed",
        },
        {
            "name": "aurora-k4-fixed-2",
            "mode": "aurora-replay",
            "verification": "token-major",
            "block_size": 2,
            "policy": "fixed",
        },
        {
            "name": "aurora-k4-fixed-4",
            "mode": "aurora-replay",
            "verification": "token-major",
            "block_size": 4,
            "policy": "fixed",
        },
        {
            "name": "aurora-k4-adaptive-token",
            "mode": "aurora-replay",
            "verification": "token-major",
            "block_size": 4,
            "policy": "adaptive",
        },
        {
            "name": "aurora-k4-fixed-2-expert",
            "mode": "aurora-replay",
            "verification": "expert-major",
            "block_size": 2,
            "policy": "fixed",
        },
        {
            "name": "aurora-k4-adaptive-expert",
            "mode": "aurora-replay",
            "verification": "expert-major",
            "block_size": 4,
            "policy": "adaptive",
        },
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic(
    artifact: Path,
    runner: Path,
    output: Path,
    case: dict[str, object],
) -> dict:
    command = [
        str(runner),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--backend", "cpu",
        "--diagnostics", "true",
        "--speculative-mode", str(case["mode"]),
        "--speculative-verification", str(case["verification"]),
        "--speculative-block-size", str(case["block_size"]),
    ]
    if case["mode"] == "aurora-replay":
        command.extend([
            "--aurora-draft-k", "4",
            "--aurora-block-policy", str(case["policy"]),
        ])
    command.extend(["--json", str(output)])
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "AURORA diagnostic failed")
    return json.loads(output.read_text(encoding="utf-8"))


def _maximum_absolute_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("AURORA final state shape diverged")
    return max((abs(actual - expected) for actual, expected in zip(left, right)),
               default=0.0)


def run_ablation(
    runner: Path,
    *,
    output_dir: Path,
    warmups: int,
    samples: int,
) -> dict:
    if warmups < 0 or samples <= 0:
        raise ValueError("warmups must be non-negative and samples must be positive")
    runner = Path(runner).resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="k3x-b0017-") as temporary:
        root = Path(temporary)
        config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
        source = root / "source-top16"
        write_source_checkpoint(source, config=config)
        artifact = root / "top16.k3x"
        convert(source, artifact, chunk_bytes=257)
        artifact_sha256 = _sha256(artifact)
        baseline_case = aurora_matrix()[0]
        baseline = _diagnostic(
            artifact, runner, root / "natural-diagnostic.json", baseline_case
        )

        records: list[dict] = []
        for case in aurora_matrix():
            mode = str(case["mode"])
            verification = str(case["verification"])
            block_size = int(case["block_size"])
            policy = str(case["policy"])
            benchmark = benchmark_once(
                artifact,
                runner,
                warmups,
                samples,
                backend="cpu",
                dense_precision="fp32",
                speculative_mode=mode,
                speculative_verification=verification,
                speculative_block_size=block_size,
                aurora_draft_k=4 if mode == "aurora-replay" else 0,
                aurora_block_policy=policy if mode == "aurora-replay" else "fixed",
            )
            name = str(case["name"])
            raw_json = output_dir / f"{name}.json"
            raw_csv = output_dir / f"{name}.csv"
            write_results(benchmark, raw_json, raw_csv)
            diagnostic = baseline if mode == "none" else _diagnostic(
                artifact, runner, root / f"{name}-diagnostic.json", case
            )
            token_parity = diagnostic["token_ids"] == baseline["token_ids"]
            route_parity = (
                diagnostic["routed_experts"] == baseline["routed_experts"]
                and diagnostic["routed_k"] == baseline["routed_k"]
            )
            state_error = _maximum_absolute_error(
                diagnostic["final_state"], baseline["final_state"]
            )
            if not token_parity or not route_parity or state_error > 1.0e-6:
                raise RuntimeError(f"{name} diverged from natural target execution")
            if tuple(benchmark.token_ids) != tuple(baseline["token_ids"]):
                raise RuntimeError(f"{name} benchmark tokens diverged")
            if benchmark.reader_completed_bytes <= 0:
                raise RuntimeError(f"{name} lacks target Reader evidence")
            if mode == "aurora-replay" and (
                benchmark.draft_reader_completed_bytes <= 0
                or benchmark.draft_reader_read_calls <= 0
                or benchmark.draft_routing_decisions <= 0
            ):
                raise RuntimeError(f"{name} lacks separated draft evidence")

            payload = asdict(benchmark)
            payload.update(
                name=name,
                token_parity=token_parity,
                final_state_max_abs_error=state_error,
                committed_route_parity=route_parity,
                raw_json_sha256=_sha256(raw_json),
                raw_csv_sha256=_sha256(raw_csv),
            )
            records.append(payload)

    baseline_decode = records[0]["decode_tokens_per_second"]
    for record in records:
        record["decode_delta_percent"] = (
            record["decode_tokens_per_second"] / baseline_decode - 1.0
        ) * 100.0
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "synthetic-milestone-one",
        "evidence": "measured",
        "benchmark": "B-0017",
        "warmups": warmups,
        "samples": samples,
        "artifact_sha256": artifact_sha256,
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = list(dict.fromkeys(key for record in records for key in record))
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    summary = run_ablation(
        args.runner,
        output_dir=args.output,
        warmups=args.warmups,
        samples=args.samples,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
