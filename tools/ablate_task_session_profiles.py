# task/session prior가 exact expert cache에 미치는 영향을 B-0011로 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from tools.benchmark_synthetic import benchmark_once, write_results


TARGET_PROMPT = "1,7,3,9"
CONFLICTING_PROMPTS = ("2,2,2,2", "5,11,17,23", "49,48,47,46")
RUNTIME_METADATA = "TASK=coding,LANG=cpp,PHASE=debug,REPO=k3x"


def task_session_profile_matrix() -> tuple[dict[str, str], ...]:
    return (
        {"name": "lfu", "policy": "lfu", "profile_kind": "none"},
        {
            "name": "least-stale",
            "policy": "least-stale",
            "profile_kind": "none",
        },
        {
            "name": "profiled-cold",
            "policy": "profiled",
            "profile_kind": "cold",
        },
        {
            "name": "profiled-helpful",
            "policy": "profiled",
            "profile_kind": "helpful",
        },
        {
            "name": "profiled-conflicting",
            "policy": "profiled",
            "profile_kind": "conflicting",
        },
    )


def _run_profile_seed(
    artifact: Path, runner: Path, prompt: str, profile: Path, output: Path
) -> None:
    subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--prompt-ids",
            prompt,
            "--generate",
            "0",
            "--mode",
            "incremental",
            "--runtime-metadata",
            RUNTIME_METADATA,
            "--runtime-profile-out",
            str(profile),
            "--json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _hot_bank(path: Path) -> tuple[tuple[int, int], ...]:
    hot: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] == "HOT":
            hot.append((int(fields[1]), int(fields[2])))
    if not hot:
        raise RuntimeError("prepared runtime profile has an empty hot bank")
    return tuple(hot)


def prepare_task_profiles(
    artifact: Path, runner: Path, output_dir: Path
) -> dict[str, object]:
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    helpful = profile_dir / "helpful.k3xp"
    _run_profile_seed(
        artifact, runner, TARGET_PROMPT, helpful, profile_dir / "helpful.json"
    )
    helpful_hot = set(_hot_bank(helpful))

    candidates: list[tuple[int, str, Path]] = []
    for index, prompt in enumerate(CONFLICTING_PROMPTS):
        path = profile_dir / f"candidate-{index}.k3xp"
        _run_profile_seed(
            artifact, runner, prompt, path, profile_dir / f"candidate-{index}.json"
        )
        overlap = len(helpful_hot.intersection(_hot_bank(path)))
        candidates.append((overlap, prompt, path))
    overlap, prompt, selected = min(candidates, key=lambda item: (item[0], item[1]))
    conflicting = profile_dir / "conflicting.k3xp"
    selected.replace(conflicting)
    for _, _, path in candidates:
        if path.exists():
            path.unlink()
    return {
        "helpful": helpful,
        "conflicting": conflicting,
        "conflicting_prompt": prompt,
        "hot_overlap": overlap,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_record(
    record: dict[str, object], policy: str, profile_kind: str,
    capacity_bytes: int,
) -> None:
    if (record["l1_expert_cache_mode"], record["l1_expert_cache_bytes"]) != (
        policy,
        capacity_bytes,
    ):
        raise RuntimeError("profile cache option identity changed")
    if int(record["reader_requested_bytes"]) <= 0 or (
        record["reader_requested_bytes"] != record["reader_completed_bytes"]
    ):
        raise RuntimeError("profile cache reader accounting failed")
    if policy != "profiled":
        if any(
            int(record[field])
            for field in (
                "runtime_profile_metadata_count",
                "runtime_profile_live_observations",
                "runtime_profile_load_bytes",
                "runtime_profile_save_bytes",
            )
        ):
            raise RuntimeError("reference policy unexpectedly observed a profile")
        return
    if int(record["runtime_profile_live_observations"]) <= 0:
        raise RuntimeError("profile live observations were not recorded")
    weight = float(record["runtime_profile_prior_weight"])
    if profile_kind == "cold":
        if weight != 0.0 or int(record["runtime_profile_load_bytes"]) != 0:
            raise RuntimeError("cold profile prior weight changed")
    elif not (0.0 < weight < 1.0) or int(record["runtime_profile_load_bytes"]) <= 0:
        raise RuntimeError("profile prior weight is outside the expected range")
    if int(record["runtime_profile_save_bytes"]) <= 0:
        raise RuntimeError("profile save telemetry is missing")


def run_task_session_profile_ablation(
    artifact: Path,
    runner: Path,
    *,
    warmup: int,
    iterations: int,
    capacity_bytes: int,
    prior_strength: int,
    output_dir: Path,
    environment_label: str = "local-synthetic-smoke",
) -> dict[str, object]:
    if capacity_bytes <= 0 or prior_strength <= 0:
        raise ValueError("capacity and prior strength must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = prepare_task_profiles(artifact, runner, output_dir)
    helpful = Path(profiles["helpful"])
    conflicting = Path(profiles["conflicting"])

    cases: list[dict[str, object]] = []
    baseline: dict[str, object] | None = None
    for configuration in task_session_profile_matrix():
        name = configuration["name"]
        policy = configuration["policy"]
        profile_kind = configuration["profile_kind"]
        profile_in = (
            helpful if profile_kind == "helpful"
            else conflicting if profile_kind == "conflicting"
            else None
        )
        profile_out = (
            output_dir / "profiles" / f"{name}-observed.k3xp"
            if policy == "profiled" else None
        )
        benchmark = benchmark_once(
            artifact,
            runner,
            warmup,
            iterations,
            backend="cpu",
            dense_precision="fp32",
            l1_expert_cache=policy,
            l1_expert_cache_bytes=capacity_bytes,
            l2_io="pread",
            l2_cache="buffered",
            l2_expert_schedule="blocking",
            profile_prior_strength=prior_strength,
            runtime_metadata=RUNTIME_METADATA if profile_kind == "cold" else "",
            runtime_profile_in=profile_in,
            runtime_profile_out=profile_out,
        )
        write_results(
            benchmark, output_dir / f"{name}.json", output_dir / f"{name}.csv"
        )
        record = asdict(benchmark)
        _validate_record(record, policy, profile_kind, capacity_bytes)
        if baseline is None:
            baseline = record
        if tuple(record["token_ids"]) != tuple(baseline["token_ids"]):
            raise RuntimeError("task profile token parity failed")
        if tuple(record["routed_experts"]) != tuple(baseline["routed_experts"]):
            raise RuntimeError("task profile routing parity failed")
        if (
            record["max_absolute_error"] != baseline["max_absolute_error"]
            or record["max_relative_error"] != baseline["max_relative_error"]
        ):
            raise RuntimeError("task profile numerical parity failed")
        cases.append(
            {"name": name, "status": "measured", "parity_status": "exact", **record}
        )

    summary: dict[str, object] = {
        "benchmark_id": "B-0011",
        "artifact_sha256": _sha256(Path(artifact)),
        "environment_label": environment_label,
        "capacity_bytes": capacity_bytes,
        "prior_strength": prior_strength,
        "target_prompt": TARGET_PROMPT,
        "conflicting_prompt": profiles["conflicting_prompt"],
        "hot_overlap": profiles["hot_overlap"],
        "helpful_profile_sha256": _sha256(helpful),
        "conflicting_profile_sha256": _sha256(conflicting),
        "supported_cases": len(cases),
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--capacity-bytes", type=int, default=13056)
    parser.add_argument("--prior-strength", type=int, default=4)
    parser.add_argument("--environment-label", default="local-synthetic-smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_task_session_profile_ablation(
        args.artifact,
        args.runner,
        warmup=args.warmup,
        iterations=args.iterations,
        capacity_bytes=args.capacity_bytes,
        prior_strength=args.prior_strength,
        output_dir=args.output_dir,
        environment_label=args.environment_label,
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
