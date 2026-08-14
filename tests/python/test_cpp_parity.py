# 독립 C++ runtime의 greedy token이 PyTorch golden과 일치하는지 검증합니다.
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import build_synthetic_model
from k3x_ref.fixtures import write_source_checkpoint
from k3x_ref.storage_fixture import write_bounded_expert_source


import pytest

from conftest import cpp_binary


def cpu_only_build() -> bool:
    return cpp_binary("test_backend_unavailable").is_file()


def test_cpp_runner_rejects_storage_fixture_before_graph_execution(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    output = tmp_path / "must-not-exist.json"
    convert(source, artifact, chunk_bytes=193 * 1024)

    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model",
            str(artifact),
            "--json",
            str(output),
            "--prompt-ids",
            "1,7,3,9",
            "--generate",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "NON_EXECUTABLE_ARTIFACT"
    assert not output.exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--backend", "warp"], "unknown backend: warp"),
        (["--dense-precision", "fp8"], "unknown dense precision: fp8"),
    ],
)
def test_cpp_runner_rejects_unknown_backend_values(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--cuda-allocation", "pool"], "unknown CUDA allocation mode: pool"),
        (["--cuda-weights", "lru"], "unknown CUDA weight mode: lru"),
        (["--cuda-batching", "graph"], "unknown CUDA batching mode: graph"),
        (["--cuda-boundary", "layer"], "unknown CUDA boundary mode: layer"),
        (["--cuda-transfer", "queue"], "unknown CUDA transfer mode: queue"),
        (["--cuda-moe-fusion", "graph"], "unknown CUDA MoE fusion mode: graph"),
        (
            ["--cuda-weight-validation", "cached"],
            "unknown CUDA weight validation mode: cached",
        ),
        (
            ["--cuda-resident-bytes", "-1"],
            "invalid CUDA resident byte capacity: -1",
        ),
        (
            ["--cuda-pinned-bytes", "-1"],
            "invalid CUDA pinned byte capacity: -1",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_execution_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpp_runner_rejects_admission_validation_without_resident_moe_layer() -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), "--cuda-weight-validation", "admission"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "admission CUDA weight validation requires resident cuda-custom "
        "moe-layer execution"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend", "cuda-custom", "--cuda-batching", "resident-grid"],
        [
            "--backend", "cuda-custom",
            "--cuda-allocation", "reused",
            "--cuda-weights", "transient",
            "--cuda-batching", "resident-grid",
            "--cuda-boundary", "ffn-block",
        ],
        [
            "--backend", "cuda-custom",
            "--cuda-allocation", "per-operation",
            "--cuda-weights", "resident",
            "--cuda-resident-bytes", "1",
            "--cuda-batching", "resident-grid",
            "--cuda-boundary", "ffn-block",
        ],
    ],
)
def test_cpp_runner_rejects_open_resident_grid_contract(
    arguments: list[str],
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "resident-grid batching requires resident cuda-custom ffn-block or "
        "moe-layer "
        "reused synchronous execution with fusion none"
    )


def test_cpp_runner_rejects_draft_batching_without_persistent_aurora() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--aurora-draft-batching", "resident-grid",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "speculative mode none does not accept speculative options"
    )


def test_cpp_runner_rejects_unknown_aurora_draft_boundary() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--aurora-draft-boundary", "warp",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "unknown AURORA draft boundary: warp"


def test_cpp_runner_rejects_draft_boundary_without_persistent_cuda() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--speculative-mode", "aurora-persistent",
            "--speculative-block-size", "2",
            "--aurora-draft-k", "4",
            "--aurora-draft-boundary", "moe-layer",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "AURORA draft boundary requires persistent cuda-custom draft backend"
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--l1-expert-cache", "clock"], "unknown L1 expert cache mode: clock"),
        (
            ["--l1-expert-cache-bytes", "-1"],
            "invalid L1 expert cache byte capacity: -1",
        ),
        (
            ["--l1-expert-cache", "static"],
            "static L1 expert cache requires a positive byte capacity",
        ),
        (
            ["--l1-expert-cache", "least-stale"],
            "least-stale L1 expert cache requires a positive byte capacity",
        ),
        (
            ["--l1-expert-cache-bytes", "1"],
            "disabled L1 expert cache requires a zero byte capacity",
        ),
        (
            ["--profile-prior-strength", "0"],
            "invalid profile prior strength: 0",
        ),
        (
            ["--profile-prior-strength", "-1"],
            "invalid profile prior strength: -1",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_l1_expert_cache_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    "cache_mode", ["static", "lru", "lfu", "least-stale", "profiled"]
)
def test_cpp_runner_accepts_l1_expert_cache_for_cpu(cache_mode: str) -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--l1-expert-cache",
            cache_mode,
            "--l1-expert-cache-bytes",
            "65536",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("bad-key=value", "INVALID_STATE: invalid runtime metadata"),
        ("TASK=", "invalid runtime metadata: TASK="),
        (
            "TASK=coding,TASK=debug",
            "invalid runtime metadata: duplicate key TASK",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_runtime_metadata(
    metadata: str, message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), "--runtime-metadata", metadata],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--l2-io", "thread-pool"], "unknown L2 I/O engine: thread-pool"),
        (["--l2-cache", "mmap"], "unknown L2 cache mode: mmap"),
        (["--l2-schedule", "fifo"], "unknown L2 expert schedule mode: fifo"),
        (["--l2-queue-depth", "0"], "L2 queue depth must be positive"),
        (["--l2-queue-depth", "-1"], "invalid L2 queue depth: -1"),
        (
            ["--l2-queue-depth", "1025"],
            "L2 queue depth exceeds maximum: 1025",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_l2_reader_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpp_runner_accepts_default_l2_reader_options() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--l2-io",
            "pread",
            "--l2-cache",
            "buffered",
            "--l2-queue-depth",
            "8",
            "--l2-schedule",
            "blocking",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend", "cpu", "--cuda-allocation", "reused"],
        ["--backend", "cpu", "--cuda-weights", "resident", "--cuda-resident-bytes", "1"],
        ["--backend", "cpu", "--cuda-batching", "grouped"],
        ["--backend", "cpu", "--cuda-resident-bytes", "1"],
        ["--backend", "cpu", "--cuda-transfer", "prefetch"],
        ["--backend", "cpu", "--cuda-pinned-bytes", "1"],
    ],
)
def test_cpp_runner_rejects_cuda_execution_options_for_cpu(
    arguments: list[str],
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "CUDA execution options require a CUDA backend"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--backend", "cuda-dense", "--cuda-weights", "resident"],
            "resident CUDA weights require a positive resident byte capacity",
        ),
        (
            ["--backend", "cuda-dense", "--cuda-resident-bytes", "1"],
            "transient CUDA weights require a zero resident byte capacity",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_weight_capacity_combinations(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize("backend", ["cpu", "cuda-dense"])
def test_cpp_runner_rejects_ffn_block_boundary_without_custom_cuda(
    backend: str,
) -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend",
            backend,
            "--cuda-boundary",
            "ffn-block",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "ffn-block boundary requires cuda-custom"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--backend", "cpu", "--cuda-moe-fusion", "routed-accumulate"],
            "routed-accumulate fusion requires cuda-custom",
        ),
        (
            [
                "--backend", "cuda-dense",
                "--cuda-moe-fusion", "routed-accumulate",
            ],
            "routed-accumulate fusion requires cuda-custom",
        ),
        (
            [
                "--backend", "cuda-custom",
                "--cuda-moe-fusion", "routed-accumulate",
            ],
            "routed-accumulate fusion requires ffn-block boundary",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_moe_fusion_combinations(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--backend", "cuda-custom", "--cuda-pinned-bytes", "1"],
            "synchronous CUDA transfer requires a zero pinned byte capacity",
        ),
        (
            ["--backend", "cuda-custom", "--cuda-transfer", "prefetch"],
            "prefetch CUDA transfer requires a positive pinned byte capacity",
        ),
        (
            [
                "--backend", "cuda-dense", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires cuda-custom",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires ffn-block boundary",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "prefetch", "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires reused allocation",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-boundary", "ffn-block",
                "--cuda-allocation", "reused", "--cuda-weights", "resident",
                "--cuda-resident-bytes", "1", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires transient weights",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_transfer_combinations(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpp_runner_accepts_exact_cuda_prefetch_capability_combination() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend", "cuda-custom",
            "--cuda-boundary", "ffn-block",
            "--cuda-allocation", "reused",
            "--cuda-weights", "transient",
            "--cuda-transfer", "prefetch",
            "--cuda-pinned-bytes", "1048576",
        ],
        capture_output=True,
        text=True,
    )
    if cpu_only_build():
        assert result.returncode == 4
        assert result.stderr.startswith("BACKEND_UNAVAILABLE")
    else:
        assert result.returncode == 3


def test_cpu_build_reports_explicit_cuda_request_as_unavailable() -> None:
    if not cpu_only_build():
        pytest.skip("CPU-build contract is exercised only against build-cpu")
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), "--backend", "cuda-custom"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.startswith("BACKEND_UNAVAILABLE")


def test_cpp_runner_rejects_bf16_for_cpu_backend() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend",
            "cpu",
            "--dense-precision",
            "bf16",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "bf16 dense precision requires a CUDA backend"


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_cpp_generation_matches_python_golden(
    synthetic_source: Path, tmp_path: Path, mode: str
) -> None:
    runner = cpp_binary("k3x_run")
    assert runner.exists(), "build k3x_run before running cross-language parity"
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "6", "--mode", mode, "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = build_synthetic_model().generate_greedy(
        [1, 7, 3, 9], 6, mode == "incremental"
    )
    assert result["token_ids"] == expected
    assert result["read_bytes"] > 0
    assert result["decode_nanoseconds"] > 0
    assert result["backend"] == "cpu"
    assert result["device"] == "CPU"
    assert result["dense_precision"] == "fp32"
    assert result["cuda_allocation"] == "per-operation"
    assert result["cuda_weights"] == "transient"
    assert result["cuda_batching"] == "scalar"
    assert result["cuda_weight_validation"] == "per-call"
    assert result["cuda_resident_bytes"] == 0
    assert result["device_allocation_count"] == 0
    assert result["weight_cache_hits"] == 0
    assert result["immutable_validation_scans"] == 0
    assert result["immutable_validation_hits"] == 0
    assert result["immutable_validation_bytes"] == 0
    assert result["immutable_validation_nanoseconds"] == 0


def test_cpp_scripted_speculation_preserves_greedy_execution(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    baseline_output = tmp_path / "baseline.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    common = [
        str(runner),
        "--model",
        str(artifact),
        "--prompt-ids",
        "1,7,3,9",
        "--generate",
        "6",
        "--mode",
        "incremental",
        "--diagnostics",
        "true",
    ]
    subprocess.run([*common, "--json", str(baseline_output)], check=True)
    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    tokens = baseline["token_ids"]

    perfect_script = f"{tokens[0]}:{tokens[1]},{tokens[2]};{tokens[3]}:{tokens[4]}"
    wrong_first = (tokens[1] + 1) % 64
    wrong_late = (tokens[4] + 1) % 64
    mixed_script = (
        f"{tokens[0]}:{wrong_first},{tokens[2]};"
        f"{tokens[1]}:;"
        f"{tokens[2]}:{tokens[3]},{wrong_late};"
        f"{tokens[4]}:"
    )
    cases = (
        ("perfect", perfect_script, 2, 3, 3, 5, 2),
        ("mixed", mixed_script, 4, 4, 1, 5, 2),
    )
    for name, script, blocks, proposed, accepted, committed, maximum in cases:
        output = tmp_path / f"{name}.json"
        subprocess.run(
            [
                *common,
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-script",
                script,
                "--json",
                str(output),
            ],
            check=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["token_ids"] == baseline["token_ids"]
        assert result["final_state"] == baseline["final_state"]
        assert result["routed_experts"] == baseline["routed_experts"]
        assert result["routed_k"] == baseline["routed_k"]
        assert result["reader_read_calls"] == baseline["reader_read_calls"]
        assert result["reader_completed_bytes"] == baseline["reader_completed_bytes"]
        assert result["l1_expert_cache_hits"] == baseline["l1_expert_cache_hits"]
        assert result["l1_expert_cache_misses"] == baseline["l1_expert_cache_misses"]
        assert result["speculative_mode"] == "scripted-reference"
        assert result["speculative_block_size"] == 2
        assert result["speculative_verification_blocks"] == blocks
        assert result["speculative_proposed_draft_tokens"] == proposed
        assert result["speculative_accepted_draft_tokens"] == accepted
        assert result["speculative_committed_tokens"] == committed
        assert result["speculative_max_proposal_tokens"] == maximum
        assert result["target_decode_forward_calls"] == 5
        assert result["speculative_acceptance_rate"] == pytest.approx(
            accepted / proposed
        )

    assert baseline["speculative_mode"] == "none"
    assert baseline["speculative_block_size"] == 0
    assert baseline["speculative_verification_blocks"] == 0
    assert baseline["speculative_acceptance_rate"] is None
    assert baseline["target_decode_forward_calls"] == 5
    assert baseline["aurora_draft_backend"] == "none"


def test_cpp_aurora_replay_preserves_natural_target_execution(
    tmp_path: Path,
) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    runner = cpp_binary("k3x_run")
    baseline_output = tmp_path / "natural.json"
    common = [
        str(runner), "--model", str(artifact),
        "--prompt-ids", "1,7,3,9", "--generate", "6",
        "--mode", "incremental", "--diagnostics", "true",
    ]
    subprocess.run([*common, "--json", str(baseline_output)], check=True)
    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    for policy in ("fixed", "adaptive"):
        results = {}
        for mode in ("aurora-replay", "aurora-persistent"):
            aurora_output = tmp_path / f"{mode}-{policy}.json"
            subprocess.run(
                [
                    *common,
                    "--speculative-mode", mode,
                    "--speculative-block-size", "2",
                    "--aurora-draft-k", "4",
                    "--aurora-block-policy", policy,
                    "--json", str(aurora_output),
                ],
                check=True,
            )
            aurora = json.loads(aurora_output.read_text(encoding="utf-8"))
            results[mode] = aurora
            assert aurora["token_ids"] == baseline["token_ids"]
            assert aurora["final_state"] == baseline["final_state"]
            assert aurora["routed_experts"] == baseline["routed_experts"]
            assert aurora["routed_k"] == baseline["routed_k"]
            assert aurora["reader_completed_bytes"] == baseline["reader_completed_bytes"]
            assert aurora["speculative_mode"] == mode
            assert aurora["aurora_draft_k"] == 4
            assert aurora["aurora_block_policy"] == policy
            assert aurora["aurora_draft_backend"] == "cpu"
            assert aurora["draft_proposal_calls"] > 0
            assert aurora["draft_candidate_tokens"] > 0
            assert aurora["draft_reader_read_calls"] > 0
            assert aurora["draft_reader_completed_bytes"] > 0
            assert aurora["draft_routing_decisions"] > 0
        replay = results["aurora-replay"]
        persistent = results["aurora-persistent"]
        assert replay["draft_replayed_context_tokens"] > 0
        assert replay["draft_context_prefill_tokens"] == 0
        assert replay["draft_incremental_forward_calls"] == 0
        assert persistent["draft_replayed_context_tokens"] == 0
        assert persistent["draft_context_prefill_tokens"] == 5
        assert persistent["draft_incremental_forward_calls"] > 0
        assert persistent["draft_kda_checkpoint_bytes"] > 0
        assert persistent["draft_reader_completed_bytes"] < replay[
            "draft_reader_completed_bytes"
        ]


def test_cpp_aurora_replay_preflight_rejects_invalid_combinations(
    tmp_path: Path,
) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    runner = cpp_binary("k3x_run")
    common = [
        str(runner), "--model", str(artifact),
        "--prompt-ids", "1,7,3,9", "--generate", "6",
    ]
    invalid_arguments = (
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "2",
         "--aurora-draft-k", "4", "--aurora-block-policy", "unknown"],
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "2",
         "--aurora-draft-k", "0"],
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "2",
         "--aurora-draft-k", "16"],
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "3",
         "--aurora-draft-k", "4"],
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "2",
         "--aurora-draft-k", "4", "--routing-mode", "fixed",
         "--routing-fixed-k", "4"],
        ["--speculative-mode", "none", "--aurora-draft-k", "4"],
        ["--speculative-mode", "aurora-replay", "--speculative-block-size", "2",
         "--aurora-draft-k", "4", "--mode", "full"],
        ["--speculative-mode", "aurora-persistent", "--speculative-block-size", "2",
         "--aurora-draft-k", "4", "--mode", "full"],
        ["--speculative-mode", "aurora-persistent", "--speculative-block-size", "2",
         "--aurora-draft-k", "4", "--routing-mode", "fixed",
         "--routing-fixed-k", "4"],
        ["--speculative-mode", "aurora-persistent", "--speculative-block-size", "3",
         "--aurora-draft-k", "4"],
        ["--speculative-mode", "aurora-persistent", "--speculative-block-size", "2",
         "--aurora-draft-k", "16"],
    )
    for index, arguments in enumerate(invalid_arguments):
        output = tmp_path / f"invalid-{index}.json"
        result = subprocess.run(
            [*common, *arguments, "--json", str(output)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert not output.exists()

    natural8 = SyntheticK3Config.default().replace(num_experts=12, top_k=8)
    source8 = tmp_path / "source-top8"
    write_source_checkpoint(source8, config=natural8)
    artifact8 = tmp_path / "top8.k3x"
    convert(source8, artifact8, chunk_bytes=257)
    output8 = tmp_path / "invalid-equal-k.json"
    result8 = subprocess.run(
        [
            str(runner), "--model", str(artifact8),
            "--prompt-ids", "1,7,3,9", "--generate", "6",
            "--speculative-mode", "aurora-replay",
            "--speculative-block-size", "2",
            "--aurora-draft-k", "8",
            "--json", str(output8),
        ],
        capture_output=True,
        text=True,
    )
    assert result8.returncode == 2
    assert not output8.exists()


def test_cpp_expert_major_speculation_preserves_exact_state_and_reuses_payloads(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    baseline_output = tmp_path / "baseline.json"
    token_major_output = tmp_path / "token-major.json"
    expert_major_output = tmp_path / "expert-major.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    common = [
        str(runner),
        "--model",
        str(artifact),
        "--prompt-ids",
        "1,7,3,9",
        "--generate",
        "6",
        "--mode",
        "incremental",
        "--diagnostics",
        "true",
    ]
    subprocess.run([*common, "--json", str(baseline_output)], check=True)
    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    tokens = baseline["token_ids"]
    script = f"{tokens[0]}:{tokens[1]},{tokens[2]};{tokens[3]}:{tokens[4]}"
    speculative = [
        "--speculative-mode",
        "scripted-reference",
        "--speculative-block-size",
        "2",
        "--speculative-script",
        script,
    ]
    subprocess.run(
        [*common, *speculative, "--json", str(token_major_output)],
        check=True,
    )
    subprocess.run(
        [
            *common,
            *speculative,
            "--speculative-verification",
            "expert-major",
            "--json",
            str(expert_major_output),
        ],
        check=True,
    )
    token_major = json.loads(token_major_output.read_text(encoding="utf-8"))
    expert_major = json.loads(expert_major_output.read_text(encoding="utf-8"))
    assert expert_major["token_ids"] == baseline["token_ids"]
    assert expert_major["final_state"] == baseline["final_state"]
    assert expert_major["routed_experts"] == baseline["routed_experts"]
    assert expert_major["routed_k"] == baseline["routed_k"]
    assert expert_major["speculative_verification"] == "expert-major"
    assert token_major["speculative_verification"] == "token-major"
    assert expert_major["target_block_forward_calls"] == 2
    assert expert_major["target_positions_evaluated"] == 5
    assert expert_major["target_positions_discarded"] == 0
    assert expert_major["expert_major_payload_loads"] == expert_major[
        "expert_major_unique_experts_sum"
    ]
    assert expert_major["expert_major_assignments"] > expert_major[
        "expert_major_payload_loads"
    ]
    assert expert_major["expert_major_reused_assignments"] == (
        expert_major["expert_major_assignments"]
        - expert_major["expert_major_payload_loads"]
    )
    assert expert_major["reader_requested_bytes"] < token_major[
        "reader_requested_bytes"
    ]
    assert expert_major["evaluated_routed_experts"]
    assert expert_major["evaluated_routed_k"]

    unused_output = tmp_path / "unused.json"
    unused = subprocess.run(
        [
            *common,
            "--speculative-mode",
            "scripted-reference",
            "--speculative-block-size",
            "2",
            "--speculative-script",
            f"{script};{tokens[5]}:",
            "--json",
            str(unused_output),
        ],
        capture_output=True,
        text=True,
    )
    assert unused.returncode == 4
    assert unused.stderr.strip() == "INVALID_STATE: unused scripted draft proposals"
    assert not unused_output.exists()


def test_cuda_expert_major_speculation_preserves_exact_state_and_unions_h2d(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA expert-major parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    common = [
        str(runner),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--diagnostics", "true",
        "--backend", "cuda-custom",
        "--dense-precision", "fp32",
        "--cuda-boundary", "ffn-block",
        "--cuda-allocation", "reused",
        "--cuda-weights", "transient",
        "--cuda-transfer", "synchronous",
        "--cuda-moe-fusion", "none",
    ]

    def run(name: str, extra: list[str]) -> dict:
        output = tmp_path / f"{name}.json"
        subprocess.run([*common, *extra, "--json", str(output)], check=True)
        return json.loads(output.read_text(encoding="utf-8"))

    greedy = run("cuda-greedy", [])
    tokens = greedy["token_ids"]
    perfect = f"{tokens[0]}:{tokens[1]},{tokens[2]};{tokens[3]}:{tokens[4]}"
    mixed = (
        f"{tokens[0]}:{tokens[1] ^ 1},{tokens[2]};"
        f"{tokens[1]}:;"
        f"{tokens[2]}:{tokens[3]},{tokens[4] ^ 1};"
        f"{tokens[4]}:"
    )
    for case, script in (("perfect", perfect), ("mixed", mixed)):
        speculative = [
            "--speculative-mode", "scripted-reference",
            "--speculative-block-size", "2",
            "--speculative-script", script,
        ]
        token_major = run(f"cuda-token-{case}", speculative)
        expert_major = run(
            f"cuda-expert-{case}",
            [
                *speculative,
                "--speculative-verification", "expert-major",
            ],
        )
        assert expert_major["token_ids"] == greedy["token_ids"]
        assert expert_major["final_state"] == greedy["final_state"]
        assert expert_major["routed_experts"] == greedy["routed_experts"]
        assert expert_major["routed_k"] == greedy["routed_k"]
        assert expert_major["speculative_verification"] == "expert-major"
        assert token_major["speculative_verification"] == "token-major"
        assert expert_major["speculative_accepted_draft_tokens"] == token_major[
            "speculative_accepted_draft_tokens"
        ]
        assert expert_major["speculative_committed_tokens"] == token_major[
            "speculative_committed_tokens"
        ]
        assert expert_major["batched_expert_ffn_calls"] == expert_major[
            "expert_major_payload_loads"
        ]
        assert expert_major["batched_expert_ffn_tokens"] == expert_major[
            "expert_major_assignments"
        ]
        assert token_major["batched_expert_ffn_calls"] == 0
        assert token_major["batched_expert_ffn_tokens"] == 0
        if case == "perfect":
            assert expert_major["weight_h2d_bytes"] < token_major[
                "weight_h2d_bytes"
            ]

    rejected_output = tmp_path / "unsupported-cuda-expert-major.json"
    rejected = subprocess.run(
        [
            str(runner),
            "--speculative-mode", "scripted-reference",
            "--speculative-block-size", "2",
            "--speculative-verification", "expert-major",
            "--backend", "cuda-custom",
            "--json", str(rejected_output),
        ],
        capture_output=True,
        text=True,
    )
    assert rejected.returncode == 2
    assert rejected.stderr.strip() == (
        "CUDA expert-major verification requires ffn-block boundary"
    )
    assert not rejected_output.exists()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--speculative-mode", "warp"], "unknown speculative mode: warp"),
        (
            ["--aurora-draft-backend", "cuda-custom"],
            "speculative mode none does not accept speculative options",
        ),
        (
            ["--speculative-mode", "none",
             "--aurora-draft-resident-bytes", "1"],
            "speculative mode none does not accept speculative options",
        ),
        (
            ["--speculative-mode", "aurora-replay",
             "--aurora-draft-resident-bytes", "1"],
            "AURORA draft residency requires persistent cuda-custom draft backend",
        ),
        (
            ["--speculative-mode", "aurora-persistent",
             "--aurora-draft-resident-bytes", "1"],
            "AURORA draft residency requires persistent cuda-custom draft backend",
        ),
        (
            ["--aurora-draft-resident-bytes", "-1"],
            "invalid AURORA draft resident byte capacity: -1",
        ),
        (
            [
                "--speculative-mode", "aurora-replay",
                "--speculative-block-size", "2",
                "--aurora-draft-k", "4",
                "--aurora-draft-backend", "cuda-custom",
            ],
            "AURORA replay requires CPU draft backend",
        ),
        (
            [
                "--speculative-mode", "aurora-persistent",
                "--speculative-block-size", "2",
                "--aurora-draft-k", "4",
                "--aurora-draft-backend", "warp",
            ],
            "unknown AURORA draft backend: warp",
        ),
        (
            ["--speculative-mode", "scripted-reference"],
            "scripted-reference speculation requires a positive block size",
        ),
        (
            ["--speculative-block-size", "2"],
            "speculative mode none requires block size 0 and an empty script",
        ),
        (
            ["--speculative-script", "1:2"],
            "speculative mode none requires block size 0 and an empty script",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--mode",
                "full",
            ],
            "scripted-reference speculation requires incremental mode",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-script",
                "1",
            ],
            "invalid speculative script record: 1",
        ),
        (
            ["--speculative-verification", "warp"],
            "unknown speculative verification mode: warp",
        ),
        (
            ["--speculative-verification", "expert-major"],
            "expert-major verification requires scripted-reference or AURORA replay speculation",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-verification",
                "expert-major",
                "--backend",
                "cuda-dense",
            ],
            "expert-major verification requires CPU or cuda-custom backend",
        ),
        (
            [
                "--speculative-mode", "scripted-reference",
                "--speculative-block-size", "2",
                "--speculative-verification", "expert-major",
                "--backend", "cuda-custom",
                "--cuda-boundary", "ffn-block",
            ],
            "CUDA expert-major verification requires reused allocation",
        ),
        (
            [
                "--speculative-mode", "scripted-reference",
                "--speculative-block-size", "2",
                "--speculative-verification", "expert-major",
                "--backend", "cuda-custom",
                "--cuda-boundary", "ffn-block",
                "--cuda-allocation", "reused",
                "--cuda-weights", "resident",
                "--cuda-resident-bytes", "65536",
            ],
            "CUDA expert-major verification requires transient weights",
        ),
        (
            [
                "--speculative-mode", "scripted-reference",
                "--speculative-block-size", "2",
                "--speculative-verification", "expert-major",
                "--backend", "cuda-custom",
                "--cuda-boundary", "ffn-block",
                "--cuda-allocation", "reused",
                "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1048576",
            ],
            "CUDA expert-major verification requires synchronous transfer",
        ),
        (
            [
                "--speculative-mode", "scripted-reference",
                "--speculative-block-size", "2",
                "--speculative-verification", "expert-major",
                "--backend", "cuda-custom",
                "--cuda-boundary", "ffn-block",
                "--cuda-allocation", "reused",
                "--cuda-moe-fusion", "routed-accumulate",
            ],
            "CUDA expert-major verification requires CUDA MoE fusion none",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-verification",
                "expert-major",
                "--l1-expert-cache",
                "static",
                "--l1-expert-cache-bytes",
                "65536",
            ],
            "expert-major verification requires disabled L1 expert cache",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-verification",
                "expert-major",
                "--l2-schedule",
                "deadline",
            ],
            "expert-major verification requires blocking L2 scheduling",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-verification",
                "expert-major",
                "--routing-mode",
                "fixed",
                "--routing-fixed-k",
                "4",
            ],
            "expert-major verification requires natural routing",
        ),
        (
            [
                "--speculative-mode",
                "scripted-reference",
                "--speculative-block-size",
                "2",
                "--speculative-verification",
                "expert-major",
                "--runtime-metadata",
                "TASK=coding",
            ],
            "expert-major verification does not support runtime profiles",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_speculative_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_static_l1_cache_preserves_cpu_graph_and_avoids_reader_calls(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    cases = (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
        ("lru", "lru", 3264),
        ("lfu", "lfu", 3264),
        ("least-stale", "least-stale", 3264),
    )
    for name, cache_mode, capacity in cases:
        output = tmp_path / f"cpu-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    policies = [results[name] for name in ("lru", "lfu", "least-stale")]
    assert static["token_ids"] == disabled["token_ids"] == [43, 32, 28, 49, 9, 28]
    assert static["prefill_routed_experts"] == disabled["prefill_routed_experts"]
    np.testing.assert_allclose(static["prefill_logits"], disabled["prefill_logits"])
    assert disabled["l1_expert_cache_hits"] == 0
    assert disabled["l1_expert_cache_misses"] == 0
    assert disabled["l1_expert_cache_evictions"] == 0
    assert disabled["l1_expert_cache_collision_misses"] == 0
    assert disabled["l1_expert_cache_resident_bytes"] == 0
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert static["l1_expert_cache_evictions"] == 0
    assert static["l1_expert_cache_collision_misses"] == 0
    assert 0 < static["l1_expert_cache_resident_bytes"] <= 65536
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert static["reader_read_calls"] == static["read_calls"]
    assert static["reader_completed_bytes"] == static["read_bytes"]
    assert static["reader_requested_bytes"] >= static["reader_completed_bytes"]
    assert tiny["token_ids"] == disabled["token_ids"]
    assert tiny["prefill_routed_experts"] == disabled["prefill_routed_experts"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_misses"] > 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["l1_expert_cache_resident_bytes"] == 0
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]
    for policy in policies:
        assert policy["token_ids"] == disabled["token_ids"]
        assert policy["prefill_routed_experts"] == disabled["prefill_routed_experts"]
        np.testing.assert_allclose(policy["prefill_logits"], disabled["prefill_logits"])
        assert policy["l1_expert_cache_misses"] > 0
        assert policy["l1_expert_cache_bypasses"] == 0
        assert policy["l1_expert_cache_evictions"] > 0
        assert 0 <= policy["l1_expert_cache_collision_misses"] <= policy[
            "l1_expert_cache_misses"
        ]
        assert policy["l1_expert_cache_resident_bytes"] <= 3264


def test_runtime_session_reuses_l1_experts_across_generations(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(cpp_binary("test_model_session")), str(artifact)],
        check=True,
    )


def test_runtime_profile_metadata_is_not_prompt_and_round_trips_session(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    baseline_output = tmp_path / "baseline.json"
    profiled_output = tmp_path / "profiled.json"
    resumed_output = tmp_path / "resumed.json"
    first_profile = tmp_path / "first.k3xp"
    resumed_profile = tmp_path / "resumed.k3xp"
    convert(synthetic_source, artifact, chunk_bytes=257)

    common = [
        str(cpp_binary("k3x_run")),
        "--model",
        str(artifact),
        "--prompt-ids",
        "1,7,3,9",
        "--generate",
        "6",
        "--mode",
        "incremental",
        "--diagnostics",
        "true",
    ]
    subprocess.run([*common, "--json", str(baseline_output)], check=True)
    subprocess.run(
        [
            *common,
            "--l1-expert-cache",
            "profiled",
            "--l1-expert-cache-bytes",
            "13056",
            "--profile-prior-strength",
            "4",
            "--runtime-metadata",
            "TASK=coding,LANG=cpp,PHASE=debug,REPO=k3x",
            "--runtime-profile-out",
            str(first_profile),
            "--json",
            str(profiled_output),
        ],
        check=True,
    )
    subprocess.run(
        [
            *common,
            "--l1-expert-cache",
            "profiled",
            "--l1-expert-cache-bytes",
            "13056",
            "--profile-prior-strength",
            "4",
            "--runtime-profile-in",
            str(first_profile),
            "--runtime-metadata",
            "PHASE=test",
            "--runtime-profile-out",
            str(resumed_profile),
            "--json",
            str(resumed_output),
        ],
        check=True,
    )

    baseline = json.loads(baseline_output.read_text(encoding="utf-8"))
    profiled = json.loads(profiled_output.read_text(encoding="utf-8"))
    resumed = json.loads(resumed_output.read_text(encoding="utf-8"))
    assert baseline["runtime_profile_live_observations"] == 0
    for payload in (profiled, resumed):
        assert payload["token_ids"] == baseline["token_ids"]
        assert payload["prefill_routed_experts"] == baseline["prefill_routed_experts"]
        assert payload["prefill_logits"] == baseline["prefill_logits"]
        assert payload["prefill_state"] == baseline["prefill_state"]
    assert profiled["runtime_profile_metadata_count"] == 4
    assert profiled["runtime_profile_prior_weight"] == 0.0
    assert profiled["runtime_profile_live_observations"] > 0
    assert profiled["runtime_profile_save_bytes"] == first_profile.stat().st_size
    assert resumed["runtime_profile_metadata_count"] == 4
    assert 0.0 < resumed["runtime_profile_prior_weight"] < 1.0
    assert resumed["runtime_profile_load_bytes"] == first_profile.stat().st_size
    assert resumed["runtime_profile_save_bytes"] == resumed_profile.stat().st_size


@pytest.mark.parametrize(
    ("cache_mode", "capacity"),
    [("static", 65536), ("lru", 3264), ("lfu", 3264), ("least-stale", 3264)],
)
def test_deadline_expert_schedule_preserves_exact_runtime_contract(
    synthetic_source: Path, tmp_path: Path, cache_mode: str, capacity: int
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    records: dict[str, dict] = {}
    for schedule in ("blocking", "deadline"):
        output = tmp_path / f"{cache_mode}-{schedule}.json"
        subprocess.run(
            [
                str(cpp_binary("k3x_run")),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--l2-schedule", schedule,
                "--json", str(output),
            ],
            check=True,
        )
        records[schedule] = json.loads(output.read_text(encoding="utf-8"))

    blocking = records["blocking"]
    deadline = records["deadline"]
    for field in (
        "token_ids",
        "prefill_routed_experts",
        "l1_expert_cache_hits",
        "l1_expert_cache_misses",
        "reader_read_calls",
        "reader_requested_bytes",
        "reader_completed_bytes",
    ):
        assert deadline[field] == blocking[field]
    assert blocking["l2_expert_schedule"] == "blocking"
    assert blocking["expert_load_submissions"] == 0
    assert blocking["expert_load_completions"] == 0
    assert deadline["l2_expert_schedule"] == "deadline"
    assert deadline["expert_load_submissions"] > 0
    assert deadline["expert_load_inline_resident_hits"] == deadline[
        "l1_expert_cache_hits"
    ]
    assert deadline["expert_load_completions"] == deadline["expert_load_submissions"]
    assert (
        deadline["expert_load_ready_before_use"]
        + deadline["expert_load_late_at_use"]
        == deadline["expert_load_submissions"]
    )
    assert deadline["expert_load_requested_bytes"] > 0
    assert deadline["expert_load_queue_high_water"] > 0
    assert deadline["expert_load_worker_nanoseconds"] > 0
    assert deadline["expert_load_exposed_wait_nanoseconds"] >= 0


@pytest.mark.skipif(
    os.environ.get("K3X_TEST_IO_URING") != "1",
    reason="requires the optional liburing build",
)
def test_io_uring_batches_exact_experts_and_reuses_session_cache(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "io-uring-session.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [
            str(cpp_binary("test_model_session")),
            str(artifact),
            "io-uring",
        ],
        check=True,
    )


@pytest.mark.skipif(
    os.environ.get("K3X_TEST_DIRECT") != "1",
    reason="requires a Linux filesystem with STATX_DIOALIGN",
)
@pytest.mark.parametrize("mode", ["direct", "io-uring-direct"])
def test_direct_modes_preserve_exact_experts_and_session_cache(
    synthetic_source: Path, tmp_path: Path, mode: str
) -> None:
    if mode == "io-uring-direct" and os.environ.get("K3X_TEST_IO_URING") != "1":
        pytest.skip("requires the optional liburing build")
    artifact = tmp_path / f"{mode}-session.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(cpp_binary("test_model_session")), str(artifact), mode],
        check=True,
    )


def test_cpp_prefill_layers_logits_and_state_match_python(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--diagnostics", "true",
         "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    model = build_synthetic_model()
    expected_logits, expected_state, expected_layers = model.prefill_with_trace(
        torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    )
    np.testing.assert_allclose(
        result["prefill_logits"], expected_logits.numpy().reshape(-1), atol=1e-6, rtol=1e-6
    )
    for actual, expected in zip(
        result["prefill_layer_outputs"], expected_layers, strict=True
    ):
        np.testing.assert_allclose(
            actual, expected.numpy().reshape(-1), atol=1e-6, rtol=1e-6
        )
    state_values: list[np.ndarray] = []
    for state in expected_state.attention:
        tensors = (
            (state.conv_q, state.conv_k, state.conv_v, state.recurrent)
            if hasattr(state, "recurrent")
            else (state.keys, state.values, state.shared_keys)
        )
        state_values.extend(tensor.numpy().reshape(-1) for tensor in tensors)
    np.testing.assert_allclose(
        result["prefill_state"], np.concatenate(state_values), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    (
        "backend",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_resident_bytes",
        "tolerance",
    ),
    [
        (backend, "fp32", allocation, weights, batching,
         8 * 1024 * 1024 if weights == "resident" else 0,
         1e-5 if backend == "cuda-dense" else 1e-4)
        for backend in ("cuda-dense", "cuda-custom")
        for allocation in ("per-operation", "reused")
        for weights in ("transient", "resident")
        for batching in ("scalar", "grouped")
    ]
    + [
        ("cuda-dense", "bf16", "reused", "resident", "grouped",
         8 * 1024 * 1024, 2e-2),
        ("cuda-custom", "bf16", "reused", "resident", "grouped",
         8 * 1024 * 1024, 2e-2),
    ],
)
def test_cuda_backends_match_synthetic_graph_and_tokens(
    synthetic_source: Path,
    tmp_path: Path,
    backend: str,
    dense_precision: str,
    cuda_allocation: str,
    cuda_weights: str,
    cuda_batching: str,
    cuda_resident_bytes: int,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / (
        f"{backend}-{dense_precision}-{cuda_allocation}-"
        f"{cuda_weights}-{cuda_batching}.json"
    )
    convert(synthetic_source, artifact, chunk_bytes=257)
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
            "--diagnostics",
            "true",
            "--backend",
            backend,
            "--dense-precision",
            dense_precision,
            "--cuda-allocation",
            cuda_allocation,
            "--cuda-weights",
            cuda_weights,
            "--cuda-batching",
            cuda_batching,
            "--cuda-resident-bytes",
            str(cuda_resident_bytes),
            "--json",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    model = build_synthetic_model()
    expected_tokens = model.generate_greedy([1, 7, 3, 9], 6, True)
    expected_logits, expected_state, expected_layers = model.prefill_with_trace(
        torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    )
    assert result["token_ids"] == expected_tokens == [43, 32, 28, 49, 9, 28]
    assert result["backend"] == backend
    assert result["device"] != "CPU"
    assert result["dense_precision"] == dense_precision
    assert result["cuda_allocation"] == cuda_allocation
    assert result["cuda_weights"] == cuda_weights
    assert result["cuda_batching"] == cuda_batching
    assert result["cuda_resident_bytes"] == cuda_resident_bytes
    assert result["kernel_nanoseconds"] > 0
    assert result["host_to_device_bytes"] > 0
    assert result["device_to_host_bytes"] > 0
    assert result["peak_vram_bytes"] > 0
    assert result["failed_operations"] == 0
    if cuda_batching == "grouped":
        assert result["grouped_projection_calls"] > 0
        assert result["grouped_projection_members"] > result["grouped_projection_calls"]
    if cuda_weights == "resident":
        assert result["weight_cache_misses"] > 0
        assert result["resident_weight_bytes"] <= cuda_resident_bytes
    np.testing.assert_allclose(
        result["prefill_logits"],
        expected_logits.numpy().reshape(-1),
        atol=tolerance,
        rtol=tolerance,
    )
    for actual, expected in zip(
        result["prefill_layer_outputs"], expected_layers, strict=True
    ):
        np.testing.assert_allclose(
            actual,
            expected.numpy().reshape(-1),
            atol=tolerance,
            rtol=tolerance,
        )
    state_values: list[np.ndarray] = []
    for state in expected_state.attention:
        tensors = (
            (state.conv_q, state.conv_k, state.conv_v, state.recurrent)
            if hasattr(state, "recurrent")
            else (state.keys, state.values, state.shared_keys)
        )
        state_values.extend(tensor.numpy().reshape(-1) for tensor in tensors)
    np.testing.assert_allclose(
        result["prefill_state"],
        np.concatenate(state_values),
        atol=tolerance,
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    ("dense_precision", "cuda_batching", "tolerance"),
    [
        ("fp32", "scalar", 1e-4),
        ("fp32", "grouped", 1e-4),
        ("bf16", "scalar", 2e-2),
        ("bf16", "grouped", 2e-2),
    ],
)
def test_cuda_ffn_block_matches_operation_graph_and_routing(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    cuda_batching: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA FFN parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)

    results: dict[str, dict] = {}
    cases = (
        (
            "operation", "operation", "resident", 8 * 1024 * 1024,
            "synchronous", 0, "none",
        ),
        (
            "synchronous", "ffn-block", "transient", 0,
            "synchronous", 0, "none",
        ),
        (
            "prefetch", "ffn-block", "transient", 0,
            "prefetch", 1024 * 1024, "none",
        ),
        (
            "fused-synchronous", "ffn-block", "transient", 0,
            "synchronous", 0, "routed-accumulate",
        ),
        (
            "fused-prefetch", "ffn-block", "transient", 0,
            "prefetch", 1024 * 1024, "routed-accumulate",
        ),
    )
    for (
        name, boundary, weights, resident_bytes, transfer, pinned_bytes, fusion
    ) in cases:
        output = tmp_path / f"{name}-{dense_precision}-{cuda_batching}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", weights,
                "--cuda-batching", cuda_batching,
                "--cuda-boundary", boundary,
                "--cuda-resident-bytes", str(resident_bytes),
                "--cuda-transfer", transfer,
                "--cuda-pinned-bytes", str(pinned_bytes),
                "--cuda-moe-fusion", fusion,
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    reference = results["operation"]
    synchronous = results["synchronous"]
    prefetch = results["prefetch"]
    fused_synchronous = results["fused-synchronous"]
    fused_prefetch = results["fused-prefetch"]
    for name, candidate in (
        ("synchronous", synchronous),
        ("prefetch", prefetch),
        ("synchronous", fused_synchronous),
        ("prefetch", fused_prefetch),
    ):
        assert candidate["cuda_boundary"] == "ffn-block"
        assert candidate["cuda_transfer"] == name
        assert candidate["token_ids"] == reference["token_ids"] == [43, 32, 28, 49, 9, 28]
        assert candidate["prefill_routed_experts"] == reference["prefill_routed_experts"]
        assert candidate["ffn_block_calls"] > 0
        assert candidate["ffn_block_experts"] > 0
        np.testing.assert_allclose(
            candidate["prefill_logits"], reference["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        for actual, expected in zip(
            candidate["prefill_layer_outputs"],
            reference["prefill_layer_outputs"],
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, expected, atol=tolerance, rtol=tolerance
            )
        np.testing.assert_allclose(
            candidate["prefill_state"], reference["prefill_state"],
            atol=tolerance, rtol=tolerance,
        )
    assert synchronous["cuda_pinned_bytes"] == 0
    assert synchronous["pinned_host_bytes"] == 0
    assert synchronous["async_prefetch_calls"] == 0
    assert synchronous["async_prefetch_bytes"] == 0
    assert synchronous["transfer_stream_wait_count"] == 0
    assert prefetch["cuda_pinned_bytes"] == 1024 * 1024
    assert prefetch["pinned_host_bytes"] == 1024 * 1024
    assert prefetch["async_prefetch_calls"] > 0
    assert prefetch["async_prefetch_bytes"] > 0
    assert prefetch["transfer_stream_wait_count"] == prefetch["async_prefetch_calls"]
    assert prefetch["stream_synchronization_count"] <= synchronous[
        "stream_synchronization_count"
    ]
    assert reference["cuda_moe_fusion"] == "none"
    assert synchronous["fused_moe_calls"] == 0
    assert prefetch["fused_moe_calls"] == 0
    for candidate, unfused in (
        (fused_synchronous, synchronous),
        (fused_prefetch, prefetch),
    ):
        assert candidate["cuda_moe_fusion"] == "routed-accumulate"
        assert 0 < candidate["fused_moe_calls"] < candidate["ffn_block_calls"]
        assert candidate["fused_moe_experts"] == candidate["ffn_block_experts"]
        assert candidate["device_to_host_bytes"] < unfused[
            "device_to_host_bytes"
        ]


@pytest.mark.parametrize(
    ("dense_precision", "tolerance"), [("fp32", 1e-4), ("bf16", 2e-2)]
)
def test_static_l1_cache_preserves_cuda_ffn_block_and_exact_bypass(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA L1 parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    for name, cache_mode, capacity in (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
    ):
        output = tmp_path / f"cuda-{dense_precision}-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", "transient",
                "--cuda-batching", "scalar",
                "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "synchronous",
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    for candidate in (static, tiny):
        assert candidate["token_ids"] == disabled["token_ids"]
        assert candidate["prefill_routed_experts"] == disabled[
            "prefill_routed_experts"
        ]
        np.testing.assert_allclose(
            candidate["prefill_logits"], disabled["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        assert candidate["host_to_device_bytes"] == disabled[
            "host_to_device_bytes"
        ]
        assert candidate["ffn_block_calls"] == disabled["ffn_block_calls"]
        assert candidate["ffn_block_experts"] == disabled["ffn_block_experts"]
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]


@pytest.mark.parametrize(
    ("dense_precision", "cuda_batching", "tolerance"),
    [
        ("fp32", "scalar", 1e-4),
        ("fp32", "grouped", 1e-4),
        ("bf16", "scalar", 2e-2),
        ("bf16", "grouped", 2e-2),
    ],
)
def test_static_l1_cache_feeds_exact_cuda_prefetch(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    cuda_batching: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA L1 prefetch parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    for name, cache_mode, capacity in (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
    ):
        output = tmp_path / f"prefetch-{dense_precision}-{cuda_batching}-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", "transient",
                "--cuda-batching", cuda_batching,
                "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", str(1024 * 1024),
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    for candidate in (static, tiny):
        assert candidate["token_ids"] == disabled["token_ids"]
        assert candidate["prefill_routed_experts"] == disabled[
            "prefill_routed_experts"
        ]
        np.testing.assert_allclose(
            candidate["prefill_logits"], disabled["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        for field in (
            "host_to_device_bytes",
            "weight_h2d_bytes",
            "activation_h2d_bytes",
            "async_prefetch_calls",
            "async_prefetch_bytes",
            "transfer_stream_wait_count",
            "ffn_block_calls",
            "ffn_block_experts",
        ):
            assert candidate[field] == disabled[field]
        assert candidate["async_prefetch_calls"] > 0
        assert candidate["transfer_stream_wait_count"] == candidate[
            "async_prefetch_calls"
        ]
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]


def test_cpp_runner_rejects_corrupt_model_before_generation(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    valid = tmp_path / "valid.k3x"
    corrupt = tmp_path / "corrupt.k3x"
    convert(synthetic_source, valid, chunk_bytes=257)
    shutil.copyfile(valid, corrupt)
    first = K3XReader.open(valid).tensor_records[0]
    with corrupt.open("r+b") as stream:
        stream.seek(first.data_offset)
        value = stream.read(1)
        stream.seek(first.data_offset)
        stream.write(bytes([value[0] ^ 1]))
    result = subprocess.run(
        [str(runner), "--model", str(corrupt), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--json", str(tmp_path / "x.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert result.stderr.strip() == "DATA_CRC_MISMATCH"


def test_cpp_first_generated_token_is_not_counted_as_decode(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["token_ids"] == [43]
    assert result["decode_nanoseconds"] == 0
