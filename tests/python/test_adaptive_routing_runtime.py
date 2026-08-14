# C++ adaptive Top-K CLI와 exact cold-rescue 경로를 검증합니다.
import json
import subprocess
from pathlib import Path

import torch

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import build_synthetic_model, write_source_checkpoint
from k3x_ref.model import SyntheticK3Model
from k3x_ref.routing_policy import RoutingMode, RoutingPolicyConfig


def _top16_artifact(tmp_path: Path) -> tuple[Path, SyntheticK3Model]:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    return artifact, build_synthetic_model(config=config)


def _run(artifact: Path, output: Path, *arguments: str) -> dict:
    subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model", str(artifact),
            "--prompt-ids", "1,7,3,9",
            "--generate", "6",
            "--mode", "incremental",
            "--diagnostics", "true",
            *arguments,
            "--json", str(output),
        ],
        check=True,
    )
    return json.loads(output.read_text(encoding="utf-8"))


def test_cpp_fixed_top16_matches_natural_and_fixed4_reference(tmp_path: Path) -> None:
    artifact, natural_model = _top16_artifact(tmp_path)
    natural = _run(artifact, tmp_path / "natural.json")
    fixed16 = _run(
        artifact,
        tmp_path / "fixed16.json",
        "--routing-mode", "fixed",
        "--routing-fixed-k", "16",
    )
    fixed4 = _run(
        artifact,
        tmp_path / "fixed4.json",
        "--routing-mode", "fixed",
        "--routing-fixed-k", "4",
    )
    fixed4_model = SyntheticK3Model(
        natural_model.cfg,
        natural_model.weights,
        routing_policy=RoutingPolicyConfig(mode=RoutingMode.FIXED, fixed_k=4),
    )

    assert fixed16["token_ids"] == natural["token_ids"]
    assert fixed16["prefill_routed_experts"] == natural["prefill_routed_experts"]
    assert fixed16["prefill_logits"] == natural["prefill_logits"]
    assert fixed16["prefill_state"] == natural["prefill_state"]
    assert fixed16["prefill_routed_k"] == [16] * 12
    assert fixed4["prefill_routed_k"] == [4] * 12
    assert fixed4["routing_average_top_k"] == 4.0
    assert fixed4["token_ids"] == fixed4_model.generate_greedy(
        [1, 7, 3, 9], 6, incremental=True
    )
    assert fixed4["routing_selected_experts"] == fixed4["routing_decisions"] * 4


def test_adaptive_escalation_and_rescue_never_reorder_selected_experts(
    tmp_path: Path,
) -> None:
    artifact, _ = _top16_artifact(tmp_path)
    adaptive = _run(
        artifact,
        tmp_path / "adaptive.json",
        "--routing-mode", "adaptive",
        "--routing-mass-target", "0.75",
    )
    escalated = _run(
        artifact,
        tmp_path / "escalated.json",
        "--routing-mode", "fixed",
        "--routing-fixed-k", "4",
        "--routing-agent-failures", "2",
    )
    uncached = _run(
        artifact,
        tmp_path / "uncached.json",
        "--routing-mode", "fixed",
        "--routing-fixed-k", "4",
    )
    rescued = _run(
        artifact,
        tmp_path / "rescued.json",
        "--routing-mode", "fixed",
        "--routing-fixed-k", "4",
        "--l1-expert-cache", "lru",
        "--l1-expert-cache-bytes", "6528",
    )

    assert min(escalated["prefill_routed_k"]) >= 12
    assert escalated["routing_quality_floor_k"] == 12
    assert escalated["routing_quality_escalated_decisions"] > 0
    assert escalated["routing_average_top_k"] == 12.0
    assert rescued["prefill_routed_experts"] == uncached["prefill_routed_experts"]
    assert rescued["token_ids"] == uncached["token_ids"]
    assert rescued["cold_rescue_count"] > 0
    assert rescued["cold_rescue_count"] <= rescued["l1_expert_cache_misses"]
    assert uncached["cold_rescue_count"] == 0


def test_critical_routing_forces_natural_top16(tmp_path: Path) -> None:
    artifact, _ = _top16_artifact(tmp_path)
    critical = _run(
        artifact,
        tmp_path / "critical.json",
        "--routing-mode", "adaptive",
        "--routing-mass-target", "0.5",
        "--routing-critical", "true",
    )
    assert critical["prefill_routed_k"] == [16] * 12
    assert critical["routing_quality_floor_k"] == 16


def test_natural_reference_ignores_quality_floor_on_default_top2(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    baseline = _run(artifact, tmp_path / "baseline.json")
    critical = _run(
        artifact,
        tmp_path / "critical.json",
        "--routing-critical", "true",
    )
    assert critical["token_ids"] == baseline["token_ids"]
    assert critical["prefill_routed_experts"] == baseline["prefill_routed_experts"]
    assert critical["prefill_routed_k"] == [2] * 12
    assert critical["routing_quality_escalated_decisions"] == 0


def test_cpp_runner_rejects_invalid_routing_options() -> None:
    cases = [
        (["--routing-mode", "resident-first"], "unknown routing mode: resident-first"),
        (["--routing-mode", "fixed"], "fixed routing requires K4, K6, K8, K12, or K16"),
        (["--routing-fixed-k", "4"], "natural routing requires --routing-fixed-k 0"),
        (["--routing-mass-target", "nan"], "invalid routing mass target: nan"),
        (["--routing-min-boundary-gap", "1.1"], "invalid routing boundary gap: 1.1"),
        (["--routing-agent-failures", "-1"], "invalid routing agent failure count: -1"),
        (["--routing-critical", "yes"], "invalid routing critical flag: yes"),
    ]
    for arguments, message in cases:
        result = subprocess.run(
            [str(cpp_binary("k3x_run")), *arguments],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert result.stderr.strip() == message
