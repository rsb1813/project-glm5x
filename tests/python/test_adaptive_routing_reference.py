# Top-16 합성 fixture와 PyTorch adaptive routing oracle을 검증합니다.
import hashlib
from pathlib import Path

import torch

from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import build_synthetic_model, write_source_checkpoint
from k3x_ref.model import SyntheticK3Model
from k3x_ref.routing_policy import RoutingMode, RoutingPolicyConfig, select_routing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_explicit_default_config_keeps_source_artifact_identical(tmp_path: Path) -> None:
    implicit = tmp_path / "implicit"
    explicit = tmp_path / "explicit"
    implicit_manifest = write_source_checkpoint(implicit)
    explicit_manifest = write_source_checkpoint(
        explicit, config=SyntheticK3Config.default()
    )

    assert implicit_manifest == explicit_manifest
    for shard in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        assert _sha256(implicit / shard) == _sha256(explicit / shard)


def test_top16_fixture_materializes_24_experts(tmp_path: Path) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    model = build_synthetic_model(config=config)
    manifest = write_source_checkpoint(tmp_path / "top16", config=config)

    assert model.cfg == config
    assert manifest["config"]["num_experts"] == 24
    assert manifest["config"]["top_k"] == 16
    for layer in model.weights.layers[1:]:
        assert len(layer.feed_forward.experts) == 24


def test_torch_policy_uses_mass_entropy_boundary_and_quality_floor() -> None:
    scores = torch.full((16,), 1.0e-9, dtype=torch.float32)
    scores[:8] = 1.0
    bias = torch.zeros_like(scores)

    adaptive = select_routing(
        scores,
        bias,
        natural_top_k=16,
        config=RoutingPolicyConfig(
            mode=RoutingMode.ADAPTIVE,
            mass_target=0.75,
            minimum_boundary_gap=0.1,
        ),
    )
    assert adaptive.selected_k == 8
    assert adaptive.expert_ids.tolist() == list(range(8))
    assert adaptive.entropy_effective_support <= 8.0001
    assert adaptive.boundary_confidence > 0.9
    assert torch.allclose(adaptive.normalized_weights.sum(), torch.tensor(1.0))

    escalated = select_routing(
        scores,
        bias,
        natural_top_k=16,
        config=RoutingPolicyConfig(
            mode=RoutingMode.ADAPTIVE,
            mass_target=0.75,
            quality_floor_k=12,
        ),
    )
    assert escalated.selected_k == 12
    assert escalated.quality_floor_escalated


def test_torch_policy_keeps_bias_out_of_contribution_weights() -> None:
    scores = torch.tensor([0.8, 0.8, 0.4, 0.7, 0.2, 0.1])
    bias = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    decision = select_routing(
        scores,
        bias,
        natural_top_k=4,
        config=RoutingPolicyConfig(mode=RoutingMode.NATURAL),
    )

    assert decision.full_order.tolist() == [2, 0, 1, 3, 4, 5]
    assert decision.expert_ids.tolist() == [2, 0, 1, 3]
    assert torch.allclose(
        decision.normalized_weights,
        torch.tensor([0.4, 0.8, 0.8, 0.7]) / 2.7,
        atol=1.0e-6,
    )


def test_natural_policy_ignores_an_external_floor_above_checkpoint_k() -> None:
    scores = torch.tensor([0.8, 0.7, 0.6, 0.5], dtype=torch.float32)
    decision = select_routing(
        scores,
        torch.zeros_like(scores),
        natural_top_k=2,
        config=RoutingPolicyConfig(
            mode=RoutingMode.NATURAL,
            quality_floor_k=16,
        ),
    )
    assert decision.selected_k == 2
    assert not decision.quality_floor_escalated


def test_top16_reference_runs_fixed_and_adaptive_end_to_end() -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    natural = build_synthetic_model(config=config)
    fixed16 = SyntheticK3Model(
        config,
        natural.weights,
        routing_policy=RoutingPolicyConfig(mode=RoutingMode.FIXED, fixed_k=16),
    )
    fixed4 = SyntheticK3Model(
        config,
        natural.weights,
        routing_policy=RoutingPolicyConfig(mode=RoutingMode.FIXED, fixed_k=4),
    )
    adaptive = SyntheticK3Model(
        config,
        natural.weights,
        routing_policy=RoutingPolicyConfig(
            mode=RoutingMode.ADAPTIVE,
            mass_target=0.75,
        ),
    )
    prompt = torch.tensor([[1, 7, 3, 9]], dtype=torch.long)

    natural_logits, natural_state, natural_layers = natural.prefill_with_trace(prompt)
    fixed16_logits, fixed16_state, fixed16_layers = fixed16.prefill_with_trace(prompt)
    fixed4_logits, _, _ = fixed4.prefill_with_trace(prompt)
    adaptive_logits, _, _ = adaptive.prefill_with_trace(prompt)

    assert torch.equal(fixed16_logits, natural_logits)
    assert all(
        torch.equal(left, right)
        for left, right in zip(fixed16_layers, natural_layers, strict=True)
    )
    assert fixed16.state_sha256(fixed16_state) == natural.state_sha256(natural_state)
    assert not torch.equal(fixed4_logits, natural_logits)
    assert torch.isfinite(adaptive_logits).all()
