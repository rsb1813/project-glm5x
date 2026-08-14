# PyTorch 기준으로 자연 및 축소 Top-K routing 결정을 계산합니다.
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch


class RoutingMode(str, Enum):
    NATURAL = "natural"
    FIXED = "fixed"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class RoutingPolicyConfig:
    mode: RoutingMode = RoutingMode.NATURAL
    fixed_k: int = 0
    mass_target: float = 0.9
    minimum_boundary_gap: float = 0.0
    quality_floor_k: int = 0


@dataclass(frozen=True)
class RoutingDecision:
    full_order: torch.Tensor
    expert_ids: torch.Tensor
    normalized_weights: torch.Tensor
    natural_top_k: int
    selected_k: int
    normalized_entropy: float
    entropy_effective_support: float
    selected_cumulative_mass: float
    boundary_confidence: float
    quality_floor_escalated: bool


_K_LADDER = (4, 6, 8, 12, 16)
_COMPARISON_EPSILON = 1.0e-5


def _boundary_confidence(probabilities: torch.Tensor, selected_k: int) -> float:
    if selected_k >= probabilities.numel():
        return 1.0
    gap = max(
        0.0,
        float(probabilities[selected_k - 1] - probabilities[selected_k]),
    )
    return gap / max(float(probabilities[0]), torch.finfo(torch.float32).tiny)


def validate_routing_policy(
    natural_top_k: int,
    expert_count: int,
    config: RoutingPolicyConfig,
) -> None:
    if not 0 < natural_top_k <= expert_count:
        raise ValueError("invalid routing dimensions")
    if (
        not math.isfinite(config.mass_target)
        or not 0.0 < config.mass_target <= 1.0
        or not math.isfinite(config.minimum_boundary_gap)
        or not 0.0 <= config.minimum_boundary_gap <= 1.0
    ):
        raise ValueError("invalid adaptive routing threshold")
    if config.quality_floor_k and (
        config.quality_floor_k not in _K_LADDER
        or (
            config.mode is not RoutingMode.NATURAL
            and config.quality_floor_k > natural_top_k
        )
    ):
        raise ValueError("invalid routing quality floor")
    if config.mode is RoutingMode.FIXED and (
        config.fixed_k not in _K_LADDER or config.fixed_k > natural_top_k
    ):
        raise ValueError("invalid fixed routing K")
    if config.mode is RoutingMode.ADAPTIVE and natural_top_k != 16:
        raise ValueError("adaptive routing requires natural Top-16")


def select_routing(
    scores: torch.Tensor,
    correction_bias: torch.Tensor,
    natural_top_k: int,
    config: RoutingPolicyConfig,
) -> RoutingDecision:
    if (
        scores.ndim != 1
        or correction_bias.ndim != 1
        or scores.shape != correction_bias.shape
        or scores.numel() == 0
        or not 0 < natural_top_k <= scores.numel()
    ):
        raise ValueError("invalid routing dimensions")
    scores = scores.to(torch.float32)
    correction_bias = correction_bias.to(device=scores.device, dtype=torch.float32)
    if (
        not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(correction_bias).all())
        or bool((scores < 0.0).any())
        or bool((scores > 1.0).any())
    ):
        raise ValueError("invalid routing score")
    validate_routing_policy(natural_top_k, scores.numel(), config)

    adjusted = scores + correction_bias
    if not bool(torch.isfinite(adjusted).all()):
        raise ValueError("invalid adjusted routing score")
    full_order = torch.argsort(adjusted, descending=True, stable=True)
    natural_scores = scores[full_order[:natural_top_k]]
    natural_denominator = natural_scores.sum()
    if not bool(torch.isfinite(natural_denominator)) or float(natural_denominator) <= 0:
        raise ValueError("zero natural routing mass")
    probabilities = natural_scores / natural_denominator
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * positive.log()).sum())
    normalized_entropy = (
        0.0 if natural_top_k == 1 else entropy / math.log(natural_top_k)
    )
    effective_support = math.exp(entropy)

    selected_k = natural_top_k
    unfloored_selected_k = natural_top_k
    if config.mode is RoutingMode.FIXED:
        unfloored_selected_k = config.fixed_k
        selected_k = max(config.fixed_k, config.quality_floor_k)
    elif config.mode is RoutingMode.ADAPTIVE:
        def choose_adaptive(quality_floor: int) -> int:
            cumulative = 0.0
            for slot, probability in enumerate(probabilities):
                cumulative += float(probability)
                candidate = slot + 1
                if candidate not in _K_LADDER or candidate < quality_floor:
                    continue
                confidence = _boundary_confidence(probabilities, candidate)
                if (
                    cumulative + _COMPARISON_EPSILON >= config.mass_target
                    and candidate + _COMPARISON_EPSILON >= effective_support
                    and confidence + _COMPARISON_EPSILON
                    >= config.minimum_boundary_gap
                ):
                    return candidate
            return natural_top_k

        unfloored_selected_k = choose_adaptive(0)
        selected_k = choose_adaptive(config.quality_floor_k)

    expert_ids = full_order[:selected_k]
    selected_scores = scores[expert_ids]
    selected_denominator = selected_scores.sum()
    if not bool(torch.isfinite(selected_denominator)) or float(selected_denominator) <= 0:
        raise ValueError("zero selected routing mass")
    return RoutingDecision(
        full_order=full_order,
        expert_ids=expert_ids,
        normalized_weights=selected_scores / selected_denominator,
        natural_top_k=natural_top_k,
        selected_k=selected_k,
        normalized_entropy=normalized_entropy,
        entropy_effective_support=effective_support,
        selected_cumulative_mass=float(probabilities[:selected_k].sum()),
        boundary_confidence=_boundary_confidence(probabilities, selected_k),
        quality_floor_escalated=selected_k > unfloored_selected_k,
    )
