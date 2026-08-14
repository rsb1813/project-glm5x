# K3 router 선택과 Stable LatentMoE branch 결합을 검증합니다.
import torch

from k3x_ref.config import SyntheticK3Config
from k3x_ref.moe import (
    ExpertWeights,
    LatentMoEWeights,
    PackedMatrix,
    route,
    stable_latent_moe,
)


def _zero_matrix(rows: int, cols: int) -> PackedMatrix:
    return PackedMatrix(bytes(rows * cols // 2), bytes([127]) * (rows * cols // 32), rows, cols)


def test_router_bias_changes_selection_but_not_selected_weights() -> None:
    hidden = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bias = torch.tensor([0.0, 0.0, 0.9])
    got = route(hidden, weight, bias, top_k=2, routed_scale=1.0)
    assert got.expert_ids.tolist() == [[2, 0]]
    expected = torch.tensor([[0.5, 0.8807971]])
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(got.weights, expected, atol=1e-6)


def test_stable_latent_moe_keeps_shared_branch_in_hidden_space() -> None:
    cfg = SyntheticK3Config.default()
    zero_expert = ExpertWeights(
        gate=_zero_matrix(cfg.expert_intermediate_size, cfg.routed_latent_size),
        up=_zero_matrix(cfg.expert_intermediate_size, cfg.routed_latent_size),
        down=_zero_matrix(cfg.routed_latent_size, cfg.expert_intermediate_size),
    )
    shared_gate = torch.zeros(cfg.expert_intermediate_size, cfg.hidden_size)
    shared_up = torch.zeros_like(shared_gate)
    shared_down = torch.zeros(cfg.hidden_size, cfg.expert_intermediate_size)
    shared_gate[0, 0] = shared_up[0, 0] = shared_down[0, 0] = 1.0
    weights = LatentMoEWeights(
        router_weight=torch.zeros(cfg.num_experts, cfg.hidden_size),
        correction_bias=torch.zeros(cfg.num_experts),
        routed_down_proj=torch.zeros(cfg.routed_latent_size, cfg.hidden_size),
        routed_up_proj=torch.zeros(cfg.hidden_size, cfg.routed_latent_size),
        routed_norm=torch.ones(cfg.routed_latent_size),
        experts=(zero_expert,) * cfg.num_experts,
        shared_gate=shared_gate,
        shared_up=shared_up,
        shared_down=shared_down,
    )
    hidden = torch.zeros(1, 1, cfg.hidden_size)
    hidden[..., 0] = 1.0
    got = stable_latent_moe(hidden, weights, cfg)
    gate = 4.0 * torch.tanh(torch.tensor(0.25)) * torch.sigmoid(torch.tensor(1.0))
    up = 25.0 * torch.tanh(torch.tensor(0.04))
    expected_first = gate * up
    assert torch.allclose(got[..., 0], expected_first.reshape(1, 1), atol=1e-6)
    assert torch.count_nonzero(got[..., 1:]) == 0
