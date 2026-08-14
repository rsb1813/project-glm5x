# GLM-5.2 layer-10 MoE reference의 공식 라우팅과 지연 expert 로드를 검증합니다.

from __future__ import annotations

import torch
import torch.nn.functional as F

from glm5x_ref.layer10_moe import GLM5XExpertWeights, GLM5XLayer10MoEReference


def _weights(expert_id: int, *, hidden: int = 3, intermediate: int = 2) -> GLM5XExpertWeights:
    base = float(expert_id + 1)
    gate = torch.tensor(
        [[base, 0.5, -0.25], [-0.5, base * 0.25, 0.75]], dtype=torch.float32
    )[:intermediate, :hidden]
    up = torch.tensor(
        [[0.25, base, 0.5], [0.75, -0.5, base * 0.2]], dtype=torch.float32
    )[:intermediate, :hidden]
    down = torch.tensor(
        [[base, -0.25], [0.5, base * 0.1], [0.25, 0.75]], dtype=torch.float32
    )[:hidden, :intermediate]
    return GLM5XExpertWeights(gate_proj=gate, up_proj=up, down_proj=down)


def test_glm5x_moe_matches_official_router_and_shared_expert() -> None:
    hidden = torch.tensor(
        [[0.5, -1.0, 0.25], [1.0, 0.25, -0.75]], dtype=torch.float32
    )
    router_weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    correction_bias = torch.tensor([0.0, 0.1, -0.2, 0.05], dtype=torch.float32)
    shared = GLM5XExpertWeights(
        gate_proj=torch.tensor([[0.5, 0.25, -0.5], [0.75, -0.25, 0.5]]),
        up_proj=torch.tensor([[1.0, -0.5, 0.25], [0.25, 0.5, 1.0]]),
        down_proj=torch.tensor([[0.5, -0.25], [0.75, 0.5], [-0.5, 0.25]]),
    )
    model = GLM5XLayer10MoEReference(
        router_weight=router_weight,
        correction_bias=correction_bias,
        expert_loader=lambda expert_id: _weights(expert_id),
        shared_expert=shared,
        top_k=2,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
    )

    result = model(hidden)

    scores = torch.sigmoid(F.linear(hidden, router_weight))
    choice_scores = scores + correction_bias
    expected_indices = torch.topk(choice_scores, k=2, dim=-1, sorted=False).indices
    expected_weights = scores.gather(1, expected_indices)
    expected_weights = expected_weights / (expected_weights.sum(dim=-1, keepdim=True) + 1e-20)
    expected_weights = expected_weights * 2.5
    expected = torch.zeros_like(hidden)
    for token_index in range(hidden.shape[0]):
        for slot in range(2):
            expert = _weights(int(expected_indices[token_index, slot]))
            gate = F.linear(hidden[token_index], expert.gate_proj)
            up = F.linear(hidden[token_index], expert.up_proj)
            routed = F.linear(torch.nn.functional.silu(gate) * up, expert.down_proj)
            expected[token_index] += routed * expected_weights[token_index, slot]
    gate = F.linear(hidden, shared.gate_proj)
    up = F.linear(hidden, shared.up_proj)
    expected += F.linear(torch.nn.functional.silu(gate) * up, shared.down_proj)

    torch.testing.assert_close(result.topk_indices, expected_indices)
    torch.testing.assert_close(result.topk_weights, expected_weights)
    torch.testing.assert_close(result.output, expected)


def test_glm5x_moe_loads_each_selected_expert_once() -> None:
    hidden = torch.ones((3, 3), dtype=torch.float32)
    router_weight = torch.tensor(
        [[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 2.0]],
        dtype=torch.float32,
    )
    calls: list[int] = []

    def load(expert_id: int) -> GLM5XExpertWeights:
        calls.append(expert_id)
        return _weights(expert_id)

    model = GLM5XLayer10MoEReference(
        router_weight=router_weight,
        correction_bias=torch.zeros(4),
        expert_loader=load,
        shared_expert=_weights(0),
        top_k=2,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
    )
    result = model(hidden)

    assert sorted(calls) == sorted(set(result.topk_indices.flatten().tolist()))
    assert result.expert_load_count == len(calls)
    model(hidden)
    assert len(calls) == result.expert_load_count


def test_glm5x_expert_major_matches_reference_loop() -> None:
    hidden = torch.tensor(
        [
            [0.5, -1.0, 0.25],
            [1.0, 0.25, -0.75],
            [-0.25, 0.75, 0.5],
            [0.1, -0.4, 0.9],
        ],
        dtype=torch.float32,
    )
    router_weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    correction_bias = torch.tensor([0.0, 0.1, -0.2, 0.05], dtype=torch.float32)
    shared = _weights(0)

    def make(mode: str) -> GLM5XLayer10MoEReference:
        return GLM5XLayer10MoEReference(
            router_weight=router_weight,
            correction_bias=correction_bias,
            expert_loader=lambda expert_id: _weights(expert_id),
            shared_expert=shared,
            top_k=2,
            routed_scaling_factor=2.5,
            n_group=1,
            topk_group=1,
            execution_mode=mode,
        )

    reference = make("loop")(hidden)
    grouped = make("expert_major")(hidden)

    torch.testing.assert_close(grouped.router_logits, reference.router_logits)
    torch.testing.assert_close(grouped.topk_indices, reference.topk_indices)
    torch.testing.assert_close(grouped.topk_weights, reference.topk_weights)
    torch.testing.assert_close(grouped.output, reference.output, rtol=1e-5, atol=1e-5)
    assert grouped.loaded_experts == reference.loaded_experts


def test_glm5x_batched_expert_loader_preserves_serial_output() -> None:
    hidden = torch.tensor(
        [[0.5, -1.0, 0.25], [1.0, 0.25, -0.75], [-0.25, 0.75, 0.5]],
        dtype=torch.float32,
    )
    router_weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.5, 0.25]],
        dtype=torch.float32,
    )
    correction_bias = torch.tensor([0.0, 0.1, -0.2, 0.05], dtype=torch.float32)
    shared = _weights(0)
    batch_calls: list[tuple[int, ...]] = []

    serial = GLM5XLayer10MoEReference(
        router_weight=router_weight,
        correction_bias=correction_bias,
        expert_loader=lambda expert_id: _weights(expert_id),
        shared_expert=shared,
        top_k=2,
        routed_scaling_factor=2.5,
    )

    def load_batch(expert_ids: tuple[int, ...]):
        batch_calls.append(expert_ids)
        return {expert_id: _weights(expert_id) for expert_id in expert_ids}

    batched = GLM5XLayer10MoEReference(
        router_weight=router_weight,
        correction_bias=correction_bias,
        expert_loader=lambda expert_id: _weights(expert_id),
        expert_batch_loader=load_batch,
        expert_load_workers=4,
        shared_expert=shared,
        top_k=2,
        routed_scaling_factor=2.5,
    )

    expected = serial(hidden)
    actual = batched(hidden)

    torch.testing.assert_close(actual.output, expected.output)
    torch.testing.assert_close(actual.topk_indices, expected.topk_indices)
    torch.testing.assert_close(actual.topk_weights, expected.topk_weights)
    assert actual.loaded_experts == expected.loaded_experts
    assert batch_calls == [tuple(sorted(set(expected.topk_indices.flatten().tolist())))]
