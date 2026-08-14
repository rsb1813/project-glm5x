# GLM5X 다층 decoder 및 incremental greedy 생성 경로를 검증합니다.
from __future__ import annotations

import torch

from glm5x_ref import GLM5XDecoderModelReference
from test_glm5x_layer_reference import _make_layer


def test_model_prefill_incremental_and_greedy_generation_match() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(73)
    model = GLM5XDecoderModelReference(
        embedding=torch.randn(16, 8),
        layers=(layer, layer),
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
    )
    prompt = [1, 4, 7]
    full = model.forward_tokens(torch.tensor(prompt))
    state = model.empty_state()
    for index, token in enumerate(prompt):
        step = model.forward_token(token, state)
        torch.testing.assert_close(step.logits, full.logits[:, index : index + 1], rtol=1e-5, atol=1e-5)
        state = step.state

    expected = prompt[:]
    state = model.empty_state()
    forward = model.forward_tokens(torch.tensor(expected))
    state = forward.state
    for _ in range(2):
        token = int(torch.argmax(forward.logits[:, -1, :], dim=-1).item())
        expected.append(token)
        forward = model.forward_token(token, state)
        state = forward.state
    assert model.generate(prompt, 2) == expected


def test_model_reference_can_load_one_layer_at_a_time() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(79)
    calls: list[int] = []

    def load_layer(layer_id: int):
        calls.append(layer_id)
        return layer

    model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=torch.randn(16, 8),
        layer_count=2,
        layer_loader=load_layer,
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
        rope_dim=2,
    )
    forward = model.forward_tokens(torch.tensor([1, 2]))
    assert calls == [0, 1]
    assert model.layer_count == 2
    assert len(forward.layers) == 2


def test_model_reference_can_retain_trunk_layers_between_forwards() -> None:
    layer, _, _ = _make_layer()
    torch.manual_seed(83)
    calls: list[int] = []

    def load_layer(layer_id: int):
        calls.append(layer_id)
        return layer

    model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=torch.randn(16, 8),
        layer_count=2,
        layer_loader=load_layer,
        final_norm=torch.ones(8),
        lm_head=torch.randn(16, 8),
        rope_dim=2,
        layer_cache_capacity=2,
    )
    first = model.forward_tokens(torch.tensor([1, 2]))
    second = model.forward_tokens(torch.tensor([1, 2]))
    torch.testing.assert_close(first.logits, second.logits)
    assert calls == [0, 1]
    assert model.layer_cache_capacity == 2
    assert model.cached_layer_count == 2

    evicted_calls: list[int] = []

    def load_evicted(layer_id: int):
        evicted_calls.append(layer_id)
        return layer

    evicted_model = GLM5XDecoderModelReference.from_layer_loader(
        embedding=model.embedding,
        layer_count=2,
        layer_loader=load_evicted,
        final_norm=model.final_norm,
        lm_head=model.lm_head,
        rope_dim=2,
        layer_cache_capacity=1,
    )
    evicted_model.forward_tokens(torch.tensor([1, 2]))
    evicted_model.forward_tokens(torch.tensor([1, 2]))
    assert evicted_calls == [0, 1, 0, 1]
    assert evicted_model.cached_layer_count == 1
