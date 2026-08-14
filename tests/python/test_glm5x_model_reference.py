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
