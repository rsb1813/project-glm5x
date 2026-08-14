# synthetic GLM5X fixture의 incremental state와 greedy parity를 검증합니다.

import torch

from glm5x_ref.toy import GLM5XSyntheticConfig, GLM5XSyntheticModel


def test_full_and_incremental_logits_match() -> None:
    torch.manual_seed(7)
    model = GLM5XSyntheticModel(GLM5XSyntheticConfig.tiny())
    tokens = torch.tensor([1, 4, 2, 9], dtype=torch.long)

    full_logits, full_routes = model.forward(tokens)
    state = model.empty_state()
    incremental_logits = []
    incremental_routes = []
    for token in tokens:
        logits, state, routes = model.forward_token(token, state)
        incremental_logits.append(logits)
        incremental_routes.append(routes)

    assert torch.allclose(full_logits, torch.stack(incremental_logits), atol=1e-6)
    for expected_layers, actual_layers in zip(full_routes, incremental_routes):
        for expected, actual in zip(expected_layers, actual_layers):
            assert torch.equal(expected, actual)


def test_greedy_generation_is_reproducible() -> None:
    torch.manual_seed(11)
    model = GLM5XSyntheticModel(GLM5XSyntheticConfig.tiny())

    first = model.generate([1, 3, 5], max_new_tokens=6)
    second = model.generate([1, 3, 5], max_new_tokens=6)

    assert first == second
    assert len(first) == 9


def test_router_selects_exact_top_k_experts() -> None:
    torch.manual_seed(13)
    cfg = GLM5XSyntheticConfig.tiny()
    model = GLM5XSyntheticModel(cfg)

    _, _, routes = model.forward_token(torch.tensor(2), model.empty_state())

    assert len(routes) == cfg.num_layers
    assert all(route.shape == (cfg.top_k,) for route in routes)
    assert all(int(route.max()) < cfg.num_experts for route in routes)
