# 합성 K3 전체 그래프의 full 및 incremental greedy 결과를 검증합니다.
import numpy as np
import torch

from k3x_ref.fixtures import build_synthetic_model, write_golden
from k3x_ref.kda import KDAState
from k3x_ref.mla import MLAState


def test_controlled_model_generates_independent_token_literal() -> None:
    model = build_synthetic_model(seed=20260808, controlled=True)
    expected = [5, 5, 5, 5, 5, 5]
    assert model.generate_greedy([1, 7, 3, 9], 6, incremental=False) == expected
    assert model.generate_greedy([1, 7, 3, 9], 6, incremental=True) == expected


def test_seeded_model_full_and_incremental_states_match() -> None:
    model = build_synthetic_model(seed=20260808)
    prompt = torch.tensor([[1, 7, 3, 9]])
    full_logits, full_state = model.prefill(prompt)

    state = model.empty_state(batch_size=1, dtype=torch.float32, device=prompt.device)
    pieces = []
    for token in prompt[0]:
        logits, state = model.decode(token.reshape(1), state)
        pieces.append(logits)

    assert torch.allclose(full_logits, torch.cat(pieces, dim=1), atol=1e-6, rtol=1e-6)
    for full_item, incremental_item in zip(
        full_state.attention, state.attention, strict=True
    ):
        if isinstance(full_item, KDAState) and isinstance(incremental_item, KDAState):
            tensors = zip(
                (full_item.conv_q, full_item.conv_k, full_item.conv_v, full_item.recurrent),
                (
                    incremental_item.conv_q,
                    incremental_item.conv_k,
                    incremental_item.conv_v,
                    incremental_item.recurrent,
                ),
                strict=True,
            )
        elif isinstance(full_item, MLAState) and isinstance(incremental_item, MLAState):
            tensors = zip(
                (full_item.keys, full_item.values, full_item.shared_keys),
                (
                    incremental_item.keys,
                    incremental_item.values,
                    incremental_item.shared_keys,
                ),
                strict=True,
            )
        else:
            raise AssertionError("attention state types differ")
        for full_tensor, incremental_tensor in tensors:
            assert torch.allclose(
                full_tensor, incremental_tensor, atol=1e-6, rtol=1e-6
            )


def test_seeded_model_greedy_tokens_match_between_modes() -> None:
    model = build_synthetic_model(seed=20260808)
    full = model.generate_greedy([1, 7, 3, 9], 6, incremental=False)
    incremental = model.generate_greedy([1, 7, 3, 9], 6, incremental=True)
    assert full == incremental


def test_golden_fixture_contains_every_layer_and_state_tensor(tmp_path) -> None:
    model = build_synthetic_model(seed=20260808)
    output = tmp_path / "golden.npz"
    write_golden(output, model, [1, 7, 3, 9])
    with np.load(output) as golden:
        for layer_index in range(4):
            assert f"layer_{layer_index}" in golden
        assert "state.0.recurrent" in golden
        assert "state.3.keys" in golden
        assert "state.3.values" in golden
        assert "state.3.shared_keys" in golden
