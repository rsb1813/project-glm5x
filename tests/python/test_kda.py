# KDA 감쇠와 delta 갱신 및 incremental state 동등성을 검증합니다.
import pytest
import torch

from k3x_ref.config import SyntheticK3Config
from k3x_ref.kda import KDAWeights, empty_kda_state, kda_decode, kda_prefill, kda_step


def _weights(cfg: SyntheticK3Config) -> KDAWeights:
    generator = torch.Generator().manual_seed(20260808)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator) * 0.05

    projection = cfg.kda_heads * cfg.kda_head_dim
    return KDAWeights(
        q_proj=rand(projection, cfg.hidden_size),
        k_proj=rand(projection, cfg.hidden_size),
        v_proj=rand(projection, cfg.hidden_size),
        q_conv=rand(projection, cfg.short_conv_kernel_size),
        k_conv=rand(projection, cfg.short_conv_kernel_size),
        v_conv=rand(projection, cfg.short_conv_kernel_size),
        f_a_proj=rand(cfg.kda_head_dim, cfg.hidden_size),
        f_b_proj=rand(projection, cfg.kda_head_dim),
        b_proj=rand(cfg.kda_heads, cfg.hidden_size),
        a_log=torch.linspace(0.0, 0.3, cfg.kda_heads),
        dt_bias=rand(cfg.kda_heads, cfg.kda_head_dim),
        g_proj=rand(projection, cfg.hidden_size),
        o_norm=torch.linspace(0.9, 1.1, cfg.kda_head_dim),
        o_proj=rand(cfg.hidden_size, projection),
    )


def test_kda_step_reads_output_from_updated_state() -> None:
    state = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[3.0, 2.0]])
    decay = torch.ones_like(k)
    out, updated = kda_step(state, q, k, v, decay, torch.tensor([0.5]))
    assert torch.equal(updated, torch.tensor([[[2.0, 1.0], [0.0, 1.0]]]))
    assert torch.equal(out, torch.tensor([[2.0, 1.0]]))


@pytest.mark.parametrize("length", [1, 2, 5])
def test_kda_prefill_matches_incremental_decode(length: int) -> None:
    cfg = SyntheticK3Config.default()
    weights = _weights(cfg)
    x = torch.arange(length * cfg.hidden_size, dtype=torch.float32).reshape(
        1, length, cfg.hidden_size
    ) / 100.0

    full_output, full_state = kda_prefill(x, weights, None, cfg)
    state = empty_kda_state(1, cfg, x.dtype, x.device)
    pieces = []
    for index in range(length):
        output, state = kda_decode(x[:, index : index + 1], weights, state, cfg)
        pieces.append(output)

    assert torch.equal(full_output, torch.cat(pieces, dim=1))
    assert torch.equal(full_state.conv_q, state.conv_q)
    assert torch.equal(full_state.conv_k, state.conv_k)
    assert torch.equal(full_state.conv_v, state.conv_v)
    assert torch.equal(full_state.recurrent, state.recurrent)

