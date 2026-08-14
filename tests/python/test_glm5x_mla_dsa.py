# GLM5X 공식 MLA와 DSA incremental 상태의 수식 parity를 검증합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from glm5x_ref.mla_dsa import (
    GLM5XMLAReference,
    GLM5XMLAWeights,
)
from glm5x_ref.official_dsa import (
    GLM5XOfficialDSAIndexer,
    GLM5XOfficialDSAState,
    build_glm_indexer_rope,
)


def _rmsnorm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    work = values.float()
    return (work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)).to(
        values.dtype
    ) * weight.to(values.dtype)


def _rope_interleaved(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    half = values.shape[-1] // 2
    angles = cos[..., :half]
    sine = sin[..., :half]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.cat((even * angles - odd * sine, odd * angles + even * sine), dim=-1)


@dataclass(frozen=True)
class _SyntheticMLA:
    weights: GLM5XMLAWeights
    cos: torch.Tensor
    sin: torch.Tensor
    hidden: torch.Tensor


def _synthetic_mla() -> _SyntheticMLA:
    torch.manual_seed(17)
    hidden_size = 8
    heads = 2
    q_rank = 4
    kv_rank = 3
    nope = 2
    rope = 2
    value = 2
    rand = lambda *shape: torch.randn(shape, dtype=torch.float32)  # noqa: E731
    weights = GLM5XMLAWeights(
        q_a_proj=rand(q_rank, hidden_size),
        q_a_norm=rand(q_rank),
        q_b_proj=rand(heads * (nope + rope), q_rank),
        kv_a_proj=rand(kv_rank + rope, hidden_size),
        kv_a_norm=rand(kv_rank),
        kv_b_proj=rand(heads * (nope + value), kv_rank),
        o_proj=rand(hidden_size, heads * value),
        num_heads=heads,
        qk_nope_head_dim=nope,
        qk_rope_head_dim=rope,
        v_head_dim=value,
        rms_norm_eps=1e-5,
    )
    positions = torch.arange(4, dtype=torch.float32).view(1, 4)
    inverse = 1.0 / (10000.0 ** (torch.arange(0, rope, 2) / rope))
    frequencies = positions[..., None] * inverse
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    hidden = rand(1, 4, hidden_size)
    return _SyntheticMLA(weights, frequencies.cos(), frequencies.sin(), hidden)


def _independent_mla(
    sample: _SyntheticMLA,
) -> torch.Tensor:
    weights = sample.weights
    hidden = sample.hidden
    q_resid = _rmsnorm(
        F.linear(hidden, weights.q_a_proj), weights.q_a_norm, weights.rms_norm_eps
    )
    q_states = F.linear(q_resid, weights.q_b_proj).view(1, 4, 2, 4).transpose(1, 2)
    q_pass, q_rot = q_states.split((2, 2), dim=-1)
    compressed = F.linear(hidden, weights.kv_a_proj)
    kv_nope, k_rot = compressed.split((3, 2), dim=-1)
    kv_nope = _rmsnorm(kv_nope, weights.kv_a_norm, weights.rms_norm_eps)
    q_rot = _rope_interleaved(q_rot, sample.cos.unsqueeze(1), sample.sin.unsqueeze(1))
    k_rot = _rope_interleaved(k_rot, sample.cos, sample.sin).unsqueeze(1)
    kv_expanded = F.linear(kv_nope, weights.kv_b_proj).view(1, 4, 2, 4).transpose(1, 2)
    k_nope, values = kv_expanded.split((2, 2), dim=-1)
    keys = torch.cat((k_nope, k_rot.expand(-1, 2, -1, -1)), dim=-1)
    queries = torch.cat((q_pass, q_rot), dim=-1)
    scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) * (4.0**-0.5)
    causal = torch.arange(4)[None, :] > torch.arange(4)[:, None]
    scores = scores.masked_fill(causal.view(1, 1, 4, 4), torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
    attended = torch.matmul(probabilities, values).transpose(1, 2).reshape(1, 4, 4)
    return F.linear(attended, weights.o_proj)


def test_glm5x_mla_matches_independent_official_prefill_formula() -> None:
    sample = _synthetic_mla()
    actual = GLM5XMLAReference(sample.weights)(
        sample.hidden,
        (sample.cos, sample.sin),
        position_ids=torch.arange(4).view(1, 4),
    )
    torch.testing.assert_close(actual.output, _independent_mla(sample), rtol=1e-5, atol=1e-5)
    assert actual.q_resid.shape == (1, 4, 4)
    assert actual.state.kv_nope.shape == (1, 1, 4, 3)


def test_glm5x_mla_incremental_state_matches_prefill_last_token() -> None:
    sample = _synthetic_mla()
    model = GLM5XMLAReference(sample.weights)
    full = model(
        sample.hidden,
        (sample.cos, sample.sin),
        position_ids=torch.arange(4).view(1, 4),
    )
    first = model(
        sample.hidden[:, :3],
        (sample.cos[:, :3], sample.sin[:, :3]),
        position_ids=torch.arange(3).view(1, 3),
    )
    last = model(
        sample.hidden[:, 3:],
        (sample.cos[:, 3:], sample.sin[:, 3:]),
        position_ids=torch.tensor([[3]]),
        state=first.state,
    )
    torch.testing.assert_close(last.output, full.output[:, 3:], rtol=1e-5, atol=1e-5)
    assert last.state.kv_nope.shape[-2] == 4


def test_glm5x_mla_reuses_precomputed_q_residual_without_output_drift() -> None:
    sample = _synthetic_mla()
    model = GLM5XMLAReference(sample.weights)
    q_resid = model.q_residual(sample.hidden)
    baseline = model(
        sample.hidden,
        (sample.cos, sample.sin),
        position_ids=torch.arange(4).view(1, 4),
    )
    reused = model(
        sample.hidden,
        (sample.cos, sample.sin),
        position_ids=torch.arange(4).view(1, 4),
        q_residual=q_resid,
    )
    torch.testing.assert_close(reused.q_resid, q_resid)
    torch.testing.assert_close(reused.output, baseline.output)
    torch.testing.assert_close(reused.state.kv_nope, baseline.state.kv_nope)


def test_glm5x_mla_sparse_topk_matches_dense_masked_attention() -> None:
    sample = _synthetic_mla()
    dense = GLM5XMLAReference(sample.weights)
    sparse = GLM5XMLAReference(sample.weights, use_sparse_topk=True)
    topk = torch.tensor(
        [[[0, 1], [0, 2], [1, 3], [2, 3]]], dtype=torch.int32
    )
    kwargs = {
        "position_ids": torch.arange(4).view(1, 4),
        "topk_indices": topk,
    }
    dense_result = dense(sample.hidden, (sample.cos, sample.sin), **kwargs)
    sparse_result = sparse(sample.hidden, (sample.cos, sample.sin), **kwargs)
    torch.testing.assert_close(sparse_result.output, dense_result.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sparse_result.state.kv_nope, dense_result.state.kv_nope)


def test_glm5x_mla_sparse_topk_incremental_matches_dense_state() -> None:
    sample = _synthetic_mla()
    dense = GLM5XMLAReference(sample.weights)
    sparse = GLM5XMLAReference(sample.weights, use_sparse_topk=True)
    first_kwargs = {
        "position_ids": torch.arange(3).view(1, 3),
        "topk_indices": torch.tensor([[[0, 1], [0, 2], [1, 2]]], dtype=torch.int32),
    }
    dense_first = dense(
        sample.hidden[:, :3],
        (sample.cos[:, :3], sample.sin[:, :3]),
        **first_kwargs,
    )
    sparse_first = sparse(
        sample.hidden[:, :3],
        (sample.cos[:, :3], sample.sin[:, :3]),
        **first_kwargs,
    )
    last_kwargs = {
        "position_ids": torch.tensor([[3]]),
        "topk_indices": torch.tensor([[[0, 2]]], dtype=torch.int32),
    }
    dense_last = dense(
        sample.hidden[:, 3:],
        (sample.cos[:, 3:], sample.sin[:, 3:]),
        state=dense_first.state,
        **last_kwargs,
    )
    sparse_last = sparse(
        sample.hidden[:, 3:],
        (sample.cos[:, 3:], sample.sin[:, 3:]),
        state=sparse_first.state,
        **last_kwargs,
    )
    torch.testing.assert_close(sparse_last.output, dense_last.output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sparse_last.state.kv_nope, dense_last.state.kv_nope)


def _synthetic_indexer() -> GLM5XOfficialDSAIndexer:
    torch.manual_seed(23)
    return GLM5XOfficialDSAIndexer(
        wq_b=torch.randn(8, 3),
        wk=torch.randn(4, 6),
        k_norm_weight=torch.randn(4),
        k_norm_bias=torch.randn(4),
        weights_proj=torch.randn(2, 6),
        qk_rope_head_dim=2,
        index_topk=1,
        indexer_rope_interleave=True,
    )


def test_glm5x_dsa_incremental_indexer_matches_causal_prefill() -> None:
    indexer = _synthetic_indexer()
    hidden = torch.randn(1, 4, 6)
    q_resid = torch.randn(1, 4, 3)
    position_ids = torch.arange(4).view(1, 4)
    cos, sin = build_glm_indexer_rope(position_ids, rope_dim=2, rope_theta=10000.0)
    full = indexer.select_topk(hidden, q_resid, (cos, sin), position_ids)
    first, state = indexer.select_topk_incremental(
        hidden[:, :3],
        q_resid[:, :3],
        (cos[:, :3], sin[:, :3]),
        position_ids[:, :3],
    )
    last, next_state = indexer.select_topk_incremental(
        hidden[:, 3:],
        q_resid[:, 3:],
        (cos[:, 3:], sin[:, 3:]),
        position_ids[:, 3:],
        state=state,
    )
    # ReLU index scores can tie at zero for an early query; the valid non-tied
    # prefix and the final incremental query must still match exactly.
    torch.testing.assert_close(first[:, :2], full[:, :2])
    torch.testing.assert_close(last, full[:, 3:])
    assert isinstance(next_state, GLM5XOfficialDSAState)
    assert next_state.keys.shape[1] == 4
