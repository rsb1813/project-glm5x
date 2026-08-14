# Kimi Delta Attention의 ShortConv와 recurrent state 전이를 구현합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from k3x_ref.config import SyntheticK3Config
from k3x_ref.ops import rms_norm


@dataclass(frozen=True)
class KDAWeights:
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_conv: torch.Tensor
    k_conv: torch.Tensor
    v_conv: torch.Tensor
    f_a_proj: torch.Tensor
    f_b_proj: torch.Tensor
    b_proj: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    g_proj: torch.Tensor
    o_norm: torch.Tensor
    o_proj: torch.Tensor


@dataclass(frozen=True)
class KDAState:
    conv_q: torch.Tensor
    conv_k: torch.Tensor
    conv_v: torch.Tensor
    recurrent: torch.Tensor


def empty_kda_state(
    batch_size: int,
    cfg: SyntheticK3Config,
    dtype: torch.dtype,
    device: torch.device,
) -> KDAState:
    projection = cfg.kda_heads * cfg.kda_head_dim
    history_shape = (batch_size, cfg.short_conv_kernel_size - 1, projection)
    recurrent_shape = (
        batch_size,
        cfg.kda_heads,
        cfg.kda_head_dim,
        cfg.kda_head_dim,
    )
    return KDAState(
        conv_q=torch.zeros(history_shape, dtype=dtype, device=device),
        conv_k=torch.zeros(history_shape, dtype=dtype, device=device),
        conv_v=torch.zeros(history_shape, dtype=dtype, device=device),
        recurrent=torch.zeros(recurrent_shape, dtype=torch.float32, device=device),
    )


def _short_conv_step(
    projected: torch.Tensor,
    history: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.cat((history, projected), dim=1)
    convolved = (window * weight.transpose(0, 1).unsqueeze(0)).sum(dim=1)
    return functional.silu(convolved).unsqueeze(1), window[:, 1:, :]


def kda_step(
    state: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    decayed = state.to(torch.float32) * decay.to(torch.float32).unsqueeze(-1)
    prediction = torch.matmul(k.to(torch.float32).unsqueeze(-2), decayed).squeeze(-2)
    delta = (v.to(torch.float32) - prediction) * beta.to(torch.float32).unsqueeze(-1)
    updated = decayed + k.to(torch.float32).unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.matmul(q.to(torch.float32).unsqueeze(-2), updated).squeeze(-2)
    return output, updated


def kda_decode(
    x_one: torch.Tensor,
    weights: KDAWeights,
    state: KDAState,
    cfg: SyntheticK3Config,
) -> tuple[torch.Tensor, KDAState]:
    if x_one.ndim != 3 or x_one.shape[1] != 1 or x_one.shape[2] != cfg.hidden_size:
        raise ValueError("x_one must have shape [batch, 1, hidden_size]")

    q_projected = functional.linear(x_one, weights.q_proj)
    k_projected = functional.linear(x_one, weights.k_proj)
    v_projected = functional.linear(x_one, weights.v_proj)
    q_conv, conv_q = _short_conv_step(q_projected, state.conv_q, weights.q_conv)
    k_conv, conv_k = _short_conv_step(k_projected, state.conv_k, weights.k_conv)
    v_conv, conv_v = _short_conv_step(v_projected, state.conv_v, weights.v_conv)

    shape = (x_one.shape[0], 1, cfg.kda_heads, cfg.kda_head_dim)
    q = q_conv.reshape(shape).to(torch.float32)
    k = k_conv.reshape(shape).to(torch.float32)
    v = v_conv.reshape(shape).to(torch.float32)
    scale = cfg.kda_head_dim**-0.5
    ones = torch.ones(cfg.kda_head_dim, dtype=torch.float32, device=x_one.device)
    q = rms_norm(q, ones, 1.0e-6).to(torch.float32) * (scale * scale)
    k = rms_norm(k, ones, 1.0e-6).to(torch.float32) * scale

    forget = functional.linear(
        functional.linear(x_one, weights.f_a_proj), weights.f_b_proj
    ).reshape(shape)
    gate_argument = forget.to(torch.float32) + weights.dt_bias.to(torch.float32)
    a = weights.a_log.to(torch.float32).exp().reshape(1, 1, cfg.kda_heads, 1)
    log_decay = cfg.kda_gate_lower_bound * torch.sigmoid(a * gate_argument)
    decay = log_decay.exp()
    beta = torch.sigmoid(functional.linear(x_one, weights.b_proj).to(torch.float32))

    recurrent_output, recurrent = kda_step(
        state.recurrent,
        q[:, 0],
        k[:, 0],
        v[:, 0],
        decay[:, 0],
        beta[:, 0],
    )
    normalized = rms_norm(
        recurrent_output, weights.o_norm, cfg.rms_norm_eps
    ).to(torch.float32)
    output_gate = torch.sigmoid(
        functional.linear(x_one, weights.g_proj).reshape(shape).to(torch.float32)
    )
    output = functional.linear(
        (normalized.unsqueeze(1) * output_gate).reshape(
            x_one.shape[0], 1, cfg.kda_heads * cfg.kda_head_dim
        ),
        weights.o_proj,
    )
    return output, KDAState(conv_q, conv_k, conv_v, recurrent)


def kda_prefill(
    x: torch.Tensor,
    weights: KDAWeights,
    state: KDAState | None,
    cfg: SyntheticK3Config,
) -> tuple[torch.Tensor, KDAState]:
    if x.ndim != 3 or x.shape[2] != cfg.hidden_size:
        raise ValueError("x must have shape [batch, sequence, hidden_size]")
    current = state or empty_kda_state(x.shape[0], cfg, x.dtype, x.device)
    outputs = []
    for index in range(x.shape[1]):
        output, current = kda_decode(x[:, index : index + 1], weights, current, cfg)
        outputs.append(output)
    return torch.cat(outputs, dim=1), current
