# Gated MLA의 NoPE attention과 incremental KV state를 구현합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from k3x_ref.config import SyntheticK3Config
from k3x_ref.ops import rms_norm


@dataclass(frozen=True)
class MLAWeights:
    q_a_proj: torch.Tensor
    q_a_norm: torch.Tensor
    q_b_proj: torch.Tensor
    kv_a_proj: torch.Tensor
    kv_a_norm: torch.Tensor
    kv_b_proj: torch.Tensor
    g_proj: torch.Tensor
    o_proj: torch.Tensor


@dataclass(frozen=True)
class MLAState:
    keys: torch.Tensor
    values: torch.Tensor
    shared_keys: torch.Tensor
    length: int


def empty_mla_state(
    batch_size: int,
    cfg: SyntheticK3Config,
    dtype: torch.dtype,
    device: torch.device,
) -> MLAState:
    return MLAState(
        keys=torch.empty(
            (batch_size, cfg.mla_heads, 0, cfg.qk_nope_head_dim),
            dtype=dtype,
            device=device,
        ),
        values=torch.empty(
            (batch_size, cfg.mla_heads, 0, cfg.v_head_dim),
            dtype=dtype,
            device=device,
        ),
        shared_keys=torch.empty(
            (batch_size, 1, 0, cfg.qk_rope_head_dim),
            dtype=dtype,
            device=device,
        ),
        length=0,
    )


def mla_decode(
    x_one: torch.Tensor,
    weights: MLAWeights,
    state: MLAState,
    cfg: SyntheticK3Config,
) -> tuple[torch.Tensor, MLAState]:
    if x_one.ndim != 3 or x_one.shape[1] != 1 or x_one.shape[2] != cfg.hidden_size:
        raise ValueError("x_one must have shape [batch, 1, hidden_size]")
    if not cfg.mla_use_nope:
        raise ValueError("Kimi K3 MLA requires NoPE")

    batch_size = x_one.shape[0]
    query_width = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    q_latent = rms_norm(
        functional.linear(x_one, weights.q_a_proj),
        weights.q_a_norm,
        cfg.rms_norm_eps,
    )
    query = functional.linear(q_latent, weights.q_b_proj).reshape(
        batch_size, 1, cfg.mla_heads, query_width
    ).transpose(1, 2)
    query_main, query_extra = torch.split(
        query, (cfg.qk_nope_head_dim, cfg.qk_rope_head_dim), dim=-1
    )

    compressed = functional.linear(x_one, weights.kv_a_proj)
    kv_latent, shared_key = torch.split(
        compressed, (cfg.kv_lora_rank, cfg.qk_rope_head_dim), dim=-1
    )
    kv_latent = rms_norm(kv_latent, weights.kv_a_norm, cfg.rms_norm_eps)
    expanded = functional.linear(kv_latent, weights.kv_b_proj).reshape(
        batch_size,
        1,
        cfg.mla_heads,
        cfg.qk_nope_head_dim + cfg.v_head_dim,
    ).transpose(1, 2)
    key, value = torch.split(
        expanded, (cfg.qk_nope_head_dim, cfg.v_head_dim), dim=-1
    )
    shared_key = shared_key.reshape(
        batch_size, 1, 1, cfg.qk_rope_head_dim
    )

    keys = torch.cat((state.keys, key), dim=2)
    values = torch.cat((state.values, value), dim=2)
    shared_keys = torch.cat((state.shared_keys, shared_key), dim=2)
    scale = query_width**-0.5
    main_scores = torch.matmul(
        query_main.to(torch.float32), keys.to(torch.float32).transpose(-2, -1)
    )
    extra_scores = torch.matmul(
        query_extra.to(torch.float32), shared_keys.to(torch.float32).transpose(-2, -1)
    )
    probabilities = torch.softmax((main_scores + extra_scores) * scale, dim=-1)
    attended = torch.matmul(probabilities, values.to(torch.float32))
    merged = attended.transpose(1, 2).reshape(
        batch_size, 1, cfg.mla_heads * cfg.v_head_dim
    )
    output_gate = torch.sigmoid(functional.linear(x_one, weights.g_proj).to(torch.float32))
    output = functional.linear(merged * output_gate, weights.o_proj)
    return output, MLAState(keys, values, shared_keys, state.length + 1)


def mla_prefill(
    x: torch.Tensor,
    weights: MLAWeights,
    state: MLAState | None,
    cfg: SyntheticK3Config,
) -> tuple[torch.Tensor, MLAState]:
    if x.ndim != 3 or x.shape[2] != cfg.hidden_size:
        raise ValueError("x must have shape [batch, sequence, hidden_size]")
    current = state or empty_mla_state(x.shape[0], cfg, x.dtype, x.device)
    outputs = []
    for index in range(x.shape[1]):
        output, current = mla_decode(x[:, index : index + 1], weights, current, cfg)
        outputs.append(output)
    return torch.cat(outputs, dim=1), current
