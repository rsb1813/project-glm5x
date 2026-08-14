# GLM-5.2 공식 MLA와 incremental compressed KV 상태를 구현합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _rms_norm(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    work = values.to(torch.float32)
    normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    return normalized.to(values.dtype) * weight.to(device=values.device, dtype=values.dtype)


def _apply_interleaved_rope(
    values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    if values.ndim < 3 or cos.shape != sin.shape:
        raise ValueError("GLM5X_MLA_ROPE_SHAPE_MISMATCH")
    if cos.ndim == 3:
        if cos.shape[0] != values.shape[0] or cos.shape[1] != values.shape[-2]:
            raise ValueError("GLM5X_MLA_ROPE_POSITION_SHAPE_MISMATCH")
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.ndim == values.ndim:
        if cos.shape[0] != values.shape[0] or cos.shape[-2:] != values.shape[-2:]:
            raise ValueError("GLM5X_MLA_ROPE_POSITION_SHAPE_MISMATCH")
        if cos.shape[-3] not in (1, values.shape[-3]):
            raise ValueError("GLM5X_MLA_ROPE_HEAD_SHAPE_MISMATCH")
    else:
        raise ValueError("GLM5X_MLA_ROPE_POSITION_SHAPE_MISMATCH")
    rope_dim = values.shape[-1]
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("GLM5X_MLA_ROPE_DIMENSION")
    if cos.shape[-1] < rope_dim:
        raise ValueError("GLM5X_MLA_ROPE_WIDTH")
    half = rope_dim // 2
    angles = cos[..., :half]
    sine = sin[..., :half]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    return torch.cat((even * angles - odd * sine, odd * angles + even * sine), dim=-1)


@dataclass(frozen=True)
class GLM5XMLAWeights:
    """GLM MLA projection tensors in safetensors linear-layer orientation."""

    q_a_proj: torch.Tensor
    q_a_norm: torch.Tensor
    q_b_proj: torch.Tensor
    kv_a_proj: torch.Tensor
    kv_a_norm: torch.Tensor
    kv_b_proj: torch.Tensor
    o_proj: torch.Tensor
    num_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    q_a_bias: torch.Tensor | None = None
    kv_a_bias: torch.Tensor | None = None
    o_bias: torch.Tensor | None = None
    rms_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        q_a_proj = torch.as_tensor(self.q_a_proj)
        q_a_norm = torch.as_tensor(self.q_a_norm)
        q_b_proj = torch.as_tensor(self.q_b_proj)
        kv_a_proj = torch.as_tensor(self.kv_a_proj)
        kv_a_norm = torch.as_tensor(self.kv_a_norm)
        kv_b_proj = torch.as_tensor(self.kv_b_proj)
        o_proj = torch.as_tensor(self.o_proj)
        if any(value.ndim != 2 for value in (q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj)):
            raise ValueError("GLM5X_MLA_PROJECTION_RANK")
        if q_a_norm.ndim != 1 or kv_a_norm.ndim != 1:
            raise ValueError("GLM5X_MLA_NORM_RANK")
        if q_a_proj.shape[0] != q_a_norm.shape[0] or q_b_proj.shape[1] != q_a_proj.shape[0]:
            raise ValueError("GLM5X_MLA_Q_SHAPE")
        if not isinstance(self.qk_rope_head_dim, int) or self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim % 2:
            raise ValueError("GLM5X_MLA_ROPE_DIM")
        if kv_a_proj.shape[0] <= kv_a_norm.shape[0] or kv_a_proj.shape[0] != kv_a_norm.shape[0] + self.qk_rope_head_dim:
            raise ValueError("GLM5X_MLA_KV_A_SHAPE")
        if kv_b_proj.shape[1] != kv_a_norm.shape[0]:
            raise ValueError("GLM5X_MLA_KV_B_SHAPE")
        if not isinstance(self.num_heads, int) or self.num_heads <= 0:
            raise ValueError("GLM5X_MLA_HEAD_COUNT")
        if not isinstance(self.qk_nope_head_dim, int) or self.qk_nope_head_dim <= 0:
            raise ValueError("GLM5X_MLA_NOPE_DIM")
        if not isinstance(self.v_head_dim, int) or self.v_head_dim <= 0:
            raise ValueError("GLM5X_MLA_VALUE_DIM")
        if q_b_proj.shape[0] != self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim):
            raise ValueError("GLM5X_MLA_Q_HEAD_SHAPE")
        if kv_b_proj.shape[0] != self.num_heads * (self.qk_nope_head_dim + self.v_head_dim):
            raise ValueError("GLM5X_MLA_KV_HEAD_SHAPE")
        if o_proj.shape[1] != self.num_heads * self.v_head_dim:
            raise ValueError("GLM5X_MLA_OUTPUT_HEAD_SHAPE")
        if q_b_proj.shape[0] % 2 or kv_b_proj.shape[0] % 2:
            raise ValueError("GLM5X_MLA_HEAD_SHAPE")
        if o_proj.shape[1] <= 0 or o_proj.shape[0] != q_a_proj.shape[1]:
            raise ValueError("GLM5X_MLA_OUTPUT_SHAPE")
        for name, bias, width in (
            ("q_a_bias", self.q_a_bias, q_a_proj.shape[0]),
            ("kv_a_bias", self.kv_a_bias, kv_a_proj.shape[0]),
            ("o_bias", self.o_bias, o_proj.shape[0]),
        ):
            if bias is not None and torch.as_tensor(bias).shape != (width,):
                raise ValueError(f"GLM5X_MLA_{name.upper()}_SHAPE")
        if not isinstance(self.rms_norm_eps, (float, int)) or self.rms_norm_eps <= 0:
            raise ValueError("GLM5X_MLA_NORM_EPS")

    @property
    def hidden_size(self) -> int:
        return int(torch.as_tensor(self.q_a_proj).shape[1])

    @property
    def q_lora_rank(self) -> int:
        return int(torch.as_tensor(self.q_a_proj).shape[0])

    @property
    def kv_lora_rank(self) -> int:
        return int(torch.as_tensor(self.kv_a_norm).shape[0])

    
@dataclass(frozen=True)
class GLM5XMLAState:
    """Compressed MLA KV state with already-positioned shared rotary keys."""

    kv_nope: torch.Tensor
    k_rot: torch.Tensor
    positions: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.positions.shape[-1])


@dataclass(frozen=True)
class GLM5XMLAForward:
    output: torch.Tensor
    q_resid: torch.Tensor
    state: GLM5XMLAState
    topk_indices: torch.Tensor | None = None


class GLM5XMLAReference:
    """Exact eager MLA reference with optional DSA-selected attention positions."""

    def __init__(self, weights: GLM5XMLAWeights) -> None:
        self.weights = weights

    def _linear(
        self, values: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        weight = torch.as_tensor(weight).to(device=values.device)
        if values.dtype != weight.dtype:
            values = values.to(weight.dtype)
        if bias is not None:
            bias = torch.as_tensor(bias).to(device=values.device, dtype=weight.dtype)
        return F.linear(values, weight, bias)

    def q_residual(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.weights.hidden_size:
            raise ValueError("GLM5X_MLA_HIDDEN_SHAPE")
        return _rms_norm(
            self._linear(hidden_states, self.weights.q_a_proj, self.weights.q_a_bias),
            self.weights.q_a_norm,
            self.weights.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: torch.Tensor | None = None,
        state: GLM5XMLAState | None = None,
        topk_indices: torch.Tensor | None = None,
        q_residual: torch.Tensor | None = None,
    ) -> GLM5XMLAForward:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.weights.hidden_size:
            raise ValueError("GLM5X_MLA_HIDDEN_SHAPE")
        batch_size, seq_length, _ = hidden_states.shape
        cos, sin = (torch.as_tensor(item) for item in position_embeddings)
        if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[:2] != (batch_size, seq_length):
            raise ValueError("GLM5X_MLA_POSITION_EMBEDDING_SHAPE")
        if position_ids is None:
            start = 0 if state is None else state.length
            position_ids = torch.arange(start, start + seq_length, device=hidden_states.device).view(1, -1)
            position_ids = position_ids.expand(batch_size, -1)
        else:
            position_ids = torch.as_tensor(position_ids, device=hidden_states.device)
            if position_ids.shape != (batch_size, seq_length):
                raise ValueError("GLM5X_MLA_POSITION_ID_SHAPE")

        if q_residual is None:
            q_resid = self.q_residual(hidden_states)
        else:
            q_resid = torch.as_tensor(q_residual, device=hidden_states.device)
            if q_resid.shape != (batch_size, seq_length, self.weights.q_lora_rank):
                raise ValueError("GLM5X_MLA_Q_RESIDUAL_SHAPE")
        qk_head_dim = self.weights.qk_nope_head_dim + self.weights.qk_rope_head_dim
        q_states = self._linear(q_resid, self.weights.q_b_proj, None).view(
            batch_size, seq_length, self.weights.num_heads, qk_head_dim
        ).transpose(1, 2)
        q_pass, q_rot = q_states.split(
            (self.weights.qk_nope_head_dim, self.weights.qk_rope_head_dim), dim=-1
        )
        compressed = self._linear(hidden_states, self.weights.kv_a_proj, self.weights.kv_a_bias)
        kv_nope, k_rot = compressed.split(
            (self.weights.kv_lora_rank, self.weights.qk_rope_head_dim), dim=-1
        )
        kv_nope = _rms_norm(kv_nope, self.weights.kv_a_norm, self.weights.rms_norm_eps)
        q_rot = _apply_interleaved_rope(q_rot, cos, sin)
        k_rot = _apply_interleaved_rope(k_rot.unsqueeze(1), cos, sin)
        current_kv_nope = kv_nope.unsqueeze(1)

        if state is None:
            all_kv_nope = current_kv_nope
            all_k_rot = k_rot
            key_positions = position_ids
        else:
            self._validate_state(state, batch_size)
            all_kv_nope = torch.cat((state.kv_nope, current_kv_nope), dim=2)
            all_k_rot = torch.cat((state.k_rot, k_rot), dim=2)
            key_positions = torch.cat((state.positions, position_ids), dim=1)

        expanded = self._linear(all_kv_nope, self.weights.kv_b_proj, None).view(
            batch_size, -1, self.weights.num_heads, self.weights.qk_nope_head_dim + self.weights.v_head_dim
        ).transpose(1, 2)
        k_nope, values = expanded.split((self.weights.qk_nope_head_dim, self.weights.v_head_dim), dim=-1)
        keys = torch.cat((k_nope, all_k_rot.expand(-1, self.weights.num_heads, -1, -1)), dim=-1)
        queries = torch.cat((q_pass, q_rot), dim=-1)
        scores = torch.matmul(queries.to(torch.float32), keys.to(torch.float32).transpose(-1, -2))
        scores = scores * (qk_head_dim ** -0.5)
        causal = key_positions[:, None, :] > position_ids[:, :, None]
        scores = scores.masked_fill(causal[:, None, :, :], torch.finfo(scores.dtype).min)
        if topk_indices is not None:
            topk_indices = torch.as_tensor(topk_indices, device=hidden_states.device)
            if topk_indices.ndim != 3 or topk_indices.shape[:2] != (batch_size, seq_length):
                raise ValueError("GLM5X_MLA_TOPK_SHAPE")
            if torch.any(topk_indices < 0) or torch.any(topk_indices >= key_positions.shape[1]):
                raise ValueError("GLM5X_MLA_TOPK_RANGE")
            selected = torch.zeros(
                (batch_size, seq_length, key_positions.shape[1]), dtype=torch.bool, device=hidden_states.device
            )
            selected.scatter_(2, topk_indices.to(torch.long), True)
            scores = scores.masked_fill(~selected[:, None, :, :], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(values.dtype)
        attended = torch.matmul(probabilities, values).transpose(1, 2).reshape(
            batch_size, seq_length, self.weights.num_heads * self.weights.v_head_dim
        )
        output = self._linear(attended, self.weights.o_proj, self.weights.o_bias)
        next_state = GLM5XMLAState(all_kv_nope, all_k_rot, key_positions)
        return GLM5XMLAForward(output, q_resid, next_state, topk_indices)

    @staticmethod
    def _validate_state(state: GLM5XMLAState, batch_size: int) -> None:
        if state.kv_nope.ndim != 4 or state.k_rot.ndim != 4 or state.positions.ndim != 2:
            raise ValueError("GLM5X_MLA_STATE_RANK")
        if state.kv_nope.shape[0] != batch_size or state.k_rot.shape[0] != batch_size:
            raise ValueError("GLM5X_MLA_STATE_BATCH")
        if state.kv_nope.shape[2] != state.k_rot.shape[2] or state.positions.shape[1] != state.kv_nope.shape[2]:
            raise ValueError("GLM5X_MLA_STATE_LENGTH")
