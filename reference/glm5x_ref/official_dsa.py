# GLM-5.2 공식 DSA indexer projection과 Top-K reference를 제공합니다.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
import torch

from .int4 import GLM5XInt4Weight, linear, weight_shape


def build_glm_indexer_rope(
    position_ids: torch.Tensor,
    *,
    rope_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.as_tensor(position_ids)
    if position_ids.ndim != 2 or rope_dim <= 0 or rope_dim % 2:
        raise ValueError("INVALID_INDEXER_ROPE_SHAPE")
    if not isinstance(rope_theta, (float, int)) or rope_theta <= 0:
        raise ValueError("INVALID_INDEXER_ROPE_THETA")
    inverse = 1.0 / (
        float(rope_theta)
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=position_ids.device) / rope_dim)
    )
    frequencies = position_ids.to(torch.float32)[..., None] * inverse
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


@dataclass(frozen=True)
class GLM5XOfficialDSAState:
    """Positioned DSA index keys retained across incremental decoding."""

    keys: torch.Tensor
    positions: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.positions.shape[-1])


@dataclass(frozen=True)
class GLM5XOfficialDSAIndexer:
    """Reference implementation of the GLM-MoE-DSA indexer formula."""

    wq_b: torch.Tensor | GLM5XInt4Weight
    wk: torch.Tensor | GLM5XInt4Weight
    k_norm_weight: torch.Tensor
    k_norm_bias: torch.Tensor
    weights_proj: torch.Tensor | GLM5XInt4Weight
    qk_rope_head_dim: int = 64
    index_topk: int = 2048
    indexer_rope_interleave: bool = True
    k_norm_eps: float = 1e-6

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        layer_id: int,
        source_layer: int | None = None,
        qk_rope_head_dim: int = 64,
        index_topk: int = 2048,
        indexer_rope_interleave: bool = True,
        k_norm_eps: float = 1e-6,
    ) -> "GLM5XOfficialDSAIndexer":
        if not isinstance(layer_id, int) or layer_id < 0:
            raise ValueError("INVALID_DSA_LAYER_ID")
        source_layer = layer_id if source_layer is None else source_layer
        if not isinstance(source_layer, int) or source_layer < 0:
            raise ValueError("INVALID_DSA_SOURCE_LAYER")
        prefix = f"model.layers.{source_layer}.self_attn.indexer"
        names = {
            "wq_b": f"{prefix}.wq_b.weight",
            "wk": f"{prefix}.wk.weight",
            "k_norm_weight": f"{prefix}.k_norm.weight",
            "k_norm_bias": f"{prefix}.k_norm.bias",
            "weights_proj": f"{prefix}.weights_proj.weight",
        }
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                tensors = {key: handle.get_tensor(name) for key, name in names.items()}
        except (OSError, KeyError, RuntimeError) as exc:
            raise ValueError("DSA_INDEXER_TENSOR_MISSING") from exc
        return cls(
            **tensors,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            indexer_rope_interleave=indexer_rope_interleave,
            k_norm_eps=k_norm_eps,
        )

    def __post_init__(self) -> None:
        wq_b = weight_shape(self.wq_b)
        wk = weight_shape(self.wk)
        k_norm_weight = torch.as_tensor(self.k_norm_weight)
        k_norm_bias = torch.as_tensor(self.k_norm_bias)
        weights_proj = weight_shape(self.weights_proj)
        if len(wq_b) != 2 or len(wk) != 2 or len(weights_proj) != 2:
            raise ValueError("DSA_INDEXER_WEIGHTS_MUST_BE_RANK_TWO")
        if k_norm_weight.ndim != 1 or k_norm_bias.ndim != 1:
            raise ValueError("DSA_INDEXER_NORM_MUST_BE_RANK_ONE")
        head_dim = wk[0]
        if k_norm_weight.shape != (head_dim,) or k_norm_bias.shape != (head_dim,):
            raise ValueError("DSA_INDEXER_NORM_SHAPE_MISMATCH")
        if weights_proj[1] != wk[1]:
            raise ValueError("DSA_WEIGHTS_PROJ_SHAPE_MISMATCH")
        if wq_b[0] != weights_proj[0] * head_dim:
            raise ValueError("DSA_WQ_B_SHAPE_MISMATCH")
        if self.qk_rope_head_dim <= 0 or self.qk_rope_head_dim > head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("DSA_INVALID_ROPE_HEAD_DIM")
        if self.index_topk <= 0 or not isinstance(self.index_topk, int):
            raise ValueError("DSA_INVALID_TOPK")
        if self.k_norm_eps <= 0:
            raise ValueError("DSA_INVALID_NORM_EPS")

    @property
    def hidden_size(self) -> int:
        return int(weight_shape(self.wk)[1])

    @property
    def index_n_heads(self) -> int:
        return int(weight_shape(self.weights_proj)[0])

    @property
    def index_head_dim(self) -> int:
        return int(weight_shape(self.wk)[0])

    @property
    def q_lora_rank(self) -> int:
        return int(weight_shape(self.wq_b)[1])

    def project_query(self, q_resid: torch.Tensor) -> torch.Tensor:
        q_resid = torch.as_tensor(q_resid)
        if q_resid.ndim == 0 or q_resid.shape[-1] != self.q_lora_rank:
            raise ValueError("DSA_Q_RESID_WIDTH_MISMATCH")
        projected = linear(q_resid, self.wq_b)
        return projected.reshape(*q_resid.shape[:-1], self.index_n_heads, self.index_head_dim)

    def project_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim == 0 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("DSA_INDEXER_HIDDEN_WIDTH_MISMATCH")
        projected = linear(hidden_states, self.wk)
        norm_weight = torch.as_tensor(self.k_norm_weight).to(device=hidden_states.device)
        norm_bias = torch.as_tensor(self.k_norm_bias).to(device=hidden_states.device)
        return torch.nn.functional.layer_norm(
            projected,
            (self.index_head_dim,),
            norm_weight,
            norm_bias,
            self.k_norm_eps,
        )

    def _rotate(
        self,
        q_rot: torch.Tensor,
        k_rot: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rope_dim = self.qk_rope_head_dim
        cos = torch.as_tensor(cos)
        sin = torch.as_tensor(sin)
        if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[-1] < rope_dim // 2:
            raise ValueError("DSA_INDEXER_ROPE_SHAPE_MISMATCH")
        angles = cos[..., : rope_dim // 2].unsqueeze(2)
        sine = sin[..., : rope_dim // 2].unsqueeze(2)
        q_rot = q_rot[..., :rope_dim]
        k_rot = k_rot[..., :rope_dim]
        if self.indexer_rope_interleave:
            q_even, q_odd = q_rot[..., 0::2], q_rot[..., 1::2]
            k_even, k_odd = k_rot[..., 0::2], k_rot[..., 1::2]
            return (
                torch.cat((q_even * angles - q_odd * sine, q_odd * angles + q_even * sine), dim=-1),
                torch.cat((k_even * angles - k_odd * sine, k_odd * angles + k_even * sine), dim=-1),
            )
        q_first, q_second = q_rot[..., : rope_dim // 2], q_rot[..., rope_dim // 2 :]
        k_first, k_second = k_rot[..., : rope_dim // 2], k_rot[..., rope_dim // 2 :]
        full_cos = torch.cat((angles, angles), dim=-1)
        full_sin = torch.cat((sine, sine), dim=-1)
        return (
            q_rot * full_cos + torch.cat((-q_second, q_first), dim=-1) * full_sin,
            k_rot * full_cos + torch.cat((-k_second, k_first), dim=-1) * full_sin,
        )

    def select_topk(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = torch.as_tensor(hidden_states)
        q_resid = torch.as_tensor(q_resid)
        position_ids = torch.as_tensor(position_ids)
        if hidden_states.ndim != 3 or q_resid.shape != (*hidden_states.shape[:-1], self.q_lora_rank):
            raise ValueError("DSA_INDEXER_INPUT_SHAPE_MISMATCH")
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError("DSA_POSITION_SHAPE_MISMATCH")
        q = self.project_query(q_resid)
        k = self.project_keys(hidden_states).unsqueeze(2)
        q_rot, q_pass = q[..., : self.qk_rope_head_dim], q[..., self.qk_rope_head_dim :]
        k_rot, k_pass = k[..., : self.qk_rope_head_dim], k[..., self.qk_rope_head_dim :]
        q_rot, k_rot = self._rotate(q_rot, k_rot, *position_embeddings)
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)
        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1))
        scores = torch.relu(scores) * (self.index_head_dim**-0.5)
        weights = linear(hidden_states, self.weights_proj).float() * (self.index_n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        if attention_mask is not None:
            index_scores = index_scores + torch.as_tensor(attention_mask)
        else:
            if key_positions is None:
                key_positions = torch.arange(k.shape[1], device=k.device)
            key_positions = torch.as_tensor(key_positions, device=k.device)
            if key_positions.ndim == 1:
                key_positions = key_positions.unsqueeze(0)
            causal = key_positions[:, None, :] > position_ids[:, :, None]
            index_scores = index_scores.masked_fill(causal, float("-inf"))
        topk = min(self.index_topk, index_scores.shape[-1])
        return torch.topk(index_scores, k=topk, dim=-1).indices.to(torch.int32)

    @staticmethod
    def _apply_interleaved_rope(
        values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int
    ) -> torch.Tensor:
        if cos.shape != sin.shape or values.shape[:2] != cos.shape[:2]:
            raise ValueError("DSA_INCREMENTAL_ROPE_SHAPE_MISMATCH")
        if cos.ndim != 3 or values.ndim not in (3, 4) or rope_dim % 2:
            raise ValueError("DSA_INCREMENTAL_ROPE_RANK")
        half = rope_dim // 2
        angles = cos[..., :half]
        sine = sin[..., :half]
        if values.ndim == 4:
            angles = angles.unsqueeze(2)
            sine = sine.unsqueeze(2)
        rotated = values[..., :rope_dim]
        even, odd = rotated[..., 0::2], rotated[..., 1::2]
        encoded = torch.cat((even * angles - odd * sine, odd * angles + even * sine), dim=-1)
        return torch.cat((encoded, values[..., rope_dim:]), dim=-1)

    def select_topk_incremental(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        *,
        state: GLM5XOfficialDSAState | None = None,
    ) -> tuple[torch.Tensor, GLM5XOfficialDSAState]:
        """Run exact indexer scoring while appending only the new key positions."""
        hidden_states = torch.as_tensor(hidden_states)
        q_resid = torch.as_tensor(q_resid)
        position_ids = torch.as_tensor(position_ids)
        if hidden_states.ndim != 3 or q_resid.shape != (*hidden_states.shape[:-1], self.q_lora_rank):
            raise ValueError("DSA_INCREMENTAL_INPUT_SHAPE_MISMATCH")
        if position_ids.shape != hidden_states.shape[:2]:
            raise ValueError("DSA_INCREMENTAL_POSITION_SHAPE_MISMATCH")
        cos, sin = (torch.as_tensor(item) for item in position_embeddings)
        if cos.shape != sin.shape or cos.shape[:2] != position_ids.shape:
            raise ValueError("DSA_INCREMENTAL_POSITION_EMBEDDING_SHAPE")
        batch_size, seq_length, _ = hidden_states.shape
        if state is not None:
            if state.keys.ndim != 3 or state.positions.ndim != 2:
                raise ValueError("DSA_INCREMENTAL_STATE_RANK")
            if state.keys.shape[0] != batch_size or state.positions.shape != state.keys.shape[:2]:
                raise ValueError("DSA_INCREMENTAL_STATE_SHAPE")

        q = self.project_query(q_resid)
        q = self._apply_interleaved_rope(q, cos, sin, self.qk_rope_head_dim)
        current_keys = self.project_keys(hidden_states)
        current_keys = self._apply_interleaved_rope(
            current_keys, cos, sin, self.qk_rope_head_dim
        )
        if state is None:
            keys = current_keys
            key_positions = position_ids
        else:
            keys = torch.cat((state.keys, current_keys), dim=1)
            key_positions = torch.cat((state.positions, position_ids), dim=1)
        scores = torch.matmul(
            q.to(torch.float32), keys.to(torch.float32).transpose(-1, -2).unsqueeze(1)
        )
        scores = torch.relu(scores) * (self.index_head_dim**-0.5)
        weights = linear(hidden_states, self.weights_proj).float() * (self.index_n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
        causal = key_positions[:, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))
        topk = min(self.index_topk, index_scores.shape[-1])
        indices = torch.topk(index_scores, k=topk, dim=-1).indices.to(torch.int32)
        return indices, GLM5XOfficialDSAState(keys=keys, positions=key_positions)
