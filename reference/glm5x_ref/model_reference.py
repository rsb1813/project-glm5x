# GLM5X 여러 decoder layer와 final logits의 CPU reference 실행을 제공합니다.
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .layer_reference import GLM5XDecoderLayerForward, GLM5XDecoderLayerReference
from .mla_dsa import GLM5XMLAState
from .official_dsa import GLM5XOfficialDSAState


@dataclass(frozen=True)
class GLM5XDecoderState:
    attention: tuple[GLM5XMLAState | None, ...]
    dsa: tuple[GLM5XOfficialDSAState | None, ...]


@dataclass(frozen=True)
class GLM5XModelForward:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    state: GLM5XDecoderState
    layers: tuple[GLM5XDecoderLayerForward, ...]


def _rope_embeddings(
    start: int,
    count: int,
    *,
    rope_dim: int,
    rope_theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(start, start + count, dtype=torch.float32, device=device)
    inverse = 1.0 / (
        rope_theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    frequencies = positions[:, None] * inverse
    frequencies = torch.cat((frequencies, frequencies), dim=-1).view(1, count, rope_dim)
    return frequencies.cos(), frequencies.sin()


class GLM5XDecoderModelReference:
    """여러 GLM decoder layer, final RMSNorm, LM head의 상태 보존 reference입니다."""

    def __init__(
        self,
        *,
        embedding: torch.Tensor,
        layers: Sequence[GLM5XDecoderLayerReference],
        final_norm: torch.Tensor,
        lm_head: torch.Tensor,
        rope_theta: float = 10000.0,
    ) -> None:
        embedding = torch.as_tensor(embedding)
        final_norm = torch.as_tensor(final_norm)
        lm_head = torch.as_tensor(lm_head)
        if embedding.ndim != 2 or embedding.shape[0] == 0 or embedding.shape[1] == 0:
            raise ValueError("GLM5X_MODEL_EMBEDDING_SHAPE")
        if not layers:
            raise ValueError("GLM5X_MODEL_LAYERS_REQUIRED")
        hidden_size = int(embedding.shape[1])
        if final_norm.shape != (hidden_size,):
            raise ValueError("GLM5X_MODEL_FINAL_NORM_SHAPE")
        if lm_head.ndim != 2 or lm_head.shape[1] != hidden_size or lm_head.shape[0] == 0:
            raise ValueError("GLM5X_MODEL_LM_HEAD_SHAPE")
        if any(layer.attention.weights.hidden_size != hidden_size for layer in layers):
            raise ValueError("GLM5X_MODEL_LAYER_HIDDEN_SHAPE")
        if rope_theta <= 0.0:
            raise ValueError("GLM5X_MODEL_ROPE_THETA")
        self.embedding = embedding
        self.layers = tuple(layers)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.rope_theta = float(rope_theta)

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def rope_dim(self) -> int:
        return int(self.layers[0].attention.weights.qk_rope_head_dim)

    def empty_state(self) -> GLM5XDecoderState:
        return GLM5XDecoderState(
            attention=tuple(None for _ in self.layers),
            dsa=tuple(None for _ in self.layers),
        )

    def _forward_hidden(
        self,
        hidden_states: torch.Tensor,
        state: GLM5XDecoderState,
        *,
        position_start: int,
    ) -> GLM5XModelForward:
        if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
            raise ValueError("GLM5X_MODEL_HIDDEN_SHAPE")
        if len(state.attention) != len(self.layers) or len(state.dsa) != len(self.layers):
            raise ValueError("GLM5X_MODEL_STATE_LAYERS")
        position_embeddings = _rope_embeddings(
            position_start,
            int(hidden_states.shape[1]),
            rope_dim=self.rope_dim,
            rope_theta=self.rope_theta,
            device=hidden_states.device,
        )
        layer_forwards: list[GLM5XDecoderLayerForward] = []
        next_attention: list[GLM5XMLAState | None] = []
        next_dsa: list[GLM5XOfficialDSAState | None] = []
        current = hidden_states
        position_ids = torch.arange(
            position_start,
            position_start + hidden_states.shape[1],
            dtype=torch.long,
            device=hidden_states.device,
        ).view(1, -1)
        for index, layer in enumerate(self.layers):
            result = layer(
                current,
                position_embeddings,
                position_ids=position_ids,
                attention_state=state.attention[index],
                dsa_state=state.dsa[index],
            )
            layer_forwards.append(result)
            current = result.output
            next_attention.append(result.attention_state)
            next_dsa.append(result.dsa_state)
        normalized = current
        squares = normalized.to(torch.float32).square().mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(squares + 1e-5)
        normalized = normalized * self.final_norm.to(device=normalized.device, dtype=normalized.dtype)
        logits = torch.matmul(
            normalized.to(torch.float32), self.lm_head.to(device=normalized.device, dtype=torch.float32).t()
        )
        return GLM5XModelForward(
            logits=logits,
            hidden_states=current,
            state=GLM5XDecoderState(tuple(next_attention), tuple(next_dsa)),
            layers=tuple(layer_forwards),
        )

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        state: GLM5XDecoderState | None = None,
    ) -> GLM5XModelForward:
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("GLM5X_MODEL_TOKEN_SHAPE")
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
            raise ValueError("GLM5X_MODEL_TOKEN_RANGE")
        state = self.empty_state() if state is None else state
        start = 0
        if state.attention and state.attention[0] is not None:
            start = state.attention[0].length
        hidden = self.embedding.to(device=tokens.device)[tokens].unsqueeze(0)
        return self._forward_hidden(hidden, state, position_start=start)

    def forward_token(
        self,
        token: int | torch.Tensor,
        state: GLM5XDecoderState,
    ) -> GLM5XModelForward:
        token_tensor = torch.as_tensor(token, dtype=torch.long).reshape(-1)
        if token_tensor.numel() != 1:
            raise ValueError("GLM5X_MODEL_SINGLE_TOKEN_REQUIRED")
        return self.forward_tokens(token_tensor, state)

    @torch.no_grad()
    def generate(self, prompt: Sequence[int], max_new_tokens: int) -> list[int]:
        if not prompt or max_new_tokens < 0:
            raise ValueError("GLM5X_MODEL_GENERATION_ARGUMENTS")
        state = self.empty_state()
        result = [int(token) for token in prompt]
        forward = self.forward_tokens(torch.tensor(result, dtype=torch.long), state)
        state = forward.state
        for _ in range(max_new_tokens):
            token = int(torch.argmax(forward.logits[:, -1, :], dim=-1).item())
            result.append(token)
            forward = self.forward_token(token, state)
            state = forward.state
        return result
