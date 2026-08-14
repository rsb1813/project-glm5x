# 작은 GLM5X MoE fixture의 recurrent attention, routing, 상태, 생성을 구현합니다.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GLM5XSyntheticConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    expert_hidden_size: int

    @classmethod
    def tiny(cls) -> "GLM5XSyntheticConfig":
        return cls(
            vocab_size=32,
            hidden_size=16,
            num_layers=2,
            num_experts=4,
            top_k=2,
            expert_hidden_size=24,
        )

    def validate(self) -> None:
        values = (
            self.vocab_size,
            self.hidden_size,
            self.num_layers,
            self.num_experts,
            self.top_k,
            self.expert_hidden_size,
        )
        if any(value <= 0 for value in values):
            raise ValueError("INVALID_SYNTHETIC_CONFIG")
        if self.top_k > self.num_experts:
            raise ValueError("INVALID_SYNTHETIC_TOP_K")


@dataclass(frozen=True)
class _AttentionState:
    key_sum: torch.Tensor
    value_sum: torch.Tensor
    count: int


@dataclass(frozen=True)
class GLM5XSyntheticState:
    attention: tuple[_AttentionState, ...]


class _ToyLayer(nn.Module):
    def __init__(self, config: GLM5XSyntheticConfig) -> None:
        super().__init__()
        h = config.hidden_size
        e = config.expert_hidden_size
        self.input_norm = nn.LayerNorm(h)
        self.q_proj = nn.Linear(h, h, bias=False)
        self.k_proj = nn.Linear(h, h, bias=False)
        self.v_proj = nn.Linear(h, h, bias=False)
        self.o_proj = nn.Linear(h, h, bias=False)
        self.post_attention_norm = nn.LayerNorm(h)
        self.router = nn.Linear(h, config.num_experts, bias=False)
        self.expert_up = nn.ModuleList(nn.Linear(h, e, bias=False) for _ in range(config.num_experts))
        self.expert_down = nn.ModuleList(nn.Linear(e, h, bias=False) for _ in range(config.num_experts))
        self.shared_up = nn.Linear(h, e, bias=False)
        self.shared_down = nn.Linear(e, h, bias=False)

    def forward_token(
        self,
        x: torch.Tensor,
        state: _AttentionState,
        top_k: int,
    ) -> tuple[torch.Tensor, _AttentionState, torch.Tensor]:
        normalized = self.input_norm(x)
        key = self.k_proj(normalized)
        value = self.v_proj(normalized)
        count = state.count + 1
        key_sum = state.key_sum + key
        value_sum = state.value_sum + value
        context = value_sum / count
        attention = self.o_proj(self.q_proj(normalized) * torch.sigmoid(context))
        x = x + attention

        routed = self.post_attention_norm(x)
        scores = self.router(routed)
        values, indices = torch.topk(scores, k=top_k, dim=-1)
        weights = torch.softmax(values, dim=-1)
        mixture = torch.zeros_like(x)
        for weight, index in zip(weights, indices):
            expert = int(index.item())
            mixture = mixture + weight * self.expert_down[expert](
                F.silu(self.expert_up[expert](routed))
            )
        shared = self.shared_down(F.silu(self.shared_up(routed)))
        return x + mixture + shared, _AttentionState(key_sum, value_sum, count), indices


class GLM5XSyntheticModel(nn.Module):
    """실제 checkpoint가 아닌 GLM5X routing/state contract fixture입니다."""

    def __init__(self, config: GLM5XSyntheticConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(_ToyLayer(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def empty_state(self) -> GLM5XSyntheticState:
        zero = torch.zeros(self.config.hidden_size)
        return GLM5XSyntheticState(
            tuple(
                _AttentionState(zero.clone(), zero.clone(), 0)
                for _ in range(self.config.num_layers)
            )
        )

    def forward_token(
        self,
        token: torch.Tensor | int,
        state: GLM5XSyntheticState,
    ) -> tuple[torch.Tensor, GLM5XSyntheticState, tuple[torch.Tensor, ...]]:
        token_tensor = torch.as_tensor(token, dtype=torch.long)
        x = self.embedding(token_tensor)
        next_attention: list[_AttentionState] = []
        routes: list[torch.Tensor] = []
        for layer, layer_state in zip(self.layers, state.attention):
            x, next_state, selected = layer.forward_token(
                x, layer_state, self.config.top_k
            )
            next_attention.append(next_state)
            routes.append(selected.detach())
        logits = self.lm_head(self.final_norm(x))
        return logits, GLM5XSyntheticState(tuple(next_attention)), tuple(routes)

    def forward(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, ...], ...]]:
        state = self.empty_state()
        logits: list[torch.Tensor] = []
        routes: list[tuple[torch.Tensor, ...]] = []
        for token in tokens.reshape(-1):
            output, state, selected = self.forward_token(token, state)
            logits.append(output)
            routes.append(selected)
        return torch.stack(logits), tuple(routes)

    @torch.no_grad()
    def generate(self, prompt: list[int], max_new_tokens: int) -> list[int]:
        state = self.empty_state()
        generated = list(prompt)
        logits = None
        for token in prompt:
            logits, state, _ = self.forward_token(token, state)
        for _ in range(max_new_tokens):
            assert logits is not None
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)
            logits, state, _ = self.forward_token(next_token, state)
        return generated
