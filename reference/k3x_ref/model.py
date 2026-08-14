# 합성 K3 decoder layer와 stateful greedy generation을 구성합니다.
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as functional

from k3x_ref.attn_res import apply_attn_res
from k3x_ref.config import SyntheticK3Config
from k3x_ref.kda import KDAState, KDAWeights, empty_kda_state, kda_prefill
from k3x_ref.mla import MLAState, MLAWeights, empty_mla_state, mla_prefill
from k3x_ref.moe import LatentMoEWeights, stable_latent_moe
from k3x_ref.ops import rms_norm, situ_glu
from k3x_ref.routing_policy import RoutingPolicyConfig


@dataclass(frozen=True)
class DenseWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class AttnResWeights:
    norm: torch.Tensor
    projection: torch.Tensor


AttentionWeights: TypeAlias = KDAWeights | MLAWeights
FeedForwardWeights: TypeAlias = DenseWeights | LatentMoEWeights
AttentionState: TypeAlias = KDAState | MLAState


@dataclass(frozen=True)
class LayerWeights:
    input_norm: torch.Tensor
    post_attention_norm: torch.Tensor
    attention: AttentionWeights
    feed_forward: FeedForwardWeights
    self_attention_residual: AttnResWeights
    mlp_residual: AttnResWeights


@dataclass(frozen=True)
class ModelWeights:
    embeddings: torch.Tensor
    layers: tuple[LayerWeights, ...]
    output_residual: AttnResWeights
    final_norm: torch.Tensor
    lm_head: torch.Tensor


@dataclass(frozen=True)
class ModelState:
    attention: tuple[AttentionState, ...]
    position: int


def dense_mlp(
    hidden: torch.Tensor,
    weights: DenseWeights,
    cfg: SyntheticK3Config,
) -> torch.Tensor:
    gate = functional.linear(hidden, weights.gate)
    up = functional.linear(hidden, weights.up)
    return functional.linear(
        situ_glu(
            gate,
            up,
            cfg.activation_situ_beta,
            cfg.activation_situ_linear_beta,
        ),
        weights.down,
    )


class SyntheticK3Model:
    def __init__(
        self,
        cfg: SyntheticK3Config,
        weights: ModelWeights,
        routing_policy: RoutingPolicyConfig | None = None,
    ):
        cfg.validate()
        if len(weights.layers) != len(cfg.layer_kinds):
            raise ValueError("weights.layers must match layer_kinds")
        self.cfg = cfg
        self.weights = weights
        self.routing_policy = routing_policy

    def empty_state(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> ModelState:
        states: list[AttentionState] = []
        for kind in self.cfg.layer_kinds:
            if kind == "kda":
                states.append(empty_kda_state(batch_size, self.cfg, dtype, device))
            else:
                states.append(empty_mla_state(batch_size, self.cfg, dtype, device))
        return ModelState(tuple(states), 0)

    def _forward(
        self,
        token_ids: torch.Tensor,
        state: ModelState | None,
        capture_layers: bool,
    ) -> tuple[torch.Tensor, ModelState, tuple[torch.Tensor, ...]]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence]")
        hidden = functional.embedding(token_ids, self.weights.embeddings)
        current = state or self.empty_state(
            token_ids.shape[0], hidden.dtype, hidden.device
        )
        if len(current.attention) != len(self.weights.layers):
            raise ValueError("state layer count mismatch")

        batch, sequence, width = hidden.shape
        block_sources = torch.empty(
            (batch * sequence, 0, width), dtype=hidden.dtype, device=hidden.device
        )
        next_states: list[AttentionState] = []
        layer_outputs: list[torch.Tensor] = []
        for layer_index, (kind, layer, layer_state) in enumerate(
            zip(self.cfg.layer_kinds, self.weights.layers, current.attention, strict=True)
        ):
            prefix_sum = hidden
            if block_sources.shape[1]:
                attention_input = apply_attn_res(
                    prefix_sum.reshape(-1, width),
                    block_sources,
                    layer.self_attention_residual.norm,
                    layer.self_attention_residual.projection,
                    self.cfg.rms_norm_eps,
                ).reshape(batch, sequence, width)
            else:
                attention_input = prefix_sum

            pushed = layer_index % self.cfg.attn_res_block_size == 0
            if pushed:
                block_sources = torch.cat(
                    (block_sources, prefix_sum.reshape(-1, width).unsqueeze(1)), dim=1
                )
            normalized = rms_norm(
                attention_input, layer.input_norm, self.cfg.rms_norm_eps
            )
            if kind == "kda":
                if not isinstance(layer.attention, KDAWeights) or not isinstance(
                    layer_state, KDAState
                ):
                    raise TypeError("KDA layer weight or state type mismatch")
                attention_output, next_state = kda_prefill(
                    normalized, layer.attention, layer_state, self.cfg
                )
            else:
                if not isinstance(layer.attention, MLAWeights) or not isinstance(
                    layer_state, MLAState
                ):
                    raise TypeError("MLA layer weight or state type mismatch")
                attention_output, next_state = mla_prefill(
                    normalized, layer.attention, layer_state, self.cfg
                )
            next_states.append(next_state)
            prefix_sum = attention_output if pushed else prefix_sum + attention_output

            ffn_input = apply_attn_res(
                prefix_sum.reshape(-1, width),
                block_sources,
                layer.mlp_residual.norm,
                layer.mlp_residual.projection,
                self.cfg.rms_norm_eps,
            ).reshape(batch, sequence, width)
            normalized_ffn = rms_norm(
                ffn_input, layer.post_attention_norm, self.cfg.rms_norm_eps
            )
            if isinstance(layer.feed_forward, DenseWeights):
                ffn_output = dense_mlp(normalized_ffn, layer.feed_forward, self.cfg)
            else:
                ffn_output = stable_latent_moe(
                    normalized_ffn,
                    layer.feed_forward,
                    self.cfg,
                    self.routing_policy,
                )
            hidden = prefix_sum + ffn_output
            if capture_layers:
                layer_outputs.append(hidden.detach().clone())

        hidden = apply_attn_res(
            hidden.reshape(-1, width),
            block_sources,
            self.weights.output_residual.norm,
            self.weights.output_residual.projection,
            self.cfg.rms_norm_eps,
        ).reshape(batch, sequence, width)
        hidden = rms_norm(hidden, self.weights.final_norm, self.cfg.rms_norm_eps)
        logits = functional.linear(hidden, self.weights.lm_head).to(torch.float32)
        return (
            logits,
            ModelState(tuple(next_states), current.position + sequence),
            tuple(layer_outputs),
        )

    def prefill(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, ModelState]:
        logits, state, _ = self._forward(token_ids, None, False)
        return logits, state

    def prefill_with_trace(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ModelState, tuple[torch.Tensor, ...]]:
        return self._forward(token_ids, None, True)

    def decode(
        self, token_ids: torch.Tensor, state: ModelState
    ) -> tuple[torch.Tensor, ModelState]:
        logits, next_state, _ = self._forward(token_ids.reshape(-1, 1), state, False)
        return logits, next_state

    def generate_greedy(
        self,
        prompt_ids: list[int],
        count: int,
        incremental: bool,
    ) -> list[int]:
        if not prompt_ids or count < 0:
            raise ValueError("prompt_ids must be non-empty and count non-negative")
        if count == 0:
            return []
        generated: list[int] = []
        if incremental:
            prompt = torch.tensor([prompt_ids], dtype=torch.long)
            logits, state = self.prefill(prompt)
            for index in range(count):
                token = int(torch.argmax(logits[0, -1]))
                generated.append(token)
                if index + 1 < count:
                    logits, state = self.decode(torch.tensor([token]), state)
            return generated

        sequence = list(prompt_ids)
        for _ in range(count):
            logits, _ = self.prefill(torch.tensor([sequence], dtype=torch.long))
            token = int(torch.argmax(logits[0, -1]))
            generated.append(token)
            sequence.append(token)
        return generated

    @staticmethod
    def state_sha256(state: ModelState) -> str:
        digest = hashlib.sha256()
        digest.update(state.position.to_bytes(8, "little"))
        for item in state.attention:
            tensors = (
                (item.conv_q, item.conv_k, item.conv_v, item.recurrent)
                if isinstance(item, KDAState)
                else (item.keys, item.values, item.shared_keys)
            )
            if isinstance(item, MLAState):
                digest.update(item.length.to_bytes(8, "little"))
            for tensor in tensors:
                contiguous = tensor.detach().cpu().contiguous()
                digest.update(str(tuple(contiguous.shape)).encode("ascii"))
                digest.update(str(contiguous.dtype).encode("ascii"))
                digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

