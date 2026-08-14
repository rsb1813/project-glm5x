# GLM5X 한 레이어의 residual, DSA, MLA, MoE 순서를 CPU reference로 실행합니다.
from __future__ import annotations

from dataclasses import dataclass

import torch

from .layer10_moe import GLM5XLayer10MoEReference, GLM5XMoEForward
from .mla_dsa import GLM5XMLAForward, GLM5XMLAReference, GLM5XMLAState, _rms_norm
from .official_dsa import GLM5XOfficialDSAIndexer, GLM5XOfficialDSAState


@dataclass(frozen=True)
class GLM5XDecoderLayerForward:
    output: torch.Tensor
    attention: GLM5XMLAForward
    moe: GLM5XMoEForward
    attention_state: GLM5XMLAState
    dsa_state: GLM5XOfficialDSAState | None
    topk_indices: torch.Tensor | None


class GLM5XDecoderLayerReference:
    """Official decoder-layer ordering with exact incremental state boundaries."""

    def __init__(
        self,
        *,
        input_layernorm: torch.Tensor,
        attention: GLM5XMLAReference,
        post_attention_layernorm: torch.Tensor,
        moe: GLM5XLayer10MoEReference,
        dsa_indexer: GLM5XOfficialDSAIndexer | None = None,
    ) -> None:
        input_layernorm = torch.as_tensor(input_layernorm)
        post_attention_layernorm = torch.as_tensor(post_attention_layernorm)
        if input_layernorm.ndim != 1 or post_attention_layernorm.shape != input_layernorm.shape:
            raise ValueError("GLM5X_LAYER_NORM_SHAPE")
        if input_layernorm.shape != (attention.weights.hidden_size,):
            raise ValueError("GLM5X_LAYER_HIDDEN_SHAPE")
        if dsa_indexer is not None and dsa_indexer.hidden_size != attention.weights.hidden_size:
            raise ValueError("GLM5X_LAYER_DSA_HIDDEN_SHAPE")
        if dsa_indexer is not None and dsa_indexer.q_lora_rank != attention.weights.q_lora_rank:
            raise ValueError("GLM5X_LAYER_DSA_Q_RANK")
        self.input_layernorm = input_layernorm
        self.attention = attention
        self.post_attention_layernorm = post_attention_layernorm
        self.moe = moe
        self.dsa_indexer = dsa_indexer

    def __call__(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        position_ids: torch.Tensor | None = None,
        attention_state: GLM5XMLAState | None = None,
        dsa_state: GLM5XOfficialDSAState | None = None,
    ) -> GLM5XDecoderLayerForward:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.attention.weights.hidden_size:
            raise ValueError("GLM5X_LAYER_INPUT_SHAPE")
        normalized = _rms_norm(
            hidden_states, self.input_layernorm, self.attention.weights.rms_norm_eps
        )
        q_resid = self.attention.q_residual(normalized)
        if self.dsa_indexer is None:
            if dsa_state is not None:
                raise ValueError("GLM5X_LAYER_DSA_STATE_WITHOUT_INDEXER")
            topk_indices = None
            next_dsa_state = None
        else:
            topk_indices, next_dsa_state = self.dsa_indexer.select_topk_incremental(
                normalized,
                q_resid,
                position_embeddings,
                torch.arange(
                    attention_state.length if attention_state is not None else 0,
                    (attention_state.length if attention_state is not None else 0) + hidden_states.shape[1],
                    device=hidden_states.device,
                ).view(1, -1).expand(hidden_states.shape[0], -1)
                if position_ids is None
                else position_ids,
                state=dsa_state,
            )
        attention = self.attention(
            normalized,
            position_embeddings,
            position_ids=position_ids,
            state=attention_state,
            topk_indices=topk_indices,
        )
        post_attention = hidden_states + attention.output
        moe_input = _rms_norm(
            post_attention,
            self.post_attention_layernorm,
            self.attention.weights.rms_norm_eps,
        )
        moe = self.moe(moe_input)
        output = post_attention + moe.output
        return GLM5XDecoderLayerForward(
            output=output,
            attention=attention,
            moe=moe,
            attention_state=attention.state,
            dsa_state=next_dsa_state,
            topk_indices=topk_indices,
        )
