# GLM5X 한 레이어의 residual, DSA, MLA, MoE 순서를 CPU reference로 실행합니다.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import torch

from glm5x_converter.bundle import GLM5XExpertBundle

from .layer10_moe import (
    GLM5XDenseMlpReference,
    GLM5XExpertWeights,
    GLM5XLayer10MoEReference,
    GLM5XMoEForward,
    _collect_tensor_refs,
)
from .mla_dsa import GLM5XMLAForward, GLM5XMLAReference, GLM5XMLAState, GLM5XMLAWeights, _rms_norm
from .official_dsa import GLM5XOfficialDSAIndexer, GLM5XOfficialDSAState


@dataclass(frozen=True)
class GLM5XDecoderLayerForward:
    output: torch.Tensor
    moe_input: torch.Tensor
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
        moe: GLM5XLayer10MoEReference | GLM5XDenseMlpReference,
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

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        layer_id: int = 10,
        cache_experts: bool = False,
        num_heads: int = 64,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        index_topk: int = 2048,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        expert_intermediate_size: int = 2048,
        hidden_size: int = 6144,
        rms_norm_eps: float = 1e-5,
        verify_payloads: bool = True,
        verify_root: bool = True,
        mlp_type: str = "sparse",
        indexer_source_layer: int | None = None,
        indexer_rope_interleave: bool = True,
        device: torch.device | str | None = None,
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
    ) -> "GLM5XDecoderLayerReference":
        bundle = GLM5XExpertBundle.open(
            bundle_path, verify_payloads=verify_payloads, verify_root=verify_root
        )
        return cls.from_open_bundle(
            bundle,
            tensor_refs=_collect_tensor_refs(bundle),
            layer_id=layer_id,
            cache_experts=cache_experts,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            index_topk=index_topk,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            expert_intermediate_size=expert_intermediate_size,
            hidden_size=hidden_size,
            rms_norm_eps=rms_norm_eps,
            mlp_type=mlp_type,
            indexer_source_layer=indexer_source_layer,
            indexer_rope_interleave=indexer_rope_interleave,
            device=device,
            execution_mode=execution_mode,
            expert_load_workers=expert_load_workers,
        )

    @classmethod
    def bundle_layer_loader(
        cls,
        bundle_path: str | Path,
        *,
        cache_experts: bool = False,
        num_heads: int = 64,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        index_topk: int = 2048,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        expert_intermediate_size: int = 2048,
        hidden_size: int = 6144,
        rms_norm_eps: float = 1e-5,
        verify_payloads: bool = True,
        verify_root: bool = True,
        mlp_type: str = "sparse",
        indexer_source_layer: int | None = None,
        indexer_rope_interleave: bool = True,
        device: torch.device | str | None = None,
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
    ) -> Callable[[int], "GLM5XDecoderLayerReference"]:
        """Open and validate one bundle once, then provide individual layers."""
        bundle = GLM5XExpertBundle.open(
            bundle_path, verify_payloads=verify_payloads, verify_root=verify_root
        )
        refs = _collect_tensor_refs(bundle)

        def load(layer_id: int) -> "GLM5XDecoderLayerReference":
            return cls.from_open_bundle(
                bundle,
                tensor_refs=refs,
                layer_id=layer_id,
                cache_experts=cache_experts,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                index_topk=index_topk,
                top_k=top_k,
                routed_scaling_factor=routed_scaling_factor,
                expert_intermediate_size=expert_intermediate_size,
                hidden_size=hidden_size,
                rms_norm_eps=rms_norm_eps,
                mlp_type=mlp_type,
                indexer_source_layer=indexer_source_layer,
                indexer_rope_interleave=indexer_rope_interleave,
                device=device,
                execution_mode=execution_mode,
                expert_load_workers=expert_load_workers,
            )

        return load

    @classmethod
    def from_open_bundle(
        cls,
        bundle: GLM5XExpertBundle,
        *,
        tensor_refs: Mapping[str, tuple[object, object]],
        layer_id: int = 10,
        cache_experts: bool = False,
        num_heads: int = 64,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        index_topk: int = 2048,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        expert_intermediate_size: int = 2048,
        hidden_size: int = 6144,
        rms_norm_eps: float = 1e-5,
        mlp_type: str = "sparse",
        indexer_source_layer: int | None = None,
        indexer_rope_interleave: bool = True,
        device: torch.device | str | None = None,
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
    ) -> "GLM5XDecoderLayerReference":
        if mlp_type not in {"dense", "sparse"}:
            raise ValueError("GLM5X_LAYER_MLP_TYPE")
        target = None if device is None else torch.device(device)

        def read(name: str) -> torch.Tensor:
            value = GLM5XLayer10MoEReference._read_tensor(tensor_refs, name)
            return value if target is None else value.to(device=target)

        prefix = f"model.layers.{layer_id}"
        attention_prefix = f"{prefix}.self_attn"
        attention = GLM5XMLAReference(
            GLM5XMLAWeights(
                q_a_proj=read(f"{attention_prefix}.q_a_proj.weight"),
                q_a_norm=read(f"{attention_prefix}.q_a_layernorm.weight"),
                q_b_proj=read(f"{attention_prefix}.q_b_proj.weight"),
                kv_a_proj=read(f"{attention_prefix}.kv_a_proj_with_mqa.weight"),
                kv_a_norm=read(f"{attention_prefix}.kv_a_layernorm.weight"),
                kv_b_proj=read(f"{attention_prefix}.kv_b_proj.weight"),
                o_proj=read(f"{attention_prefix}.o_proj.weight"),
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                rms_norm_eps=rms_norm_eps,
            )
        )
        indexer_layer = layer_id if indexer_source_layer is None else indexer_source_layer
        indexer_prefix = f"model.layers.{indexer_layer}.self_attn.indexer"
        indexer = GLM5XOfficialDSAIndexer(
            wq_b=read(f"{indexer_prefix}.wq_b.weight"),
            wk=read(f"{indexer_prefix}.wk.weight"),
            k_norm_weight=read(f"{indexer_prefix}.k_norm.weight"),
            k_norm_bias=read(f"{indexer_prefix}.k_norm.bias"),
            weights_proj=read(f"{indexer_prefix}.weights_proj.weight"),
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            indexer_rope_interleave=indexer_rope_interleave,
        )
        if mlp_type == "dense":
            moe = GLM5XDenseMlpReference(
                GLM5XExpertWeights(
                    gate_proj=read(f"{prefix}.mlp.gate_proj.weight"),
                    up_proj=read(f"{prefix}.mlp.up_proj.weight"),
                    down_proj=read(f"{prefix}.mlp.down_proj.weight"),
                )
            )
        else:
            moe = GLM5XLayer10MoEReference._from_open_bundle(
                bundle,
                tensor_refs=tensor_refs,
                layer_id=layer_id,
                cache_experts=cache_experts,
                top_k=top_k,
                routed_scaling_factor=routed_scaling_factor,
                expert_intermediate_size=expert_intermediate_size,
                hidden_size=hidden_size,
                device=target,
                execution_mode=execution_mode,
                expert_load_workers=expert_load_workers,
            )
        return cls(
            input_layernorm=read(f"{prefix}.input_layernorm.weight"),
            attention=attention,
            post_attention_layernorm=read(f"{prefix}.post_attention_layernorm.weight"),
            moe=moe,
            dsa_indexer=indexer,
        )

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
            moe_input=moe_input,
            attention=attention,
            moe=moe,
            attention_state=attention.state,
            dsa_state=next_dsa_state,
            topk_indices=topk_indices,
        )
