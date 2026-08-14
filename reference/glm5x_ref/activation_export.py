# GLM5X layer reference의 MoE 입력과 출력 기준값을 GLM5XACT로 내보냅니다.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from glm5x_converter.activation import write_bf16_activation

from .layer10_moe import (
    GLM5XExpertWeights,
    GLM5XLayer10MoEReference,
    GLM5XMoEForward,
)
from .layer_reference import GLM5XDecoderLayerReference


def _position_embeddings(
    token_count: int, *, rope_dim: int = 64, rope_theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(token_count, dtype=torch.long).view(1, -1)
    inverse = 1.0 / (
        rope_theta
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    frequencies = position_ids.to(torch.float32)[..., None] * inverse
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def _fp32_moe_reference(moe: GLM5XLayer10MoEReference) -> GLM5XLayer10MoEReference:
    def load_expert(expert_id: int) -> GLM5XExpertWeights:
        expert = moe.expert_loader(expert_id)
        return GLM5XExpertWeights(
            gate_proj=expert.gate_proj.to(torch.float32),
            up_proj=expert.up_proj.to(torch.float32),
            down_proj=expert.down_proj.to(torch.float32),
        )

    shared = moe.shared_expert
    return GLM5XLayer10MoEReference(
        router_weight=moe.router_weight.to(torch.float32),
        correction_bias=moe.correction_bias.to(torch.float32),
        expert_loader=load_expert,
        shared_expert=GLM5XExpertWeights(
            gate_proj=shared.gate_proj.to(torch.float32),
            up_proj=shared.up_proj.to(torch.float32),
            down_proj=shared.down_proj.to(torch.float32),
        ),
        top_k=moe.top_k,
        routed_scaling_factor=moe.routed_scaling_factor,
        n_group=moe.n_group,
        topk_group=moe.topk_group,
        norm_topk_prob=moe.norm_topk_prob,
        cache_experts=False,
    )


def _mixed_bf16_mlp(hidden: torch.Tensor, expert: GLM5XExpertWeights) -> torch.Tensor:
    work = hidden.to(torch.bfloat16).to(torch.float32)
    gate = F.linear(work, expert.gate_proj.to(torch.float32))
    up = F.linear(work, expert.up_proj.to(torch.float32))
    activated = (F.silu(gate) * up).to(torch.bfloat16)
    return F.linear(activated.to(torch.float32), expert.down_proj.to(torch.float32))


def _mixed_bf16_moe(
    hidden: torch.Tensor,
    moe: GLM5XLayer10MoEReference,
    routing: GLM5XMoEForward,
) -> torch.Tensor:
    flat = hidden.reshape(-1, moe.hidden_size).to(torch.float32)
    output = torch.zeros_like(flat)
    topk_indices = routing.topk_indices.reshape(-1, moe.top_k)
    topk_weights = routing.topk_weights.reshape(-1, moe.top_k)
    for expert_id_tensor in torch.unique(topk_indices, sorted=True):
        expert_id = int(expert_id_tensor)
        expert = moe.expert_loader(expert_id)
        slot_mask = topk_indices == expert_id
        token_indices, slots = torch.where(slot_mask)
        routed = _mixed_bf16_mlp(flat[token_indices], expert)
        weighted = routed * topk_weights[token_indices, slots].unsqueeze(-1)
        output.index_add_(0, token_indices, weighted)
    output += _mixed_bf16_mlp(flat, moe.shared_expert)
    return output.reshape(hidden.shape)


def export_moe_activation(
    bundle_path: str | Path,
    moe_input_path: str | Path,
    moe_output_path: str | Path,
    *,
    token_count: int = 2,
    seed: int = 17,
    layer_id: int = 10,
) -> dict[str, object]:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    torch.manual_seed(seed)
    hidden = torch.randn((1, token_count, 6144), dtype=torch.float32).bfloat16()
    position_ids = torch.arange(token_count, dtype=torch.long).view(1, -1)
    position_embeddings = _position_embeddings(token_count)
    layer = GLM5XDecoderLayerReference.from_bundle(
        bundle_path,
        layer_id=layer_id,
        cache_experts=False,
    )
    result = layer(
        hidden,
        position_embeddings,
        position_ids=position_ids,
    )
    moe_input = result.moe_input[0].detach().to(torch.bfloat16).cpu()
    fp32_moe = _fp32_moe_reference(layer.moe)
    rounded_moe_input = moe_input.unsqueeze(0).to(torch.float32)
    fp32_output = fp32_moe(rounded_moe_input)
    mixed_output = _mixed_bf16_moe(moe_input, layer.moe, fp32_output)
    moe_output = mixed_output.detach().to(torch.bfloat16).cpu()
    write_bf16_activation(moe_input_path, moe_input)
    write_bf16_activation(moe_output_path, moe_output)
    return {
        "bundle": str(bundle_path),
        "layer_id": layer_id,
        "token_count": token_count,
        "hidden_size": int(moe_input.shape[-1]),
        "seed": seed,
        "input_dtype": str(moe_input.dtype),
        "output_dtype": str(moe_output.dtype),
        "unique_routed_experts": int(torch.unique(fp32_output.topk_indices).numel()),
        "routed_experts": sorted(
            int(value) for value in torch.unique(fp32_output.topk_indices).tolist()
        ),
        "route_experts": [
            [int(value) for value in row]
            for row in fp32_output.topk_indices[0].tolist()
        ],
        "route_contributions": [
            [float(value) for value in row]
            for row in fp32_output.topk_weights[0].tolist()
        ],
        "moe_input": str(moe_input_path),
        "moe_output": str(moe_output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--moe-input", required=True, type=Path)
    parser.add_argument("--moe-output", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = export_moe_activation(
        args.bundle,
        args.moe_input,
        args.moe_output,
        token_count=args.tokens,
        seed=args.seed,
        layer_id=args.layer,
    )
    encoded = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.metadata is None:
        print(encoded, end="")
    else:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
