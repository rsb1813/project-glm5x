# 결정적 합성 K3 weight와 source checkpoint 및 golden fixture를 생성합니다.
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from k3x_ref.config import SyntheticK3Config
from k3x_ref.kda import KDAState, KDAWeights
from k3x_ref.mla import MLAState, MLAWeights
from k3x_ref.model import (
    AttnResWeights,
    DenseWeights,
    LayerWeights,
    ModelWeights,
    SyntheticK3Model,
)
from k3x_ref.moe import ExpertWeights, LatentMoEWeights, PackedMatrix
from k3x_ref.mxfp4 import decode_mxfp4
from k3x_ref.ops import situ_glu


def _packed_random(
    rows: int, cols: int, generator: torch.Generator, scale_byte: int = 120
) -> PackedMatrix:
    codes = torch.randint(0, 16, (rows * cols,), generator=generator, dtype=torch.uint8)
    packed = (codes[0::2] | (codes[1::2] << 4)).tolist()
    scales = bytes([scale_byte]) * (rows * cols // 32)
    return PackedMatrix(bytes(packed), scales, rows, cols)


def _packed_zero(rows: int, cols: int) -> PackedMatrix:
    return PackedMatrix(bytes(rows * cols // 2), bytes([127]) * (rows * cols // 32), rows, cols)


def _build_weights(
    cfg: SyntheticK3Config,
    seed: int,
    controlled: bool,
) -> ModelWeights:
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int, scale: float = 0.03) -> torch.Tensor:
        if controlled:
            return torch.zeros(shape)
        return torch.randn(shape, generator=generator) * scale

    projection = cfg.kda_heads * cfg.kda_head_dim
    query_width = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    layers: list[LayerWeights] = []
    for layer_index, kind in enumerate(cfg.layer_kinds):
        if kind == "kda":
            q_proj = rand(projection, cfg.hidden_size)
            k_proj = rand(projection, cfg.hidden_size)
            v_proj = rand(projection, cfg.hidden_size)
            q_conv = rand(projection, cfg.short_conv_kernel_size)
            k_conv = rand(projection, cfg.short_conv_kernel_size)
            v_conv = rand(projection, cfg.short_conv_kernel_size)
            o_proj = rand(cfg.hidden_size, projection)
            if controlled:
                q_proj.copy_(torch.eye(projection, cfg.hidden_size))
                k_proj.copy_(q_proj)
                v_proj.copy_(q_proj)
                q_conv[:, -1] = 1.0
                k_conv[:, -1] = 1.0
                v_conv[:, -1] = 1.0
                o_proj.copy_(torch.eye(cfg.hidden_size, projection))
            attention: KDAWeights | MLAWeights = KDAWeights(
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                q_conv=q_conv,
                k_conv=k_conv,
                v_conv=v_conv,
                f_a_proj=rand(cfg.kda_head_dim, cfg.hidden_size),
                f_b_proj=rand(projection, cfg.kda_head_dim),
                b_proj=rand(cfg.kda_heads, cfg.hidden_size),
                a_log=rand(cfg.kda_heads),
                dt_bias=rand(cfg.kda_heads, cfg.kda_head_dim),
                g_proj=rand(projection, cfg.hidden_size),
                o_norm=torch.ones(cfg.kda_head_dim),
                o_proj=o_proj,
            )
        else:
            attention = MLAWeights(
                q_a_proj=rand(cfg.q_lora_rank, cfg.hidden_size),
                q_a_norm=torch.ones(cfg.q_lora_rank),
                q_b_proj=rand(cfg.mla_heads * query_width, cfg.q_lora_rank),
                kv_a_proj=rand(cfg.kv_lora_rank + cfg.qk_rope_head_dim, cfg.hidden_size),
                kv_a_norm=torch.ones(cfg.kv_lora_rank),
                kv_b_proj=rand(
                    cfg.mla_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
                    cfg.kv_lora_rank,
                ),
                g_proj=rand(cfg.mla_heads * cfg.v_head_dim, cfg.hidden_size),
                o_proj=rand(cfg.hidden_size, cfg.mla_heads * cfg.v_head_dim),
            )

        if layer_index in cfg.dense_layers:
            feed_forward: DenseWeights | LatentMoEWeights = DenseWeights(
                gate=rand(cfg.dense_intermediate_size, cfg.hidden_size),
                up=rand(cfg.dense_intermediate_size, cfg.hidden_size),
                down=rand(cfg.hidden_size, cfg.dense_intermediate_size),
            )
        else:
            experts = tuple(
                ExpertWeights(
                    gate=(
                        _packed_zero(cfg.expert_intermediate_size, cfg.routed_latent_size)
                        if controlled
                        else _packed_random(
                            cfg.expert_intermediate_size,
                            cfg.routed_latent_size,
                            generator,
                        )
                    ),
                    up=(
                        _packed_zero(cfg.expert_intermediate_size, cfg.routed_latent_size)
                        if controlled
                        else _packed_random(
                            cfg.expert_intermediate_size,
                            cfg.routed_latent_size,
                            generator,
                        )
                    ),
                    down=(
                        _packed_zero(cfg.routed_latent_size, cfg.expert_intermediate_size)
                        if controlled
                        else _packed_random(
                            cfg.routed_latent_size,
                            cfg.expert_intermediate_size,
                            generator,
                        )
                    ),
                )
                for _ in range(cfg.num_experts)
            )
            feed_forward = LatentMoEWeights(
                router_weight=rand(cfg.num_experts, cfg.hidden_size),
                correction_bias=rand(cfg.num_experts),
                routed_down_proj=rand(cfg.routed_latent_size, cfg.hidden_size),
                routed_up_proj=rand(cfg.hidden_size, cfg.routed_latent_size),
                routed_norm=torch.ones(cfg.routed_latent_size),
                experts=experts,
                shared_gate=rand(cfg.expert_intermediate_size, cfg.hidden_size),
                shared_up=rand(cfg.expert_intermediate_size, cfg.hidden_size),
                shared_down=rand(cfg.hidden_size, cfg.expert_intermediate_size),
            )
        layers.append(
            LayerWeights(
                input_norm=torch.ones(cfg.hidden_size),
                post_attention_norm=torch.ones(cfg.hidden_size),
                attention=attention,
                feed_forward=feed_forward,
                self_attention_residual=AttnResWeights(
                    torch.ones(cfg.hidden_size), rand(cfg.hidden_size, scale=0.01)
                ),
                mlp_residual=AttnResWeights(
                    torch.ones(cfg.hidden_size), rand(cfg.hidden_size, scale=0.01)
                ),
            )
        )

    embeddings = rand(cfg.vocab_size, cfg.hidden_size, scale=0.05)
    lm_head = rand(cfg.vocab_size, cfg.hidden_size, scale=0.05)
    if controlled:
        embeddings.fill_(1.0)
        lm_head.zero_()
        lm_head[5].fill_(1.0)
    return ModelWeights(
        embeddings=embeddings,
        layers=tuple(layers),
        output_residual=AttnResWeights(
            torch.ones(cfg.hidden_size), rand(cfg.hidden_size, scale=0.01)
        ),
        final_norm=torch.ones(cfg.hidden_size),
        lm_head=lm_head,
    )


def build_synthetic_model(
    seed: int = 20260808,
    controlled: bool = False,
    config: SyntheticK3Config | None = None,
) -> SyntheticK3Model:
    cfg = config or SyntheticK3Config.default()
    return SyntheticK3Model(cfg, _build_weights(cfg, seed, controlled))


def _collect_tensors(
    value: Any,
    prefix: str,
    output: dict[str, torch.Tensor],
    packed_shapes: dict[str, list[int]],
) -> None:
    if isinstance(value, torch.Tensor):
        output[prefix] = value.detach().cpu().contiguous()
    elif isinstance(value, PackedMatrix):
        output[f"{prefix}.weight_packed"] = torch.tensor(list(value.packed), dtype=torch.uint8)
        output[f"{prefix}.weight_scale"] = torch.tensor(list(value.scales), dtype=torch.uint8)
        packed_shapes[prefix] = [value.rows, value.cols]
    elif is_dataclass(value):
        for field in fields(value):
            _collect_tensors(
                getattr(value, field.name), f"{prefix}.{field.name}", output, packed_shapes
            )
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            _collect_tensors(item, f"{prefix}.{index}", output, packed_shapes)


def write_source_checkpoint(
    path: Path,
    seed: int = 20260808,
    config: SyntheticK3Config | None = None,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    model = build_synthetic_model(seed, config=config)
    tensors: dict[str, torch.Tensor] = {}
    packed_shapes: dict[str, list[int]] = {}
    _collect_tensors(model.weights, "model", tensors, packed_shapes)
    names = sorted(tensors)
    midpoint = (len(names) + 1) // 2
    shard_names = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    weight_map: dict[str, str] = {}
    for shard_index, shard_name in enumerate(shard_names):
        shard_tensor_names = names[:midpoint] if shard_index == 0 else names[midpoint:]
        save_file({name: tensors[name] for name in shard_tensor_names}, path / shard_name)
        weight_map.update({name: shard_name for name in shard_tensor_names})
    manifest = {
        "format": "synthetic-k3-source-v1",
        "seed": seed,
        "config": model.cfg.__dict__,
        "weight_map": weight_map,
        "packed_shapes": packed_shapes,
    }
    manifest_path = path / "source-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def write_golden(
    path: Path,
    model: SyntheticK3Model,
    prompt_ids: list[int],
) -> None:
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    logits, state, layers = model.prefill_with_trace(prompt)
    tokens = model.generate_greedy(prompt_ids, 6, incremental=True)
    primitive_gate = torch.tensor([[-1.0, 2.0]])
    primitive_up = torch.tensor([[3.0, 4.0]])
    arrays: dict[str, np.ndarray] = {
        "logits": logits.numpy(),
        "token_ids": np.asarray(tokens, dtype=np.uint32),
        "state_sha256": np.frombuffer(bytes.fromhex(model.state_sha256(state)), dtype=np.uint8),
        "situ_literal": situ_glu(primitive_gate, primitive_up, 1.0, 1.0).numpy(),
        "mxfp4_literal": decode_mxfp4(bytes([0x10] + [0] * 15), bytes([127]), 1, 32).numpy(),
    }
    arrays.update({f"layer_{index}": value.numpy() for index, value in enumerate(layers)})
    for index, item in enumerate(state.attention):
        if isinstance(item, KDAState):
            arrays[f"state.{index}.conv_q"] = item.conv_q.numpy()
            arrays[f"state.{index}.conv_k"] = item.conv_k.numpy()
            arrays[f"state.{index}.conv_v"] = item.conv_v.numpy()
            arrays[f"state.{index}.recurrent"] = item.recurrent.numpy()
        elif isinstance(item, MLAState):
            arrays[f"state.{index}.keys"] = item.keys.numpy()
            arrays[f"state.{index}.values"] = item.values.numpy()
            arrays[f"state.{index}.shared_keys"] = item.shared_keys.numpy()
            arrays[f"state.{index}.length"] = np.asarray(item.length, dtype=np.uint64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_digest_manifest(root: Path) -> Path:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "manifest.sha256"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output = root / "manifest.sha256"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
