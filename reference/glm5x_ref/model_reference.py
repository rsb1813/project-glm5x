# GLM5X 여러 decoder layer와 final logits의 CPU reference 실행을 제공합니다.
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from glm5x_converter.bundle import GLM5XExpertBundle, GLM5XExpertPayloadCacheStats

from .layer_reference import GLM5XDecoderLayerForward, GLM5XDecoderLayerReference
from .layer10_moe import (
    GLM5XExpertTensorCache,
    GLM5XExpertTensorCacheStats,
    GLM5XLayer10MoEReference,
    GLM5XTrunkTensorCache,
    GLM5XTrunkTensorCacheStats,
    _collect_tensor_refs,
)
from .packed_cache import GLM5XPackedExpertCache
from .model import GLM5XModelDescriptor
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


def _positive_config_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"GLM5X_BUNDLE_CONFIG_{key.upper()}")
    return value


def _positive_config_float(
    config: Mapping[str, object], key: str, *, default: float | None = None
) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"GLM5X_BUNDLE_CONFIG_{key.upper()}")
    return float(value)


def _bundle_mlp_types(config: Mapping[str, object], layer_count: int) -> tuple[str, ...]:
    raw_types = config.get("mlp_layer_types")
    if raw_types is not None:
        if (
            not isinstance(raw_types, list)
            or len(raw_types) != layer_count
            or any(item not in {"dense", "sparse"} for item in raw_types)
        ):
            raise ValueError("GLM5X_BUNDLE_MLP_TYPES")
        return tuple(str(item) for item in raw_types)
    first_dense = config.get("first_k_dense_replace", 0)
    if not isinstance(first_dense, int) or isinstance(first_dense, bool) or first_dense < 0:
        raise ValueError("GLM5X_BUNDLE_FIRST_DENSE")
    if first_dense > layer_count:
        raise ValueError("GLM5X_BUNDLE_FIRST_DENSE")
    return ("dense",) * first_dense + ("sparse",) * (layer_count - first_dense)


def _bundle_indexer_sources(config: Mapping[str, object], layer_count: int) -> tuple[int, ...]:
    raw_types = config.get("indexer_types")
    if raw_types is None:
        return tuple(range(layer_count))
    if (
        not isinstance(raw_types, list)
        or len(raw_types) != layer_count
        or any(item not in {"full", "shared"} for item in raw_types)
    ):
        raise ValueError("GLM5X_BUNDLE_INDEXER_TYPES")
    sources: list[int] = []
    for layer_id, indexer_type in enumerate(raw_types):
        if indexer_type == "full":
            sources.append(layer_id)
            continue
        if not sources:
            raise ValueError("GLM5X_BUNDLE_INDEXER_SOURCE")
        sources.append(sources[-1])
    return tuple(sources)


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
        self._layer_loader: Callable[[int], GLM5XDecoderLayerReference] | None = None
        self._layer_count = len(self.layers)
        self._layer_cache_capacity = 0
        self._layer_cache: OrderedDict[int, GLM5XDecoderLayerReference] = OrderedDict()
        self._trunk_tensor_cache = None
        self._packed_expert_cache = None
        self._rope_dim = int(self.layers[0].attention.weights.qk_rope_head_dim)
        self.final_norm = final_norm
        self.lm_head = lm_head
        self._prepared_lm_head: torch.Tensor | None = None
        self.rope_theta = float(rope_theta)

    @classmethod
    def from_layer_loader(
        cls,
        *,
        embedding: torch.Tensor,
        layer_count: int,
        layer_loader: Callable[[int], GLM5XDecoderLayerReference],
        final_norm: torch.Tensor,
        lm_head: torch.Tensor,
        rope_dim: int = 64,
        rope_theta: float = 10000.0,
        layer_cache_capacity: int = 0,
    ) -> "GLM5XDecoderModelReference":
        """Create a reference model whose layer weights are loaded per forward.

        ``layer_cache_capacity`` keeps validated layer objects (including their
        non-expert trunk tensors) between forwards. Zero preserves the strict
        out-of-core behavior and does not retain loaded layers.
        """
        if not isinstance(layer_count, int) or layer_count <= 0:
            raise ValueError("GLM5X_MODEL_LAYER_COUNT")
        if not callable(layer_loader):
            raise ValueError("GLM5X_MODEL_LAYER_LOADER")
        if not isinstance(rope_dim, int) or rope_dim <= 0 or rope_dim % 2:
            raise ValueError("GLM5X_MODEL_ROPE_DIM")
        if not isinstance(layer_cache_capacity, int) or layer_cache_capacity < 0:
            raise ValueError("GLM5X_MODEL_LAYER_CACHE_CAPACITY")
        embedding = torch.as_tensor(embedding)
        final_norm = torch.as_tensor(final_norm)
        lm_head = torch.as_tensor(lm_head)
        if embedding.ndim != 2 or embedding.shape[0] == 0 or embedding.shape[1] == 0:
            raise ValueError("GLM5X_MODEL_EMBEDDING_SHAPE")
        hidden_size = int(embedding.shape[1])
        if final_norm.shape != (hidden_size,):
            raise ValueError("GLM5X_MODEL_FINAL_NORM_SHAPE")
        if lm_head.ndim != 2 or lm_head.shape[1] != hidden_size or lm_head.shape[0] == 0:
            raise ValueError("GLM5X_MODEL_LM_HEAD_SHAPE")
        if rope_theta <= 0.0:
            raise ValueError("GLM5X_MODEL_ROPE_THETA")
        instance = cls.__new__(cls)
        instance.embedding = embedding
        instance.layers = None
        instance._layer_loader = layer_loader
        instance._layer_count = layer_count
        instance._rope_dim = rope_dim
        instance._layer_cache_capacity = min(layer_cache_capacity, layer_count)
        instance._layer_cache = OrderedDict()
        instance._expert_bundle = None
        instance._expert_device_cache = None
        instance._trunk_tensor_cache = None
        instance._packed_expert_cache = None
        instance.final_norm = final_norm
        instance.lm_head = lm_head
        instance._prepared_lm_head = None
        instance.rope_theta = float(rope_theta)
        return instance

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        config: Mapping[str, object],
        embedding: torch.Tensor | None = None,
        final_norm: torch.Tensor | None = None,
        lm_head: torch.Tensor | None = None,
        cache_experts: bool = False,
        verify_payloads: bool = True,
        verify_root: bool = True,
        layer_cache_capacity: int = 0,
        device: torch.device | str | None = None,
        execution_mode: str = "loop",
        use_sparse_topk: bool = False,
        expert_load_workers: int = 1,
        expert_cache_capacity_bytes: int = 0,
        expert_device_cache_capacity_bytes: int = 0,
        expert_device_cache_policy: str = "lru",
        expert_device_cache_protected_entries_per_layer: int = 0,
        trunk_cache_capacity_bytes: int = 0,
        packed_expert_cache_path: str | Path | None = None,
        routing_top_k: int | None = None,
        proxy_mode: str = "none",
        proxy_top_k: int | None = None,
        expert_precision: str = "bf16",
        trunk_precision: str = "bf16",
        grouped_nvfp4: bool = False,
    ) -> "GLM5XDecoderModelReference":
        """Build an out-of-core model factory from one validated GLM bundle.

        Embedding, final norm, and LM-head tensors are read once when they are
        present in the bundle. Bounded probes may provide those three tensors
        explicitly while still loading decoder layers from real artifacts.
        Decoder layers remain provider-owned and are constructed only when
        requested; dense/sparse MLP kind and shared indexer source are resolved
        from the official configuration.
        """
        descriptor = GLM5XModelDescriptor.from_config(config)
        if (
            not isinstance(expert_device_cache_capacity_bytes, int)
            or isinstance(expert_device_cache_capacity_bytes, bool)
            or expert_device_cache_capacity_bytes < 0
        ):
            raise ValueError("GLM5X_BUNDLE_EXPERT_DEVICE_CACHE_CAPACITY")
        if expert_device_cache_policy not in {"lru", "layer_balanced"}:
            raise ValueError("GLM5X_BUNDLE_EXPERT_DEVICE_CACHE_POLICY")
        if (
            not isinstance(expert_device_cache_protected_entries_per_layer, int)
            or isinstance(expert_device_cache_protected_entries_per_layer, bool)
            or expert_device_cache_protected_entries_per_layer < 0
        ):
            raise ValueError("GLM5X_BUNDLE_EXPERT_DEVICE_CACHE_PROTECTED_ENTRIES")
        if (
            expert_device_cache_policy == "layer_balanced"
            and expert_device_cache_protected_entries_per_layer <= 0
        ):
            raise ValueError("GLM5X_BUNDLE_EXPERT_DEVICE_CACHE_PROTECTED_ENTRIES")
        if (
            not isinstance(trunk_cache_capacity_bytes, int)
            or isinstance(trunk_cache_capacity_bytes, bool)
            or trunk_cache_capacity_bytes < 0
        ):
            raise ValueError("GLM5X_BUNDLE_TRUNK_CACHE_CAPACITY")
        if trunk_precision not in {"bf16", "int4"}:
            raise ValueError("GLM5X_INVALID_TRUNK_PRECISION")
        if expert_precision not in {"bf16", "fp8", "int4", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
            raise ValueError("GLM5X_INVALID_EXPERT_PRECISION")
        if routing_top_k is not None and (
            not isinstance(routing_top_k, int)
            or isinstance(routing_top_k, bool)
            or routing_top_k <= 0
            or routing_top_k > descriptor.top_k
        ):
            raise ValueError("GLM5X_BUNDLE_ROUTING_TOP_K")
        effective_top_k = descriptor.top_k if routing_top_k is None else routing_top_k
        if proxy_mode not in {"none", "shared"}:
            raise ValueError("GLM5X_BUNDLE_PROXY_MODE")
        if proxy_top_k is None:
            proxy_top_k = effective_top_k
        if (
            not isinstance(proxy_top_k, int)
            or isinstance(proxy_top_k, bool)
            or proxy_top_k <= 0
            or proxy_top_k > effective_top_k
        ):
            raise ValueError("GLM5X_BUNDLE_PROXY_TOP_K")
        if proxy_mode == "none" and proxy_top_k != effective_top_k:
            raise ValueError("GLM5X_BUNDLE_PROXY_TOP_K_WITHOUT_PROXY")
        layer_count = descriptor.hidden_layers
        hidden_size = descriptor.hidden_size
        num_heads = _positive_config_int(config, "num_attention_heads")
        qk_nope_head_dim = _positive_config_int(config, "qk_nope_head_dim")
        qk_rope_head_dim = _positive_config_int(config, "qk_rope_head_dim")
        v_head_dim = _positive_config_int(config, "v_head_dim")
        if descriptor.index_topk <= 0:
            raise ValueError("GLM5X_BUNDLE_CONFIG_INDEX_TOPK")
        if descriptor.moe_intermediate_size <= 0:
            raise ValueError("GLM5X_BUNDLE_CONFIG_MOE_INTERMEDIATE_SIZE")
        rms_norm_eps = _positive_config_float(config, "rms_norm_eps", default=1e-5)
        routed_scaling_factor = _positive_config_float(
            config, "routed_scaling_factor", default=2.5
        )
        rope_parameters = config.get("rope_parameters", {})
        if not isinstance(rope_parameters, Mapping):
            raise ValueError("GLM5X_BUNDLE_ROPE_PARAMETERS")
        rope_theta_value = rope_parameters.get("rope_theta", config.get("rope_theta", 10000.0))
        if not isinstance(rope_theta_value, (int, float)) or isinstance(rope_theta_value, bool):
            raise ValueError("GLM5X_BUNDLE_ROPE_THETA")
        rope_theta = float(rope_theta_value)
        if rope_theta <= 0.0:
            raise ValueError("GLM5X_BUNDLE_ROPE_THETA")
        indexer_rope_interleave = config.get(
            "indexer_rope_interleave", config.get("rope_interleave", True)
        )
        if not isinstance(indexer_rope_interleave, bool):
            raise ValueError("GLM5X_BUNDLE_INDEXER_ROPE")
        mlp_types = _bundle_mlp_types(config, layer_count)
        indexer_sources = _bundle_indexer_sources(config, layer_count)

        bundle = GLM5XExpertBundle.open(
            bundle_path,
            verify_payloads=verify_payloads,
            verify_root=verify_root,
            expert_cache_capacity_bytes=expert_cache_capacity_bytes,
        )
        tensor_refs = _collect_tensor_refs(bundle)
        expert_device_cache = (
            GLM5XExpertTensorCache(
                expert_device_cache_capacity_bytes,
                policy=expert_device_cache_policy,
                protected_entries_per_layer=expert_device_cache_protected_entries_per_layer,
            )
            if expert_device_cache_capacity_bytes
            else None
        )
        trunk_tensor_cache = (
            GLM5XTrunkTensorCache(trunk_cache_capacity_bytes)
            if trunk_cache_capacity_bytes
            else None
        )
        packed_expert_cache = (
            GLM5XPackedExpertCache(packed_expert_cache_path)
            if packed_expert_cache_path is not None
            and expert_precision in {"int4", "fp8", "mxfp4", "nvfp4", "nvfp4_gate_up"}
            else None
        )
        read = lambda name: GLM5XLayer10MoEReference._read_tensor(tensor_refs, name)  # noqa: E731
        target = None if device is None else torch.device(device)
        if embedding is None:
            embedding = read("model.embed_tokens.weight")
        if final_norm is None:
            final_norm = read("model.norm.weight")
        if lm_head is None:
            lm_head = read("lm_head.weight")
        if target is not None:
            embedding = embedding.to(device=target)
            final_norm = final_norm.to(device=target)
            lm_head = lm_head.to(device=target)

        def load_layer(layer_id: int) -> GLM5XDecoderLayerReference:
            if not isinstance(layer_id, int) or isinstance(layer_id, bool):
                raise ValueError("GLM5X_BUNDLE_LAYER_ID")
            if layer_id < 0 or layer_id >= layer_count:
                raise ValueError("GLM5X_BUNDLE_LAYER_ID")
            return GLM5XDecoderLayerReference.from_open_bundle(
                bundle,
                tensor_refs=tensor_refs,
                layer_id=layer_id,
                cache_experts=cache_experts,
                num_heads=num_heads,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                index_topk=descriptor.index_topk,
                top_k=effective_top_k,
                routed_scaling_factor=routed_scaling_factor,
                expert_intermediate_size=descriptor.moe_intermediate_size,
                hidden_size=hidden_size,
                rms_norm_eps=rms_norm_eps,
                mlp_type=mlp_types[layer_id],
                indexer_source_layer=indexer_sources[layer_id],
                indexer_rope_interleave=indexer_rope_interleave,
                device=target,
                execution_mode=execution_mode,
                use_sparse_topk=use_sparse_topk,
                expert_load_workers=expert_load_workers,
                expert_device_cache=expert_device_cache,
                trunk_tensor_cache=trunk_tensor_cache,
                packed_expert_cache=packed_expert_cache,
                expert_precision=expert_precision,
                trunk_precision=trunk_precision,
                proxy_mode=proxy_mode,
                proxy_top_k=proxy_top_k,
                grouped_nvfp4=grouped_nvfp4,
            )

        instance = cls.from_layer_loader(
            embedding=embedding,
            layer_count=layer_count,
            layer_loader=load_layer,
            final_norm=final_norm,
            lm_head=lm_head,
            rope_dim=qk_rope_head_dim,
            rope_theta=rope_theta,
            layer_cache_capacity=layer_cache_capacity,
        )
        instance._expert_bundle = bundle
        instance._expert_device_cache = expert_device_cache
        instance._trunk_tensor_cache = trunk_tensor_cache
        instance._packed_expert_cache = packed_expert_cache
        instance._routing_top_k = effective_top_k
        return instance

    @property
    def vocab_size(self) -> int:
        return int(self.embedding.shape[0])

    @property
    def hidden_size(self) -> int:
        return int(self.embedding.shape[1])

    @property
    def rope_dim(self) -> int:
        return self._rope_dim

    @property
    def layer_count(self) -> int:
        return self._layer_count

    @property
    def layer_cache_capacity(self) -> int:
        return self._layer_cache_capacity

    @property
    def expert_payload_cache_stats(self) -> GLM5XExpertPayloadCacheStats:
        bundle = getattr(self, "_expert_bundle", None)
        if bundle is None:
            return GLM5XExpertPayloadCacheStats(0, 0, 0, 0, 0, 0)
        return bundle.expert_payload_cache_stats

    @property
    def expert_device_cache_stats(self) -> GLM5XExpertTensorCacheStats:
        cache = getattr(self, "_expert_device_cache", None)
        if cache is None:
            return GLM5XExpertTensorCacheStats(0, 0, 0, 0, 0, 0)
        return cache.stats

    @property
    def trunk_tensor_cache_stats(self) -> GLM5XTrunkTensorCacheStats:
        cache = getattr(self, "_trunk_tensor_cache", None)
        if cache is None:
            return GLM5XTrunkTensorCacheStats(0, 0, 0, 0, 0, 0)
        return cache.stats

    @property
    def packed_expert_cache_stats(self):
        cache = getattr(self, "_packed_expert_cache", None)
        if cache is None:
            return None
        return cache.stats

    @property
    def bundle_read_stats(self):
        bundle = getattr(self, "_expert_bundle", None)
        if bundle is None:
            return None
        return bundle.read_stats

    @property
    def prepared_lm_head(self) -> torch.Tensor | None:
        return self._prepared_lm_head

    @property
    def cached_layer_count(self) -> int:
        return len(self._layer_cache)

    def _load_layer(self, index: int) -> GLM5XDecoderLayerReference:
        if self._layer_loader is None:
            assert self.layers is not None
            layer = self.layers[index]
        else:
            layer = self._layer_cache.get(index)
            if layer is not None:
                self._layer_cache.move_to_end(index)
            else:
                layer = self._layer_loader(index)
        if not isinstance(layer, GLM5XDecoderLayerReference):
            raise ValueError("GLM5X_MODEL_LAYER_LOADER_RETURN_TYPE")
        if layer.attention.weights.hidden_size != self.hidden_size:
            raise ValueError("GLM5X_MODEL_LAYER_HIDDEN_SHAPE")
        if layer.attention.weights.qk_rope_head_dim != self.rope_dim:
            raise ValueError("GLM5X_MODEL_LAYER_ROPE_DIM")
        if self._layer_loader is not None and self._layer_cache_capacity:
            self._layer_cache[index] = layer
            self._layer_cache.move_to_end(index)
            while len(self._layer_cache) > self._layer_cache_capacity:
                self._layer_cache.popitem(last=False)
        return layer

    def empty_state(self) -> GLM5XDecoderState:
        return GLM5XDecoderState(
            attention=tuple(None for _ in range(self.layer_count)),
            dsa=tuple(None for _ in range(self.layer_count)),
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
        if len(state.attention) != self.layer_count or len(state.dsa) != self.layer_count:
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
        for index in range(self.layer_count):
            layer = self._load_layer(index)
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
        normalized = normalized * self.final_norm.to(
            device=normalized.device, dtype=normalized.dtype
        )
        prepared_lm_head = self._prepared_lm_head
        if prepared_lm_head is None or prepared_lm_head.device != normalized.device:
            prepared_lm_head = self.lm_head.to(
                device=normalized.device, dtype=torch.float32
            )
            self.lm_head = prepared_lm_head
            self._prepared_lm_head = prepared_lm_head
        logits = torch.matmul(normalized.to(torch.float32), prepared_lm_head.t())
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
        tokens = torch.as_tensor(tokens, dtype=torch.long, device=self.embedding.device)
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError("GLM5X_MODEL_TOKEN_SHAPE")
        if torch.any(tokens < 0) or torch.any(tokens >= self.vocab_size):
            raise ValueError("GLM5X_MODEL_TOKEN_RANGE")
        state = self.empty_state() if state is None else state
        start = 0
        if state.attention and state.attention[0] is not None:
            start = state.attention[0].length
        hidden = self.embedding[tokens].unsqueeze(0)
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
