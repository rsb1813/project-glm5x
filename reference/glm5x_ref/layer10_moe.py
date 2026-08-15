# GLM-5.2 layer-10 MoE의 공식 라우터와 지연 expert payload 실행을 제공합니다.

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from k3x_converter.format import DType, K3XError
from k3x_converter.reader import K3XReader
from glm5x_converter.bundle import GLM5XExpertBundle

from .int4 import GLM5XInt4Weight, linear, quantize_int4_weight, weight_shape
from .nvfp4 import (
    GLM5XNVFP4Weight,
    linear_nvfp4,
    linear_nvfp4_pair,
    linear_nvfp4_pair_from_activation,
    quantize_nvfp4_activation,
    quantize_nvfp4_weight,
)
from .nvfp4_batched import linear_nvfp4_gate_up_batched_from_activation
from k3x_ref.mxfp4 import decode_mxfp4, encode_mxfp4
from .packed_cache import GLM5XPackedExpertCache


def _tensor_from_readonly_buffer(data: bytes, dtype: torch.dtype) -> torch.Tensor:
    """Decode large payloads without a second host copy; keep tiny fixtures warning-free."""
    source = bytearray(data) if len(data) <= 4096 else memoryview(data)
    return torch.frombuffer(source, dtype=dtype)


@dataclass(frozen=True)
class GLM5XExpertWeights:
    """한 routed/shared expert의 gate, up, down projection입니다."""

    gate_proj: torch.Tensor | GLM5XInt4Weight | GLM5XNVFP4Weight
    up_proj: torch.Tensor | GLM5XInt4Weight | GLM5XNVFP4Weight
    down_proj: torch.Tensor | GLM5XInt4Weight | GLM5XNVFP4Weight
    gate_scale: torch.Tensor | None = None
    up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        gate = self.gate_proj
        up = self.up_proj
        down = self.down_proj
        gate_shape = weight_shape(gate)
        up_shape = weight_shape(up)
        down_shape = weight_shape(down)
        if len(gate_shape) != 2 or len(up_shape) != 2 or len(down_shape) != 2:
            raise ValueError("GLM5X_EXPERT_WEIGHTS_MUST_BE_RANK_TWO")
        if gate_shape != up_shape or down_shape != (gate_shape[1], gate_shape[0]):
            raise ValueError("GLM5X_EXPERT_WEIGHT_SHAPE_MISMATCH")
        if gate.dtype != up.dtype or gate.dtype != down.dtype:
            raise ValueError("GLM5X_EXPERT_WEIGHT_DTYPE_MISMATCH")
        scales = (self.gate_scale, self.up_scale, self.down_scale)
        if any(scale is not None for scale in scales):
            if any(scale is None for scale in scales):
                raise ValueError("GLM5X_EXPERT_SCALE_SET_INCOMPLETE")
            for weight, scale in zip((gate, up, down), scales):
                assert scale is not None
                value = torch.as_tensor(scale)
                if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError("GLM5X_EXPERT_SCALE_DTYPE")
                if value.shape != (weight_shape(weight)[0], 1):
                    raise ValueError("GLM5X_EXPERT_SCALE_SHAPE")

    @property
    def is_fp8(self) -> bool:
        return (
            self.gate_proj.dtype == torch.float8_e4m3fn
            and self.up_proj.dtype == torch.float8_e4m3fn
            and self.down_proj.dtype == torch.float8_e4m3fn
            and self.gate_scale is not None
            and self.up_scale is not None
            and self.down_scale is not None
        )

    @property
    def is_nvfp4(self) -> bool:
        return all(
            isinstance(weight, GLM5XNVFP4Weight)
            for weight in (self.gate_proj, self.up_proj, self.down_proj)
        )


@dataclass(frozen=True)
class GLM5XExpertTensorCacheStats:
    capacity_bytes: int
    resident_bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int
    bypasses: int
    promotions: int


class GLM5XExpertTensorCache:
    """Bounded exact decoded expert tensors shared across layer objects."""

    def __init__(
        self,
        capacity_bytes: int,
        *,
        policy: str = "lru",
        protected_entries_per_layer: int = 0,
    ) -> None:
        if (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes <= 0
        ):
            raise ValueError("GLM5X_EXPERT_DEVICE_CACHE_CAPACITY")
        if policy not in {"lru", "layer_balanced", "stable_hot_bank"}:
            raise ValueError("GLM5X_EXPERT_DEVICE_CACHE_POLICY")
        if (
            not isinstance(protected_entries_per_layer, int)
            or isinstance(protected_entries_per_layer, bool)
            or protected_entries_per_layer < 0
        ):
            raise ValueError("GLM5X_EXPERT_DEVICE_CACHE_PROTECTED_ENTRIES")
        if policy in {"layer_balanced", "stable_hot_bank"} and protected_entries_per_layer <= 0:
            raise ValueError("GLM5X_EXPERT_DEVICE_CACHE_PROTECTED_ENTRIES")
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.protected_entries_per_layer = protected_entries_per_layer
        self._entries: OrderedDict[
            tuple[int, int], tuple[GLM5XExpertWeights, int]
        ] = OrderedDict()
        self._protected_by_layer: dict[int, set[tuple[int, int]]] = {}
        self._access_counts: dict[tuple[int, int], int] = {}
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bypasses = 0
        self._promotions = 0
        self._lock = Lock()

    @staticmethod
    def _size(expert: GLM5XExpertWeights) -> int:
        return sum(
            tensor.num_bytes()
            if isinstance(tensor, (GLM5XInt4Weight, GLM5XNVFP4Weight))
            else int(tensor.numel()) * int(tensor.element_size())
            for tensor in (expert.gate_proj, expert.up_proj, expert.down_proj)
        )

    def get(self, key: tuple[int, int]) -> GLM5XExpertWeights | None:
        with self._lock:
            if self.policy == "stable_hot_bank":
                self._access_counts[key] = self._access_counts.get(key, 0) + 1
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry[0]

    def put(self, key: tuple[int, int], expert: GLM5XExpertWeights) -> None:
        size = self._size(expert)
        if size > self.capacity_bytes:
            return
        with self._lock:
            if self.policy == "stable_hot_bank":
                layer = int(key[0])
                protected = self._protected_by_layer.setdefault(layer, set())
                previous = self._entries.get(key)
                if previous is not None:
                    prospective = self._resident_bytes - previous[1] + size
                    if prospective > self.capacity_bytes:
                        self._bypasses += 1
                        return
                    self._entries[key] = (expert, size)
                    self._entries.move_to_end(key)
                    self._resident_bytes = prospective
                    protected.add(key)
                    return

                victim = None
                if len(protected) >= self.protected_entries_per_layer:
                    victim = min(
                        protected,
                        key=lambda item: (self._access_counts.get(item, 0), item),
                    )
                    if self._access_counts.get(key, 0) <= self._access_counts.get(victim, 0):
                        self._bypasses += 1
                        return
                victim_size = 0 if victim is None else self._entries[victim][1]
                prospective = self._resident_bytes - victim_size + size
                if prospective > self.capacity_bytes:
                    self._bypasses += 1
                    return
                if victim is not None:
                    self._entries.pop(victim)
                    protected.remove(victim)
                    self._resident_bytes -= victim_size
                    self._evictions += 1
                    self._promotions += 1
                self._entries[key] = (expert, size)
                protected.add(key)
                self._resident_bytes += size
                return

            previous = self._entries.pop(key, None)
            if previous is not None:
                self._resident_bytes -= previous[1]
            protected = self._protected_by_layer.setdefault(int(key[0]), set())
            was_protected = key in protected
            if self.policy == "layer_balanced":
                while self._resident_bytes + size > self.capacity_bytes:
                    candidate = None
                    for existing_key in self._entries:
                        layer_count = sum(
                            1
                            for layer_key in self._entries
                            if int(layer_key[0]) == int(existing_key[0])
                        )
                        if (
                            layer_count > self.protected_entries_per_layer
                            and existing_key
                            not in self._protected_by_layer.get(int(existing_key[0]), set())
                        ):
                            candidate = existing_key
                            break
                    if candidate is None:
                        if previous is not None:
                            self._entries[key] = previous
                            self._resident_bytes += previous[1]
                        return
                    _, evicted_size = self._entries.pop(candidate)
                    self._resident_bytes -= evicted_size
                    self._protected_by_layer.get(int(candidate[0]), set()).discard(candidate)
                    self._evictions += 1
            else:
                while self._entries and self._resident_bytes + size > self.capacity_bytes:
                    _, (_, evicted_size) = self._entries.popitem(last=False)
                    self._resident_bytes -= evicted_size
                    self._evictions += 1
            self._entries[key] = (expert, size)
            if (
                self.policy == "layer_balanced"
                and (was_protected or len(protected) < self.protected_entries_per_layer)
            ):
                protected.add(key)
            self._resident_bytes += size

    @property
    def stats(self) -> GLM5XExpertTensorCacheStats:
        with self._lock:
            return GLM5XExpertTensorCacheStats(
                capacity_bytes=self.capacity_bytes,
                resident_bytes=self._resident_bytes,
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                bypasses=self._bypasses,
                promotions=self._promotions,
            )


@dataclass(frozen=True)
class GLM5XTrunkTensorCacheStats:
    capacity_bytes: int
    resident_bytes: int
    entries: int
    hits: int
    misses: int
    evictions: int


class GLM5XTrunkTensorCache:
    """Bounded exact CPU cache for non-expert decoder-layer tensors."""

    def __init__(self, capacity_bytes: int) -> None:
        if (
            not isinstance(capacity_bytes, int)
            or isinstance(capacity_bytes, bool)
            or capacity_bytes <= 0
        ):
            raise ValueError("GLM5X_TRUNK_CACHE_CAPACITY")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[str, tuple[torch.Tensor, int]] = OrderedDict()
        self._resident_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = Lock()

    @staticmethod
    def _size(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def get(self, name: str) -> torch.Tensor | None:
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(name)
            self._hits += 1
            return entry[0]

    def put(self, name: str, tensor: torch.Tensor) -> None:
        tensor = torch.as_tensor(tensor)
        if tensor.device.type != "cpu":
            raise ValueError("GLM5X_TRUNK_CACHE_CPU_ONLY")
        size = self._size(tensor)
        if size > self.capacity_bytes:
            return
        with self._lock:
            previous = self._entries.pop(name, None)
            if previous is not None:
                self._resident_bytes -= previous[1]
            while self._entries and self._resident_bytes + size > self.capacity_bytes:
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._resident_bytes -= evicted_size
                self._evictions += 1
            self._entries[name] = (tensor, size)
            self._resident_bytes += size

    @property
    def stats(self) -> GLM5XTrunkTensorCacheStats:
        with self._lock:
            return GLM5XTrunkTensorCacheStats(
                capacity_bytes=self.capacity_bytes,
                resident_bytes=self._resident_bytes,
                entries=len(self._entries),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )


@dataclass(frozen=True)
class GLM5XMoEForward:
    output: torch.Tensor
    router_logits: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    loaded_experts: tuple[int, ...]

    @property
    def expert_load_count(self) -> int:
        return len(self.loaded_experts)


class GLM5XDenseMlpReference:
    """GLM-5.2 first dense MLP layers with the shared forward contract."""

    def __init__(self, weights: GLM5XExpertWeights) -> None:
        if not isinstance(weights, GLM5XExpertWeights):
            raise ValueError("GLM5X_DENSE_MLP_WEIGHTS_REQUIRED")
        self.weights = weights

    @property
    def hidden_size(self) -> int:
        return int(self.weights.gate_proj.shape[1])

    def __call__(self, hidden_states: torch.Tensor) -> GLM5XMoEForward:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("GLM5X_DENSE_MLP_HIDDEN_SHAPE")
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        gate = linear(flat, self.weights.gate_proj)
        up = linear(flat, self.weights.up_proj)
        output = linear(F.silu(gate) * up, self.weights.down_proj)
        empty_router = torch.empty(
            (*original_shape[:-1], 0), dtype=torch.float32, device=output.device
        )
        empty_indices = torch.empty(
            (*original_shape[:-1], 0), dtype=torch.long, device=output.device
        )
        return GLM5XMoEForward(
            output=output.reshape(original_shape),
            router_logits=empty_router,
            topk_indices=empty_indices,
            topk_weights=empty_router.clone(),
            loaded_experts=(),
        )


ExpertLoader = Callable[[int], GLM5XExpertWeights]
ExpertBatchLoader = Callable[[Sequence[int]], Mapping[int, GLM5XExpertWeights]]


class GLM5XLayer10MoEReference:
    """공식 GLM/DeepSeek V3 MoE를 token-major CPU reference로 실행합니다."""

    def __init__(
        self,
        *,
        router_weight: torch.Tensor,
        correction_bias: torch.Tensor,
        expert_loader: ExpertLoader,
        shared_expert: GLM5XExpertWeights,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        cache_experts: bool = True,
        execution_mode: str = "loop",
        expert_batch_loader: ExpertBatchLoader | None = None,
        expert_load_workers: int = 1,
        expert_device_cache: GLM5XExpertTensorCache | None = None,
        expert_precision: str = "bf16",
        trunk_precision: str = "bf16",
        proxy_mode: str = "none",
        proxy_top_k: int | None = None,
        grouped_nvfp4: bool = False,
    ) -> None:
        router_weight = torch.as_tensor(router_weight)
        correction_bias = torch.as_tensor(correction_bias)
        if router_weight.ndim != 2 or router_weight.numel() == 0:
            raise ValueError("GLM5X_ROUTER_WEIGHT_SHAPE")
        if correction_bias.shape != (router_weight.shape[0],):
            raise ValueError("GLM5X_ROUTER_BIAS_SHAPE")
        if not callable(expert_loader):
            raise ValueError("GLM5X_EXPERT_LOADER_REQUIRED")
        if not isinstance(top_k, int) or not 0 < top_k <= router_weight.shape[0]:
            raise ValueError("GLM5X_INVALID_TOP_K")
        if not isinstance(n_group, int) or n_group <= 0 or router_weight.shape[0] % n_group:
            raise ValueError("GLM5X_INVALID_ROUTER_GROUPS")
        experts_per_group = router_weight.shape[0] // n_group
        if experts_per_group < 2:
            raise ValueError("GLM5X_ROUTER_GROUP_TOO_SMALL")
        if not isinstance(topk_group, int) or not 0 < topk_group <= n_group:
            raise ValueError("GLM5X_INVALID_TOPK_GROUP")
        if top_k > topk_group * experts_per_group:
            raise ValueError("GLM5X_TOP_K_EXCEEDS_GROUP_BUDGET")
        if routed_scaling_factor <= 0:
            raise ValueError("GLM5X_INVALID_ROUTED_SCALE")
        if not isinstance(cache_experts, bool):
            raise ValueError("GLM5X_INVALID_EXPERT_CACHE_FLAG")
        if not isinstance(grouped_nvfp4, bool):
            raise ValueError("GLM5X_INVALID_GROUPED_NVFP4_FLAG")
        if execution_mode not in {"loop", "expert_major"}:
            raise ValueError("GLM5X_INVALID_EXECUTION_MODE")
        if expert_precision not in {"bf16", "fp8", "int4", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
            raise ValueError("GLM5X_INVALID_EXPERT_PRECISION")
        if proxy_mode not in {"none", "shared"}:
            raise ValueError("GLM5X_INVALID_PROXY_MODE")
        if proxy_top_k is None:
            proxy_top_k = top_k
        if (
            not isinstance(proxy_top_k, int)
            or isinstance(proxy_top_k, bool)
            or proxy_top_k <= 0
            or proxy_top_k > top_k
        ):
            raise ValueError("GLM5X_INVALID_PROXY_TOP_K")
        if proxy_mode == "none" and proxy_top_k != top_k:
            raise ValueError("GLM5X_PROXY_TOP_K_WITHOUT_PROXY")
        if expert_batch_loader is not None and not callable(expert_batch_loader):
            raise ValueError("GLM5X_INVALID_EXPERT_BATCH_LOADER")
        if (
            not isinstance(expert_load_workers, int)
            or isinstance(expert_load_workers, bool)
            or expert_load_workers <= 0
        ):
            raise ValueError("GLM5X_INVALID_EXPERT_LOAD_WORKERS")
        if expert_device_cache is not None and not isinstance(
            expert_device_cache, GLM5XExpertTensorCache
        ):
            raise ValueError("GLM5X_INVALID_EXPERT_DEVICE_CACHE")
        self.router_weight = router_weight
        self.correction_bias = correction_bias.to(torch.float32)
        self.expert_loader = expert_loader
        self.shared_expert = (
            self._quantize_expert_fp8(shared_expert)
            if expert_precision == "fp8"
            else self._quantize_expert_int4(shared_expert)
            if expert_precision == "int4"
            else self._quantize_expert_mxfp4(shared_expert)
            if expert_precision == "mxfp4"
            else self._quantize_expert_nvfp4(shared_expert)
            if expert_precision == "nvfp4"
            else shared_expert
        )
        self.top_k = top_k
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = bool(norm_topk_prob)
        self.cache_experts = cache_experts
        self.execution_mode = execution_mode
        self.expert_batch_loader = expert_batch_loader
        self.expert_load_workers = expert_load_workers
        self.expert_device_cache = expert_device_cache
        self.expert_precision = expert_precision
        self.proxy_mode = proxy_mode
        self.proxy_top_k = proxy_top_k
        self.grouped_nvfp4 = grouped_nvfp4
        self._expert_cache: dict[int, GLM5XExpertWeights] = {}

    @property
    def hidden_size(self) -> int:
        return int(self.router_weight.shape[1])

    @property
    def num_experts(self) -> int:
        return int(self.router_weight.shape[0])

    def _load_expert(self, expert_id: int) -> tuple[GLM5XExpertWeights, bool]:
        if self.cache_experts and expert_id in self._expert_cache:
            return self._expert_cache[expert_id], False
        expert = self.expert_loader(int(expert_id))
        if not isinstance(expert, GLM5XExpertWeights):
            raise ValueError("GLM5X_EXPERT_LOADER_RETURN_TYPE")
        if self.expert_precision == "fp8":
            expert = self._quantize_expert_fp8(expert)
        elif self.expert_precision == "int4":
            expert = self._quantize_expert_int4(expert, device=self._device_for_expert(expert))
        elif self.expert_precision == "mxfp4":
            expert = self._quantize_expert_mxfp4(
                expert, device=self._device_for_expert(expert)
            )
        elif self.expert_precision == "nvfp4":
            expert = self._quantize_expert_nvfp4(
                expert, device=self._device_for_expert(expert)
            )
        elif self.expert_precision == "nvfp4_gate_up":
            expert = self._quantize_expert_nvfp4_gate_up(
                expert, device=self._device_for_expert(expert)
            )
        if expert.gate_proj.shape[1] != self.hidden_size:
            raise ValueError("GLM5X_EXPERT_HIDDEN_SIZE_MISMATCH")
        if self.cache_experts:
            self._expert_cache[expert_id] = expert
        return expert, True

    def _load_experts(
        self, expert_ids: Sequence[int]
    ) -> tuple[dict[int, GLM5XExpertWeights], tuple[int, ...]]:
        experts: dict[int, GLM5XExpertWeights] = {}
        pending: list[int] = []
        for expert_id in expert_ids:
            if self.cache_experts and expert_id in self._expert_cache:
                experts[expert_id] = self._expert_cache[expert_id]
            else:
                pending.append(expert_id)

        loaded: list[int] = []
        if self.expert_batch_loader is None:
            for expert_id in pending:
                expert, did_load = self._load_expert(expert_id)
                experts[expert_id] = expert
                if did_load:
                    loaded.append(expert_id)
            return experts, tuple(loaded)

        batch = self.expert_batch_loader(tuple(pending)) if pending else {}
        if not isinstance(batch, Mapping) or set(batch) != set(pending):
            raise ValueError("GLM5X_EXPERT_BATCH_LOADER_RESULT")
        for expert_id in pending:
            expert = batch[expert_id]
            if not isinstance(expert, GLM5XExpertWeights):
                raise ValueError("GLM5X_EXPERT_LOADER_RETURN_TYPE")
            if self.expert_precision == "fp8":
                expert = self._quantize_expert_fp8(expert)
            elif self.expert_precision == "int4":
                expert = self._quantize_expert_int4(
                    expert, device=self._device_for_expert(expert)
                )
            elif self.expert_precision == "mxfp4":
                expert = self._quantize_expert_mxfp4(
                    expert, device=self._device_for_expert(expert)
                )
            elif self.expert_precision == "nvfp4":
                expert = self._quantize_expert_nvfp4(
                    expert, device=self._device_for_expert(expert)
                )
            elif self.expert_precision == "nvfp4_gate_up":
                expert = self._quantize_expert_nvfp4_gate_up(
                    expert, device=self._device_for_expert(expert)
                )
            if expert.gate_proj.shape[1] != self.hidden_size:
                raise ValueError("GLM5X_EXPERT_HIDDEN_SIZE_MISMATCH")
            experts[expert_id] = expert
            if self.cache_experts:
                self._expert_cache[expert_id] = expert
            loaded.append(expert_id)
        return experts, tuple(loaded)

    @staticmethod
    def _quantize_expert_fp8(expert: GLM5XExpertWeights) -> GLM5XExpertWeights:
        """Create an experimental row-scaled E4M3 expert copy."""
        if expert.is_fp8:
            return expert

        def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            source = weight.detach().to(dtype=torch.float32)
            scale = source.abs().amax(dim=1, keepdim=True).div(448.0).clamp_min(1e-12)
            return (source / scale).to(torch.float8_e4m3fn), scale

        gate, gate_scale = quantize(expert.gate_proj)
        up, up_scale = quantize(expert.up_proj)
        down, down_scale = quantize(expert.down_proj)
        return GLM5XExpertWeights(
            gate_proj=gate,
            up_proj=up,
            down_proj=down,
            gate_scale=gate_scale,
            up_scale=up_scale,
            down_scale=down_scale,
        )

    @staticmethod
    def _quantize_expert_mxfp4(
        expert: GLM5XExpertWeights,
        *,
        device: torch.device | str | None = None,
    ) -> GLM5XExpertWeights:
        """Reference-only E2M1/E8M0 pack/decode; native CUDA FP4 is separate."""
        if any(isinstance(weight, GLM5XInt4Weight) for weight in (
            expert.gate_proj, expert.up_proj, expert.down_proj
        )):
            raise ValueError("GLM5X_MXFP4_REQUIRES_FLOAT_WEIGHTS")
        target = None if device is None else torch.device(device)

        def quantize(weight: torch.Tensor) -> torch.Tensor:
            tensor = torch.as_tensor(weight)
            if tensor.ndim != 2 or tensor.shape[1] % 32:
                raise ValueError("GLM5X_MXFP4_GROUP_ALIGNMENT")
            packed, scales = encode_mxfp4(
                tensor, group_size=32, scale_mode="max_abs"
            )
            decoded = decode_mxfp4(
                packed, scales, int(tensor.shape[0]), int(tensor.shape[1]), 32
            ).to(dtype=torch.bfloat16)
            return decoded if target is None else decoded.to(device=target)

        return GLM5XExpertWeights(
            gate_proj=quantize(expert.gate_proj),
            up_proj=quantize(expert.up_proj),
            down_proj=quantize(expert.down_proj),
        )

    @staticmethod
    def _quantize_expert_nvfp4(
        expert: GLM5XExpertWeights,
        *,
        device: torch.device | str | None = None,
    ) -> GLM5XExpertWeights:
        """Pack all three projections for Blackwell native NVFP4 GEMM."""
        weights = (expert.gate_proj, expert.up_proj, expert.down_proj)
        if all(isinstance(weight, GLM5XNVFP4Weight) for weight in weights):
            return expert
        if any(isinstance(weight, (GLM5XInt4Weight, GLM5XNVFP4Weight)) for weight in weights):
            raise ValueError("GLM5X_NVFP4_REQUIRES_FLOAT_WEIGHTS")
        target = None if device is None else torch.device(device)
        return GLM5XExpertWeights(
            gate_proj=quantize_nvfp4_weight(torch.as_tensor(weights[0]), device=target),
            up_proj=quantize_nvfp4_weight(torch.as_tensor(weights[1]), device=target),
            down_proj=quantize_nvfp4_weight(torch.as_tensor(weights[2]), device=target),
        )

    @staticmethod
    def _quantize_expert_nvfp4_gate_up(
        expert: GLM5XExpertWeights,
        *,
        device: torch.device | str | None = None,
    ) -> GLM5XExpertWeights:
        """Keep the sensitive down projection in BF16 while packing gate/up in NVFP4."""
        weights = (expert.gate_proj, expert.up_proj, expert.down_proj)
        if isinstance(weights[0], GLM5XNVFP4Weight) and isinstance(
            weights[1], GLM5XNVFP4Weight
        ) and not isinstance(weights[2], (GLM5XInt4Weight, GLM5XNVFP4Weight)):
            return expert
        if any(isinstance(weight, GLM5XInt4Weight) for weight in weights):
            raise ValueError("GLM5X_NVFP4_REQUIRES_FLOAT_WEIGHTS")
        target = None if device is None else torch.device(device)
        down = torch.as_tensor(weights[2])
        if target is not None:
            down = down.to(device=target)
        return GLM5XExpertWeights(
            gate_proj=quantize_nvfp4_weight(torch.as_tensor(weights[0]), device=target),
            up_proj=quantize_nvfp4_weight(torch.as_tensor(weights[1]), device=target),
            down_proj=down,
        )

    @staticmethod
    def _device_for_expert(expert: GLM5XExpertWeights) -> torch.device | None:
        for weight in (expert.gate_proj, expert.up_proj, expert.down_proj):
            if isinstance(weight, GLM5XInt4Weight):
                return weight.device
            if isinstance(weight, GLM5XNVFP4Weight):
                return weight.device
            if isinstance(weight, torch.Tensor) and weight.device.type == "cuda":
                return weight.device
        return None

    @staticmethod
    def _quantize_expert_int4(
        expert: GLM5XExpertWeights,
        *,
        device: torch.device | str | None = None,
    ) -> GLM5XExpertWeights:
        """Pack all three expert projections for CUDA TinyGEMM."""
        weights = (expert.gate_proj, expert.up_proj, expert.down_proj)
        if all(isinstance(weight, GLM5XInt4Weight) for weight in weights):
            return expert
        if any(isinstance(weight, GLM5XInt4Weight) for weight in weights):
            raise ValueError("GLM5X_INT4_EXPERT_MIXED_REPRESENTATION")
        target = None if device is None else torch.device(device)
        if target is None:
            for weight in weights:
                if isinstance(weight, torch.Tensor) and weight.device.type == "cuda":
                    target = weight.device
                    break
        if target is not None and target.type != "cuda":
            raise ValueError("GLM5X_INT4_CUDA_REQUIRED")
        if not torch.cuda.is_available():
            raise ValueError("GLM5X_INT4_CUDA_REQUIRED")
        return GLM5XExpertWeights(
            gate_proj=quantize_int4_weight(torch.as_tensor(weights[0]), device=target),
            up_proj=quantize_int4_weight(torch.as_tensor(weights[1]), device=target),
            down_proj=quantize_int4_weight(torch.as_tensor(weights[2]), device=target),
        )

    @staticmethod
    def _fp8_linear(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.ndim != 2 or weight.ndim != 2:
            raise ValueError("GLM5X_FP8_LINEAR_SHAPE")
        source = hidden.to(dtype=torch.float32)
        input_scale = source.abs().amax(dim=1, keepdim=True).div(448.0).clamp_min(1e-12)
        input_fp8 = (source / input_scale).to(torch.float8_e4m3fn)
        if hidden.device.type == "cuda":
            return torch._scaled_mm(
                input_fp8,
                weight.transpose(0, 1),
                scale_a=input_scale,
                scale_b=weight_scale.transpose(0, 1).contiguous(),
                out_dtype=torch.bfloat16,
            )
        dequantized = weight.to(dtype=torch.float32) * weight_scale.to(
            device=weight.device, dtype=torch.float32
        )
        return torch.nn.functional.linear(
            hidden.to(dtype=torch.float32), dequantized
        ).to(dtype=torch.bfloat16)

    @staticmethod
    def _mlp(hidden: torch.Tensor, expert: GLM5XExpertWeights) -> torch.Tensor:
        if expert.is_fp8:
            assert expert.gate_scale is not None
            assert expert.up_scale is not None
            assert expert.down_scale is not None
            gate = GLM5XLayer10MoEReference._fp8_linear(
                hidden, expert.gate_proj, expert.gate_scale
            )
            up = GLM5XLayer10MoEReference._fp8_linear(
                hidden, expert.up_proj, expert.up_scale
            )
            return GLM5XLayer10MoEReference._fp8_linear(
                F.silu(gate) * up, expert.down_proj, expert.down_scale
            )
        if expert.is_nvfp4:
            assert isinstance(expert.gate_proj, GLM5XNVFP4Weight)
            assert isinstance(expert.up_proj, GLM5XNVFP4Weight)
            assert isinstance(expert.down_proj, GLM5XNVFP4Weight)
            gate, up = linear_nvfp4_pair(hidden, expert.gate_proj, expert.up_proj)
            return linear_nvfp4(F.silu(gate) * up, expert.down_proj)
        if isinstance(expert.gate_proj, GLM5XNVFP4Weight) and isinstance(
            expert.up_proj, GLM5XNVFP4Weight
        ):
            gate, up = linear_nvfp4_pair(hidden, expert.gate_proj, expert.up_proj)
        else:
            gate = linear(hidden, expert.gate_proj)
            up = linear(hidden, expert.up_proj)
        return linear(F.silu(gate) * up, expert.down_proj)

    @staticmethod
    def _mlp_with_prequantized_activation(
        hidden: torch.Tensor,
        expert: GLM5XExpertWeights,
        activation: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        if (
            activation is None
            or not isinstance(expert.gate_proj, GLM5XNVFP4Weight)
            or not isinstance(expert.up_proj, GLM5XNVFP4Weight)
        ):
            return GLM5XLayer10MoEReference._mlp(hidden, expert)
        gate, up = linear_nvfp4_pair_from_activation(
            activation, expert.gate_proj, expert.up_proj
        )
        routed = F.silu(gate) * up
        if isinstance(expert.down_proj, GLM5XNVFP4Weight):
            return linear_nvfp4(routed, expert.down_proj)
        return linear(routed, expert.down_proj)

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router = self.router_weight.to(device=hidden.device, dtype=torch.float32)
        bias = self.correction_bias.to(device=hidden.device, dtype=torch.float32)
        logits = F.linear(hidden.to(torch.float32), router)
        scores = torch.sigmoid(logits)
        choice = scores + bias
        group_scores = choice.view(-1, self.n_group, self.num_experts // self.n_group).topk(
            2, dim=-1
        ).values.sum(dim=-1)
        group_indices = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        ).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_indices, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.num_experts // self.n_group)
            .reshape(-1, self.num_experts)
        )
        topk_indices = torch.topk(
            choice.masked_fill(~score_mask.bool(), float("-inf")),
            k=self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return logits, topk_indices, topk_weights

    def _run_loop(
        self,
        flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[int, ...]:
        exact_indices = topk_indices[:, : self.proxy_top_k]
        exact_weights = topk_weights[:, : self.proxy_top_k]
        expert_ids = [int(value) for value in torch.unique(exact_indices, sorted=True)]
        experts, loaded = self._load_experts(expert_ids)
        shared_activation = None
        if (
            flat.shape[0] == 1
            and flat.device.type == "cuda"
            and all(
                isinstance(experts[expert_id].gate_proj, GLM5XNVFP4Weight)
                and isinstance(experts[expert_id].up_proj, GLM5XNVFP4Weight)
                for expert_id in expert_ids
            )
        ):
            shared_activation = quantize_nvfp4_activation(flat)
        grouped_routed: dict[int, torch.Tensor] = {}
        if self.grouped_nvfp4 and shared_activation is not None and len(expert_ids) > 1:
            gate_weights = tuple(experts[expert_id].gate_proj for expert_id in expert_ids)
            up_weights = tuple(experts[expert_id].up_proj for expert_id in expert_ids)
            if all(
                isinstance(weight, GLM5XNVFP4Weight)
                for weight in gate_weights + up_weights
            ):
                grouped_gate, grouped_up = linear_nvfp4_gate_up_batched_from_activation(
                    shared_activation, gate_weights, up_weights
                )
                grouped_routed = {
                    expert_id: F.silu(grouped_gate[0, index]) * grouped_up[0, index]
                    for index, expert_id in enumerate(expert_ids)
                }
        for expert_id in expert_ids:
            expert = experts[expert_id]
            slot_mask = exact_indices == expert_id
            token_indices, slots = torch.where(slot_mask)
            if grouped_routed:
                routed_input = grouped_routed[expert_id]
                routed = (
                    linear_nvfp4(routed_input, expert.down_proj)
                    if isinstance(expert.down_proj, GLM5XNVFP4Weight)
                    else linear(routed_input, expert.down_proj)
                )
            else:
                routed = self._mlp_with_prequantized_activation(
                    flat[token_indices], expert, shared_activation
                )
            weighted = routed * exact_weights[token_indices, slots].to(
                routed.dtype
            ).unsqueeze(-1)
            output.index_add_(0, token_indices, weighted.to(output.dtype))
        return loaded

    def _run_expert_major(
        self,
        flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> tuple[int, ...]:
        exact_indices = topk_indices[:, : self.proxy_top_k]
        exact_weights = topk_weights[:, : self.proxy_top_k]
        expert_ids = [int(value) for value in torch.unique(exact_indices, sorted=True)]
        experts, loaded = self._load_experts(expert_ids)
        assignments: list[tuple[int, GLM5XExpertWeights, torch.Tensor, torch.Tensor]] = []
        for expert_id in expert_ids:
            expert = experts[expert_id]
            slot_mask = exact_indices == expert_id
            token_indices, slots = torch.where(slot_mask)
            assignments.append((expert_id, expert, token_indices, slots))

        if not assignments:
            return loaded

        if any(
            isinstance(expert.gate_proj, GLM5XInt4Weight)
            or isinstance(expert.up_proj, GLM5XInt4Weight)
            or isinstance(expert.down_proj, GLM5XInt4Weight)
            or isinstance(expert.gate_proj, GLM5XNVFP4Weight)
            or isinstance(expert.up_proj, GLM5XNVFP4Weight)
            or isinstance(expert.down_proj, GLM5XNVFP4Weight)
            for _, expert, _, _ in assignments
        ):
            for _, expert, token_indices, slots in assignments:
                routed = self._mlp(flat[token_indices], expert)
                weighted = routed * exact_weights[token_indices, slots].to(
                    routed.dtype
                ).unsqueeze(-1)
                output.index_add_(0, token_indices, weighted.to(output.dtype))
            return tuple(loaded)

        shapes = {
            (
                expert.gate_proj.dtype,
                tuple(expert.gate_proj.shape),
                tuple(expert.down_proj.shape),
            )
            for _, expert, _, _ in assignments
        }
        if len(shapes) != 1:
            for _, expert, token_indices, slots in assignments:
                routed = self._mlp(flat[token_indices], expert)
                weighted = routed * topk_weights[token_indices, slots].to(
                    routed.dtype
                ).unsqueeze(-1)
                output.index_add_(0, token_indices, weighted.to(output.dtype))
            return tuple(loaded)

        work_dtype = assignments[0][1].gate_proj.dtype
        max_assignments = max(
            int(token_indices.numel()) for _, _, token_indices, _ in assignments
        )
        expert_count = len(assignments)
        hidden_batch = torch.zeros(
            (expert_count, max_assignments, self.hidden_size),
            dtype=work_dtype,
            device=flat.device,
        )
        for group_index, (_, _, token_indices, _) in enumerate(assignments):
            hidden_batch[group_index, : token_indices.numel()] = flat[
                token_indices
            ].to(dtype=work_dtype)

        gate_weight = torch.stack(
            [expert.gate_proj for _, expert, _, _ in assignments], dim=0
        ).to(device=flat.device, dtype=work_dtype)
        up_weight = torch.stack(
            [expert.up_proj for _, expert, _, _ in assignments], dim=0
        ).to(device=flat.device, dtype=work_dtype)
        down_weight = torch.stack(
            [expert.down_proj for _, expert, _, _ in assignments], dim=0
        ).to(device=flat.device, dtype=work_dtype)
        gate = torch.bmm(hidden_batch, gate_weight.transpose(1, 2))
        up = torch.bmm(hidden_batch, up_weight.transpose(1, 2))
        routed = torch.bmm(
            F.silu(gate) * up,
            down_weight.transpose(1, 2),
        )
        for group_index, (_, _, token_indices, slots) in enumerate(assignments):
            count = token_indices.numel()
            weighted = routed[group_index, :count] * exact_weights[
                token_indices, slots
            ].to(routed.dtype).unsqueeze(-1)
            output.index_add_(0, token_indices, weighted.to(output.dtype))
        return loaded

    def __call__(self, hidden_states: torch.Tensor) -> GLM5XMoEForward:
        hidden_states = torch.as_tensor(hidden_states)
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("GLM5X_MOE_HIDDEN_SHAPE")
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits, topk_indices, topk_weights = self._route(flat)
        output = torch.zeros_like(flat)
        if self.execution_mode == "loop":
            loaded = self._run_loop(flat, topk_indices, topk_weights, output)
        else:
            loaded = self._run_expert_major(flat, topk_indices, topk_weights, output)
        shared_output = self._mlp(flat, self.shared_expert).to(output.dtype)
        if self.proxy_mode == "shared" and self.proxy_top_k < self.top_k:
            dropped_mass = topk_weights[:, self.proxy_top_k :].sum(
                dim=-1, keepdim=True
            ).to(dtype=shared_output.dtype)
            output += shared_output * (1.0 + dropped_mass)
        else:
            output += shared_output
        return GLM5XMoEForward(
            output=output.reshape(original_shape),
            router_logits=logits.reshape(*original_shape[:-1], self.num_experts),
            topk_indices=topk_indices.reshape(*original_shape[:-1], self.top_k),
            topk_weights=topk_weights.reshape(*original_shape[:-1], self.top_k),
            loaded_experts=tuple(loaded),
        )

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str | Path,
        *,
        layer_id: int = 10,
        cache_experts: bool = False,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        expert_intermediate_size: int = 2048,
        hidden_size: int = 6144,
        device: torch.device | str | None = None,
        verify_payloads: bool = True,
        verify_root: bool = True,
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
        expert_cache_capacity_bytes: int = 0,
        expert_device_cache: GLM5XExpertTensorCache | None = None,
        packed_expert_cache: GLM5XPackedExpertCache | None = None,
        expert_precision: str = "bf16",
        trunk_precision: str = "bf16",
        proxy_mode: str = "none",
        proxy_top_k: int | None = None,
        grouped_nvfp4: bool = False,
    ) -> "GLM5XLayer10MoEReference":
        bundle = GLM5XExpertBundle.open(
            bundle_path,
            verify_payloads=verify_payloads,
            verify_root=verify_root,
            expert_cache_capacity_bytes=expert_cache_capacity_bytes,
        )
        return cls._from_open_bundle(
            bundle,
            tensor_refs=_collect_tensor_refs(bundle),
            layer_id=layer_id,
            cache_experts=cache_experts,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            expert_intermediate_size=expert_intermediate_size,
            hidden_size=hidden_size,
            execution_mode=execution_mode,
            expert_load_workers=expert_load_workers,
            expert_device_cache=expert_device_cache,
            packed_expert_cache=packed_expert_cache,
            device=device,
            expert_precision=expert_precision,
            trunk_precision=trunk_precision,
            proxy_mode=proxy_mode,
            proxy_top_k=proxy_top_k,
            grouped_nvfp4=grouped_nvfp4,
        )

    @classmethod
    def _from_open_bundle(
        cls,
        bundle: GLM5XExpertBundle,
        *,
        tensor_refs: Mapping[str, tuple[K3XReader, object]],
        tensor_values: Mapping[str, torch.Tensor] | None = None,
        layer_id: int = 10,
        cache_experts: bool = False,
        top_k: int = 8,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        norm_topk_prob: bool = True,
        expert_intermediate_size: int = 2048,
        hidden_size: int = 6144,
        device: torch.device | str | None = None,
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
        expert_device_cache: GLM5XExpertTensorCache | None = None,
        packed_expert_cache: GLM5XPackedExpertCache | None = None,
        expert_precision: str = "bf16",
        trunk_precision: str = "bf16",
        proxy_mode: str = "none",
        proxy_top_k: int | None = None,
        grouped_nvfp4: bool = False,
    ) -> "GLM5XLayer10MoEReference":

        prefix = f"model.layers.{layer_id}.mlp"
        target = None if device is None else torch.device(device)
        if expert_precision not in {"bf16", "fp8", "int4", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
            raise ValueError("GLM5X_INVALID_EXPERT_PRECISION")
        if trunk_precision not in {"bf16", "int4"}:
            raise ValueError("GLM5X_INVALID_TRUNK_PRECISION")
        def read(name: str) -> torch.Tensor | GLM5XInt4Weight:
            if tensor_values is not None:
                value = tensor_values.get(name)
                if value is None:
                    raise K3XError("GLM5X_LAYER_TENSOR_NOT_FOUND", name)
                return value
            return cls._read_tensor(tensor_refs, name)

        router_weight = read(f"{prefix}.gate.weight").to(
            device=target, dtype=torch.float32
        )
        correction_bias = read(f"{prefix}.gate.e_score_correction_bias").to(
            device=target, dtype=torch.float32
        )
        shared = cls._read_expert(
            tensor_refs,
            f"{prefix}.shared_experts.gate_proj.weight",
            f"{prefix}.shared_experts.up_proj.weight",
            f"{prefix}.shared_experts.down_proj.weight",
            tensor_values=tensor_values,
        )
        if expert_precision == "fp8":
            shared = cls._quantize_expert_fp8(shared)
        elif expert_precision == "int4":
            if target is None or target.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("GLM5X_INT4_CUDA_REQUIRED")
            shared = cls._quantize_expert_int4(shared, device=target)
        elif expert_precision == "mxfp4":
            shared = cls._quantize_expert_mxfp4(shared, device=target)
        elif expert_precision == "nvfp4":
            shared = cls._quantize_expert_nvfp4(shared, device=target)
        if target is not None:
            if not isinstance(shared.gate_proj, (GLM5XInt4Weight, GLM5XNVFP4Weight)):
                shared = GLM5XExpertWeights(
                    gate_proj=shared.gate_proj.to(device=target),
                    up_proj=shared.up_proj.to(device=target),
                    down_proj=shared.down_proj.to(device=target),
                    gate_scale=(
                        None
                        if shared.gate_scale is None
                        else shared.gate_scale.to(device=target)
                    ),
                    up_scale=(
                        None
                        if shared.up_scale is None
                        else shared.up_scale.to(device=target)
                    ),
                    down_scale=(
                        None
                        if shared.down_scale is None
                        else shared.down_scale.to(device=target)
                    ),
                )

        def load_expert(expert_id: int) -> GLM5XExpertWeights:
            cache_key = (layer_id, int(expert_id))
            if expert_device_cache is not None:
                cached = expert_device_cache.get(cache_key)
                if cached is not None:
                    return cached
            source_digest = None
            if packed_expert_cache is not None and expert_precision in {"int4", "fp8", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
                source_digest = bundle.expert_source_digest(layer_id, expert_id)
                cached = packed_expert_cache.get(
                    cache_key,
                    source_digest,
                    device=target if target is not None else "cuda",
                    precision=expert_precision,
                    non_blocking=packed_expert_cache.non_blocking,
                )
                if cached is not None:
                    if expert_device_cache is not None:
                        expert_device_cache.put(cache_key, cached)
                    return cached
            try:
                payload = bundle.read_expert(layer_id, expert_id)
            except (KeyError, K3XError) as exc:
                raise K3XError("GLM5X_LAYER_EXPERT_NOT_FOUND", f"{layer_id}:{expert_id}") from exc
            expert = cls._expert_from_payload(
                payload,
                (expert_intermediate_size, hidden_size),
                (hidden_size, expert_intermediate_size),
                device=target,
                precision=expert_precision,
            )
            if packed_expert_cache is not None and expert_precision in {"int4", "fp8", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
                assert source_digest is not None
                packed_expert_cache.put(
                    cache_key, source_digest, expert, precision=expert_precision
                )
            if expert_device_cache is not None:
                expert_device_cache.put(cache_key, expert)
            return expert

        def load_experts(expert_ids: Sequence[int]) -> Mapping[int, GLM5XExpertWeights]:
            result: dict[int, GLM5XExpertWeights] = {}
            pending: list[int] = []
            for expert_id in expert_ids:
                cache_key = (layer_id, int(expert_id))
                cached = (
                    expert_device_cache.get(cache_key)
                    if expert_device_cache is not None
                    else None
                )
                if cached is None:
                    pending.append(int(expert_id))
                else:
                    result[int(expert_id)] = cached

            try:
                payload_pending: list[int] = []
                source_digests: dict[int, str] = {}
                packed_precisions = {
                    "int4",
                    "fp8",
                    "mxfp4",
                    "nvfp4",
                    "nvfp4_gate_up",
                }
                if packed_expert_cache is not None and expert_precision in packed_precisions:
                    source_digests = {
                        expert_id: bundle.expert_source_digest(layer_id, expert_id)
                        for expert_id in pending
                    }
                    cached_many = packed_expert_cache.get_many(
                        {
                            (layer_id, expert_id): digest
                            for expert_id, digest in source_digests.items()
                        },
                        device=target if target is not None else "cuda",
                        precision=expert_precision,
                        workers=expert_load_workers,
                        non_blocking=packed_expert_cache.non_blocking,
                    )
                    for (cached_layer, expert_id), cached in cached_many.items():
                        if cached_layer != layer_id:
                            raise K3XError(
                                "GLM5X_LAYER_EXPERT_CACHE_LAYER_MISMATCH",
                                str(cached_layer),
                            )
                        result[expert_id] = cached
                        if expert_device_cache is not None:
                            expert_device_cache.put((layer_id, expert_id), cached)
                payload_pending = [
                    expert_id for expert_id in pending if expert_id not in result
                ]
                payload_map = (
                    bundle.read_experts(layer_id, payload_pending)
                    if payload_pending
                    else {}
                )
            except (KeyError, K3XError) as exc:
                missing = pending[0] if pending else -1
                raise K3XError(
                    "GLM5X_LAYER_EXPERT_NOT_FOUND", f"{layer_id}:{missing}"
                ) from exc
            payloads = [
                (expert_id, payload_map[expert_id])
                for expert_id in payload_pending
            ]
            for expert_id, payload in payloads:
                expert = cls._expert_from_payload(
                    payload,
                    (expert_intermediate_size, hidden_size),
                    (hidden_size, expert_intermediate_size),
                    device=target,
                    precision=expert_precision,
                )
                if packed_expert_cache is not None and expert_precision in {"int4", "fp8", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
                    packed_expert_cache.put(
                        (layer_id, expert_id),
                        source_digests[expert_id],
                        expert,
                        precision=expert_precision,
                    )
                if expert_device_cache is not None:
                    expert_device_cache.put((layer_id, expert_id), expert)
                result[expert_id] = expert
            return result

        return cls(
            router_weight=router_weight,
            correction_bias=correction_bias,
            expert_loader=load_expert,
            shared_expert=shared,
            top_k=top_k,
            routed_scaling_factor=routed_scaling_factor,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=norm_topk_prob,
            cache_experts=cache_experts,
            execution_mode=execution_mode,
            expert_batch_loader=load_experts if expert_load_workers > 1 else None,
            expert_load_workers=expert_load_workers,
            expert_device_cache=expert_device_cache,
            expert_precision=expert_precision,
            proxy_mode=proxy_mode,
            proxy_top_k=proxy_top_k,
            grouped_nvfp4=grouped_nvfp4,
        )

    @classmethod
    def _read_tensors(
        cls,
        refs: Mapping[str, tuple[K3XReader, object]],
        names: Sequence[str],
        *,
        tensor_cache: GLM5XTrunkTensorCache | None = None,
    ) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        pending_names: list[str] = []
        for name in names:
            if tensor_cache is not None:
                cached = tensor_cache.get(name)
                if cached is not None:
                    values[name] = cached
                    continue
            pending_names.append(name)
        if not pending_names:
            return values
        grouped: dict[int, tuple[K3XReader, list[tuple[str, object]]]] = {}
        for name in pending_names:
            item = refs.get(name)
            if item is None:
                raise K3XError("GLM5X_LAYER_TENSOR_NOT_FOUND", name)
            reader, record = item
            group = grouped.setdefault(id(reader), (reader, []))
            group[1].append((name, record))

        for reader, items in grouped.values():
            payloads = reader.read_tensor_extents_many(
                [record for _, record in items]
            )
            for name, record in items:
                data, auxiliary = payloads[record.tensor_id]
                value = cls._decode_tensor(name, record, data, auxiliary)
                values[name] = value
                if tensor_cache is not None:
                    tensor_cache.put(name, value)
        return values

    @classmethod
    def _read_tensor(
        cls,
        refs: Mapping[str, tuple[K3XReader, object]],
        name: str,
        *,
        tensor_cache: GLM5XTrunkTensorCache | None = None,
    ) -> torch.Tensor:
        return cls._read_tensors(refs, (name,), tensor_cache=tensor_cache)[name]

    @staticmethod
    def _decode_tensor(
        name: str, record: object, data: bytes, auxiliary: bytes
    ) -> torch.Tensor:
        if auxiliary or record.quantization.name != "NONE":
            raise K3XError("GLM5X_LAYER_UNSUPPORTED_TENSOR", name)
        if record.dtype == DType.BF16:
            values = _tensor_from_readonly_buffer(data, torch.int16).view(torch.bfloat16)
        elif record.dtype == DType.FP32:
            values = _tensor_from_readonly_buffer(data, torch.float32)
        else:
            raise K3XError("GLM5X_LAYER_UNSUPPORTED_DTYPE", name)
        expected = 1
        for dimension in record.dimensions:
            expected *= dimension
        if values.numel() != expected:
            raise K3XError("GLM5X_LAYER_TENSOR_LENGTH", name)
        return values.reshape(record.dimensions)

    @classmethod
    def _read_expert(
        cls,
        refs: Mapping[str, tuple[K3XReader, object]],
        gate_name: str,
        up_name: str,
        down_name: str,
        *,
        tensor_values: Mapping[str, torch.Tensor] | None = None,
    ) -> GLM5XExpertWeights:
        if tensor_values is not None:
            try:
                return GLM5XExpertWeights(
                    gate_proj=tensor_values[gate_name],
                    up_proj=tensor_values[up_name],
                    down_proj=tensor_values[down_name],
                )
            except KeyError as exc:
                raise K3XError("GLM5X_LAYER_TENSOR_NOT_FOUND", str(exc)) from exc
        return GLM5XExpertWeights(
            gate_proj=cls._read_tensor(refs, gate_name),
            up_proj=cls._read_tensor(refs, up_name),
            down_proj=cls._read_tensor(refs, down_name),
        )

    @classmethod
    def _expert_from_payload(
        cls,
        payload: Mapping[str, bytes],
        intermediate_hidden: tuple[int, int],
        down_shape: tuple[int, int],
        *,
        device: torch.device | str | None = None,
        precision: str = "bf16",
    ) -> GLM5XExpertWeights:
        target = None if device is None else torch.device(device)
        if precision not in {"bf16", "fp8", "int4", "mxfp4", "nvfp4", "nvfp4_gate_up"}:
            raise ValueError("GLM5X_INVALID_EXPERT_PRECISION")

        def decode(role: str, shape: tuple[int, int]) -> torch.Tensor:
            data = payload.get(role)
            if data is None or len(data) != shape[0] * shape[1] * 2:
                raise K3XError("GLM5X_LAYER_EXPERT_PAYLOAD", role)
            return _tensor_from_readonly_buffer(data, torch.int16).view(torch.bfloat16).reshape(shape)

        expert = GLM5XExpertWeights(
            gate_proj=decode("gate_proj", intermediate_hidden),
            up_proj=decode("up_proj", intermediate_hidden),
            down_proj=decode("down_proj", down_shape),
        )
        if precision == "fp8":
            expert = cls._quantize_expert_fp8(expert)
        elif precision == "int4":
            if target is None or target.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("GLM5X_INT4_CUDA_REQUIRED")
            return cls._quantize_expert_int4(expert, device=target)
        elif precision == "mxfp4":
            expert = cls._quantize_expert_mxfp4(expert, device=target)
        elif precision == "nvfp4":
            expert = cls._quantize_expert_nvfp4(expert, device=target)
        elif precision == "nvfp4_gate_up":
            expert = cls._quantize_expert_nvfp4_gate_up(expert, device=target)
        if target is None or isinstance(
            expert.gate_proj, (GLM5XInt4Weight, GLM5XNVFP4Weight)
        ):
            return expert
        return GLM5XExpertWeights(
            gate_proj=expert.gate_proj.to(device=target),
            up_proj=expert.up_proj.to(device=target),
            down_proj=expert.down_proj.to(device=target),
            gate_scale=(
                None if expert.gate_scale is None else expert.gate_scale.to(device=target)
            ),
            up_scale=(
                None if expert.up_scale is None else expert.up_scale.to(device=target)
            ),
            down_scale=(
                None if expert.down_scale is None else expert.down_scale.to(device=target)
            ),
        )


def _collect_tensor_refs(
    bundle: GLM5XExpertBundle,
) -> dict[str, tuple[K3XReader, object]]:
    refs: dict[str, tuple[K3XReader, object]] = {}
    for artifact_key, artifact_path in bundle.artifact_paths.items():
        sidecar_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise K3XError("GLM5X_LAYER_SIDECAR_INVALID", str(sidecar_path)) from exc
        records = bundle.record_indexes[artifact_key]
        for item in sidecar.get("tensors", []):
            name = item.get("name") if isinstance(item, dict) else None
            tensor_id = item.get("tensor_id") if isinstance(item, dict) else None
            if not isinstance(name, str) or not isinstance(tensor_id, int):
                raise K3XError("GLM5X_LAYER_TENSOR_METADATA", str(sidecar_path))
            record = records.get(tensor_id)
            if record is None:
                raise K3XError("GLM5X_LAYER_TENSOR_ID_MISMATCH", name)
            if name in refs:
                raise K3XError("GLM5X_LAYER_DUPLICATE_TENSOR", name)
            refs[name] = (bundle.readers[artifact_key], record)
    return refs
