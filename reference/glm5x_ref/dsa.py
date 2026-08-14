# GLM-5.2 DSA 인덱서와 압축 KV 상태의 CPU reference 경로를 구현합니다.

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .model import GLM5XModelDescriptor
from .turboquant import TurboQuantConfig, TurboQuantKVCache, estimate_kv_storage_bytes


@dataclass(frozen=True)
class GLM5XDSAConfig:
    """Descriptor DSA metadata plus the reference KV storage policy."""

    index_topk: int
    index_topk_freq: int
    index_n_heads: int
    index_head_dim: int
    kv_config: TurboQuantConfig = TurboQuantConfig()
    index_dtype: torch.dtype = torch.bfloat16
    block_tokens: int = 256

    def __post_init__(self) -> None:
        values = (
            self.index_topk,
            self.index_topk_freq,
            self.index_n_heads,
            self.index_head_dim,
            self.block_tokens,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
            raise ValueError("INVALID_DSA_METADATA")
        if not isinstance(self.index_dtype, torch.dtype):
            raise ValueError("INVALID_DSA_INDEX_DTYPE")

    @classmethod
    def from_descriptor(
        cls,
        descriptor: GLM5XModelDescriptor,
        *,
        kv_config: TurboQuantConfig,
        index_dtype: torch.dtype = torch.bfloat16,
        block_tokens: int = 256,
    ) -> "GLM5XDSAConfig":
        values = (
            descriptor.index_topk,
            descriptor.index_topk_freq,
            descriptor.index_n_heads,
            descriptor.index_head_dim,
        )
        if any(value <= 0 for value in values):
            raise ValueError("DSA_METADATA_REQUIRED")
        return cls(
            index_topk=descriptor.index_topk,
            index_topk_freq=descriptor.index_topk_freq,
            index_n_heads=descriptor.index_n_heads,
            index_head_dim=descriptor.index_head_dim,
            kv_config=kv_config,
            index_dtype=index_dtype,
            block_tokens=block_tokens,
        )

    @property
    def index_width(self) -> int:
        return self.index_n_heads * self.index_head_dim


def estimate_dsa_state_bytes(
    *,
    tokens: int,
    index_width: int,
    key_width: int,
    value_width: int,
    config: GLM5XDSAConfig,
) -> int:
    """Estimate index-key plus compressed KV bytes without allocating the state."""

    values = (tokens, index_width, key_width, value_width)
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("INVALID_DSA_CAPACITY_SHAPE")
    index_element_bytes = torch.empty((), dtype=config.index_dtype).element_size()
    index_bytes = tokens * index_width * index_element_bytes
    kv_bytes = estimate_kv_storage_bytes(
        tokens=tokens,
        key_width=key_width,
        value_width=value_width,
        config=config.kv_config,
        block_tokens=config.block_tokens,
    )
    return index_bytes + kv_bytes


class GLM5XDSAState:
    """Reference DSA state with exact refresh and explicitly stale fast selection."""

    def __init__(self, config: GLM5XDSAConfig) -> None:
        self.config = config
        self._kv_cache = TurboQuantKVCache(config.kv_config)
        self._index_keys: torch.Tensor | None = None
        self._cached_indices = torch.empty((0,), dtype=torch.long)
        self._last_refresh_token = 0
        self._last_selection_refreshed = False

    @property
    def token_count(self) -> int:
        return self._kv_cache.token_count

    @property
    def index_width(self) -> int:
        return self.config.index_width

    @property
    def storage_bytes(self) -> int:
        index_bytes = 0 if self._index_keys is None else self._index_keys.numel() * self._index_keys.element_size()
        return index_bytes + self._kv_cache.storage_bytes

    @property
    def last_selection_refreshed(self) -> bool:
        return self._last_selection_refreshed

    def append(
        self,
        index_keys: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        index_keys = torch.as_tensor(index_keys)
        if index_keys.ndim != 2:
            raise ValueError("DSA_INDEX_KEYS_MUST_BE_RANK_TWO")
        if index_keys.shape[1] != self.config.index_width:
            raise ValueError("DSA_INDEX_WIDTH_MISMATCH")
        if index_keys.shape[0] != torch.as_tensor(keys).shape[0] or index_keys.shape[0] != torch.as_tensor(values).shape[0]:
            raise ValueError("DSA_INDEX_KV_TOKEN_COUNT_MISMATCH")
        if index_keys.shape[0] == 0:
            return
        index_keys = index_keys.to(self.config.index_dtype).contiguous()
        self._index_keys = index_keys if self._index_keys is None else torch.cat((self._index_keys, index_keys), dim=0)
        self._kv_cache.append(keys, values)

    def _refresh(self, query: torch.Tensor) -> None:
        if self._index_keys is None or self.token_count == 0:
            self._cached_indices = torch.empty((0,), dtype=torch.long)
            self._last_refresh_token = self.token_count
            return
        query = torch.as_tensor(query)
        if query.ndim != 1 or query.shape[0] != self.config.index_width:
            raise ValueError("DSA_QUERY_WIDTH_MISMATCH")
        scores = query.to(self._index_keys.device, dtype=torch.float32) @ self._index_keys.to(torch.float32).T
        scores = scores / math.sqrt(self.config.index_width)
        count = min(self.config.index_topk, self.token_count)
        self._cached_indices = torch.topk(scores, k=count, dim=-1, largest=True, sorted=True).indices
        self._last_refresh_token = self.token_count

    def select(self, query: torch.Tensor, *, reference_mode: bool = True) -> torch.Tensor:
        query = torch.as_tensor(query)
        if query.ndim != 1 or query.shape[0] != self.config.index_width:
            raise ValueError("DSA_QUERY_WIDTH_MISMATCH")
        refresh = (
            reference_mode
            or self._index_keys is None
            or self._cached_indices.numel() == 0
            or self.token_count - self._last_refresh_token >= self.config.index_topk_freq
        )
        if refresh:
            self._refresh(query)
        self._last_selection_refreshed = refresh
        return self._cached_indices.clone()

    def attend(
        self,
        index_query: torch.Tensor,
        *,
        attention_query: torch.Tensor | None = None,
        reference_mode: bool = True,
    ) -> torch.Tensor:
        selected = self.select(index_query, reference_mode=reference_mode)
        keys, values = self._kv_cache.materialize()
        if selected.numel() == 0:
            return torch.zeros(values.shape[1], dtype=values.dtype)
        query = torch.as_tensor(index_query if attention_query is None else attention_query)
        if query.ndim != 1 or query.shape[0] != keys.shape[1]:
            raise ValueError("DSA_ATTENTION_QUERY_WIDTH_MISMATCH")
        selected_keys = keys.index_select(0, selected)
        selected_values = values.index_select(0, selected)
        scores = query.to(selected_keys.dtype) @ selected_keys.T / math.sqrt(keys.shape[1])
        return torch.softmax(scores, dim=-1) @ selected_values
