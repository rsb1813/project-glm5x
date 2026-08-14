# GLM-5.2 safetensors 인덱스와 모델 descriptor의 경계를 검증합니다.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .model import GLM5XModelDescriptor


_INDEXER_COMPONENTS = frozenset(
    {"k_norm.bias", "k_norm.weight", "weights_proj.weight", "wk.weight", "wq_b.weight"}
)


def _shard_name(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("INVALID_SHARD_NAME")
    if value.endswith("/") or "\\" in value or "/" in value:
        raise ValueError("INVALID_SHARD_NAME")
    if value in {".", ".."} or not value.endswith(".safetensors"):
        raise ValueError("INVALID_SHARD_NAME")
    return value


@dataclass(frozen=True)
class GLM5XTensorManifest:
    """Bounded metadata needed before any source shard is opened."""

    descriptor: GLM5XModelDescriptor
    tensor_shards: tuple[tuple[str, str], ...]
    shard_names: tuple[str, ...]
    total_size: int

    @property
    def tensor_count(self) -> int:
        return len(self.tensor_shards)

    @property
    def shard_count(self) -> int:
        return len(self.shard_names)

    def indexer_source_layer(self, layer_id: int) -> int:
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or layer_id < 0
            or layer_id >= self.descriptor.hidden_layers
        ):
            raise ValueError("INVALID_LAYER_ID")
        if not self.descriptor.indexer_types:
            return layer_id
        if self.descriptor.indexer_types[layer_id] == "full":
            return layer_id
        for source in range(layer_id - 1, -1, -1):
            if self.descriptor.indexer_types[source] == "full":
                return source
        raise ValueError("INDEXER_FULL_SOURCE_MISSING")

    def resolve_indexer_tensor(self, layer_id: int, component: str) -> tuple[str, str]:
        if component not in _INDEXER_COMPONENTS:
            raise ValueError("INVALID_INDEXER_COMPONENT")
        source_layer = self.indexer_source_layer(layer_id)
        tensor_name = f"model.layers.{source_layer}.self_attn.indexer.{component}"
        shard = dict(self.tensor_shards).get(tensor_name)
        if shard is None:
            raise ValueError("INDEXER_TENSOR_MISSING")
        return tensor_name, shard

    @classmethod
    def from_json(
        cls,
        config: Mapping[str, object],
        index: Mapping[str, object],
    ) -> "GLM5XTensorManifest":
        descriptor = GLM5XModelDescriptor.from_config(config)
        raw_metadata = index.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("METADATA_REQUIRED")
        total_size = raw_metadata.get("total_size")
        if (
            not isinstance(total_size, int)
            or isinstance(total_size, bool)
            or total_size <= 0
        ):
            raise ValueError("INVALID_TOTAL_SIZE")

        raw_weight_map = index.get("weight_map")
        if not isinstance(raw_weight_map, Mapping) or not raw_weight_map:
            raise ValueError("WEIGHT_MAP_REQUIRED")

        entries: list[tuple[str, str]] = []
        shard_names: set[str] = set()
        for tensor_name, raw_shard in raw_weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValueError("INVALID_TENSOR_NAME")
            shard = _shard_name(raw_shard)
            entries.append((tensor_name, shard))
            shard_names.add(shard)

        entries.sort(key=lambda item: item[0])
        return cls(
            descriptor=descriptor,
            tensor_shards=tuple(entries),
            shard_names=tuple(sorted(shard_names)),
            total_size=total_size,
        )
