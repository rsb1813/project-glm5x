# GLM-5.x 실행에 필요한 모델 descriptor와 checkpoint metadata 경계를 정의합니다.

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"INVALID_{key.upper()}={value!r}")
    return value


def _first_int(config: Mapping[str, object], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = config.get(key)
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"INVALID_{key.upper()}={value!r}")
            return value
    return default


def _optional_positive_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"INVALID_{key.upper()}={value!r}")
    return value


def _optional_bool(config: Mapping[str, object], key: str) -> bool:
    value = config.get(key)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"INVALID_{key.upper()}={value!r}")
    return value


def _optional_string_tuple(
    config: Mapping[str, object], key: str, allowed: set[str]
) -> tuple[str, ...]:
    value = config.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(item not in allowed for item in value):
        raise ValueError(f"INVALID_{key.upper()}")
    return tuple(value)


@dataclass(frozen=True)
class GLM5XModelDescriptor:
    """모델 가중치와 runtime 정책 사이의 안정적인 GLM descriptor입니다."""

    model_family: str
    attention_kind: str
    hidden_layers: int
    hidden_size: int
    routed_experts: int
    top_k: int
    shared_experts: int
    vocab_size: int
    mtp_layers: int
    moe_intermediate_size: int = 0
    index_topk: int = 0
    index_topk_freq: int = 0
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_share_for_mtp_iteration: bool = False
    max_position_embeddings: int = 0
    indexer_types: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "GLM5XModelDescriptor":
        architectures = config.get("architectures", [])
        if not isinstance(architectures, list) or not any(
            isinstance(item, str) and item.startswith("GlmMoeDsa")
            for item in architectures
        ):
            raise ValueError("GLM_ARCHITECTURE_REQUIRED")

        model_type = config.get("model_type", "")
        if model_type not in {"glm_moe_dsa", "glm5", "glm_5"}:
            raise ValueError(f"GLM_MODEL_TYPE_REQUIRED={model_type!r}")

        raw_top_k = config.get("num_experts_per_tok")
        if (
            not isinstance(raw_top_k, int)
            or isinstance(raw_top_k, bool)
            or raw_top_k <= 0
        ):
            raise ValueError("INVALID_TOP_K")
        top_k = raw_top_k
        routed_experts = _positive_int(
            config,
            "n_routed_experts" if "n_routed_experts" in config else "num_experts",
        )
        if top_k > routed_experts:
            raise ValueError("INVALID_TOP_K")

        hidden_layers = _positive_int(config, "num_hidden_layers")
        indexer_types = _optional_string_tuple(
            config, "indexer_types", {"full", "shared"}
        )
        if indexer_types and len(indexer_types) != hidden_layers:
            raise ValueError("INVALID_INDEXER_TYPES_LENGTH")

        return cls(
            model_family="glm5",
            attention_kind="dsa",
            hidden_layers=hidden_layers,
            hidden_size=_positive_int(config, "hidden_size"),
            routed_experts=routed_experts,
            top_k=top_k,
            shared_experts=_first_int(
                config, "n_shared_experts", "num_shared_experts", default=0
            ),
            vocab_size=_positive_int(config, "vocab_size"),
            mtp_layers=_first_int(
                config, "num_nextn_predict_layers", "num_mtp_layers", default=0
            ),
            moe_intermediate_size=_optional_positive_int(
                config, "moe_intermediate_size"
            ),
            index_topk=_optional_positive_int(config, "index_topk"),
            index_topk_freq=_optional_positive_int(config, "index_topk_freq"),
            index_n_heads=_optional_positive_int(config, "index_n_heads"),
            index_head_dim=_optional_positive_int(config, "index_head_dim"),
            index_share_for_mtp_iteration=_optional_bool(
                config, "index_share_for_mtp_iteration"
            ),
            max_position_embeddings=_optional_positive_int(
                config, "max_position_embeddings"
            ),
            indexer_types=indexer_types,
        )
