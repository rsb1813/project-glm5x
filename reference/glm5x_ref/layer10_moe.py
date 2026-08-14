# GLM-5.2 layer-10 MoE의 공식 라우터와 지연 expert payload 실행을 제공합니다.

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from k3x_converter.format import DType, K3XError
from k3x_converter.reader import K3XReader
from glm5x_converter.bundle import GLM5XExpertBundle


@dataclass(frozen=True)
class GLM5XExpertWeights:
    """한 routed/shared expert의 gate, up, down projection입니다."""

    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def __post_init__(self) -> None:
        gate = torch.as_tensor(self.gate_proj)
        up = torch.as_tensor(self.up_proj)
        down = torch.as_tensor(self.down_proj)
        if gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
            raise ValueError("GLM5X_EXPERT_WEIGHTS_MUST_BE_RANK_TWO")
        if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
            raise ValueError("GLM5X_EXPERT_WEIGHT_SHAPE_MISMATCH")
        if gate.dtype != up.dtype or gate.dtype != down.dtype:
            raise ValueError("GLM5X_EXPERT_WEIGHT_DTYPE_MISMATCH")


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
        gate_weight = self.weights.gate_proj.to(device=flat.device)
        up_weight = self.weights.up_proj.to(device=flat.device)
        down_weight = self.weights.down_proj.to(device=flat.device)
        work = flat.to(dtype=gate_weight.dtype)
        gate = F.linear(work, gate_weight)
        up = F.linear(work, up_weight)
        output = F.linear(F.silu(gate) * up, down_weight)
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
        if execution_mode not in {"loop", "expert_major"}:
            raise ValueError("GLM5X_INVALID_EXECUTION_MODE")
        if expert_batch_loader is not None and not callable(expert_batch_loader):
            raise ValueError("GLM5X_INVALID_EXPERT_BATCH_LOADER")
        if (
            not isinstance(expert_load_workers, int)
            or isinstance(expert_load_workers, bool)
            or expert_load_workers <= 0
        ):
            raise ValueError("GLM5X_INVALID_EXPERT_LOAD_WORKERS")
        self.router_weight = router_weight
        self.correction_bias = correction_bias.to(torch.float32)
        self.expert_loader = expert_loader
        self.shared_expert = shared_expert
        self.top_k = top_k
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = bool(norm_topk_prob)
        self.cache_experts = cache_experts
        self.execution_mode = execution_mode
        self.expert_batch_loader = expert_batch_loader
        self.expert_load_workers = expert_load_workers
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
            if expert.gate_proj.shape[1] != self.hidden_size:
                raise ValueError("GLM5X_EXPERT_HIDDEN_SIZE_MISMATCH")
            experts[expert_id] = expert
            if self.cache_experts:
                self._expert_cache[expert_id] = expert
            loaded.append(expert_id)
        return experts, tuple(loaded)

    @staticmethod
    def _mlp(hidden: torch.Tensor, expert: GLM5XExpertWeights) -> torch.Tensor:
        gate_weight = expert.gate_proj.to(device=hidden.device)
        up_weight = expert.up_proj.to(device=hidden.device)
        down_weight = expert.down_proj.to(device=hidden.device)
        work = hidden.to(dtype=gate_weight.dtype)
        gate = F.linear(work, gate_weight)
        up = F.linear(work, up_weight)
        return F.linear(F.silu(gate) * up, down_weight)

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
        expert_ids = [int(value) for value in torch.unique(topk_indices, sorted=True)]
        experts, loaded = self._load_experts(expert_ids)
        for expert_id in expert_ids:
            expert = experts[expert_id]
            slot_mask = topk_indices == expert_id
            token_indices, slots = torch.where(slot_mask)
            routed = self._mlp(flat[token_indices], expert)
            weighted = routed * topk_weights[token_indices, slots].to(
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
        expert_ids = [int(value) for value in torch.unique(topk_indices, sorted=True)]
        experts, loaded = self._load_experts(expert_ids)
        assignments: list[tuple[int, GLM5XExpertWeights, torch.Tensor, torch.Tensor]] = []
        for expert_id in expert_ids:
            expert = experts[expert_id]
            slot_mask = topk_indices == expert_id
            token_indices, slots = torch.where(slot_mask)
            assignments.append((expert_id, expert, token_indices, slots))

        if not assignments:
            return loaded

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
            weighted = routed[group_index, :count] * topk_weights[
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
        output += self._mlp(flat, self.shared_expert).to(output.dtype)
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
        execution_mode: str = "loop",
        expert_load_workers: int = 1,
        expert_cache_capacity_bytes: int = 0,
    ) -> "GLM5XLayer10MoEReference":
        bundle = GLM5XExpertBundle.open(
            bundle_path, expert_cache_capacity_bytes=expert_cache_capacity_bytes
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
        )

    @classmethod
    def _from_open_bundle(
        cls,
        bundle: GLM5XExpertBundle,
        *,
        tensor_refs: Mapping[str, tuple[K3XReader, object]],
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
    ) -> "GLM5XLayer10MoEReference":

        prefix = f"model.layers.{layer_id}.mlp"
        target = None if device is None else torch.device(device)
        router_weight = cls._read_tensor(tensor_refs, f"{prefix}.gate.weight").to(
            device=target, dtype=torch.float32
        )
        correction_bias = cls._read_tensor(
            tensor_refs, f"{prefix}.gate.e_score_correction_bias"
        ).to(device=target, dtype=torch.float32)
        shared = cls._read_expert(
            tensor_refs,
            f"{prefix}.shared_experts.gate_proj.weight",
            f"{prefix}.shared_experts.up_proj.weight",
            f"{prefix}.shared_experts.down_proj.weight",
        )
        if target is not None:
            shared = GLM5XExpertWeights(
                *(tensor.to(device=target) for tensor in (
                    shared.gate_proj,
                    shared.up_proj,
                    shared.down_proj,
                ))
            )

        def load_expert(expert_id: int) -> GLM5XExpertWeights:
            try:
                payload = bundle.read_expert(layer_id, expert_id)
            except (KeyError, K3XError) as exc:
                raise K3XError("GLM5X_LAYER_EXPERT_NOT_FOUND", f"{layer_id}:{expert_id}") from exc
            return cls._expert_from_payload(
                payload,
                (expert_intermediate_size, hidden_size),
                (hidden_size, expert_intermediate_size),
                device=target,
            )

        def load_experts(expert_ids: Sequence[int]) -> Mapping[int, GLM5XExpertWeights]:
            def read_one(expert_id: int) -> tuple[int, Mapping[str, bytes]]:
                try:
                    return int(expert_id), bundle.read_expert(layer_id, int(expert_id))
                except (KeyError, K3XError) as exc:
                    raise K3XError(
                        "GLM5X_LAYER_EXPERT_NOT_FOUND", f"{layer_id}:{expert_id}"
                    ) from exc

            if len(expert_ids) <= 1 or expert_load_workers == 1:
                payloads = [read_one(expert_id) for expert_id in expert_ids]
            else:
                with ThreadPoolExecutor(
                    max_workers=min(expert_load_workers, len(expert_ids))
                ) as executor:
                    payloads = list(executor.map(read_one, expert_ids))
            return {
                expert_id: cls._expert_from_payload(
                    payload,
                    (expert_intermediate_size, hidden_size),
                    (hidden_size, expert_intermediate_size),
                    device=target,
                )
                for expert_id, payload in payloads
            }

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
        )

    @staticmethod
    def _read_tensor(
        refs: Mapping[str, tuple[K3XReader, object]], name: str
    ) -> torch.Tensor:
        item = refs.get(name)
        if item is None:
            raise K3XError("GLM5X_LAYER_TENSOR_NOT_FOUND", name)
        reader, record = item
        data, auxiliary = reader.read_tensor_extents(record)
        if auxiliary or record.quantization.name != "NONE":
            raise K3XError("GLM5X_LAYER_UNSUPPORTED_TENSOR", name)
        if record.dtype == DType.BF16:
            values = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16)
        elif record.dtype == DType.FP32:
            values = torch.frombuffer(bytearray(data), dtype=torch.float32)
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
    ) -> GLM5XExpertWeights:
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
    ) -> GLM5XExpertWeights:
        target = None if device is None else torch.device(device)

        def decode(role: str, shape: tuple[int, int]) -> torch.Tensor:
            data = payload.get(role)
            if data is None or len(data) != shape[0] * shape[1] * 2:
                raise K3XError("GLM5X_LAYER_EXPERT_PAYLOAD", role)
            value = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16).reshape(shape)
            return value if target is None else value.to(device=target)

        return GLM5XExpertWeights(
            gate_proj=decode("gate_proj", intermediate_hidden),
            up_proj=decode("up_proj", intermediate_hidden),
            down_proj=decode("down_proj", down_shape),
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
        records = {record.tensor_id: record for record in bundle.readers[artifact_key].tensor_records}
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
