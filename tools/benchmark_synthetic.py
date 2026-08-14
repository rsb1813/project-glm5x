# K3X synthetic runtime의 latency, I/O, RSS와 state 크기를 재현 가능하게 측정합니다.
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import struct
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from k3x_converter.reader import K3XReader


@dataclass(frozen=True)
class BenchmarkRecord:
    scope: str
    evidence: str
    platform: str
    iterations: int
    prompt_tokens: int
    generated_tokens: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    ttft_ms: float
    peak_rss_bytes: int
    file_read_bytes_per_token: float
    backend: str
    device: str
    dense_precision: str
    cuda_allocation: str
    cuda_weights: str
    cuda_batching: str
    cuda_boundary: str
    cuda_transfer: str
    cuda_resident_bytes: int
    cuda_pinned_bytes: int
    kernel_nanoseconds: int
    host_to_device_bytes: int
    weight_h2d_bytes: int
    activation_h2d_bytes: int
    device_to_host_bytes: int
    peak_vram_bytes: int | None
    device_allocation_count: int
    device_free_count: int
    stream_synchronization_count: int
    weight_cache_hits: int
    weight_cache_misses: int
    weight_cache_bypasses: int
    resident_weight_bytes: int
    peak_resident_weight_bytes: int
    scratch_bytes: int
    peak_scratch_bytes: int
    grouped_projection_calls: int
    grouped_projection_members: int
    ffn_block_calls: int
    ffn_block_experts: int
    pinned_host_bytes: int
    peak_pinned_host_bytes: int
    async_prefetch_calls: int
    async_prefetch_bytes: int
    async_prefetch_ready_before_use: int
    async_prefetch_late_at_use: int
    transfer_stream_wait_count: int
    pinned_staging_nanoseconds: int
    transfer_device_nanoseconds: int
    transfer_stall_nanoseconds: int
    async_engine_count: int
    device_overlap: bool
    max_absolute_error: float | None
    max_relative_error: float | None
    kda_state_bytes: int
    mla_kv_bytes: int
    per_layer_nanoseconds: tuple[int, ...]
    token_ids: tuple[int, ...]
    routed_experts: tuple[int, ...]
    cuda_moe_fusion: str = "none"
    cuda_weight_validation: str = "per-call"
    immutable_validation_scans: int = 0
    immutable_validation_hits: int = 0
    immutable_validation_bytes: int = 0
    immutable_validation_nanoseconds: int = 0
    fused_moe_calls: int = 0
    fused_moe_experts: int = 0
    batched_expert_ffn_calls: int = 0
    batched_expert_ffn_tokens: int = 0
    resident_grid_calls: int = 0
    resident_grid_experts: int = 0
    resident_grid_tokens: int = 0
    resident_grid_expert_tokens: int = 0
    resident_grid_kernel_launches: int = 0
    resident_grid_fallbacks: int = 0
    resident_grid_descriptor_h2d_bytes: int = 0
    resident_moe_layer_calls: int = 0
    resident_moe_layer_experts: int = 0
    resident_moe_layer_kernel_launches: int = 0
    resident_moe_layer_fallbacks: int = 0
    resident_moe_layer_contribution_h2d_bytes: int = 0
    l1_expert_cache_mode: str = "disabled"
    l1_expert_cache_bytes: int = 0
    l1_expert_cache_hits: int = 0
    l1_expert_cache_misses: int = 0
    l1_expert_cache_bypasses: int = 0
    l1_expert_cache_evictions: int = 0
    l1_expert_cache_collision_misses: int = 0
    l1_expert_cache_resident_bytes: int = 0
    peak_l1_expert_cache_resident_bytes: int = 0
    runtime_profile_metadata_count: int = 0
    runtime_profile_prior_weight: float = 0.0
    runtime_profile_live_observations: int = 0
    runtime_profile_load_bytes: int = 0
    runtime_profile_save_bytes: int = 0
    runtime_profile_load_nanoseconds: int = 0
    runtime_profile_save_nanoseconds: int = 0
    l2_expert_schedule: str = "blocking"
    expert_load_submissions: int = 0
    expert_load_inline_resident_hits: int = 0
    expert_load_completions: int = 0
    expert_load_ready_before_use: int = 0
    expert_load_late_at_use: int = 0
    expert_load_estimated_deadline_misses: int = 0
    expert_load_requested_bytes: int = 0
    expert_load_queue_high_water: int = 0
    expert_load_worker_nanoseconds: int = 0
    expert_load_exposed_wait_nanoseconds: int = 0
    reader_read_calls: int = 0
    reader_requested_bytes: int = 0
    reader_completed_bytes: int = 0
    l2_io_engine: str = "pread"
    l2_cache_mode: str = "buffered"
    l2_queue_depth: int = 8
    l2_direct_memory_alignment: int = 0
    l2_direct_offset_alignment: int = 0
    reader_batch_submissions: int = 0
    reader_storage_submitted_bytes: int = 0
    reader_storage_completed_bytes: int = 0
    reader_completions: int = 0
    reader_short_reads: int = 0
    reader_failures: int = 0
    reader_storage_nanoseconds: int = 0
    process_io_available: bool = False
    process_rchar_bytes: int | None = None
    process_read_bytes: int | None = None
    routing_mode: str = "natural"
    routing_natural_top_k: int = 0
    routing_fixed_k: int = 0
    routing_mass_target: float = 0.9
    routing_min_boundary_gap: float = 0.0
    routing_quality_floor_k: int = 0
    routing_agent_failures: int = 0
    routing_critical: bool = False
    routing_decisions: int = 0
    routing_selected_experts: int = 0
    routing_average_top_k: float = 0.0
    routing_average_normalized_entropy: float = 0.0
    routing_average_selected_mass: float = 0.0
    routing_average_boundary_confidence: float = 0.0
    routing_quality_escalated_decisions: int = 0
    cold_rescue_count: int = 0
    routed_k: tuple[int, ...] = ()
    natural_token_parity: bool | None = None
    first_decision_natural_prefix: bool | None = None
    natural_routing_prefix_rate: float | None = None
    natural_prefill_logits_max_abs_error: float | None = None
    natural_prefill_state_max_abs_error: float | None = None
    speculative_mode: str = "none"
    speculative_verification: str = "token-major"
    speculative_block_size: int = 0
    aurora_draft_k: int = 0
    aurora_block_policy: str = "none"
    aurora_draft_backend: str = "none"
    draft_cuda_resident_bytes: int = 0
    draft_device: str = "CPU"
    draft_cuda_allocation: str = "per-operation"
    draft_cuda_weights: str = "transient"
    draft_cuda_batching: str = "scalar"
    draft_cuda_boundary: str = "operation"
    draft_cuda_transfer: str = "synchronous"
    draft_cuda_moe_fusion: str = "none"
    draft_cuda_weight_validation: str = "per-call"
    draft_immutable_validation_scans: int = 0
    draft_immutable_validation_hits: int = 0
    draft_immutable_validation_bytes: int = 0
    draft_immutable_validation_nanoseconds: int = 0
    draft_kernel_nanoseconds: int = 0
    draft_host_to_device_bytes: int = 0
    draft_weight_h2d_bytes: int = 0
    draft_activation_h2d_bytes: int = 0
    draft_device_to_host_bytes: int = 0
    draft_peak_vram_bytes: int = 0
    draft_device_allocation_count: int = 0
    draft_stream_synchronization_count: int = 0
    draft_weight_cache_hits: int = 0
    draft_weight_cache_misses: int = 0
    draft_weight_cache_bypasses: int = 0
    draft_resident_weight_bytes: int = 0
    draft_peak_resident_weight_bytes: int = 0
    draft_resident_grid_calls: int = 0
    draft_resident_grid_experts: int = 0
    draft_resident_grid_tokens: int = 0
    draft_resident_grid_expert_tokens: int = 0
    draft_resident_grid_kernel_launches: int = 0
    draft_resident_grid_fallbacks: int = 0
    draft_resident_grid_descriptor_h2d_bytes: int = 0
    draft_resident_moe_layer_calls: int = 0
    draft_resident_moe_layer_experts: int = 0
    draft_resident_moe_layer_kernel_launches: int = 0
    draft_resident_moe_layer_fallbacks: int = 0
    draft_resident_moe_layer_contribution_h2d_bytes: int = 0
    draft_ffn_block_calls: int = 0
    draft_ffn_block_experts: int = 0
    draft_proposal_calls: int = 0
    draft_candidate_tokens: int = 0
    draft_replayed_context_tokens: int = 0
    draft_generation_nanoseconds: int = 0
    draft_reader_read_calls: int = 0
    draft_reader_completed_bytes: int = 0
    draft_routing_decisions: int = 0
    draft_routing_selected_experts: int = 0
    draft_selected_length_1: int = 0
    draft_selected_length_2: int = 0
    draft_selected_length_4: int = 0
    draft_scheduler_growths: int = 0
    draft_scheduler_backoffs: int = 0
    draft_context_prefill_tokens: int = 0
    draft_incremental_forward_calls: int = 0
    draft_rollback_events: int = 0
    draft_mla_positions_cropped: int = 0
    draft_kda_checkpoint_bytes: int = 0
    speculative_verification_blocks: int = 0
    speculative_proposed_draft_tokens: int = 0
    speculative_accepted_draft_tokens: int = 0
    speculative_committed_tokens: int = 0
    speculative_max_proposal_tokens: int = 0
    target_decode_forward_calls: int = 0
    target_block_forward_calls: int = 0
    target_positions_evaluated: int = 0
    target_positions_discarded: int = 0
    expert_major_unique_experts_sum: int = 0
    expert_major_unique_experts_max: int = 0
    expert_major_assignments: int = 0
    expert_major_reused_assignments: int = 0
    expert_major_payload_loads: int = 0
    evaluated_routed_experts: tuple[int, ...] = ()
    evaluated_routed_k: tuple[int, ...] = ()
    speculative_acceptance_rate: float | None = None

    def __post_init__(self) -> None:
        if self.scope not in {
            "synthetic-milestone-zero",
            "synthetic-milestone-one",
        }:
            raise ValueError("measured records must use a synthetic-milestone scope")
        if self.evidence != "measured":
            raise ValueError("benchmark evidence must be measured")


def write_results(record: BenchmarkRecord, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_payload = dict(payload)
    csv_payload["per_layer_nanoseconds"] = ";".join(
        str(value) for value in record.per_layer_nanoseconds
    )
    csv_payload["token_ids"] = ";".join(str(value) for value in record.token_ids)
    csv_payload["routed_experts"] = ";".join(
        str(value) for value in record.routed_experts
    )
    csv_payload["routed_k"] = ";".join(str(value) for value in record.routed_k)
    csv_payload["evaluated_routed_experts"] = ";".join(
        str(value) for value in record.evaluated_routed_experts
    )
    csv_payload["evaluated_routed_k"] = ";".join(
        str(value) for value in record.evaluated_routed_k
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=csv_payload.keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(csv_payload)


def _run_process(
    artifact: Path,
    runner: Path,
    generated_tokens: int,
    output: Path,
    *,
    backend: str,
    dense_precision: str,
    cuda_allocation: str,
    cuda_weights: str,
    cuda_batching: str,
    cuda_boundary: str,
    cuda_transfer: str,
    cuda_resident_bytes: int,
    cuda_pinned_bytes: int,
    cuda_moe_fusion: str,
    cuda_weight_validation: str,
    l1_expert_cache: str,
    l1_expert_cache_bytes: int,
    l2_io: str,
    l2_cache: str,
    l2_queue_depth: int,
    l2_expert_schedule: str,
    profile_prior_strength: int = 64,
    runtime_metadata: str = "",
    runtime_profile_in: Path | None = None,
    runtime_profile_out: Path | None = None,
    routing_mode: str = "natural",
    routing_fixed_k: int = 0,
    routing_mass_target: float = 0.9,
    routing_min_boundary_gap: float = 0.0,
    routing_agent_failures: int = 0,
    routing_critical: bool = False,
    speculative_mode: str = "none",
    speculative_verification: str = "token-major",
    speculative_block_size: int = 0,
    speculative_script: str = "",
    aurora_draft_k: int = 0,
    aurora_block_policy: str = "fixed",
    aurora_draft_backend: str = "cpu",
    aurora_draft_resident_bytes: int = 0,
    aurora_draft_batching: str = "grouped",
    aurora_draft_boundary: str = "ffn-block",
    diagnostics: bool = False,
) -> tuple[dict, int, float]:
    command = [
        str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
        "--generate", str(generated_tokens), "--mode", "incremental",
        "--backend", backend, "--dense-precision", dense_precision,
        "--cuda-allocation", cuda_allocation,
        "--cuda-weights", cuda_weights,
        "--cuda-batching", cuda_batching,
        "--cuda-boundary", cuda_boundary,
        "--cuda-transfer", cuda_transfer,
        "--cuda-resident-bytes", str(cuda_resident_bytes),
        "--cuda-pinned-bytes", str(cuda_pinned_bytes),
        "--cuda-moe-fusion", cuda_moe_fusion,
        "--cuda-weight-validation", cuda_weight_validation,
        "--l1-expert-cache", l1_expert_cache,
        "--l1-expert-cache-bytes", str(l1_expert_cache_bytes),
        "--profile-prior-strength", str(profile_prior_strength),
        "--l2-io", l2_io,
        "--l2-cache", l2_cache,
        "--l2-queue-depth", str(l2_queue_depth),
        "--l2-schedule", l2_expert_schedule,
        "--routing-mode", routing_mode,
        "--routing-fixed-k", str(routing_fixed_k),
        "--routing-mass-target", str(routing_mass_target),
        "--routing-min-boundary-gap", str(routing_min_boundary_gap),
        "--routing-agent-failures", str(routing_agent_failures),
        "--routing-critical", str(routing_critical).lower(),
        "--speculative-mode", speculative_mode,
        "--speculative-verification", speculative_verification,
        "--speculative-block-size", str(speculative_block_size),
        "--speculative-script", speculative_script,
    ]
    if speculative_mode in ("aurora-replay", "aurora-persistent"):
        command.extend([
            "--aurora-draft-k", str(aurora_draft_k),
            "--aurora-block-policy", aurora_block_policy,
            "--aurora-draft-backend", aurora_draft_backend,
        ])
    if aurora_draft_resident_bytes > 0:
        command.extend([
            "--aurora-draft-resident-bytes",
            str(aurora_draft_resident_bytes),
        ])
    if (
        speculative_mode == "aurora-persistent"
        and aurora_draft_backend == "cuda-custom"
    ):
        command.extend([
            "--aurora-draft-batching", aurora_draft_batching,
            "--aurora-draft-boundary", aurora_draft_boundary,
        ])
    if runtime_metadata:
        command.extend(["--runtime-metadata", runtime_metadata])
    if runtime_profile_in is not None:
        command.extend(["--runtime-profile-in", str(runtime_profile_in)])
    if runtime_profile_out is not None:
        command.extend(["--runtime-profile-out", str(runtime_profile_out)])
    if diagnostics:
        command.extend(["--diagnostics", "true"])
    command.extend(["--json", str(output)])
    started = time.perf_counter_ns()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed_peak = 0
    observed = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            observed_peak = max(observed_peak, observed.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.001)
    stdout, stderr = process.communicate()
    wall_ms = (time.perf_counter_ns() - started) / 1.0e6
    if process.returncode:
        raise RuntimeError(
            f"runner failed with {process.returncode}: "
            f"{stderr.decode(errors='replace')} {stdout.decode(errors='replace')}"
        )
    return json.loads(output.read_text(encoding="utf-8")), observed_peak, wall_ms


def _state_sizes(artifact: Path, context_tokens: int) -> tuple[int, int]:
    config = K3XReader.open(artifact).model_config
    values = struct.unpack_from("<20I", config)
    layer_count, kda_heads, kda_dim, conv_kernel = values[2:6]
    mla_heads, q_main, q_extra, value_dim = values[6], values[9], values[10], values[11]
    kda_layers = layer_count - 1
    kda_per_layer = (
        3 * (conv_kernel - 1) * kda_heads * kda_dim
        + kda_heads * kda_dim * kda_dim
    ) * 4
    mla_per_token = (mla_heads * (q_main + value_dim) + q_extra) * 4
    return kda_layers * kda_per_layer, context_tokens * mla_per_token


def _flatten_numbers(value: object) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    return [float(value)]


def _numerical_errors(reference: dict, candidate: dict) -> tuple[float, float]:
    if candidate["token_ids"] != reference["token_ids"]:
        raise RuntimeError("backend token sequence diverged from the CPU reference")
    reference_values: list[float] = []
    candidate_values: list[float] = []
    for key in ("prefill_layer_outputs", "prefill_logits", "prefill_state"):
        reference_values.extend(_flatten_numbers(reference[key]))
        candidate_values.extend(_flatten_numbers(candidate[key]))
    if len(reference_values) != len(candidate_values):
        raise RuntimeError("backend diagnostic shape diverged from the CPU reference")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for expected, actual in zip(reference_values, candidate_values, strict=True):
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1.0e-30)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def benchmark_once(
    artifact: Path,
    runner: Path,
    warmup: int,
    iterations: int,
    *,
    backend: str = "cpu",
    dense_precision: str = "fp32",
    cuda_allocation: str = "per-operation",
    cuda_weights: str = "transient",
    cuda_batching: str = "scalar",
    cuda_boundary: str = "operation",
    cuda_transfer: str = "synchronous",
    cuda_moe_fusion: str = "none",
    cuda_weight_validation: str = "per-call",
    cuda_resident_bytes: int = 0,
    cuda_pinned_bytes: int = 0,
    l1_expert_cache: str = "disabled",
    l1_expert_cache_bytes: int = 0,
    l2_io: str = "pread",
    l2_cache: str = "buffered",
    l2_queue_depth: int = 8,
    l2_expert_schedule: str = "blocking",
    profile_prior_strength: int = 64,
    runtime_metadata: str = "",
    runtime_profile_in: Path | None = None,
    runtime_profile_out: Path | None = None,
    routing_mode: str = "natural",
    routing_fixed_k: int = 0,
    routing_mass_target: float = 0.9,
    routing_min_boundary_gap: float = 0.0,
    routing_agent_failures: int = 0,
    routing_critical: bool = False,
    speculative_mode: str = "none",
    speculative_verification: str = "token-major",
    speculative_block_size: int = 0,
    speculative_script: str = "",
    aurora_draft_k: int = 0,
    aurora_block_policy: str = "fixed",
    aurora_draft_backend: str = "cpu",
    aurora_draft_resident_bytes: int = 0,
    aurora_draft_batching: str = "grouped",
    aurora_draft_boundary: str = "ffn-block",
) -> BenchmarkRecord:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    artifact, runner = Path(artifact).resolve(), Path(runner).resolve()
    prompt_tokens, generated_tokens = 4, 6
    samples: list[dict] = []
    peaks: list[int] = []
    ttft_samples: list[float] = []
    routed_experts: tuple[int, ...] = ()
    routed_k: tuple[int, ...] = ()
    evaluated_routed_experts: tuple[int, ...] = ()
    evaluated_routed_k: tuple[int, ...] = ()
    with tempfile.TemporaryDirectory(prefix="k3x-benchmark-") as temporary:
        root = Path(temporary)
        for index in range(warmup + iterations):
            sample, peak, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / f"run-{index}.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_boundary=cuda_boundary,
                cuda_transfer=cuda_transfer,
                cuda_moe_fusion=cuda_moe_fusion,
                cuda_weight_validation=cuda_weight_validation,
                cuda_resident_bytes=cuda_resident_bytes,
                cuda_pinned_bytes=cuda_pinned_bytes,
                l1_expert_cache=l1_expert_cache,
                l1_expert_cache_bytes=l1_expert_cache_bytes,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                profile_prior_strength=profile_prior_strength,
                runtime_metadata=runtime_metadata,
                runtime_profile_in=runtime_profile_in,
                runtime_profile_out=(
                    root / f"profile-run-{index}.k3xp"
                    if runtime_profile_out is not None else None
                ),
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script=speculative_script,
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
            )
            _, ttft_peak, ttft = _run_process(
                artifact,
                runner,
                1,
                root / f"ttft-{index}.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_boundary=cuda_boundary,
                cuda_transfer=cuda_transfer,
                cuda_moe_fusion=cuda_moe_fusion,
                cuda_weight_validation=cuda_weight_validation,
                cuda_resident_bytes=cuda_resident_bytes,
                cuda_pinned_bytes=cuda_pinned_bytes,
                l1_expert_cache=l1_expert_cache,
                l1_expert_cache_bytes=l1_expert_cache_bytes,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                profile_prior_strength=profile_prior_strength,
                runtime_metadata=runtime_metadata,
                runtime_profile_in=runtime_profile_in,
                runtime_profile_out=(
                    root / f"profile-ttft-{index}.k3xp"
                    if runtime_profile_out is not None else None
                ),
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script="",
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
            )
            if index >= warmup:
                samples.append(sample)
                peaks.append(max(peak, ttft_peak))
                ttft_samples.append(ttft)
        if backend == "cpu":
            max_absolute_error = 0.0
            max_relative_error = 0.0
            diagnostic, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "diagnostic.json",
                backend="cpu",
                dense_precision="fp32",
                cuda_allocation="per-operation",
                cuda_weights="transient",
                cuda_batching="scalar",
                cuda_boundary="operation",
                cuda_transfer="synchronous",
                cuda_moe_fusion="none",
                cuda_weight_validation="per-call",
                cuda_resident_bytes=0,
                cuda_pinned_bytes=0,
                l1_expert_cache="disabled",
                l1_expert_cache_bytes=0,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script=speculative_script,
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
                diagnostics=True,
            )
            if diagnostic["token_ids"] != samples[0]["token_ids"]:
                raise RuntimeError("CPU diagnostic token sequence diverged")
            routed_experts = tuple(diagnostic["prefill_routed_experts"])
            routed_k = tuple(diagnostic["prefill_routed_k"])
            evaluated_routed_experts = tuple(
                diagnostic["evaluated_routed_experts"]
            )
            evaluated_routed_k = tuple(diagnostic["evaluated_routed_k"])
        else:
            reference, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "reference.json",
                backend="cpu",
                dense_precision="fp32",
                cuda_allocation="per-operation",
                cuda_weights="transient",
                cuda_batching="scalar",
                cuda_boundary="operation",
                cuda_transfer="synchronous",
                cuda_moe_fusion="none",
                cuda_weight_validation="per-call",
                cuda_resident_bytes=0,
                cuda_pinned_bytes=0,
                l1_expert_cache="disabled",
                l1_expert_cache_bytes=0,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script=speculative_script,
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
                diagnostics=True,
            )
            candidate, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "candidate.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_boundary=cuda_boundary,
                cuda_transfer=cuda_transfer,
                cuda_moe_fusion=cuda_moe_fusion,
                cuda_weight_validation=cuda_weight_validation,
                cuda_resident_bytes=cuda_resident_bytes,
                cuda_pinned_bytes=cuda_pinned_bytes,
                l1_expert_cache=l1_expert_cache,
                l1_expert_cache_bytes=l1_expert_cache_bytes,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script=speculative_script,
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
                diagnostics=True,
            )
            max_absolute_error, max_relative_error = _numerical_errors(
                reference, candidate
            )
            routed_experts = tuple(candidate["prefill_routed_experts"])
            routed_k = tuple(candidate["prefill_routed_k"])
            evaluated_routed_experts = tuple(
                candidate["evaluated_routed_experts"]
            )
            evaluated_routed_k = tuple(candidate["evaluated_routed_k"])
        if runtime_profile_out is not None:
            materialized, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "materialized-profile.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_boundary=cuda_boundary,
                cuda_transfer=cuda_transfer,
                cuda_moe_fusion=cuda_moe_fusion,
                cuda_weight_validation=cuda_weight_validation,
                cuda_resident_bytes=cuda_resident_bytes,
                cuda_pinned_bytes=cuda_pinned_bytes,
                l1_expert_cache=l1_expert_cache,
                l1_expert_cache_bytes=l1_expert_cache_bytes,
                l2_io=l2_io,
                l2_cache=l2_cache,
                l2_queue_depth=l2_queue_depth,
                l2_expert_schedule=l2_expert_schedule,
                profile_prior_strength=profile_prior_strength,
                runtime_metadata=runtime_metadata,
                runtime_profile_in=runtime_profile_in,
                runtime_profile_out=runtime_profile_out,
                routing_mode=routing_mode,
                routing_fixed_k=routing_fixed_k,
                routing_mass_target=routing_mass_target,
                routing_min_boundary_gap=routing_min_boundary_gap,
                routing_agent_failures=routing_agent_failures,
                routing_critical=routing_critical,
                speculative_mode=speculative_mode,
                speculative_verification=speculative_verification,
                speculative_block_size=speculative_block_size,
                speculative_script=speculative_script,
                aurora_draft_k=aurora_draft_k,
                aurora_block_policy=aurora_block_policy,
                aurora_draft_backend=aurora_draft_backend,
                aurora_draft_resident_bytes=aurora_draft_resident_bytes,
                aurora_draft_batching=aurora_draft_batching,
                aurora_draft_boundary=aurora_draft_boundary,
            )
            if (
                materialized["token_ids"] != samples[0]["token_ids"]
                or materialized["runtime_profile_save_bytes"]
                != samples[0]["runtime_profile_save_bytes"]
                or Path(runtime_profile_out).stat().st_size
                != samples[0]["runtime_profile_save_bytes"]
            ):
                raise RuntimeError("materialized runtime profile diverged")
    deterministic_fields = (
        "backend",
        "device",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_boundary",
        "cuda_transfer",
        "cuda_resident_bytes",
        "cuda_pinned_bytes",
        "l1_expert_cache_mode",
        "l1_expert_cache_bytes",
        "l1_expert_cache_hits",
        "l1_expert_cache_misses",
        "l1_expert_cache_bypasses",
        "l1_expert_cache_evictions",
        "l1_expert_cache_collision_misses",
        "l1_expert_cache_resident_bytes",
        "peak_l1_expert_cache_resident_bytes",
        "runtime_profile_metadata_count",
        "runtime_profile_prior_weight",
        "runtime_profile_live_observations",
        "runtime_profile_load_bytes",
        "runtime_profile_save_bytes",
        "routing_mode",
        "routing_natural_top_k",
        "routing_fixed_k",
        "routing_mass_target",
        "routing_min_boundary_gap",
        "routing_quality_floor_k",
        "routing_agent_failures",
        "routing_critical",
        "routing_decisions",
        "routing_selected_experts",
        "routing_average_top_k",
        "routing_average_normalized_entropy",
        "routing_average_selected_mass",
        "routing_average_boundary_confidence",
        "routing_quality_escalated_decisions",
        "cold_rescue_count",
        "l2_expert_schedule",
        "expert_load_submissions",
        "expert_load_inline_resident_hits",
        "expert_load_completions",
        "expert_load_requested_bytes",
        "reader_read_calls",
        "reader_requested_bytes",
        "reader_completed_bytes",
        "l2_io_engine",
        "l2_cache_mode",
        "l2_queue_depth",
        "l2_direct_memory_alignment",
        "l2_direct_offset_alignment",
        "reader_batch_submissions",
        "reader_storage_submitted_bytes",
        "reader_storage_completed_bytes",
        "reader_completions",
        "reader_short_reads",
        "reader_failures",
        "process_io_available",
        "device_allocation_count",
        "device_free_count",
        "stream_synchronization_count",
        "weight_cache_hits",
        "weight_cache_misses",
        "weight_cache_bypasses",
        "resident_weight_bytes",
        "peak_resident_weight_bytes",
        "scratch_bytes",
        "peak_scratch_bytes",
        "weight_h2d_bytes",
        "activation_h2d_bytes",
        "grouped_projection_calls",
        "grouped_projection_members",
        "resident_grid_calls",
        "resident_grid_experts",
        "resident_grid_tokens",
        "resident_grid_expert_tokens",
        "resident_grid_kernel_launches",
        "resident_grid_fallbacks",
        "resident_grid_descriptor_h2d_bytes",
        "resident_moe_layer_calls",
        "resident_moe_layer_experts",
        "resident_moe_layer_kernel_launches",
        "resident_moe_layer_fallbacks",
        "resident_moe_layer_contribution_h2d_bytes",
        "ffn_block_calls",
        "ffn_block_experts",
        "cuda_moe_fusion",
        "cuda_weight_validation",
        "immutable_validation_scans",
        "immutable_validation_hits",
        "immutable_validation_bytes",
        "fused_moe_calls",
        "fused_moe_experts",
        "batched_expert_ffn_calls",
        "batched_expert_ffn_tokens",
        "pinned_host_bytes",
        "peak_pinned_host_bytes",
        "async_prefetch_calls",
        "async_prefetch_bytes",
        "transfer_stream_wait_count",
        "async_engine_count",
        "device_overlap",
        "speculative_mode",
        "speculative_verification",
        "speculative_block_size",
        "aurora_draft_k",
        "aurora_block_policy",
        "aurora_draft_backend",
        "draft_cuda_resident_bytes",
        "draft_device",
        "draft_cuda_allocation",
        "draft_cuda_weights",
        "draft_cuda_batching",
        "draft_cuda_boundary",
        "draft_cuda_transfer",
        "draft_cuda_moe_fusion",
        "draft_cuda_weight_validation",
        "draft_immutable_validation_scans",
        "draft_immutable_validation_hits",
        "draft_immutable_validation_bytes",
        "draft_host_to_device_bytes",
        "draft_weight_h2d_bytes",
        "draft_activation_h2d_bytes",
        "draft_device_to_host_bytes",
        "draft_peak_vram_bytes",
        "draft_device_allocation_count",
        "draft_stream_synchronization_count",
        "draft_weight_cache_hits",
        "draft_weight_cache_misses",
        "draft_weight_cache_bypasses",
        "draft_resident_weight_bytes",
        "draft_peak_resident_weight_bytes",
        "draft_resident_grid_calls",
        "draft_resident_grid_experts",
        "draft_resident_grid_tokens",
        "draft_resident_grid_expert_tokens",
        "draft_resident_grid_kernel_launches",
        "draft_resident_grid_fallbacks",
        "draft_resident_grid_descriptor_h2d_bytes",
        "draft_resident_moe_layer_calls",
        "draft_resident_moe_layer_experts",
        "draft_resident_moe_layer_kernel_launches",
        "draft_resident_moe_layer_fallbacks",
        "draft_resident_moe_layer_contribution_h2d_bytes",
        "draft_ffn_block_calls",
        "draft_ffn_block_experts",
        "draft_proposal_calls",
        "draft_candidate_tokens",
        "draft_replayed_context_tokens",
        "draft_reader_read_calls",
        "draft_reader_completed_bytes",
        "draft_routing_decisions",
        "draft_routing_selected_experts",
        "draft_selected_length_1",
        "draft_selected_length_2",
        "draft_selected_length_4",
        "draft_scheduler_growths",
        "draft_scheduler_backoffs",
        "draft_context_prefill_tokens",
        "draft_incremental_forward_calls",
        "draft_rollback_events",
        "draft_mla_positions_cropped",
        "draft_kda_checkpoint_bytes",
        "speculative_verification_blocks",
        "speculative_proposed_draft_tokens",
        "speculative_accepted_draft_tokens",
        "speculative_committed_tokens",
        "speculative_max_proposal_tokens",
        "target_decode_forward_calls",
        "target_block_forward_calls",
        "target_positions_evaluated",
        "target_positions_discarded",
        "expert_major_unique_experts_sum",
        "expert_major_unique_experts_max",
        "expert_major_assignments",
        "expert_major_reused_assignments",
        "expert_major_payload_loads",
        "speculative_acceptance_rate",
        "token_ids",
    )
    if any(
        any(item[field] != samples[0][field] for field in deterministic_fields)
        for item in samples[1:]
    ):
        raise RuntimeError("runner metadata changed across benchmark samples")
    expected_options = (
        backend,
        dense_precision,
        cuda_allocation,
        cuda_weights,
        cuda_batching,
        cuda_boundary,
        cuda_transfer,
        cuda_moe_fusion,
        cuda_weight_validation,
        cuda_resident_bytes,
        cuda_pinned_bytes,
        l1_expert_cache,
        l1_expert_cache_bytes,
        l2_expert_schedule,
        l2_io,
        l2_cache,
        l2_queue_depth,
        routing_mode,
        routing_fixed_k,
        routing_mass_target,
        routing_min_boundary_gap,
        routing_agent_failures,
        routing_critical,
        speculative_mode,
        speculative_verification,
        speculative_block_size,
        aurora_draft_k if speculative_mode in ("aurora-replay", "aurora-persistent") else 0,
        aurora_block_policy if speculative_mode in ("aurora-replay", "aurora-persistent") else "none",
        aurora_draft_backend
        if speculative_mode in ("aurora-replay", "aurora-persistent")
        else "none",
        aurora_draft_resident_bytes,
        aurora_draft_batching
        if speculative_mode == "aurora-persistent"
        and aurora_draft_backend == "cuda-custom"
        else "scalar",
    )
    option_fields = (
        "backend",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_boundary",
        "cuda_transfer",
        "cuda_moe_fusion",
        "cuda_weight_validation",
        "cuda_resident_bytes",
        "cuda_pinned_bytes",
        "l1_expert_cache_mode",
        "l1_expert_cache_bytes",
        "l2_expert_schedule",
        "l2_io_engine",
        "l2_cache_mode",
        "l2_queue_depth",
        "routing_mode",
        "routing_fixed_k",
        "routing_mass_target",
        "routing_min_boundary_gap",
        "routing_agent_failures",
        "routing_critical",
        "speculative_mode",
        "speculative_verification",
        "speculative_block_size",
        "aurora_draft_k",
        "aurora_block_policy",
        "aurora_draft_backend",
        "draft_cuda_resident_bytes",
        "draft_cuda_batching",
    )
    observed_options = tuple(samples[0][field] for field in option_fields)
    float_option_fields = {"routing_mass_target", "routing_min_boundary_gap"}
    options_match = all(
        math.isclose(observed, expected, rel_tol=1.0e-6, abs_tol=1.0e-7)
        if field in float_option_fields
        else observed == expected
        for field, observed, expected in zip(
            option_fields, observed_options, expected_options, strict=True
        )
    )
    if not options_match:
        raise RuntimeError("runner metadata did not match requested benchmark options")
    if any(
        item["async_prefetch_ready_before_use"]
        + item["async_prefetch_late_at_use"]
        != item["async_prefetch_calls"]
        for item in samples
    ):
        raise RuntimeError("runner async readiness accounting is inconsistent")
    prefill_ns = statistics.median(item["prefill_nanoseconds"] for item in samples)
    decode_ns = statistics.median(item["decode_nanoseconds"] for item in samples)
    layer_count = len(samples[0]["per_layer_nanoseconds"])
    layer_ns = tuple(
        int(statistics.median(item["per_layer_nanoseconds"][index] for item in samples))
        for index in range(layer_count)
    )
    kda_state, mla_kv = _state_sizes(artifact, prompt_tokens + generated_tokens - 1)
    return BenchmarkRecord(
        scope="synthetic-milestone-one",
        evidence="measured",
        platform=f"{platform.system()} {platform.release()} / {platform.machine()}",
        iterations=iterations,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_tokens_per_second=prompt_tokens * 1.0e9 / prefill_ns,
        decode_tokens_per_second=(generated_tokens - 1) * 1.0e9 / decode_ns,
        ttft_ms=statistics.median(ttft_samples),
        peak_rss_bytes=max(peaks),
        file_read_bytes_per_token=statistics.median(
            item["read_bytes"] / generated_tokens for item in samples
        ),
        backend=samples[0]["backend"],
        device=samples[0]["device"],
        dense_precision=samples[0]["dense_precision"],
        cuda_allocation=samples[0]["cuda_allocation"],
        cuda_weights=samples[0]["cuda_weights"],
        cuda_batching=samples[0]["cuda_batching"],
        cuda_boundary=samples[0]["cuda_boundary"],
        cuda_transfer=samples[0]["cuda_transfer"],
        cuda_moe_fusion=samples[0]["cuda_moe_fusion"],
        cuda_weight_validation=samples[0]["cuda_weight_validation"],
        immutable_validation_scans=samples[0]["immutable_validation_scans"],
        immutable_validation_hits=samples[0]["immutable_validation_hits"],
        immutable_validation_bytes=samples[0]["immutable_validation_bytes"],
        immutable_validation_nanoseconds=int(statistics.median(
            item["immutable_validation_nanoseconds"] for item in samples
        )),
        cuda_resident_bytes=samples[0]["cuda_resident_bytes"],
        cuda_pinned_bytes=samples[0]["cuda_pinned_bytes"],
        kernel_nanoseconds=int(
            statistics.median(item["kernel_nanoseconds"] for item in samples)
        ),
        host_to_device_bytes=int(
            statistics.median(item["host_to_device_bytes"] for item in samples)
        ),
        weight_h2d_bytes=samples[0]["weight_h2d_bytes"],
        activation_h2d_bytes=samples[0]["activation_h2d_bytes"],
        device_to_host_bytes=int(
            statistics.median(item["device_to_host_bytes"] for item in samples)
        ),
        peak_vram_bytes=max(item["peak_vram_bytes"] for item in samples),
        device_allocation_count=samples[0]["device_allocation_count"],
        device_free_count=samples[0]["device_free_count"],
        stream_synchronization_count=samples[0]["stream_synchronization_count"],
        weight_cache_hits=samples[0]["weight_cache_hits"],
        weight_cache_misses=samples[0]["weight_cache_misses"],
        weight_cache_bypasses=samples[0]["weight_cache_bypasses"],
        resident_weight_bytes=samples[0]["resident_weight_bytes"],
        peak_resident_weight_bytes=samples[0]["peak_resident_weight_bytes"],
        scratch_bytes=samples[0]["scratch_bytes"],
        peak_scratch_bytes=samples[0]["peak_scratch_bytes"],
        grouped_projection_calls=samples[0]["grouped_projection_calls"],
        grouped_projection_members=samples[0]["grouped_projection_members"],
        resident_grid_calls=samples[0]["resident_grid_calls"],
        resident_grid_experts=samples[0]["resident_grid_experts"],
        resident_grid_tokens=samples[0]["resident_grid_tokens"],
        resident_grid_expert_tokens=samples[0]["resident_grid_expert_tokens"],
        resident_grid_kernel_launches=samples[0]["resident_grid_kernel_launches"],
        resident_grid_fallbacks=samples[0]["resident_grid_fallbacks"],
        resident_grid_descriptor_h2d_bytes=samples[0][
            "resident_grid_descriptor_h2d_bytes"
        ],
        resident_moe_layer_calls=samples[0]["resident_moe_layer_calls"],
        resident_moe_layer_experts=samples[0]["resident_moe_layer_experts"],
        resident_moe_layer_kernel_launches=samples[0][
            "resident_moe_layer_kernel_launches"
        ],
        resident_moe_layer_fallbacks=samples[0][
            "resident_moe_layer_fallbacks"
        ],
        resident_moe_layer_contribution_h2d_bytes=samples[0][
            "resident_moe_layer_contribution_h2d_bytes"
        ],
        ffn_block_calls=samples[0]["ffn_block_calls"],
        ffn_block_experts=samples[0]["ffn_block_experts"],
        fused_moe_calls=samples[0]["fused_moe_calls"],
        fused_moe_experts=samples[0]["fused_moe_experts"],
        batched_expert_ffn_calls=samples[0]["batched_expert_ffn_calls"],
        batched_expert_ffn_tokens=samples[0]["batched_expert_ffn_tokens"],
        pinned_host_bytes=samples[0]["pinned_host_bytes"],
        peak_pinned_host_bytes=samples[0]["peak_pinned_host_bytes"],
        async_prefetch_calls=samples[0]["async_prefetch_calls"],
        async_prefetch_bytes=samples[0]["async_prefetch_bytes"],
        async_prefetch_ready_before_use=int(statistics.median(
            item["async_prefetch_ready_before_use"] for item in samples
        )),
        async_prefetch_late_at_use=(
            samples[0]["async_prefetch_calls"]
            - int(statistics.median(
                item["async_prefetch_ready_before_use"] for item in samples
            ))
        ),
        transfer_stream_wait_count=samples[0]["transfer_stream_wait_count"],
        pinned_staging_nanoseconds=int(statistics.median(
            item["pinned_staging_nanoseconds"] for item in samples
        )),
        transfer_device_nanoseconds=int(statistics.median(
            item["transfer_device_nanoseconds"] for item in samples
        )),
        transfer_stall_nanoseconds=int(statistics.median(
            item["transfer_stall_nanoseconds"] for item in samples
        )),
        async_engine_count=samples[0]["async_engine_count"],
        device_overlap=bool(samples[0]["device_overlap"]),
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        kda_state_bytes=kda_state,
        mla_kv_bytes=mla_kv,
        per_layer_nanoseconds=layer_ns,
        token_ids=tuple(samples[0]["token_ids"]),
        routed_experts=routed_experts,
        routed_k=routed_k,
        evaluated_routed_experts=evaluated_routed_experts,
        evaluated_routed_k=evaluated_routed_k,
        l1_expert_cache_mode=samples[0]["l1_expert_cache_mode"],
        l1_expert_cache_bytes=samples[0]["l1_expert_cache_bytes"],
        l1_expert_cache_hits=samples[0]["l1_expert_cache_hits"],
        l1_expert_cache_misses=samples[0]["l1_expert_cache_misses"],
        l1_expert_cache_bypasses=samples[0]["l1_expert_cache_bypasses"],
        l1_expert_cache_evictions=samples[0]["l1_expert_cache_evictions"],
        l1_expert_cache_collision_misses=samples[0][
            "l1_expert_cache_collision_misses"
        ],
        l1_expert_cache_resident_bytes=samples[0][
            "l1_expert_cache_resident_bytes"
        ],
        peak_l1_expert_cache_resident_bytes=samples[0][
            "peak_l1_expert_cache_resident_bytes"
        ],
        runtime_profile_metadata_count=samples[0][
            "runtime_profile_metadata_count"
        ],
        runtime_profile_prior_weight=samples[0][
            "runtime_profile_prior_weight"
        ],
        runtime_profile_live_observations=samples[0][
            "runtime_profile_live_observations"
        ],
        runtime_profile_load_bytes=samples[0]["runtime_profile_load_bytes"],
        runtime_profile_save_bytes=samples[0]["runtime_profile_save_bytes"],
        runtime_profile_load_nanoseconds=int(statistics.median(
            item["runtime_profile_load_nanoseconds"] for item in samples
        )),
        runtime_profile_save_nanoseconds=int(statistics.median(
            item["runtime_profile_save_nanoseconds"] for item in samples
        )),
        l2_expert_schedule=samples[0]["l2_expert_schedule"],
        expert_load_submissions=samples[0]["expert_load_submissions"],
        expert_load_inline_resident_hits=samples[0][
            "expert_load_inline_resident_hits"
        ],
        expert_load_completions=samples[0]["expert_load_completions"],
        expert_load_ready_before_use=int(statistics.median(
            item["expert_load_ready_before_use"] for item in samples
        )),
        expert_load_late_at_use=int(statistics.median(
            item["expert_load_late_at_use"] for item in samples
        )),
        expert_load_estimated_deadline_misses=int(statistics.median(
            item["expert_load_estimated_deadline_misses"] for item in samples
        )),
        expert_load_requested_bytes=samples[0]["expert_load_requested_bytes"],
        expert_load_queue_high_water=int(statistics.median(
            item["expert_load_queue_high_water"] for item in samples
        )),
        expert_load_worker_nanoseconds=int(statistics.median(
            item["expert_load_worker_nanoseconds"] for item in samples
        )),
        expert_load_exposed_wait_nanoseconds=int(statistics.median(
            item["expert_load_exposed_wait_nanoseconds"] for item in samples
        )),
        reader_read_calls=samples[0]["reader_read_calls"],
        reader_requested_bytes=samples[0]["reader_requested_bytes"],
        reader_completed_bytes=samples[0]["reader_completed_bytes"],
        l2_io_engine=samples[0]["l2_io_engine"],
        l2_cache_mode=samples[0]["l2_cache_mode"],
        l2_queue_depth=samples[0]["l2_queue_depth"],
        l2_direct_memory_alignment=samples[0]["l2_direct_memory_alignment"],
        l2_direct_offset_alignment=samples[0]["l2_direct_offset_alignment"],
        reader_batch_submissions=samples[0]["reader_batch_submissions"],
        reader_storage_submitted_bytes=samples[0]["reader_storage_submitted_bytes"],
        reader_storage_completed_bytes=samples[0]["reader_storage_completed_bytes"],
        reader_completions=samples[0]["reader_completions"],
        reader_short_reads=samples[0]["reader_short_reads"],
        reader_failures=samples[0]["reader_failures"],
        reader_storage_nanoseconds=int(statistics.median(
            item["reader_storage_nanoseconds"] for item in samples
        )),
        process_io_available=samples[0]["process_io_available"],
        process_rchar_bytes=(
            int(statistics.median(item["process_rchar_bytes"] for item in samples))
            if samples[0]["process_io_available"] else None
        ),
        process_read_bytes=(
            int(statistics.median(item["process_read_bytes"] for item in samples))
            if samples[0]["process_io_available"] else None
        ),
        routing_mode=samples[0]["routing_mode"],
        routing_natural_top_k=samples[0]["routing_natural_top_k"],
        routing_fixed_k=samples[0]["routing_fixed_k"],
        routing_mass_target=samples[0]["routing_mass_target"],
        routing_min_boundary_gap=samples[0]["routing_min_boundary_gap"],
        routing_quality_floor_k=samples[0]["routing_quality_floor_k"],
        routing_agent_failures=samples[0]["routing_agent_failures"],
        routing_critical=bool(samples[0]["routing_critical"]),
        routing_decisions=samples[0]["routing_decisions"],
        routing_selected_experts=samples[0]["routing_selected_experts"],
        routing_average_top_k=samples[0]["routing_average_top_k"],
        routing_average_normalized_entropy=samples[0][
            "routing_average_normalized_entropy"
        ],
        routing_average_selected_mass=samples[0][
            "routing_average_selected_mass"
        ],
        routing_average_boundary_confidence=samples[0][
            "routing_average_boundary_confidence"
        ],
        routing_quality_escalated_decisions=samples[0][
            "routing_quality_escalated_decisions"
        ],
        cold_rescue_count=samples[0]["cold_rescue_count"],
        speculative_mode=samples[0]["speculative_mode"],
        speculative_verification=samples[0]["speculative_verification"],
        speculative_block_size=samples[0]["speculative_block_size"],
        aurora_draft_k=samples[0]["aurora_draft_k"],
        aurora_block_policy=samples[0]["aurora_block_policy"],
        aurora_draft_backend=samples[0]["aurora_draft_backend"],
        draft_cuda_resident_bytes=samples[0]["draft_cuda_resident_bytes"],
        draft_device=samples[0]["draft_device"],
        draft_cuda_allocation=samples[0]["draft_cuda_allocation"],
        draft_cuda_weights=samples[0]["draft_cuda_weights"],
        draft_cuda_batching=samples[0]["draft_cuda_batching"],
        draft_cuda_boundary=samples[0]["draft_cuda_boundary"],
        draft_cuda_transfer=samples[0]["draft_cuda_transfer"],
        draft_cuda_moe_fusion=samples[0]["draft_cuda_moe_fusion"],
        draft_cuda_weight_validation=samples[0][
            "draft_cuda_weight_validation"
        ],
        draft_immutable_validation_scans=samples[0][
            "draft_immutable_validation_scans"
        ],
        draft_immutable_validation_hits=samples[0][
            "draft_immutable_validation_hits"
        ],
        draft_immutable_validation_bytes=samples[0][
            "draft_immutable_validation_bytes"
        ],
        draft_immutable_validation_nanoseconds=int(statistics.median(
            item["draft_immutable_validation_nanoseconds"] for item in samples
        )),
        draft_kernel_nanoseconds=int(statistics.median(
            item["draft_kernel_nanoseconds"] for item in samples
        )),
        draft_host_to_device_bytes=samples[0]["draft_host_to_device_bytes"],
        draft_weight_h2d_bytes=samples[0]["draft_weight_h2d_bytes"],
        draft_activation_h2d_bytes=samples[0]["draft_activation_h2d_bytes"],
        draft_device_to_host_bytes=samples[0]["draft_device_to_host_bytes"],
        draft_peak_vram_bytes=max(
            item["draft_peak_vram_bytes"] for item in samples
        ),
        draft_device_allocation_count=samples[0]["draft_device_allocation_count"],
        draft_stream_synchronization_count=samples[0][
            "draft_stream_synchronization_count"
        ],
        draft_weight_cache_hits=samples[0]["draft_weight_cache_hits"],
        draft_weight_cache_misses=samples[0]["draft_weight_cache_misses"],
        draft_weight_cache_bypasses=samples[0]["draft_weight_cache_bypasses"],
        draft_resident_weight_bytes=samples[0][
            "draft_resident_weight_bytes"
        ],
        draft_peak_resident_weight_bytes=samples[0][
            "draft_peak_resident_weight_bytes"
        ],
        draft_resident_grid_calls=samples[0]["draft_resident_grid_calls"],
        draft_resident_grid_experts=samples[0]["draft_resident_grid_experts"],
        draft_resident_grid_tokens=samples[0]["draft_resident_grid_tokens"],
        draft_resident_grid_expert_tokens=samples[0][
            "draft_resident_grid_expert_tokens"
        ],
        draft_resident_grid_kernel_launches=samples[0][
            "draft_resident_grid_kernel_launches"
        ],
        draft_resident_grid_fallbacks=samples[0][
            "draft_resident_grid_fallbacks"
        ],
        draft_resident_grid_descriptor_h2d_bytes=samples[0][
            "draft_resident_grid_descriptor_h2d_bytes"
        ],
        draft_resident_moe_layer_calls=samples[0][
            "draft_resident_moe_layer_calls"
        ],
        draft_resident_moe_layer_experts=samples[0][
            "draft_resident_moe_layer_experts"
        ],
        draft_resident_moe_layer_kernel_launches=samples[0][
            "draft_resident_moe_layer_kernel_launches"
        ],
        draft_resident_moe_layer_fallbacks=samples[0][
            "draft_resident_moe_layer_fallbacks"
        ],
        draft_resident_moe_layer_contribution_h2d_bytes=samples[0][
            "draft_resident_moe_layer_contribution_h2d_bytes"
        ],
        draft_ffn_block_calls=samples[0]["draft_ffn_block_calls"],
        draft_ffn_block_experts=samples[0]["draft_ffn_block_experts"],
        draft_proposal_calls=samples[0]["draft_proposal_calls"],
        draft_candidate_tokens=samples[0]["draft_candidate_tokens"],
        draft_replayed_context_tokens=samples[0][
            "draft_replayed_context_tokens"
        ],
        draft_generation_nanoseconds=int(statistics.median(
            item["draft_generation_nanoseconds"] for item in samples
        )),
        draft_reader_read_calls=samples[0]["draft_reader_read_calls"],
        draft_reader_completed_bytes=samples[0][
            "draft_reader_completed_bytes"
        ],
        draft_routing_decisions=samples[0]["draft_routing_decisions"],
        draft_routing_selected_experts=samples[0][
            "draft_routing_selected_experts"
        ],
        draft_selected_length_1=samples[0]["draft_selected_length_1"],
        draft_selected_length_2=samples[0]["draft_selected_length_2"],
        draft_selected_length_4=samples[0]["draft_selected_length_4"],
        draft_scheduler_growths=samples[0]["draft_scheduler_growths"],
        draft_scheduler_backoffs=samples[0]["draft_scheduler_backoffs"],
        draft_context_prefill_tokens=samples[0][
            "draft_context_prefill_tokens"
        ],
        draft_incremental_forward_calls=samples[0][
            "draft_incremental_forward_calls"
        ],
        draft_rollback_events=samples[0]["draft_rollback_events"],
        draft_mla_positions_cropped=samples[0][
            "draft_mla_positions_cropped"
        ],
        draft_kda_checkpoint_bytes=samples[0][
            "draft_kda_checkpoint_bytes"
        ],
        speculative_verification_blocks=samples[0][
            "speculative_verification_blocks"
        ],
        speculative_proposed_draft_tokens=samples[0][
            "speculative_proposed_draft_tokens"
        ],
        speculative_accepted_draft_tokens=samples[0][
            "speculative_accepted_draft_tokens"
        ],
        speculative_committed_tokens=samples[0][
            "speculative_committed_tokens"
        ],
        speculative_max_proposal_tokens=samples[0][
            "speculative_max_proposal_tokens"
        ],
        target_decode_forward_calls=samples[0][
            "target_decode_forward_calls"
        ],
        target_block_forward_calls=samples[0]["target_block_forward_calls"],
        target_positions_evaluated=samples[0]["target_positions_evaluated"],
        target_positions_discarded=samples[0]["target_positions_discarded"],
        expert_major_unique_experts_sum=samples[0][
            "expert_major_unique_experts_sum"
        ],
        expert_major_unique_experts_max=samples[0][
            "expert_major_unique_experts_max"
        ],
        expert_major_assignments=samples[0]["expert_major_assignments"],
        expert_major_reused_assignments=samples[0][
            "expert_major_reused_assignments"
        ],
        expert_major_payload_loads=samples[0]["expert_major_payload_loads"],
        speculative_acceptance_rate=samples[0][
            "speculative_acceptance_rate"
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--backend", choices=("cpu", "cuda-dense", "cuda-custom"), default="cpu"
    )
    parser.add_argument(
        "--dense-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument(
        "--cuda-allocation", choices=("per-operation", "reused"),
        default="per-operation",
    )
    parser.add_argument(
        "--cuda-weights", choices=("transient", "resident"), default="transient"
    )
    parser.add_argument(
        "--cuda-batching", choices=("scalar", "grouped"), default="scalar"
    )
    parser.add_argument(
        "--cuda-boundary", choices=("operation", "ffn-block"), default="operation"
    )
    parser.add_argument(
        "--cuda-transfer", choices=("synchronous", "prefetch"),
        default="synchronous",
    )
    parser.add_argument(
        "--cuda-moe-fusion", choices=("none", "routed-accumulate"),
        default="none",
    )
    parser.add_argument(
        "--cuda-weight-validation", choices=("per-call", "admission"),
        default="per-call",
    )
    parser.add_argument("--cuda-resident-bytes", type=int, default=0)
    parser.add_argument("--cuda-pinned-bytes", type=int, default=0)
    parser.add_argument(
        "--l1-expert-cache",
        choices=("disabled", "static", "lru", "lfu", "least-stale"),
        default="disabled",
    )
    parser.add_argument("--l1-expert-cache-bytes", type=int, default=0)
    parser.add_argument("--l2-io", choices=("pread", "io-uring"), default="pread")
    parser.add_argument("--l2-cache", choices=("buffered", "direct"), default="buffered")
    parser.add_argument("--l2-queue-depth", type=int, default=8)
    parser.add_argument(
        "--l2-expert-schedule", choices=("blocking", "deadline"),
        default="blocking",
    )
    parser.add_argument(
        "--routing-mode", choices=("natural", "fixed", "adaptive"),
        default="natural",
    )
    parser.add_argument("--routing-fixed-k", type=int, default=0)
    parser.add_argument("--routing-mass-target", type=float, default=0.9)
    parser.add_argument("--routing-min-boundary-gap", type=float, default=0.0)
    parser.add_argument("--routing-agent-failures", type=int, default=0)
    parser.add_argument("--routing-critical", action="store_true")
    parser.add_argument(
        "--speculative-mode",
        choices=(
            "none", "scripted-reference", "aurora-replay",
            "aurora-persistent",
        ),
        default="none",
    )
    parser.add_argument(
        "--speculative-verification",
        choices=("token-major", "expert-major"),
        default="token-major",
    )
    parser.add_argument("--speculative-block-size", type=int, default=0)
    parser.add_argument("--speculative-script", default="")
    parser.add_argument("--aurora-draft-k", type=int, default=0)
    parser.add_argument(
        "--aurora-block-policy", choices=("fixed", "adaptive"),
        default="fixed",
    )
    parser.add_argument(
        "--aurora-draft-backend", choices=("cpu", "cuda-custom"),
        default="cpu",
    )
    parser.add_argument("--aurora-draft-resident-bytes", type=int, default=0)
    parser.add_argument("--aurora-draft-batching", default="grouped")
    parser.add_argument("--aurora-draft-boundary", default="ffn-block")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_once(
        args.artifact,
        args.runner,
        args.warmup,
        args.iterations,
        backend=args.backend,
        dense_precision=args.dense_precision,
        cuda_allocation=args.cuda_allocation,
        cuda_weights=args.cuda_weights,
        cuda_batching=args.cuda_batching,
        cuda_boundary=args.cuda_boundary,
        cuda_transfer=args.cuda_transfer,
        cuda_moe_fusion=args.cuda_moe_fusion,
        cuda_weight_validation=args.cuda_weight_validation,
        cuda_resident_bytes=args.cuda_resident_bytes,
        cuda_pinned_bytes=args.cuda_pinned_bytes,
        l1_expert_cache=args.l1_expert_cache,
        l1_expert_cache_bytes=args.l1_expert_cache_bytes,
        l2_io=args.l2_io,
        l2_cache=args.l2_cache,
        l2_queue_depth=args.l2_queue_depth,
        l2_expert_schedule=args.l2_expert_schedule,
        routing_mode=args.routing_mode,
        routing_fixed_k=args.routing_fixed_k,
        routing_mass_target=args.routing_mass_target,
        routing_min_boundary_gap=args.routing_min_boundary_gap,
        routing_agent_failures=args.routing_agent_failures,
        routing_critical=args.routing_critical,
        speculative_mode=args.speculative_mode,
        speculative_verification=args.speculative_verification,
        speculative_block_size=args.speculative_block_size,
        speculative_script=args.speculative_script,
        aurora_draft_k=args.aurora_draft_k,
        aurora_block_policy=args.aurora_block_policy,
        aurora_draft_backend=args.aurora_draft_backend,
        aurora_draft_resident_bytes=args.aurora_draft_resident_bytes,
        aurora_draft_batching=args.aurora_draft_batching,
        aurora_draft_boundary=args.aurora_draft_boundary,
    )
    write_results(result, args.json, args.csv)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
