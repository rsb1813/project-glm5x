# Synthetic benchmark 결과 schema와 measured/projected 구분을 검증합니다.
import csv
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint
from tools.ablate_cuda_residency import cuda_residency_matrix, run_ablation
from tools.ablate_cuda_ffn import ffn_boundary_matrix, run_ffn_ablation
from tools.benchmark_synthetic import BenchmarkRecord, benchmark_once, write_results


def _record() -> BenchmarkRecord:
    return BenchmarkRecord(
        scope="synthetic-milestone-zero",
        evidence="measured",
        platform="test-machine",
        iterations=3,
        prompt_tokens=4,
        generated_tokens=6,
        prefill_tokens_per_second=100.0,
        decode_tokens_per_second=50.0,
        ttft_ms=12.5,
        peak_rss_bytes=123456,
        file_read_bytes_per_token=789.0,
        backend="cpu",
        device="CPU",
        dense_precision="fp32",
        cuda_allocation="per-operation",
        cuda_weights="transient",
        cuda_batching="scalar",
        cuda_boundary="operation",
        cuda_transfer="synchronous",
        cuda_resident_bytes=0,
        cuda_pinned_bytes=0,
        kernel_nanoseconds=0,
        host_to_device_bytes=0,
        weight_h2d_bytes=0,
        activation_h2d_bytes=0,
        device_to_host_bytes=0,
        peak_vram_bytes=None,
        device_allocation_count=0,
        device_free_count=0,
        stream_synchronization_count=0,
        weight_cache_hits=0,
        weight_cache_misses=0,
        weight_cache_bypasses=0,
        resident_weight_bytes=0,
        peak_resident_weight_bytes=0,
        scratch_bytes=0,
        peak_scratch_bytes=0,
        grouped_projection_calls=0,
        grouped_projection_members=0,
        ffn_block_calls=0,
        ffn_block_experts=0,
        pinned_host_bytes=0,
        peak_pinned_host_bytes=0,
        async_prefetch_calls=0,
        async_prefetch_bytes=0,
        async_prefetch_ready_before_use=0,
        async_prefetch_late_at_use=0,
        transfer_stream_wait_count=0,
        pinned_staging_nanoseconds=0,
        transfer_device_nanoseconds=0,
        transfer_stall_nanoseconds=0,
        async_engine_count=0,
        device_overlap=False,
        max_absolute_error=None,
        max_relative_error=None,
        kda_state_bytes=1024,
        mla_kv_bytes=2048,
        per_layer_nanoseconds=(1, 2, 3, 4),
        token_ids=(43, 32, 28, 49, 9, 28),
        routed_experts=(),
    )


def test_benchmark_json_and_csv_preserve_schema(tmp_path: Path) -> None:
    json_path, csv_path = tmp_path / "result.json", tmp_path / "result.csv"
    write_results(_record(), json_path, csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "synthetic-milestone-zero"
    assert payload["evidence"] == "measured"
    assert payload["l1_expert_cache_mode"] == "disabled"
    assert payload["routing_mode"] == "natural"
    assert payload["routing_average_top_k"] == 0.0
    assert payload["cold_rescue_count"] == 0
    assert payload["speculative_mode"] == "none"
    assert payload["speculative_verification"] == "token-major"
    assert payload["speculative_block_size"] == 0
    assert payload["aurora_draft_k"] == 0
    assert payload["aurora_block_policy"] == "none"
    assert payload["aurora_draft_backend"] == "none"
    assert payload["draft_device"] == "CPU"
    assert payload["draft_cuda_allocation"] == "per-operation"
    assert payload["draft_cuda_weights"] == "transient"
    assert payload["draft_cuda_batching"] == "scalar"
    assert payload["draft_cuda_boundary"] == "operation"
    assert payload["draft_cuda_transfer"] == "synchronous"
    assert payload["draft_cuda_moe_fusion"] == "none"
    assert payload["cuda_weight_validation"] == "per-call"
    assert payload["draft_cuda_weight_validation"] == "per-call"
    for field in (
        "immutable_validation_scans",
        "immutable_validation_hits",
        "immutable_validation_bytes",
        "immutable_validation_nanoseconds",
    ):
        assert payload[field] == 0
        assert payload[f"draft_{field}"] == 0
    assert payload["draft_kernel_nanoseconds"] == 0
    assert payload["draft_host_to_device_bytes"] == 0
    assert payload["draft_weight_h2d_bytes"] == 0
    assert payload["draft_activation_h2d_bytes"] == 0
    assert payload["draft_device_to_host_bytes"] == 0
    assert payload["draft_peak_vram_bytes"] == 0
    assert payload["draft_device_allocation_count"] == 0
    assert payload["draft_stream_synchronization_count"] == 0
    assert payload["draft_weight_cache_hits"] == 0
    assert payload["draft_weight_cache_misses"] == 0
    assert payload["draft_weight_cache_bypasses"] == 0
    assert payload["draft_cuda_resident_bytes"] == 0
    assert payload["draft_resident_weight_bytes"] == 0
    assert payload["draft_peak_resident_weight_bytes"] == 0
    for field in (
        "resident_grid_calls",
        "resident_grid_experts",
        "resident_grid_tokens",
        "resident_grid_expert_tokens",
        "resident_grid_kernel_launches",
        "resident_grid_fallbacks",
        "resident_grid_descriptor_h2d_bytes",
    ):
        assert payload[field] == 0
        assert payload[f"draft_{field}"] == 0
    for field in (
        "resident_moe_layer_calls",
        "resident_moe_layer_experts",
        "resident_moe_layer_kernel_launches",
        "resident_moe_layer_fallbacks",
        "resident_moe_layer_contribution_h2d_bytes",
    ):
        assert payload[field] == 0
        assert payload[f"draft_{field}"] == 0
    assert payload["draft_proposal_calls"] == 0
    assert payload["draft_candidate_tokens"] == 0
    assert payload["draft_replayed_context_tokens"] == 0
    assert payload["draft_generation_nanoseconds"] == 0
    assert payload["draft_reader_read_calls"] == 0
    assert payload["draft_reader_completed_bytes"] == 0
    assert payload["draft_routing_decisions"] == 0
    assert payload["draft_routing_selected_experts"] == 0
    assert payload["draft_selected_length_1"] == 0
    assert payload["draft_selected_length_2"] == 0
    assert payload["draft_selected_length_4"] == 0
    assert payload["draft_scheduler_growths"] == 0
    assert payload["draft_scheduler_backoffs"] == 0
    assert payload["draft_context_prefill_tokens"] == 0
    assert payload["draft_incremental_forward_calls"] == 0
    assert payload["draft_rollback_events"] == 0
    assert payload["draft_mla_positions_cropped"] == 0
    assert payload["draft_kda_checkpoint_bytes"] == 0
    assert payload["speculative_verification_blocks"] == 0
    assert payload["target_decode_forward_calls"] == 0
    assert payload["target_block_forward_calls"] == 0
    assert payload["target_positions_evaluated"] == 0
    assert payload["target_positions_discarded"] == 0
    assert payload["expert_major_unique_experts_sum"] == 0
    assert payload["expert_major_unique_experts_max"] == 0
    assert payload["expert_major_assignments"] == 0
    assert payload["expert_major_reused_assignments"] == 0
    assert payload["expert_major_payload_loads"] == 0
    assert payload["evaluated_routed_experts"] == []
    assert payload["evaluated_routed_k"] == []
    assert payload["speculative_acceptance_rate"] is None
    assert payload["l1_expert_cache_bytes"] == 0
    assert payload["l1_expert_cache_hits"] == 0
    assert payload["l1_expert_cache_evictions"] == 0
    assert payload["l1_expert_cache_collision_misses"] == 0
    assert payload["l2_expert_schedule"] == "blocking"
    assert payload["expert_load_submissions"] == 0
    assert payload["expert_load_completions"] == 0
    assert payload["reader_read_calls"] == 0
    assert payload["reader_requested_bytes"] == 0
    assert payload["reader_completed_bytes"] == 0
    assert payload["l2_io_engine"] == "pread"
    assert payload["l2_cache_mode"] == "buffered"
    assert payload["l2_queue_depth"] == 8
    assert payload["l2_direct_memory_alignment"] == 0
    assert payload["l2_direct_offset_alignment"] == 0
    assert payload["reader_batch_submissions"] == 0
    assert payload["reader_storage_submitted_bytes"] == 0
    assert payload["reader_storage_completed_bytes"] == 0
    assert payload["reader_completions"] == 0
    assert payload["reader_short_reads"] == 0
    assert payload["reader_failures"] == 0
    assert payload["reader_storage_nanoseconds"] == 0
    assert payload["process_io_available"] is False
    assert payload["process_rchar_bytes"] is None
    assert payload["process_read_bytes"] is None
    assert isinstance(payload["peak_rss_bytes"], int)
    assert payload["backend"] == "cpu"
    assert payload["device"] == "CPU"
    assert payload["dense_precision"] == "fp32"
    assert payload["cuda_allocation"] == "per-operation"
    assert payload["cuda_weights"] == "transient"
    assert payload["cuda_batching"] == "scalar"
    assert payload["cuda_boundary"] == "operation"
    assert payload["cuda_transfer"] == "synchronous"
    assert payload["cuda_moe_fusion"] == "none"
    assert payload["fused_moe_calls"] == 0
    assert payload["fused_moe_experts"] == 0
    assert payload["batched_expert_ffn_calls"] == 0
    assert payload["batched_expert_ffn_tokens"] == 0
    assert payload["cuda_resident_bytes"] == 0
    assert payload["cuda_pinned_bytes"] == 0
    assert payload["device_allocation_count"] == 0
    assert payload["weight_h2d_bytes"] == 0
    assert payload["activation_h2d_bytes"] == 0
    assert payload["peak_vram_bytes"] is None
    assert payload["async_engine_count"] == 0
    assert payload["device_overlap"] is False
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert b"\r\n" not in csv_path.read_bytes()
    assert row["decode_tokens_per_second"] == "50.0"
    assert row["host_to_device_bytes"] == "0"
    assert row["cuda_allocation"] == "per-operation"
    assert row["weight_cache_bypasses"] == "0"
    assert row["grouped_projection_members"] == "0"
    assert row["ffn_block_calls"] == "0"
    assert row["ffn_block_experts"] == "0"
    assert row["cuda_transfer"] == "synchronous"
    assert row["cuda_moe_fusion"] == "none"
    assert row["fused_moe_calls"] == "0"
    assert row["fused_moe_experts"] == "0"
    assert row["batched_expert_ffn_calls"] == "0"
    assert row["batched_expert_ffn_tokens"] == "0"
    assert row["resident_grid_calls"] == "0"
    assert row["draft_resident_grid_calls"] == "0"
    assert row["resident_moe_layer_calls"] == "0"
    assert row["draft_resident_moe_layer_calls"] == "0"
    assert row["cuda_pinned_bytes"] == "0"
    assert row["transfer_device_nanoseconds"] == "0"
    assert row["device_overlap"] == "False"
    assert row["peak_vram_bytes"] == ""
    assert row["per_layer_nanoseconds"] == "1;2;3;4"
    assert row["token_ids"] == "43;32;28;49;9;28"
    assert row["routed_experts"] == ""
    assert row["routed_k"] == ""
    assert row["speculative_mode"] == "none"
    assert row["speculative_verification"] == "token-major"
    assert row["speculative_verification_blocks"] == "0"
    assert row["aurora_draft_k"] == "0"
    assert row["aurora_block_policy"] == "none"
    assert row["aurora_draft_backend"] == "none"
    assert row["draft_device"] == "CPU"
    assert row["draft_kernel_nanoseconds"] == "0"
    assert row["draft_weight_h2d_bytes"] == "0"
    assert row["draft_peak_vram_bytes"] == "0"
    assert row["draft_cuda_resident_bytes"] == "0"
    assert row["draft_resident_weight_bytes"] == "0"
    assert row["draft_peak_resident_weight_bytes"] == "0"
    assert row["draft_reader_completed_bytes"] == "0"
    assert row["draft_context_prefill_tokens"] == "0"
    assert row["draft_incremental_forward_calls"] == "0"
    assert row["draft_rollback_events"] == "0"
    assert row["draft_mla_positions_cropped"] == "0"
    assert row["draft_kda_checkpoint_bytes"] == "0"
    assert row["target_block_forward_calls"] == "0"
    assert row["target_positions_evaluated"] == "0"
    assert row["expert_major_payload_loads"] == "0"
    assert row["evaluated_routed_experts"] == ""
    assert row["evaluated_routed_k"] == ""
    assert row["speculative_acceptance_rate"] == ""


def test_ffn_boundary_matrix_and_runner_preserve_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matrix = ffn_boundary_matrix()
    assert tuple(item["name"] for item in matrix) == (
        "operation-scalar",
        "operation-grouped",
        "ffn-block-scalar",
        "ffn-block-grouped",
    )

    def fake_benchmark_once(*args: object, **kwargs: object) -> BenchmarkRecord:
        boundary = str(kwargs["cuda_boundary"])
        batching = str(kwargs["cuda_batching"])
        block = boundary == "ffn-block"
        grouped = batching == "grouped"
        return BenchmarkRecord(
            **{
                **_record().__dict__,
                "backend": "cuda-custom",
                "cuda_allocation": "reused",
                "cuda_weights": "resident",
                "cuda_batching": batching,
                "cuda_boundary": boundary,
                "cuda_resident_bytes": 4096,
                "device_to_host_bytes": 35 if block and grouped else 40 if block else 80 if grouped else 100,
                "stream_synchronization_count": 7 if block and grouped else 8 if block else 15 if grouped else 20,
                "ffn_block_calls": 4 if block else 0,
                "ffn_block_experts": 6 if block else 0,
            }
        )

    monkeypatch.setattr("tools.ablate_cuda_ffn.benchmark_once", fake_benchmark_once)
    summary = run_ffn_ablation(
        tmp_path / "model.k3x",
        tmp_path / "k3x_run",
        dense_precision="fp32",
        cuda_resident_bytes=4096,
        warmup=0,
        iterations=1,
        output_dir=tmp_path / "ffn-ablation",
    )
    assert all(item["parity_status"] == "exact" for item in summary["records"])
    assert summary["records"][2]["device_to_host_bytes"] < summary["records"][0]["device_to_host_bytes"]
    assert summary["records"][3]["stream_synchronization_count"] < summary["records"][1]["stream_synchronization_count"]
    assert (tmp_path / "ffn-ablation" / "summary.json").is_file()


def test_schema_accepts_explicit_milestone_one_scope() -> None:
    record = BenchmarkRecord(
        **{**_record().__dict__, "scope": "synthetic-milestone-one"}
    )
    assert record.scope == "synthetic-milestone-one"


def test_cuda_residency_matrix_changes_one_axis_at_a_time() -> None:
    matrix = cuda_residency_matrix()
    assert tuple(item["name"] for item in matrix) == (
        "reference",
        "reuse",
        "residency",
        "grouped",
    )
    assert tuple(
        (
            item["cuda_allocation"],
            item["cuda_weights"],
            item["cuda_batching"],
        )
        for item in matrix
    ) == (
        ("per-operation", "transient", "scalar"),
        ("reused", "transient", "scalar"),
        ("reused", "resident", "scalar"),
        ("reused", "resident", "grouped"),
    )


def test_cuda_residency_ablation_runs_sequential_stages_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, str, str, int]] = []

    def fake_benchmark_once(*args: object, **kwargs: object) -> BenchmarkRecord:
        observed.append(
            (
                str(kwargs["cuda_allocation"]),
                str(kwargs["cuda_weights"]),
                str(kwargs["cuda_batching"]),
                int(kwargs["cuda_resident_bytes"]),
            )
        )
        return BenchmarkRecord(
            **{
                **_record().__dict__,
                "backend": str(kwargs["backend"]),
                "cuda_allocation": str(kwargs["cuda_allocation"]),
                "cuda_weights": str(kwargs["cuda_weights"]),
                "cuda_batching": str(kwargs["cuda_batching"]),
                "cuda_resident_bytes": int(kwargs["cuda_resident_bytes"]),
                "device_allocation_count": len(observed),
            }
        )

    monkeypatch.setattr(
        "tools.ablate_cuda_residency.benchmark_once", fake_benchmark_once
    )
    output_dir = tmp_path / "ablation"
    summary = run_ablation(
        tmp_path / "model.k3x",
        tmp_path / "k3x_run",
        backend="cuda-custom",
        dense_precision="fp32",
        cuda_resident_bytes=4096,
        warmup=0,
        iterations=1,
        output_dir=output_dir,
    )
    assert observed == [
        ("per-operation", "transient", "scalar", 0),
        ("reused", "transient", "scalar", 0),
        ("reused", "resident", "scalar", 4096),
        ("reused", "resident", "grouped", 4096),
    ]
    assert [item["name"] for item in summary["records"]] == [
        "reference",
        "reuse",
        "residency",
        "grouped",
    ]
    assert summary["deltas"][0]["device_allocation_count"] == 1
    assert (output_dir / "summary.json").is_file()
    for name in ("reference", "reuse", "residency", "grouped"):
        assert (output_dir / f"{name}.json").is_file()
        assert (output_dir / f"{name}.csv").is_file()


def test_cuda_residency_ablation_requires_positive_resident_capacity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_ablation(
            tmp_path / "model.k3x",
            tmp_path / "k3x_run",
            backend="cuda-custom",
            dense_precision="fp32",
            cuda_resident_bytes=0,
            warmup=0,
            iterations=1,
            output_dir=tmp_path / "ablation",
        )


def test_schema_rejects_projected_values_as_measured() -> None:
    with pytest.raises(ValueError, match="synthetic-milestone"):
        BenchmarkRecord(**{**_record().__dict__, "scope": "projected-full-model"})


def test_benchmark_once_collects_cpu_backend_profile(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cpu",
        dense_precision="fp32",
    )
    assert record.scope == "synthetic-milestone-one"
    assert record.backend == "cpu"
    assert record.device == "CPU"
    assert record.dense_precision == "fp32"
    assert record.cuda_allocation == "per-operation"
    assert record.cuda_weights == "transient"
    assert record.cuda_batching == "scalar"
    assert record.cuda_boundary == "operation"
    assert record.cuda_transfer == "synchronous"
    assert record.cuda_resident_bytes == 0
    assert record.cuda_pinned_bytes == 0
    assert record.l1_expert_cache_mode == "disabled"
    assert record.l1_expert_cache_bytes == 0
    assert record.l1_expert_cache_hits == 0
    assert record.l1_expert_cache_misses == 0
    assert record.l1_expert_cache_bypasses == 0
    assert record.l1_expert_cache_evictions == 0
    assert record.l1_expert_cache_collision_misses == 0
    assert record.speculative_mode == "none"
    assert record.speculative_block_size == 0
    assert record.speculative_verification_blocks == 0
    assert record.target_decode_forward_calls == 5
    assert record.speculative_acceptance_rate is None


def test_benchmark_once_collects_scripted_speculative_telemetry(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cpu",
        dense_precision="fp32",
        speculative_mode="scripted-reference",
        speculative_block_size=2,
        speculative_script="43:32,28;49:9",
    )
    assert record.token_ids == (43, 32, 28, 49, 9, 28)
    assert record.speculative_mode == "scripted-reference"
    assert record.speculative_block_size == 2
    assert record.aurora_draft_k == 0
    assert record.aurora_block_policy == "none"
    assert record.draft_proposal_calls == 0
    assert record.draft_candidate_tokens == 0
    assert record.draft_reader_completed_bytes == 0
    assert record.speculative_verification_blocks == 2
    assert record.speculative_proposed_draft_tokens == 3
    assert record.speculative_accepted_draft_tokens == 3
    assert record.speculative_committed_tokens == 5
    assert record.speculative_max_proposal_tokens == 2
    assert record.target_decode_forward_calls == 5
    assert record.speculative_acceptance_rate == 1.0
    assert record.l1_expert_cache_resident_bytes == 0
    assert record.reader_read_calls > 0
    assert record.reader_requested_bytes >= record.reader_completed_bytes > 0
    assert record.l2_io_engine == "pread"
    assert record.l2_cache_mode == "buffered"
    assert record.l2_queue_depth == 8
    assert record.l2_direct_memory_alignment == 0
    assert record.l2_direct_offset_alignment == 0
    assert 0 < record.reader_batch_submissions < record.reader_read_calls
    assert record.reader_storage_submitted_bytes == record.reader_requested_bytes
    assert record.reader_storage_completed_bytes == record.reader_completed_bytes
    assert record.reader_completions == record.reader_read_calls
    assert record.reader_short_reads == 0
    assert record.reader_failures == 0
    assert record.reader_storage_nanoseconds > 0
    assert record.routed_experts
    assert record.routed_k
    assert record.routing_mode == "natural"
    assert record.routing_natural_top_k > 0
    assert record.routing_average_top_k == record.routing_natural_top_k
    assert record.routing_quality_escalated_decisions == 0
    assert record.cold_rescue_count == 0
    if record.process_io_available:
        assert record.process_rchar_bytes is not None
        assert record.process_rchar_bytes >= record.reader_completed_bytes
        assert record.process_read_bytes is not None
        assert record.process_read_bytes >= 0
    else:
        assert record.process_rchar_bytes is None
        assert record.process_read_bytes is None


def test_benchmark_once_collects_aurora_replay_telemetry(
    tmp_path: Path,
) -> None:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        speculative_mode="aurora-replay",
        speculative_block_size=2,
        aurora_draft_k=4,
        aurora_block_policy="fixed",
    )
    assert record.speculative_mode == "aurora-replay"
    assert record.aurora_draft_k == 4
    assert record.aurora_block_policy == "fixed"
    assert record.draft_proposal_calls > 0
    assert record.draft_candidate_tokens > 0
    assert record.draft_replayed_context_tokens > 0
    assert record.draft_generation_nanoseconds > 0
    assert record.draft_reader_read_calls > 0
    assert record.draft_reader_completed_bytes > 0
    assert record.draft_routing_decisions > 0
    assert record.kernel_nanoseconds == 0
    assert record.host_to_device_bytes == 0
    assert record.weight_h2d_bytes == 0
    assert record.activation_h2d_bytes == 0
    assert record.device_to_host_bytes == 0
    assert record.peak_vram_bytes == 0
    assert record.device_allocation_count == 0
    assert record.device_free_count == 0
    assert record.stream_synchronization_count == 0
    assert record.weight_cache_hits == 0
    assert record.weight_cache_misses == 0
    assert record.weight_cache_bypasses == 0
    assert record.resident_weight_bytes == 0
    assert record.peak_resident_weight_bytes == 0
    assert record.scratch_bytes == 0
    assert record.peak_scratch_bytes == 0
    assert record.grouped_projection_calls == 0
    assert record.grouped_projection_members == 0
    assert record.ffn_block_calls == 0
    assert record.ffn_block_experts == 0
    assert record.pinned_host_bytes == 0
    assert record.peak_pinned_host_bytes == 0
    assert record.async_prefetch_calls == 0
    assert record.async_prefetch_bytes == 0
    assert record.async_prefetch_ready_before_use == 0
    assert record.async_prefetch_late_at_use == 0
    assert record.transfer_stream_wait_count == 0
    assert record.pinned_staging_nanoseconds == 0
    assert record.transfer_device_nanoseconds == 0
    assert record.transfer_stall_nanoseconds == 0
    assert record.async_engine_count == 0
    assert record.device_overlap is False
    assert record.max_absolute_error == 0.0
    assert record.max_relative_error == 0.0


def test_benchmark_once_collects_resident_cuda_aurora_draft_telemetry(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA draft residency is exercised only against build-cuda")
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-resident-draft-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "resident-draft-top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        speculative_mode="aurora-persistent",
        speculative_block_size=2,
        aurora_draft_k=4,
        aurora_block_policy="fixed",
        aurora_draft_backend="cuda-custom",
        aurora_draft_resident_bytes=8388608,
    )
    assert record.draft_cuda_weights == "resident"
    assert record.draft_cuda_resident_bytes == 8388608
    assert 0 < record.draft_resident_weight_bytes <= 8388608
    assert (
        record.draft_resident_weight_bytes
        <= record.draft_peak_resident_weight_bytes
        <= 8388608
    )
    assert record.draft_weight_cache_hits > 0
    assert record.draft_weight_cache_misses > 0
    assert record.draft_weight_cache_bypasses == 0
    assert record.resident_weight_bytes == 0
    assert record.peak_resident_weight_bytes == 0


def test_benchmark_once_reports_static_l1_and_reader_accounting(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cpu",
        l1_expert_cache="static",
        l1_expert_cache_bytes=65536,
    )
    assert record.l1_expert_cache_mode == "static"
    assert record.l1_expert_cache_bytes == 65536
    assert record.l1_expert_cache_hits > 0
    assert record.l1_expert_cache_misses > 0
    assert record.l1_expert_cache_bypasses == 0
    assert record.l1_expert_cache_evictions == 0
    assert record.l1_expert_cache_collision_misses == 0
    assert 0 < record.l1_expert_cache_resident_bytes <= 65536
    assert record.peak_l1_expert_cache_resident_bytes == (
        record.l1_expert_cache_resident_bytes
    )
    assert record.reader_read_calls > 0
    assert record.reader_requested_bytes >= record.reader_completed_bytes > 0


@pytest.mark.parametrize(
    ("backend", "dense_precision", "tolerance"),
    [
        ("cuda-dense", "fp32", 1e-4),
        ("cuda-custom", "fp32", 1e-4),
        ("cuda-custom", "bf16", 2e-2),
    ],
)
def test_benchmark_once_measures_cuda_error_against_cpu(
    synthetic_source: Path,
    tmp_path: Path,
    backend: str,
    dense_precision: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA benchmark contract is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend=backend,
        dense_precision=dense_precision,
    )
    assert record.backend == backend
    assert record.device != "CPU"
    assert record.kernel_nanoseconds > 0
    assert record.host_to_device_bytes > 0
    assert record.weight_h2d_bytes > 0
    assert record.activation_h2d_bytes > 0
    assert (
        record.weight_h2d_bytes + record.activation_h2d_bytes
        == record.host_to_device_bytes
    )
    assert record.device_to_host_bytes > 0
    assert record.peak_vram_bytes > 0
    assert record.device_allocation_count > 0
    assert record.device_allocation_count == record.device_free_count
    assert record.stream_synchronization_count > 0
    assert record.scratch_bytes == 0
    assert record.max_absolute_error is not None
    assert record.max_absolute_error <= tolerance
    assert record.max_relative_error is not None
    assert record.max_relative_error >= 0.0


def test_benchmark_once_reports_reused_cuda_scratch(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA benchmark contract is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cuda-dense",
        dense_precision="fp32",
        cuda_allocation="reused",
    )
    assert record.cuda_allocation == "reused"
    assert record.device_allocation_count > record.device_free_count
    assert record.scratch_bytes > 0
    assert record.peak_scratch_bytes >= record.scratch_bytes
    assert record.max_absolute_error is not None
    assert record.max_absolute_error <= 1.0e-4


def test_benchmark_once_reports_exact_async_transfer_accounting(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA transfer accounting is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cuda-custom",
        dense_precision="fp32",
        cuda_allocation="reused",
        cuda_weights="transient",
        cuda_batching="scalar",
        cuda_boundary="ffn-block",
        cuda_transfer="prefetch",
        cuda_pinned_bytes=1024 * 1024,
    )
    assert record.cuda_transfer == "prefetch"
    assert record.cuda_pinned_bytes == 1024 * 1024
    assert record.pinned_host_bytes == 1024 * 1024
    assert record.peak_pinned_host_bytes == 1024 * 1024
    assert record.async_prefetch_calls > 0
    assert 0 < record.async_prefetch_bytes <= record.weight_h2d_bytes
    assert (
        record.async_prefetch_ready_before_use
        + record.async_prefetch_late_at_use
        == record.async_prefetch_calls
    )
    assert record.transfer_stream_wait_count == record.async_prefetch_calls
    assert record.pinned_staging_nanoseconds > 0
    assert record.transfer_device_nanoseconds > 0
    assert record.transfer_stall_nanoseconds >= 0
    assert record.async_engine_count > 0
    assert isinstance(record.device_overlap, bool)
