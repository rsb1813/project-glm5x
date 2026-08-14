# CUDA synchronous/prefetch 전송 ablation의 축 격리와 실패 검증을 테스트합니다.
from dataclasses import replace
from pathlib import Path

import pytest

from tools.ablate_cuda_transfer import run_transfer_ablation, transfer_matrix
from tools.benchmark_synthetic import BenchmarkRecord


PINNED_BYTES = 1024


def _record(*, transfer: str, batching: str) -> BenchmarkRecord:
    prefetch = transfer == "prefetch"
    return BenchmarkRecord(
        scope="synthetic-milestone-one",
        evidence="measured",
        platform="test-machine",
        iterations=3,
        prompt_tokens=4,
        generated_tokens=6,
        prefill_tokens_per_second=100.0,
        decode_tokens_per_second=55.0 if prefetch else 50.0,
        ttft_ms=12.5,
        peak_rss_bytes=123456,
        file_read_bytes_per_token=789.0,
        backend="cuda-custom",
        device="RTX test",
        dense_precision="fp32",
        cuda_allocation="reused",
        cuda_weights="transient",
        cuda_batching=batching,
        cuda_boundary="ffn-block",
        cuda_transfer=transfer,
        cuda_resident_bytes=0,
        cuda_pinned_bytes=PINNED_BYTES if prefetch else 0,
        kernel_nanoseconds=100,
        host_to_device_bytes=100,
        weight_h2d_bytes=60,
        activation_h2d_bytes=40,
        device_to_host_bytes=8,
        peak_vram_bytes=4096,
        device_allocation_count=5,
        device_free_count=0,
        stream_synchronization_count=10,
        weight_cache_hits=0,
        weight_cache_misses=0,
        weight_cache_bypasses=0,
        resident_weight_bytes=0,
        peak_resident_weight_bytes=0,
        scratch_bytes=2048,
        peak_scratch_bytes=2048,
        grouped_projection_calls=1 if batching == "grouped" else 0,
        grouped_projection_members=2 if batching == "grouped" else 0,
        ffn_block_calls=5,
        ffn_block_experts=10,
        pinned_host_bytes=PINNED_BYTES if prefetch else 0,
        peak_pinned_host_bytes=PINNED_BYTES if prefetch else 0,
        async_prefetch_calls=5 if prefetch else 0,
        async_prefetch_bytes=50 if prefetch else 0,
        async_prefetch_ready_before_use=2 if prefetch else 0,
        async_prefetch_late_at_use=3 if prefetch else 0,
        transfer_stream_wait_count=5 if prefetch else 0,
        pinned_staging_nanoseconds=100 if prefetch else 0,
        transfer_device_nanoseconds=200 if prefetch else 0,
        transfer_stall_nanoseconds=30 if prefetch else 0,
        async_engine_count=2,
        device_overlap=True,
        max_absolute_error=0.0,
        max_relative_error=0.0,
        kda_state_bytes=1024,
        mla_kv_bytes=2048,
        per_layer_nanoseconds=(1, 2, 3, 4),
        token_ids=(43, 32, 28, 49, 9, 28),
        routed_experts=(1, 2, 3, 4),
    )


def test_transfer_matrix_changes_only_transfer_and_batching() -> None:
    assert tuple(
        (item["name"], item["cuda_transfer"], item["cuda_batching"],
         item["cuda_pinned_bytes"])
        for item in transfer_matrix(PINNED_BYTES)
    ) == (
        ("synchronous-scalar", "synchronous", "scalar", 0),
        ("prefetch-scalar", "prefetch", "scalar", PINNED_BYTES),
        ("synchronous-grouped", "synchronous", "grouped", 0),
        ("prefetch-grouped", "prefetch", "grouped", PINNED_BYTES),
    )


def test_transfer_ablation_writes_raw_records_and_measured_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, str, int]] = []

    def fake_benchmark(*args: object, **kwargs: object) -> BenchmarkRecord:
        transfer = str(kwargs["cuda_transfer"])
        batching = str(kwargs["cuda_batching"])
        observed.append((transfer, batching, int(kwargs["cuda_pinned_bytes"])))
        return _record(transfer=transfer, batching=batching)

    monkeypatch.setattr("tools.ablate_cuda_transfer.benchmark_once", fake_benchmark)
    output = tmp_path / "transfer"
    summary = run_transfer_ablation(
        tmp_path / "model.k3x",
        tmp_path / "k3x_run",
        dense_precision="fp32",
        cuda_pinned_bytes=PINNED_BYTES,
        warmup=0,
        iterations=1,
        output_dir=output,
    )
    assert observed == [
        ("synchronous", "scalar", 0),
        ("prefetch", "scalar", PINNED_BYTES),
        ("synchronous", "grouped", 0),
        ("prefetch", "grouped", PINNED_BYTES),
    ]
    assert [item["parity_status"] for item in summary["records"]] == [
        "exact", "exact", "exact", "exact"
    ]
    assert [item["decode_tokens_per_second"] for item in summary["deltas"]] == [
        5.0, 5.0
    ]
    for name in (
        "synchronous-scalar", "prefetch-scalar",
        "synchronous-grouped", "prefetch-grouped",
    ):
        assert (output / f"{name}.json").is_file()
        assert (output / f"{name}.csv").is_file()
    assert (output / "summary.json").is_file()


@pytest.mark.parametrize(
    ("case_index", "changes", "message"),
    [
        (1, {"token_ids": (99,)}, "token"),
        (1, {"routed_experts": (99,)}, "routing"),
        (0, {"async_prefetch_calls": 1}, "synchronous"),
        (1, {"async_prefetch_calls": 0}, "prefetch"),
        (1, {"stream_synchronization_count": 11}, "synchronization"),
        (1, {"host_to_device_bytes": 101}, "H2D"),
        (
            1,
            {"host_to_device_bytes": 101, "weight_h2d_bytes": 61},
            "matched H2D",
        ),
        (
            1,
            {"host_to_device_bytes": 101, "activation_h2d_bytes": 41},
            "matched H2D",
        ),
        (
            1,
            {"weight_h2d_bytes": 61, "activation_h2d_bytes": 39},
            "matched H2D",
        ),
        (1, {"stream_synchronization_count": 9}, "matched synchronization"),
        (1, {"backend": "cuda-dense"}, "identity"),
        (1, {"cuda_boundary": "operation"}, "identity"),
        (1, {"cuda_allocation": "per-operation"}, "identity"),
        (1, {"cuda_weights": "resident"}, "identity"),
    ],
)
def test_transfer_ablation_rejects_invalid_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_index: int,
    changes: dict[str, object],
    message: str,
) -> None:
    calls = 0

    def fake_benchmark(*args: object, **kwargs: object) -> BenchmarkRecord:
        nonlocal calls
        index = calls
        calls += 1
        record = _record(
            transfer=str(kwargs["cuda_transfer"]),
            batching=str(kwargs["cuda_batching"]),
        )
        return replace(record, **changes) if index == case_index else record

    monkeypatch.setattr("tools.ablate_cuda_transfer.benchmark_once", fake_benchmark)
    with pytest.raises(RuntimeError, match=message):
        run_transfer_ablation(
            tmp_path / "model.k3x",
            tmp_path / "k3x_run",
            dense_precision="fp32",
            cuda_pinned_bytes=PINNED_BYTES,
            warmup=0,
            iterations=1,
            output_dir=tmp_path / "invalid",
        )


def test_transfer_ablation_rejects_missing_raw_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.ablate_cuda_transfer.benchmark_once",
        lambda *args, **kwargs: _record(
            transfer=str(kwargs["cuda_transfer"]),
            batching=str(kwargs["cuda_batching"]),
        ),
    )
    monkeypatch.setattr("tools.ablate_cuda_transfer.write_results", lambda *args: None)
    with pytest.raises(RuntimeError, match="raw"):
        run_transfer_ablation(
            tmp_path / "model.k3x",
            tmp_path / "k3x_run",
            dense_precision="fp32",
            cuda_pinned_bytes=PINNED_BYTES,
            warmup=0,
            iterations=1,
            output_dir=tmp_path / "missing",
        )
