# L1 expert cache ablation의 축 격리와 exact acceptance 계약을 검증합니다.
import os
import json
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from tools import ablate_l1_expert_cache
from tools.ablate_l1_expert_cache import l1_cache_matrix, run_l1_cache_ablation
from tools.benchmark_synthetic import BenchmarkRecord


def test_l1_cache_matrix_crosses_only_cache_and_transfer() -> None:
    assert l1_cache_matrix(65536, 1024) == (
        {
            "name": "disabled-synchronous",
            "l1_expert_cache": "disabled",
            "l1_expert_cache_bytes": 0,
            "cuda_transfer": "synchronous",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "static-synchronous",
            "l1_expert_cache": "static",
            "l1_expert_cache_bytes": 65536,
            "cuda_transfer": "synchronous",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "disabled-prefetch",
            "l1_expert_cache": "disabled",
            "l1_expert_cache_bytes": 0,
            "cuda_transfer": "prefetch",
            "cuda_pinned_bytes": 1024,
        },
        {
            "name": "static-prefetch",
            "l1_expert_cache": "static",
            "l1_expert_cache_bytes": 65536,
            "cuda_transfer": "prefetch",
            "cuda_pinned_bytes": 1024,
        },
    )


def _measured_record(name: str) -> BenchmarkRecord:
    root = Path(__file__).parents[2]
    payload = json.loads(
        (root / "results" / "b0006-l1-cache-fp32" / f"{name}.json").read_text(
            encoding="utf-8"
        )
    )
    return BenchmarkRecord(**payload)


def test_b0006_compact_manifest_matches_all_raw_records() -> None:
    root = Path(__file__).parents[2]
    compact = json.loads(
        (root / "results" / "b0006-l1-cache.json").read_text(encoding="utf-8")
    )
    field_map = {
        "decode_tok_s": "decode_tokens_per_second",
        "prefill_tok_s": "prefill_tokens_per_second",
        "ttft_ms": "ttft_ms",
        "peak_rss_bytes": "peak_rss_bytes",
        "logical_read_bytes_per_token": "file_read_bytes_per_token",
        "reader_calls": "reader_read_calls",
        "reader_bytes": "reader_completed_bytes",
        "l1_hits": "l1_expert_cache_hits",
        "l1_misses": "l1_expert_cache_misses",
        "l1_bypasses": "l1_expert_cache_bypasses",
        "l1_resident_bytes": "l1_expert_cache_resident_bytes",
        "h2d_bytes": "host_to_device_bytes",
        "d2h_bytes": "device_to_host_bytes",
        "peak_vram_bytes": "peak_vram_bytes",
        "kernel_nanoseconds": "kernel_nanoseconds",
        "async_prefetch_calls": "async_prefetch_calls",
        "transfer_stall_nanoseconds": "transfer_stall_nanoseconds",
        "max_absolute_error": "max_absolute_error",
    }
    assert len(compact["records"]) == 8
    for record in compact["records"]:
        raw = json.loads(
            (
                root
                / "results"
                / f"b0006-l1-cache-{record['precision']}"
                / f"{record['name']}.json"
            ).read_text(encoding="utf-8")
        )
        for compact_field, raw_field in field_map.items():
            assert record[compact_field] == raw[raw_field]
        assert raw["token_ids"] == compact["generated_token_ids"]
        assert raw["routed_experts"] == compact["routed_experts"]


def _fake_benchmark(changes: dict[str, tuple[str, object]] | None = None):
    changes = changes or {}

    def run(*_args, **kwargs) -> BenchmarkRecord:
        name = f"{kwargs['l1_expert_cache']}-{kwargs['cuda_transfer']}"
        record = _measured_record(name)
        if name in changes:
            field, value = changes[name]
            record = replace(record, **{field: value})
        return record

    return run


def test_l1_cache_ablation_validates_fake_records(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ablate_l1_expert_cache, "benchmark_once", _fake_benchmark()
    )
    summary = run_l1_cache_ablation(
        Path("fixture.k3x"),
        Path("runner"),
        dense_precision="fp32",
        l1_expert_cache_bytes=65536,
        cuda_pinned_bytes=1024 * 1024,
        warmup=3,
        iterations=20,
        output_dir=tmp_path,
    )
    assert len(summary["records"]) == 4
    assert all(record["parity_status"] == "exact" for record in summary["records"])


@pytest.mark.parametrize(
    ("name", "field", "value", "message"),
    [
        ("static-synchronous", "token_ids", (0,), "token parity"),
        ("static-synchronous", "routed_experts", (0,), "routing parity"),
        ("disabled-synchronous", "l1_expert_cache_hits", 1, "disabled L1"),
        ("static-synchronous", "l1_expert_cache_hits", 0, "static L1"),
        ("static-synchronous", "l1_expert_cache_bypasses", 1, "static L1"),
        ("static-synchronous", "l1_expert_cache_resident_bytes", 65537, "static L1"),
        ("static-synchronous", "reader_read_calls", 428, "logical reads"),
        ("static-synchronous", "host_to_device_bytes", 5074561, "GPU execution"),
    ],
)
def test_l1_cache_ablation_rejects_invalid_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(
        ablate_l1_expert_cache,
        "benchmark_once",
        _fake_benchmark({name: (field, value)}),
    )
    with pytest.raises(RuntimeError, match=message):
        run_l1_cache_ablation(
            Path("fixture.k3x"),
            Path("runner"),
            dense_precision="fp32",
            l1_expert_cache_bytes=65536,
            cuda_pinned_bytes=1024 * 1024,
            warmup=3,
            iterations=20,
            output_dir=tmp_path,
        )


def test_l1_cache_ablation_runs_exact_cuda_matrix(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("L1 CUDA ablation is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    summary = run_l1_cache_ablation(
        artifact,
        cpp_binary("k3x_run"),
        dense_precision="fp32",
        l1_expert_cache_bytes=65536,
        cuda_pinned_bytes=1024 * 1024,
        warmup=0,
        iterations=1,
        output_dir=tmp_path / "ablation",
    )
    assert len(summary["records"]) == 4
    assert len(summary["deltas"]) == 2
    for record in summary["records"]:
        assert record["parity_status"] == "exact"
        assert (tmp_path / "ablation" / f"{record['name']}.json").is_file()
        assert (tmp_path / "ablation" / f"{record['name']}.csv").is_file()
