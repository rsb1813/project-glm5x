# GLM5X INT4 packed expert sidecar의 원자적 저장과 재로딩을 검증합니다.
from __future__ import annotations

import pytest
import torch

from glm5x_ref.int4 import GLM5XInt4Weight, quantize_int4_weight
from glm5x_ref.layer10_moe import GLM5XExpertWeights
from glm5x_ref.nvfp4 import GLM5XNVFP4Weight, quantize_nvfp4_weight
from glm5x_ref.packed_cache import GLM5XPackedExpertCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip(tmp_path) -> None:
    torch.manual_seed(23)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
        up_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
        down_proj=quantize_int4_weight(torch.randn(256, 256, dtype=torch.bfloat16)),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-0000000000000000000000000000000000000000"
    cache.put((4, 7), digest, expert)
    loaded = cache.get((4, 7), digest, device="cuda")
    assert loaded is not None
    assert isinstance(loaded.gate_proj, GLM5XInt4Weight)
    assert loaded.gate_proj.shape == expert.gate_proj.shape
    torch.testing.assert_close(loaded.gate_proj.packed, expert.gate_proj.packed)
    torch.testing.assert_close(
        loaded.gate_proj.scale_and_zero, expert.gate_proj.scale_and_zero
    )
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_fp8(tmp_path) -> None:
    torch.manual_seed(29)
    weights = tuple(
        torch.randn(256, 256, dtype=torch.float32).to(torch.float8_e4m3fn)
        for _ in range(3)
    )
    scales = tuple(torch.rand(256, 1, dtype=torch.float32) + 0.25 for _ in range(3))
    expert = GLM5XExpertWeights(
        gate_proj=weights[0],
        up_proj=weights[1],
        down_proj=weights[2],
        gate_scale=scales[0],
        up_scale=scales[1],
        down_scale=scales[2],
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-fp8-000000000000000000000000000000000000"
    cache.put((4, 8), digest, expert, precision="fp8")
    loaded = cache.get((4, 8), digest, device="cuda", precision="fp8")
    assert loaded is not None
    assert loaded.is_fp8
    torch.testing.assert_close(loaded.gate_proj, expert.gate_proj.cuda())
    torch.testing.assert_close(loaded.gate_scale, expert.gate_scale.cuda())
    assert (tmp_path / "layer-0004-expert-0008.pf8").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_mxfp4(tmp_path) -> None:
    torch.manual_seed(37)
    expert = GLM5XExpertWeights(
        gate_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
        up_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
        down_proj=torch.randn(64, 64, dtype=torch.bfloat16, device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-mxfp4-000000000000000000000000000000000"
    cache.put((4, 9), digest, expert, precision="mxfp4")
    loaded = cache.get((4, 9), digest, device="cuda", precision="mxfp4")
    assert loaded is not None
    assert (tmp_path / "layer-0004-expert-0009.pm4").exists()
    assert loaded.gate_proj.dtype == torch.bfloat16
    assert loaded.gate_proj.shape == expert.gate_proj.shape
    assert torch.isfinite(loaded.gate_proj).all()
    assert float(
        (loaded.gate_proj.float() - expert.gate_proj.float()).norm()
        / expert.gate_proj.float().norm()
    ) < 0.25


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_nvfp4(tmp_path) -> None:
    torch.manual_seed(47)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(torch.randn(128, 256, device="cuda"), device="cuda"),
        up_proj=quantize_nvfp4_weight(torch.randn(128, 256, device="cuda"), device="cuda"),
        down_proj=quantize_nvfp4_weight(torch.randn(256, 128, device="cuda"), device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-nvfp4-000000000000000000000000000000000"
    cache.put((4, 10), digest, expert, precision="nvfp4")
    loaded = cache.get((4, 10), digest, device="cuda", precision="nvfp4")
    assert loaded is not None
    assert isinstance(loaded.gate_proj, GLM5XNVFP4Weight)
    assert (tmp_path / "layer-0004-expert-0010.pn4").exists()
    torch.testing.assert_close(loaded.gate_proj.packed, expert.gate_proj.packed)
    torch.testing.assert_close(loaded.gate_proj.scales, expert.gate_proj.scales)
    torch.testing.assert_close(loaded.gate_proj.global_scale, expert.gate_proj.global_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_round_trip_nvfp4_gate_up(tmp_path) -> None:
    torch.manual_seed(53)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(torch.randn(128, 256, device="cuda"), device="cuda"),
        up_proj=quantize_nvfp4_weight(torch.randn(128, 256, device="cuda"), device="cuda"),
        down_proj=torch.randn(256, 128, dtype=torch.bfloat16, device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digest = "source-digest-nvfp4-gate-up-000000000000000000000000000"
    cache.put((4, 11), digest, expert, precision="nvfp4_gate_up")
    loaded = cache.get((4, 11), digest, device="cuda", precision="nvfp4_gate_up")
    assert loaded is not None
    assert isinstance(loaded.gate_proj, GLM5XNVFP4Weight)
    assert isinstance(loaded.down_proj, torch.Tensor)
    assert (tmp_path / "layer-0004-expert-0011.pgu").exists()
    torch.testing.assert_close(loaded.gate_proj.packed, expert.gate_proj.packed)
    torch.testing.assert_close(loaded.up_proj.scales, expert.up_proj.scales)
    torch.testing.assert_close(loaded.down_proj, expert.down_proj)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_get_many_reads_independent_sidecars(tmp_path) -> None:
    torch.manual_seed(59)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(torch.randn(64, 128, device="cuda"), device="cuda"),
        up_proj=quantize_nvfp4_weight(torch.randn(64, 128, device="cuda"), device="cuda"),
        down_proj=torch.randn(128, 64, dtype=torch.bfloat16, device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path)
    digests = {
        (4, 12): "digest-12-000000000000",
        (4, 13): "digest-13-000000000000",
    }
    for key, digest in digests.items():
        cache.put(key, digest, expert, precision="nvfp4_gate_up")
    loaded = cache.get_many(
        digests, device="cuda", precision="nvfp4_gate_up", workers=2
    )
    assert set(loaded) == set(digests)
    assert cache.stats.hits == 2
    assert cache.stats.misses == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_reuses_verified_host_payload(tmp_path) -> None:
    torch.manual_seed(61)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(torch.randn(64, 128, device="cuda"), device="cuda"),
        up_proj=quantize_nvfp4_weight(torch.randn(64, 128, device="cuda"), device="cuda"),
        down_proj=torch.randn(128, 64, dtype=torch.bfloat16, device="cuda"),
    )
    cache = GLM5XPackedExpertCache(tmp_path, host_cache_capacity_bytes=1 << 20)
    digest = "digest-host-cache-000000000000"
    cache.put((4, 14), digest, expert, precision="nvfp4_gate_up")
    first = cache.get((4, 14), digest, device="cuda", precision="nvfp4_gate_up")
    assert first is not None
    (tmp_path / "layer-0004-expert-0014.pgu").unlink()
    second = cache.get((4, 14), digest, device="cuda", precision="nvfp4_gate_up")
    assert second is not None
    torch.testing.assert_close(second.gate_proj.packed, first.gate_proj.packed)
    assert cache.stats.host_hits == 1
    assert cache.stats.host_resident_bytes > 0


def test_packed_expert_cache_rejects_invalid_pinned_capacity(tmp_path) -> None:
    with pytest.raises(ValueError, match="PINNED_CAPACITY"):
        GLM5XPackedExpertCache(tmp_path, pinned_staging_capacity_bytes=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_non_blocking_pinned_nvfp4_round_trip(tmp_path) -> None:
    torch.manual_seed(67)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(
            torch.randn(64, 128, device="cuda"), device="cuda"
        ),
        up_proj=quantize_nvfp4_weight(
            torch.randn(64, 128, device="cuda"), device="cuda"
        ),
        down_proj=quantize_nvfp4_weight(
            torch.randn(128, 64, device="cuda"), device="cuda"
        ),
    )
    cache = GLM5XPackedExpertCache(
        tmp_path, pinned_staging_capacity_bytes=1 << 20
    )
    digest = "digest-pinned-nvfp4-000000000000000000000000"
    cache.put((4, 15), digest, expert, precision="nvfp4")
    loaded = cache.get(
        (4, 15),
        digest,
        device="cuda",
        precision="nvfp4",
        non_blocking=True,
    )
    assert loaded is not None
    torch.cuda.synchronize()
    torch.testing.assert_close(loaded.gate_proj.packed, expert.gate_proj.packed)
    torch.testing.assert_close(loaded.up_proj.scales, expert.up_proj.scales)
    torch.testing.assert_close(loaded.down_proj.global_scale, expert.down_proj.global_scale)
    assert cache.stats.pinned_staging_bytes > 0
    assert cache.stats.pinned_staging_hits == 0
    second = cache.get(
        (4, 15),
        digest,
        device="cuda",
        precision="nvfp4",
        non_blocking=True,
    )
    assert second is not None
    torch.cuda.synchronize()
    assert cache.stats.pinned_staging_hits == 1


def test_packed_expert_cache_rejects_non_boolean_telemetry_flag(tmp_path) -> None:
    with pytest.raises(ValueError, match="TELEMETRY"):
        GLM5XPackedExpertCache(tmp_path, telemetry_enabled=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_telemetry_separates_file_decode_and_h2d(tmp_path) -> None:
    torch.manual_seed(71)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(
            torch.randn(32, 64, device="cuda"), device="cuda"
        ),
        up_proj=quantize_nvfp4_weight(
            torch.randn(32, 64, device="cuda"), device="cuda"
        ),
        down_proj=quantize_nvfp4_weight(
            torch.randn(64, 32, device="cuda"), device="cuda"
        ),
    )
    cache = GLM5XPackedExpertCache(tmp_path, telemetry_enabled=True)
    digest = "digest-telemetry-000000000000000000000000"
    cache.put((4, 16), digest, expert, precision="nvfp4")
    sidecar = tmp_path / "layer-0004-expert-0016.pn4"

    loaded = cache.get((4, 16), digest, device="cuda", precision="nvfp4")
    assert loaded is not None
    torch.cuda.synchronize()

    stats = cache.stats
    assert stats.sidecar_read_calls == 1
    assert stats.sidecar_read_bytes == sidecar.stat().st_size
    assert stats.decoded_payload_bytes > 0
    assert stats.h2d_bytes == stats.decoded_payload_bytes
    assert stats.h2d_submission_nanoseconds > 0
    assert stats.h2d_event_nanoseconds > 0

    loaded_again = cache.get((4, 16), digest, device="cuda", precision="nvfp4")
    assert loaded_again is not None
    torch.cuda.synchronize()
    repeated = cache.stats
    assert repeated.sidecar_read_calls == 2
    assert repeated.sidecar_read_bytes == sidecar.stat().st_size * 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_packed_expert_cache_telemetry_does_not_count_host_hit_as_file_read(tmp_path) -> None:
    torch.manual_seed(73)
    expert = GLM5XExpertWeights(
        gate_proj=quantize_nvfp4_weight(
            torch.randn(32, 64, device="cuda"), device="cuda"
        ),
        up_proj=quantize_nvfp4_weight(
            torch.randn(32, 64, device="cuda"), device="cuda"
        ),
        down_proj=quantize_nvfp4_weight(
            torch.randn(64, 32, device="cuda"), device="cuda"
        ),
    )
    cache = GLM5XPackedExpertCache(
        tmp_path, host_cache_capacity_bytes=1 << 20, telemetry_enabled=True
    )
    digest = "digest-host-telemetry-0000000000000000000000"
    cache.put((4, 17), digest, expert, precision="nvfp4")
    sidecar = tmp_path / "layer-0004-expert-0017.pn4"
    assert cache.get((4, 17), digest, device="cuda", precision="nvfp4") is not None
    sidecar_bytes = sidecar.stat().st_size
    sidecar.unlink()
    assert cache.get((4, 17), digest, device="cuda", precision="nvfp4") is not None
    torch.cuda.synchronize()

    stats = cache.stats
    assert stats.sidecar_read_calls == 1
    assert stats.sidecar_read_bytes == sidecar_bytes
    assert stats.decoded_payload_bytes > 0
    assert stats.h2d_bytes > 0
