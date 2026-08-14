# TurboQuant reference KV cache의 압축, 복원, attention 계약을 검증합니다.

import pytest
import torch

from glm5x_ref.turboquant import (
    TurboQuantConfig,
    TurboQuantKVCache,
    estimate_kv_storage_bytes,
    quantize_vector,
)


def test_lossless_mode_round_trips_after_hadamard_rotation() -> None:
    values = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 7.0

    packed = quantize_vector(values, bits=16, seed=19)

    restored = packed.dequantize()

    assert torch.equal(restored, values)
    assert packed.storage_bytes == values.numel() * values.element_size()


def test_quantized_vector_reports_smaller_storage_and_shape() -> None:
    values = torch.randn(5, 16)

    packed = quantize_vector(values, bits=4, seed=3)
    restored = packed.dequantize()

    assert restored.shape == values.shape
    assert packed.storage_bytes < values.numel() * values.element_size()
    assert packed.effective_bits == pytest.approx(4.0)


def test_fractional_bits_use_a_two_width_channel_schedule() -> None:
    values = torch.randn(2, 16)

    packed = quantize_vector(values, bits=3.5, seed=5)

    assert packed.effective_bits == pytest.approx(3.5)
    assert packed.bit_schedule == (3, 4)
    assert packed.dequantize().shape == values.shape


def test_cache_incremental_attention_matches_materialized_path() -> None:
    torch.manual_seed(23)
    keys = torch.randn(6, 8)
    values = torch.randn(6, 8)
    query = torch.randn(8)
    cache = TurboQuantKVCache(TurboQuantConfig(bits=4, seed=11))

    cache.append(keys[:3], values[:3])
    cache.append(keys[3:], values[3:])

    expected_keys, expected_values = cache.materialize()
    expected_weights = torch.softmax(query @ expected_keys.T / 8**0.5, dim=-1)
    expected = expected_weights @ expected_values

    actual = cache.attend(query)

    assert torch.allclose(actual, expected, atol=1e-6)
    assert cache.token_count == 6
    assert cache.storage_bytes < keys.numel() * 2 * keys.element_size()


def test_config_rejects_unsupported_bit_width() -> None:
    with pytest.raises(ValueError, match="UNSUPPORTED_TURBOQUANT_BITS"):
        TurboQuantConfig(bits=5)


def test_asymmetric_key_value_bits_and_million_token_estimate() -> None:
    config = TurboQuantConfig(key_bits=6, value_bits=4, seed=29)

    compressed = estimate_kv_storage_bytes(
        tokens=1_000_000,
        key_width=256,
        value_width=256,
        config=config,
    )
    fp16 = 1_000_000 * (256 + 256) * 2

    assert compressed < fp16
    assert compressed / fp16 == pytest.approx(0.3125, abs=0.02)
