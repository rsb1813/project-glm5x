# K3 기본 연산과 MXFP4 해석을 독립 기대값으로 검증합니다.
import pytest
import torch

from k3x_ref.mxfp4 import decode_mxfp4, encode_mxfp4, mxfp4_matmul
from k3x_ref.ops import rms_norm, situ_glu


def test_rms_norm_uses_mean_square_without_centering() -> None:
    got = rms_norm(torch.tensor([[3.0, 4.0]]), torch.tensor([2.0, 0.5]), 0.0)
    assert torch.allclose(got, torch.tensor([[1.6970563, 0.5656854]]), atol=1e-6)


def test_situ_glu_multiplies_bounded_gate_and_up_branches() -> None:
    got = situ_glu(
        torch.tensor([[-1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        beta=1.0,
        linear_beta=1.0,
    )
    expected = torch.tensor([[-0.20381131, 0.84854317]])
    assert torch.allclose(got, expected, atol=1e-6)


def test_mxfp4_decodes_low_nibble_first_with_one_e8m0_scale() -> None:
    packed = bytes([0x10, 0x32] + [0x00] * 14)
    scales = bytes([127])
    got = decode_mxfp4(packed, scales, rows=1, cols=32)
    assert torch.equal(got[0, :4], torch.tensor([0.0, 0.5, 1.0, 1.5]))


def test_mxfp4_applies_one_scale_per_logical_group() -> None:
    packed = bytes([0x11] * 32)
    got = decode_mxfp4(packed, bytes([127, 128]), rows=1, cols=64)
    assert torch.equal(got[0, :32], torch.full((32,), 0.5))
    assert torch.equal(got[0, 32:], torch.full((32,), 1.0))


def test_mxfp4_rejects_reserved_e8m0_scale() -> None:
    with pytest.raises(ValueError, match="0xff"):
        decode_mxfp4(bytes(16), bytes([255]), rows=1, cols=32)


def test_mxfp4_matmul_matches_literal_dense_result() -> None:
    packed = bytes([0x21] + [0x00] * 15)
    got = mxfp4_matmul(
        torch.tensor([[2.0, 3.0] + [0.0] * 30]),
        packed,
        bytes([127]),
        rows=1,
        cols=32,
    )
    assert torch.equal(got, torch.tensor([[4.0]]))


def test_mxfp4_encoder_preserves_native_nibble_order_and_scale() -> None:
    source = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] + [0.0] * 24]
    )

    packed, scales = encode_mxfp4(source)

    assert packed[:2] == bytes([0x10, 0x32])
    assert scales == bytes([127])
    assert torch.equal(decode_mxfp4(packed, scales, 1, 32), source)


def test_mxfp4_encoder_selects_power_of_two_group_scale() -> None:
    source = torch.full((1, 32), 12.0)

    packed, scales = encode_mxfp4(source)

    assert scales == bytes([128])
    assert torch.equal(decode_mxfp4(packed, scales, 1, 32), source)


def test_mxfp4_encoder_rejects_invalid_shape_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="group_size"):
        encode_mxfp4(torch.zeros(1, 30))
    with pytest.raises(ValueError, match="finite"):
        encode_mxfp4(torch.tensor([[float("nan")] + [0.0] * 31]))


def test_mxfp4_encoder_mse_scale_mode_never_worsens_group_error() -> None:
    source = torch.randn((2, 32), generator=torch.Generator().manual_seed(31))

    max_packed, max_scales = encode_mxfp4(source, scale_mode="max_abs")
    mse_packed, mse_scales = encode_mxfp4(source, scale_mode="mse")
    max_error = (decode_mxfp4(max_packed, max_scales, 2, 32) - source).square().mean()
    mse_error = (decode_mxfp4(mse_packed, mse_scales, 2, 32) - source).square().mean()

    assert mse_error <= max_error + 1e-7


def test_mxfp4_encoder_keeps_scale_bytes_across_chunks() -> None:
    source = torch.cat((torch.full((1, 32), 1.0), torch.full((1, 32), 2.0)))

    packed, scales = encode_mxfp4(source, chunk_groups=1)

    assert scales == bytes([125, 126])
    assert torch.equal(decode_mxfp4(packed, scales, 2, 32), source)
