# Native MXFP4 E2M1 payload와 E8M0 scale을 해석하는 참조 구현입니다.
from __future__ import annotations

import torch


E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_MSE_SCALE_OFFSETS = torch.tensor([-4, -3, -2, -1, 0, 1, 2], dtype=torch.int32)


def encode_mxfp4(
    values: torch.Tensor,
    group_size: int = 32,
    *,
    chunk_groups: int = 8192,
    scale_mode: str = "max_abs",
) -> tuple[bytes, bytes]:
    """Encode a floating-point matrix into the native E2M1/E8M0 layout."""
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    rows, cols = values.shape
    if rows <= 0 or cols <= 0 or group_size <= 0:
        raise ValueError("rows, cols, and group_size must be positive")
    if cols % group_size or cols % 2:
        raise ValueError("cols must align to group_size and nibble pairs")
    if chunk_groups <= 0:
        raise ValueError("chunk_groups must be positive")
    if scale_mode not in {"max_abs", "mse"}:
        raise ValueError("scale_mode must be max_abs or mse")
    if not values.is_floating_point():
        raise ValueError("values must use a floating-point dtype")

    source = values.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(source).all():
        raise ValueError("values must be finite")

    groups = source.reshape(rows * (cols // group_size), group_size)
    maximum = groups.abs().amax(dim=1)
    nonzero = maximum > 0
    raw_exponents = torch.empty_like(maximum, dtype=torch.float32)
    raw_exponents[~nonzero] = -127.0
    raw_exponents[nonzero] = torch.ceil(torch.log2(maximum[nonzero] / 6.0))
    if torch.any(raw_exponents > 126):
        raise ValueError("values exceed the E8M0 exponent range")
    exponents = raw_exponents.clamp(min=-127, max=126).to(torch.int32)
    codebook = E2M1_VALUES

    packed_chunks: list[bytes] = []
    scale_chunks: list[bytes] = []
    for start in range(0, groups.shape[0], chunk_groups):
        stop = min(start + chunk_groups, groups.shape[0])
        group_values = groups[start:stop]
        if scale_mode == "max_abs":
            selected_exponents = exponents[start:stop]
            group_scales = torch.ldexp(
                torch.ones(stop - start, dtype=torch.float32), selected_exponents
            )
            normalized = group_values / group_scales[:, None]
            codes = (
                normalized[:, :, None] - codebook[None, None, :]
            ).abs().argmin(dim=-1)
        else:
            candidate_exponents = (
                exponents[start:stop, None] + _MSE_SCALE_OFFSETS[None, :]
            ).clamp(min=-127, max=126)
            candidate_scales = torch.ldexp(
                torch.ones_like(candidate_exponents, dtype=torch.float32),
                candidate_exponents,
            )
            normalized = group_values[:, :, None] / candidate_scales[:, None, :]
            candidate_codes = (
                normalized[:, :, :, None] - codebook[None, None, None, :]
            ).abs().argmin(dim=-1)
            reconstructed = codebook[candidate_codes] * candidate_scales[:, None, :]
            errors = (reconstructed - group_values[:, :, None]).square().mean(dim=1)
            best = errors.argmin(dim=1)
            selected_exponents = candidate_exponents.gather(1, best[:, None]).squeeze(1)
            codes = candidate_codes.gather(
                2, best[:, None, None].expand(-1, group_size, 1)
            ).squeeze(2)
        scales = (selected_exponents + 127).to(torch.uint8)
        codes = codes.to(torch.uint8)
        flat_codes = codes.reshape(-1)
        packed = flat_codes[0::2] | (flat_codes[1::2] << 4)
        packed_chunks.append(packed.numpy().tobytes())
        scale_chunks.append(scales.numpy().tobytes())
    return b"".join(packed_chunks), b"".join(scale_chunks)


def decode_mxfp4(
    packed: bytes,
    scales: bytes,
    rows: int,
    cols: int,
    group_size: int = 32,
) -> torch.Tensor:
    if rows <= 0 or cols <= 0 or group_size <= 0:
        raise ValueError("rows, cols, and group_size must be positive")
    if cols % group_size or cols % 2:
        raise ValueError("cols must align to group_size and nibble pairs")

    logical_values = rows * cols
    expected_packed = logical_values // 2
    expected_scales = logical_values // group_size
    if len(packed) != expected_packed:
        raise ValueError(f"packed length must be {expected_packed}")
    if len(scales) != expected_scales:
        raise ValueError(f"scale length must be {expected_scales}")
    if 0xFF in scales:
        raise ValueError("E8M0 scale 0xff is reserved")

    packed_tensor = torch.tensor(list(packed), dtype=torch.uint8)
    nibbles = torch.stack(
        (packed_tensor.bitwise_and(0x0F), packed_tensor.bitwise_right_shift(4)),
        dim=1,
    ).reshape(-1)
    values = E2M1_VALUES[nibbles.to(torch.long)]

    scale_exponents = torch.tensor(list(scales), dtype=torch.int32) - 127
    scale_values = torch.ldexp(
        torch.ones_like(scale_exponents, dtype=torch.float32), scale_exponents
    )
    expanded_scales = scale_values.repeat_interleave(group_size)
    return (values * expanded_scales).reshape(rows, cols)


def mxfp4_matmul(
    x: torch.Tensor,
    packed: bytes,
    scales: bytes,
    rows: int,
    cols: int,
    group_size: int = 32,
) -> torch.Tensor:
    if x.shape[-1] != cols:
        raise ValueError(f"input width must be {cols}")
    weight = decode_mxfp4(packed, scales, rows, cols, group_size)
    return x.to(torch.float32) @ weight.transpose(0, 1)
