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
