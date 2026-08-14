# GLM5X activation artifact writer의 고정 헤더와 CRC를 검증합니다.
import struct

import google_crc32c
import torch
import torch.nn.functional as F

from glm5x_converter.activation import write_bf16_activation
from glm5x_ref.activation_export import _mixed_bf16_mlp
from glm5x_ref.layer10_moe import GLM5XExpertWeights


def test_write_bf16_activation_round_trip_header(tmp_path) -> None:
    destination = tmp_path / "hidden.bin"
    tensor = torch.tensor([[1.0, -2.0, 3.5, 0.25]], dtype=torch.float32)

    write_bf16_activation(destination, tensor)

    payload = destination.read_bytes()
    header = struct.unpack("<8sIIIIHHQI", payload[:40])
    assert header[:7] == (b"GLM5XACT", 1, 40, 1, 4, 3, 0)
    assert header[7] == 8
    assert header[8] == google_crc32c.value(payload[40:])
    assert len(payload) == 48
    assert torch.frombuffer(bytearray(payload[40:]), dtype=torch.int16).view(torch.bfloat16).tolist() == [
        1.0, -2.0, 3.5, 0.25
    ]


def test_mixed_bf16_mlp_matches_explicit_reference() -> None:
    hidden = torch.tensor([[0.5, -1.25]], dtype=torch.bfloat16)
    expert = GLM5XExpertWeights(
        gate_proj=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.bfloat16),
        up_proj=torch.tensor([[1.0, 1.0], [2.0, -1.0]], dtype=torch.bfloat16),
        down_proj=torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.bfloat16),
    )
    actual = _mixed_bf16_mlp(hidden, expert)
    work = hidden.to(torch.bfloat16).to(torch.float32)
    gate = F.linear(work, expert.gate_proj.to(torch.float32))
    up = F.linear(work, expert.up_proj.to(torch.float32))
    activation = (F.silu(gate) * up).to(torch.bfloat16)
    expected = F.linear(activation.to(torch.float32), expert.down_proj.to(torch.float32))
    torch.testing.assert_close(actual, expected)
