# GLM5X activation artifact writer의 고정 헤더와 CRC를 검증합니다.
import struct

import google_crc32c
import torch

from glm5x_converter.activation import write_bf16_activation


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
    assert torch.frombuffer(payload[40:], dtype=torch.int16).view(torch.bfloat16).tolist() == [
        1.0, -2.0, 3.5, 0.25
    ]
