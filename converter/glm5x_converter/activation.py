# GLM5X reference hidden-state를 C++ 런타임으로 전달하는 BF16 artifact writer입니다.
from __future__ import annotations

import os
import struct
from pathlib import Path
from tempfile import NamedTemporaryFile

import google_crc32c
import torch


_HEADER = struct.Struct("<8sIIIIHHQI")
_MAGIC = b"GLM5XACT"
_VERSION = 1
_HEADER_BYTES = _HEADER.size
_BF16_DTYPE = 3


def write_bf16_activation(path: str | Path, tensor: torch.Tensor) -> None:
    """Write one contiguous [tokens, hidden] BF16 activation batch atomically."""
    work = torch.as_tensor(tensor).detach().to(device="cpu", dtype=torch.bfloat16)
    if work.ndim != 2:
        raise ValueError("activation tensor must be rank-2 [tokens, hidden]")
    work = work.contiguous()
    token_count, hidden_size = (int(value) for value in work.shape)
    if token_count == 0 or hidden_size == 0:
        raise ValueError("activation tensor dimensions must be non-zero")
    payload = work.view(torch.int16).numpy().tobytes(order="C")
    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        _HEADER_BYTES,
        token_count,
        hidden_size,
        _BF16_DTYPE,
        0,
        len(payload),
        google_crc32c.value(payload),
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.",
        suffix=".tmp", delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(header)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, destination)

