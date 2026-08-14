# K3 Attention Residual의 depth score와 raw source 혼합을 구현합니다.
from __future__ import annotations

import torch

from k3x_ref.ops import rms_norm


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_sources: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if block_sources.shape[:-2] != prefix_sum.shape[:-1]:
        raise ValueError("block_sources leading shape must match prefix_sum")
    values = torch.cat((block_sources, prefix_sum.unsqueeze(-2)), dim=-2)
    keys = rms_norm(values.to(torch.float32), norm_weight, eps).to(torch.float32)
    scores = torch.matmul(keys, proj_weight.to(torch.float32))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities.unsqueeze(-2), values.to(torch.float32)).squeeze(-2)
