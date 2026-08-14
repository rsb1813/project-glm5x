# Attention Residual이 normalized score로 raw source를 혼합하는지 검증합니다.
import torch

from k3x_ref.attn_res import apply_attn_res


def test_attention_residual_scores_normalized_but_mixes_raw_sources() -> None:
    prefix = torch.tensor([[1.0, 1.0]])
    sources = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
    got = apply_attn_res(
        prefix,
        sources,
        norm_weight=torch.ones(2),
        proj_weight=torch.tensor([1.0, 0.0]),
        eps=0.0,
    )
    assert torch.allclose(got, torch.tensor([[1.9227442, 0.85785025]]), atol=1e-6)

