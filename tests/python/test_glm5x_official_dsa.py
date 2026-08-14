# GLM5X 공식 DSA indexer 수식의 shape·RoPE·causal Top-K parity를 검증합니다.

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from glm5x_ref.official_dsa import GLM5XOfficialDSAIndexer


def _rope(position_ids: torch.Tensor, dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inverse = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    frequencies = position_ids.to(torch.float32)[..., None] * inverse
    frequencies = torch.cat((frequencies, frequencies), dim=-1)
    return frequencies.cos(), frequencies.sin()


def _oracle(
    hidden: torch.Tensor,
    q_resid: torch.Tensor,
    wq_b: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    heads, head_dim = weights_proj.shape[0], wk.shape[0]
    rope_dim = 2
    q = (q_resid @ wq_b.T).reshape(1, hidden.shape[1], heads, head_dim)
    k = (hidden @ wk.T).unsqueeze(2)
    k = F.layer_norm(k, (head_dim,), norm_weight, norm_bias, 1e-6)
    q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
    k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
    q_even, q_odd = q_rot[..., 0::2], q_rot[..., 1::2]
    k_even, k_odd = k_rot[..., 0::2], k_rot[..., 1::2]
    angle_cos = cos[..., : rope_dim // 2].unsqueeze(2)
    angle_sin = sin[..., : rope_dim // 2].unsqueeze(2)
    q_rot = torch.cat((q_even * angle_cos - q_odd * angle_sin,
                       q_odd * angle_cos + q_even * angle_sin), dim=-1)
    k_rot = torch.cat((k_even * angle_cos - k_odd * angle_sin,
                       k_odd * angle_cos + k_even * angle_sin), dim=-1)
    q = torch.cat((q_rot, q_pass), dim=-1)
    k = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) / head_dim**0.5
    scores = F.relu(scores)
    weights = (hidden @ weights_proj.T).float() / heads**0.5
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)
    key_positions = torch.arange(hidden.shape[1])
    causal = key_positions[None, None, :] > position_ids[:, :, None]
    index_scores = index_scores.masked_fill(causal, float("-inf"))
    return torch.topk(index_scores, k=min(topk, hidden.shape[1]), dim=-1).indices.to(torch.int32)


def test_official_indexer_matches_independent_reference_formula() -> None:
    torch.manual_seed(17)
    hidden = torch.randn(1, 4, 3, dtype=torch.float32)
    q_resid = torch.randn(1, 4, 3, dtype=torch.float32)
    wq_b = torch.randn(8, 3, dtype=torch.float32)
    wk = torch.randn(4, 3, dtype=torch.float32)
    norm_weight = torch.randn(4, dtype=torch.float32)
    norm_bias = torch.randn(4, dtype=torch.float32)
    weights_proj = torch.randn(2, 3, dtype=torch.float32)
    position_ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    cos, sin = _rope(position_ids, dim=2, theta=10_000.0)
    indexer = GLM5XOfficialDSAIndexer(
        wq_b=wq_b,
        wk=wk,
        k_norm_weight=norm_weight,
        k_norm_bias=norm_bias,
        weights_proj=weights_proj,
        qk_rope_head_dim=2,
        index_topk=3,
        indexer_rope_interleave=True,
    )

    actual = indexer.select_topk(hidden, q_resid, (cos, sin), position_ids)
    expected = _oracle(
        hidden,
        q_resid,
        wq_b,
        wk,
        norm_weight,
        norm_bias,
        weights_proj,
        cos,
        sin,
        position_ids,
        3,
    )

    assert actual.dtype == torch.int32
    assert actual.shape == (1, 4, 3)
    assert torch.equal(actual, expected)


def test_official_indexer_rejects_inconsistent_projection_shapes() -> None:
    common = {
        "wk": torch.zeros((4, 3)),
        "k_norm_weight": torch.ones(4),
        "k_norm_bias": torch.zeros(4),
        "weights_proj": torch.zeros((2, 3)),
    }
    try:
        GLM5XOfficialDSAIndexer(wq_b=torch.zeros((7, 3)), **common)
    except ValueError as error:
        assert str(error) == "DSA_WQ_B_SHAPE_MISMATCH"
    else:
        raise AssertionError("invalid wq_b shape was accepted")


def test_official_indexer_loads_only_indexer_tensors_from_a_shard(tmp_path) -> None:
    shard = tmp_path / "indexer.safetensors"
    prefix = "model.layers.0.self_attn.indexer"
    save_file(
        {
            f"{prefix}.wq_b.weight": torch.zeros((8, 3)),
            f"{prefix}.wk.weight": torch.zeros((4, 3)),
            f"{prefix}.k_norm.weight": torch.ones(4),
            f"{prefix}.k_norm.bias": torch.zeros(4),
            f"{prefix}.weights_proj.weight": torch.zeros((2, 3)),
            "model.layers.0.self_attn.q_a_proj.weight": torch.zeros((3, 3)),
        },
        str(shard),
    )

    indexer = GLM5XOfficialDSAIndexer.from_safetensors(
        shard,
        layer_id=0,
        qk_rope_head_dim=2,
        index_topk=3,
        indexer_rope_interleave=True,
    )

    assert indexer.index_n_heads == 2
    assert indexer.index_head_dim == 4
    assert indexer.q_lora_rank == 3
