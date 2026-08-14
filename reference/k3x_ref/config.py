# 합성 Kimi K3 그래프의 크기와 구조 계약을 정의합니다.
from dataclasses import dataclass, replace as dataclass_replace


@dataclass(frozen=True)
class SyntheticK3Config:
    vocab_size: int = 64
    hidden_size: int = 64
    layer_kinds: tuple[str, ...] = ("kda", "kda", "kda", "mla")
    dense_layers: tuple[int, ...] = (0,)
    kda_heads: int = 4
    kda_head_dim: int = 16
    short_conv_kernel_size: int = 4
    mla_heads: int = 4
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 8
    qk_rope_head_dim: int = 8
    v_head_dim: int = 8
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    num_experts: int = 8
    top_k: int = 2
    num_shared_experts: int = 1
    routed_latent_size: int = 32
    expert_intermediate_size: int = 32
    dense_intermediate_size: int = 96
    attn_res_block_size: int = 2
    rms_norm_eps: float = 1.0e-5
    kda_gate_lower_bound: float = -5.0
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    routed_scaling_factor: float = 1.0
    mxfp4_group_size: int = 32

    @classmethod
    def default(cls) -> "SyntheticK3Config":
        cfg = cls()
        cfg.validate()
        return cfg

    def replace(self, **changes: object) -> "SyntheticK3Config":
        return dataclass_replace(self, **changes)

    def validate(self) -> None:
        if self.layer_kinds != ("kda", "kda", "kda", "mla"):
            raise ValueError("layer_kinds must be KDA,KDA,KDA,MLA")
        if self.kda_heads * self.kda_head_dim != self.hidden_size:
            raise ValueError("kda_heads * kda_head_dim must equal hidden_size")
        if self.expert_intermediate_size % self.mxfp4_group_size:
            raise ValueError("expert_intermediate_size must align to MXFP4 group size")
        if not self.mla_use_nope:
            raise ValueError("mla_use_nope must be true for Kimi K3")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
