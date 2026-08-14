# Research and Implementation References

K3X uses pinned source snapshots for graph decisions. Performance claims from cluster-oriented implementations are not transferred to the single-PC target.

## Model and kernels

| Source | Role | Snapshot used during Milestone 0 |
|---|---|---|
| [MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) | Official release, configuration, report, and reference code | `3cb39df` |
| [Kimi K3 technical report](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf) | Architecture and training report | Repository snapshot above |
| [Hugging Face Kimi K3 configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) | Released tensor dimensions and feature flags | Inspected 2026-08-08 |
| [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA/tree/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b) | Official KDA API and optimized kernels | `1ce47ea` |
| [MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals/tree/85e22310fe5ee860b4a023de312d791de8a5a5e6) | Attention Residual method and implementation | `85e2231` |
| [vLLM Kimi K3 source](https://github.com/vllm-project/vllm/tree/44351f81/vllm/models/kimi_k3) | Production model graph and NVIDIA/AMD execution paths | `44351f81` |
| [vLLM Kimi K3 engineering article](https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md) | Production integration, KDA decode and fused residual context | Inspected 2026-08-08 |
| [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c/tree/ff11dce858a2eb8a781224facdffd33a1fa48d25) | Independent C runtime reference | `ff11dce` |
| [PipeNetwork/kimi-k3-mlx](https://github.com/PipeNetwork/kimi-k3-mlx/tree/20a4fb101ce81380ab8af0036743d49e7256c521) | Independent MLX graph reference | `20a4fb1` |

## Cache and speculative decoding research

| Work | K3X use |
|---|---|
| [SpecMD](https://arxiv.org/abs/2602.03921) | Reproduce Least-Stale expert caching and compare it with LRU/LFU |
| [EcoSpec](https://arxiv.org/abs/2607.12696) | Study acceptance probability together with marginal expert fetch cost |
| [MoE-Spec](https://arxiv.org/abs/2602.16052) | Experimental layer-wise verification expert budgets |
| [AcceptMoE](https://arxiv.org/abs/2608.02989) | Explicitly lossy verifier expert selection experiment, never the strict default |
| [DSpark paper](https://arxiv.org/abs/2607.05147) | Proposal/target separation, prefix verification, and confidence scheduling source |
| [DeepSpec](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2) | Inspected DSpark proposal, verification, target-cache crop, and draft-state update implementation |
| [SpecMoE](https://arxiv.org/abs/2604.10152) | Duplicate expert-migration coalescing and self-assisted offloaded MoE speculation evidence |

SpecMD informed the implemented experimental Least-Stale cache. DSpark and DeepSpec informed the implemented Milestone 13 token-major reference. MoE-Spec Appendix C, SpecMoE, and the official vLLM Kimi K3 multi-token paths inform the accepted Milestone 14 design; their lossy policies and reported speedups are not transferred to K3X. EcoSpec, MoE-Spec budgeting, and AcceptMoE remain future experimental inputs.

## Reproducibility policy

- A source revision is evidence only for what was inspected at that revision.
- A paper's reported speedup or quality result remains scoped to its reported setup.
- K3X defaults change only after the relevant optimization is implemented, ablated, and measured on the target workload.
- Exact routing and reference modes remain available even when an experimental fast mode exists.
