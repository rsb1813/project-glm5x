# GLM5X Project Charter

GLM5X is a dedicated out-of-core inference engine for GLM-5.x on one consumer PC. GLM-5.2 is the first executable target; GLM-5.3 must be a checkpoint-level replacement when its official weights become available.

## Goals

- Prioritize correctness, then measured end-to-end decode and prefill throughput.
- Minimize NVMe, RAM, and PCIe traffic without silently changing natural routing.
- Preserve coding and agentic quality through reference and quality gates.
- Stream conversion in bounded shard/layer units without full-model RAM or VRAM residency.
- Reuse the proven K3X storage/cache interfaces only where they are model-neutral.

## Target machine

- AMD Ryzen 7 9800X3D.
- NVIDIA RTX 5080 with 16 GB VRAM.
- 96 GB DDR5-4200 system RAM.
- Solidigm P44 Pro 2 TB NVMe.
- Native Linux is the preferred execution environment.

## Non-goals for bootstrap

- Do not download or bundle the full GLM checkpoint.
- Do not provision paid cloud resources.
- Do not claim a TPS target as measured before the benchmark is run.
- Do not keep Kimi K3 official weights in this repository.

## Quality contract

Natural GLM routing and a strict reference path remain available. Adaptive Top-K, proxy, pruning, cost-aware verification, and other lossy modes are opt-in and must report quality divergence. Every optimization must have a disable/reference mode.

## Milestones

1. Descriptor and reference graph.
2. Model-neutral GLM5X streaming format and converter.
3. Exact CPU runtime and profiler.
4. Basic CUDA DSA/MLA and Top-8 MoE backend.
5. Three-tier asynchronous storage and expert cache.
6. MTP/AURORA/DSpark expert-major verification.
7. Calibration, mixed quantization, quality modes, and ablation suite.
8. GLM-5.3 checkpoint swap and full-model validation.

