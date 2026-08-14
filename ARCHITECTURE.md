# GLM5X Architecture

## Current product boundary

GLM5X is the GLM-5.x product runtime. The migrated K3X code is treated as a storage and scheduling foundation, not as a claim that Kimi K3 execution remains supported by this repository.

### Implemented in this bootstrap

- K3X-compatible superblock, tensor/layer/expert directories, aligned extents, CRC/SHA verification, and resumable writer.
- Portable C++20 reader, cache, prefetch, routing-policy, speculative, and expert-major interfaces.
- GLM descriptor validation for DSA, Top-8, 256 routed experts, shared experts, and MTP metadata.
- GLM-5.2 tensor-manifest validation for `config.json`, safetensors `weight_map`, shard names, tensor count, and source byte total without opening a weight shard.
- A bounded RTX 5080 CUDA benchmark for GLM-5.2 expert dimensions (`hidden=6144`, `expert_intermediate=2048`, `group=32`) using the resident expert-grid backend.
- Resident expert-major batch execution now admits packed/scales through the shared `ResidentWeightTable`; repeated candidate batches can reuse exact MXFP4 weights without another weight H2D upload.
- Experimental resident BF16 expert-grid execution dequantizes exact MXFP4 values once per tensor and uses cublasLt BF16-input/FP32-output projections; native MXFP4 remains the default. A preflight budget check accounts for dense resident weights and warm BF16 keys before admission, then falls back to native without partial mixed-representation residency when the budget is insufficient.
- GLM5X converter CLI wrapper.
- CPU/reference TurboQuant-inspired KV cache with Hadamard rotation, integer and half-bit schedules, asymmetric K/V policy, incremental attention, and capacity estimation.

### In progress

- GLM model-specific extent roles and streaming conversion from the validated manifest.
- GLM-5.2 reference graph with exact DSA/MLA and MoE routing.
- Synthetic GLM-5.2 checkpoint round-trip.
- Connecting compressed KV blocks to the GLM DSA/indexer state instead of the standalone reference cache.
- Wiring GLM DSA/MTP state and exact routing around the existing expert-major batch path.

### Experimental

- MTP/AURORA draft and DSpark-compatible target verification.
- Expert-major union scheduling and cost-aware speculation.
- TurboQuant 2.5/3.5-bit KV schedules, UltraQuant-style asymmetric K/V and block-scale variants.
- Mixed weight quantization and CUDA fusion.

### Proposed

- GLM-5.3 checkpoint descriptor and calibration swap.
- Full RTX 5080 native CUDA kernels beyond the current scalar MXFP4 grid path.
- Making dequantized BF16 the default, or storing all GLM experts in BF16, is rejected until VRAM pressure and quality are measured with real shards.
- Cloud-side shard conversion through the existing SKYFORGE concept.

## Runtime data flow

The converter reads bounded source shards, validates identity, emits execution-ordered extents, and releases source memory before the next unit. Runtime residency is tiered as L0 VRAM, L1 RAM, and L2 NVMe. Cache score combines frequency, transition probability, recency, predicted use, load latency, size, residency, and speculative usefulness.

For speculative verification, GLM5X computes candidate routing first, forms a per-layer unique expert union, fetches each exact expert once, and batches candidate-token work by expert. Natural Top-8 routing remains the correctness reference.

For long context, the planned path is paged DSA state with a recent high-precision window and compressed historical KV blocks. `TurboQuantKVCache` is currently a CPU/reference implementation only. Its storage estimate is a logical bit budget plus per-row scale metadata; it is not yet a packed CUDA storage format and must not be used as a throughput claim.

## Model boundary

`GLM5XModelDescriptor` is the first model boundary. It owns model family, attention kind, layer count, hidden size, routed expert count, Top-K, shared experts, vocabulary size, MTP layer count, MoE intermediate width, DSA index Top-K/frequency/head shape, MTP sharing policy, and maximum position length. Tensor file names and source byte totals belong to `GLM5XTensorManifest` and must not be embedded in runtime kernels.

The current CUDA evidence is deliberately bounded. The shaped benchmark runs the resident MXFP4 grid and the opt-in BF16 resident grid with deterministic zero weights, so maximum absolute error is checked against a zero reference. It is a kernel/layer measurement, not a full 78-layer GLM decode result and not a quality benchmark. The BF16 path trades approximately 3.77x resident-weight bytes for the measured 2.09x grid latency improvement in the 8-expert/4-token sample.

K3-specific KDA, Attention Residual, Stable LatentMoE, 896-way Top-16 assumptions, and native Kimi MXFP4 naming are not part of the GLM5X default graph. They remain historical source context in the old K3X repository only.

## Quality modes

The strict reference path is always available. Any adaptive Top-K, proxy, pruning, or verifier-budget mode is opt-in and must report quality divergence. SHADOW and PHOENIX-style escalation are policy layers and cannot silently change the natural routing contract.

`TURBO-LONGCTX` is an experimental mode for 600k–1M context capacity. It keeps exact routing and target verification while allowing compressed historical KV. Context capacity, prefill/TTFT, and decode tok/s are recorded as separate measurements. TurboQuant does not compress the 753B model weights; expert weight streaming remains an independent bottleneck.
