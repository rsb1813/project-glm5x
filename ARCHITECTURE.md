# GLM5X Architecture

## Current product boundary

GLM5X is the GLM-5.x product runtime. The migrated K3X code is treated as a storage and scheduling foundation, not as a claim that Kimi K3 execution remains supported by this repository.

### Implemented in this bootstrap

- K3X-compatible superblock, tensor/layer/expert directories, aligned extents, CRC/SHA verification, and resumable writer.
- Portable C++20 reader, cache, prefetch, routing-policy, speculative, and expert-major interfaces.
- GLM descriptor validation for DSA, Top-8, 256 routed experts, shared experts, and MTP metadata.
- GLM5X converter CLI wrapper.
- CPU/reference TurboQuant-inspired KV cache with Hadamard rotation, integer and half-bit schedules, asymmetric K/V policy, incremental attention, and capacity estimation.

### In progress

- GLM tensor manifest and model-specific extent roles.
- GLM-5.2 reference graph with exact DSA/MLA and MoE routing.
- Synthetic GLM-5.2 checkpoint round-trip.
- Connecting compressed KV blocks to the GLM DSA/indexer state instead of the standalone reference cache.

### Experimental

- MTP/AURORA draft and DSpark-compatible target verification.
- Expert-major union scheduling and cost-aware speculation.
- TurboQuant 2.5/3.5-bit KV schedules, UltraQuant-style asymmetric K/V and block-scale variants.
- Mixed weight quantization and CUDA fusion.

### Proposed

- GLM-5.3 checkpoint descriptor and calibration swap.
- Full RTX 5080 native CUDA kernels.
- Cloud-side shard conversion through the existing SKYFORGE concept.

## Runtime data flow

The converter reads bounded source shards, validates identity, emits execution-ordered extents, and releases source memory before the next unit. Runtime residency is tiered as L0 VRAM, L1 RAM, and L2 NVMe. Cache score combines frequency, transition probability, recency, predicted use, load latency, size, residency, and speculative usefulness.

For speculative verification, GLM5X computes candidate routing first, forms a per-layer unique expert union, fetches each exact expert once, and batches candidate-token work by expert. Natural Top-8 routing remains the correctness reference.

For long context, the planned path is paged DSA state with a recent high-precision window and compressed historical KV blocks. `TurboQuantKVCache` is currently a CPU/reference implementation only. Its storage estimate is a logical bit budget plus per-row scale metadata; it is not yet a packed CUDA storage format and must not be used as a throughput claim.

## Model boundary

`GLM5XModelDescriptor` is the first model boundary. It owns model family, attention kind, layer count, hidden size, routed expert count, Top-K, shared experts, vocabulary size, and MTP layer count. Tensor file names and shapes belong to a checkpoint manifest and must not be embedded in runtime kernels.

K3-specific KDA, Attention Residual, Stable LatentMoE, 896-way Top-16 assumptions, and native Kimi MXFP4 naming are not part of the GLM5X default graph. They remain historical source context in the old K3X repository only.

## Quality modes

The strict reference path is always available. Any adaptive Top-K, proxy, pruning, or verifier-budget mode is opt-in and must report quality divergence. SHADOW and PHOENIX-style escalation are policy layers and cannot silently change the natural routing contract.

`TURBO-LONGCTX` is an experimental mode for 600k–1M context capacity. It keeps exact routing and target verification while allowing compressed historical KV. Context capacity, prefill/TTFT, and decode tok/s are recorded as separate measurements. TurboQuant does not compress the 753B model weights; expert weight streaming remains an independent bottleneck.
