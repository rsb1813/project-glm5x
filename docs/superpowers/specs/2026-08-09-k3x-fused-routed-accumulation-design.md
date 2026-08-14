# K3X Milestone 12 Fused Routed Accumulation Design

## Scope

Milestone 12 removes the largest remaining avoidable boundary inside the exact native-MXFP4 routed FFN path. The current `ffn-block` implementation runs gate, up, SiTU-GLU, and down projection on the GPU, but copies one latent vector per selected expert to the host and performs router scaling plus expert accumulation on the CPU. Released Kimi K3 selects 16 of 896 experts, so this boundary scales device-to-host traffic with Top-K even though the next operation consumes only the mixed latent vector.

The accepted experiment fuses each expert's routing scale and ordered accumulation into the existing native MXFP4 down-projection kernel. Gate/up projection and SiTU remain separate exact kernels. The final mixed latent vector is copied to the host once. The existing unfused `ffn-block` path remains the reference and default.

Primary evidence is the official [MoonshotAI Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3), which specifies native MXFP4 weights, SiTU-GLU, 896 routed experts, and natural Top-16. The official [FlashKDA implementation and design report](https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md) also shows why fusion boundaries must follow parallelism: its single-kernel prototype was slower than a two-kernel split because token-parallel and recurrence work had different parallelism. NVIDIA's [CUDA Graph programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) identifies repeated host launch overhead as the problem graphs address, but graphs do not remove the routed output copies or CPU accumulation targeted here.

## Alternatives and decision

Three boundaries were compared.

1. Fuse SiTU directly into the current one-block-per-output-row down kernel. This would recompute every SiTU activation once per output row, changing an `O(intermediate)` activation into `O(latent × intermediate)` work. A correct tiled redesign could avoid that duplication, but it is substantially broader than the measured boundary. This approach is rejected for this milestone.
2. Fuse gate and up into one launch. Both projections share the input, but the existing grouped path already uploads that input once. Combining their independent row grids saves one launch while leaving activation materialization, Top-K output copies, and CPU mixing intact. This remains a later micro-optimization.
3. Fuse routing scale and ordered expert accumulation into each down projection, then copy one mixed vector. This is accepted. It preserves the existing reduction and expert order, removes `K-1` latent-vector D2H copies, removes the CPU mixing loop, and adds no extra kernel launch.

CUDA Graph and persistent-kernel experiments remain separate follow-on axes. Combining them with this change would prevent attribution and would require stable pointer and routing-shape contracts that the streaming expert path does not yet guarantee.

## Runtime contract

Add an explicit `none|routed-accumulate` CUDA MoE fusion option. It is valid only for `cuda-custom + ffn-block`; invalid backend/boundary combinations fail closed. `none` preserves the current API behavior and remains the default.

The fused backend entry point accepts the input latent, experts in natural routing order, and one finite FP32 contribution weight per expert. A contribution weight is the already normalized unbiased router weight multiplied by the checkpoint `routed_scale`. Residency never changes expert order or weight.

For expert zero, the native down kernel stores `weight[0] × down[0]`. For every later expert, the same stream launches the kernel in order and performs `mixed[row] = fma(weight[e], down[e][row], mixed[row])`. Each output row is owned by one block, so no atomic operation is required. The backend returns one mixed latent vector after one final D2H copy and stream synchronization.

Synchronous and prepared-prefetch paths must implement the same contract. A prepared token remains single-use and identity-validated before any side effect. Empty expert groups, count mismatches, non-finite weights, malformed native MXFP4 views, and output-shape disagreement are rejected before allocation, transfer, or statistics change.

## Numerical and correctness contract

The optimization does not change routing, native MXFP4 decoding, SiTU definition, expert order, or precision. It may change the last few FP32 bits because multiplication and accumulation move from host scalar operations to device FMA. Correctness therefore requires.

- Literal kernel tests against an independent CPU ordered-accumulation oracle within the existing native-MXFP4 tolerance.
- Fused versus unfused backend parity for one and multiple experts, negative and zero contribution weights, columns beyond one CUDA block, and synchronous/prepared paths.
- Graph-level equality for token IDs and routed expert traces, with bounded logits, layer outputs, and recurrent-state error.
- Malformed input rejection before observable side effects.
- Compute Sanitizer with zero reported errors.

The unfused path remains available as the reference mode for every test and benchmark.

## Telemetry and measurement

Runtime telemetry adds the selected fusion identity, fused call count, and fused expert count. Existing D2H bytes, synchronization, kernel time, H2D bytes, cache statistics, Reader traffic, average Top-K, cold rescue count, and correctness diagnostics remain authoritative.

B-0013 compares `none` and `routed-accumulate` under matched FP32 `cuda-custom + ffn-block` settings on the deterministic natural Top-16 synthetic fixture. It also runs a released-dimension bounded expert microbenchmark using the existing 3,584 × 3,072 native-MXFP4 storage fixture, repeating one immutable expert view in a controlled 16-slot execution to measure kernel/D2H behavior without downloading real weights. Repeated-view results are explicitly labeled a kernel fixture, not a routing or full-model throughput result.

The expected D2H reduction is a design invariant, not a measured speed claim. The fused path is promoted only if target RTX 5080 measurements preserve correctness, reduce the intended traffic, and do not regress matched end-to-end decode beyond the predeclared noise threshold. Otherwise it stays experimental or is rejected in `DECISIONS.md`.

## Non-goals

This milestone does not implement KDA/MLA fusion, Attention Residual fusion, CUDA Graph replay, persistent kernels, tensor-core MXFP4 repacking, MXFP8 activations, multi-token expert-major verification, mixed trunk quantization, or a whole-layer GPU executor. It does not infer full-model TPS from the synthetic or repeated-view fixtures.
