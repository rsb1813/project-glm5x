# K3X Milestone 15 CUDA Expert-Major Design

## Status and objective

Milestone 15 extends the exact Milestone 14 expert-major verifier from its CPU reference to one deliberately narrow CUDA boundary. For every MoE layer in a speculative block, K3X will still compute natural routing for every candidate position, build the stable first-use expert union, load each expert payload once, and preserve each token's original router-slot accumulation order. The new work makes the physical CUDA execution match that scheduling contract by uploading one expert's native MXFP4 gate/up/down payload once and evaluating every assigned token before advancing to the next expert.

This milestone does not add a learned drafter, change Top-K, budget experts, prune experts, use proxies, or select a new default. Token-major verification remains the default.

## Evidence and constraints

- MoonshotAI Kimi K3 commit `3cb39dfd32e51c3328e2e4b4af21341247d06c43` documents 896 routed experts, natural Top-16, latent dimension 3,584, expert hidden dimension 3,072, SiTU-GLU, and native MXFP4 weights.
- vLLM commit `83ad767eed3be3ee7f2df63be693bfaca5c7c922` sorts token assignments into expert-grouped order and processes token tiles against one selected expert weight. Its fused MoE source explicitly describes grouped ordering as a way to promote weight-cache reuse.
- NVIDIA CUDA 13.3 cuBLAS documentation exposes grouped and batched GEMM but warns that separate calls or streams can be faster for some problem sizes. K3X therefore does not assume a generic grouped API wins for its native E2M1 plus E8M0/32 representation.
- The current K3X CUDA MXFP4 kernel performs one matrix-vector operation per launch. Repeating that API for every assignment would reload transient expert weights and would not satisfy the physical H2D-union requirement.

Primary sources:

- <https://github.com/MoonshotAI/Kimi-K3/tree/3cb39dfd32e51c3328e2e4b4af21341247d06c43>
- <https://github.com/vllm-project/vllm/blob/83ad767eed3be3ee7f2df63be693bfaca5c7c922/vllm/model_executor/layers/fused_moe/fused_moe.py>
- <https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmgroupedbatched>

## Alternatives

### Accepted: one expert, multiple tokens

Add a backend primitive that consumes a flat batch of latent inputs and one `Mxfp4MlpView`. CUDA uploads the contiguous input batch and the expert's three native weight representations once, launches gate and up with a two-dimensional `(row, token)` grid, applies the existing SiTU-GLU elementwise over the flattened batch, launches down with the same two-dimensional grid, and copies one contiguous output batch back. The runtime scatters outputs to the existing `(token_index, router_slot)` slots.

This is the smallest boundary that makes the M14 logical union a physical CUDA weight-transfer union while preserving the current arithmetic and rollback model.

### Rejected for this milestone: temporary residency around scalar calls

Temporarily pinning an expert in the existing resident table would reduce some uploads with less kernel work, but ownership, capacity misses, cache counters, and eviction semantics would become part of the experiment. It would not provide an isolated proof that one expert group causes one upload, and transient scratch reuse would still require careful lifetime rules.

### Rejected for this milestone: one persistent multi-expert block kernel

A persistent kernel could eventually reduce launch overhead further, but it combines expert union scheduling, variable assignment counts, multiple weight sets, ordered mixing, and state rollback in one change. It is not necessary to prove multi-token weight amortization and would make failures harder to attribute.

## Backend contract

Add the following virtual method to `ComputeBackend` and implement it for CPU, CUDA, and the unavailable CUDA stub.

```cpp
virtual Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_batch(
    std::span<const float> inputs,
    std::size_t batch_size,
    Mxfp4MlpView expert,
    float situ_beta,
    std::optional<float> situ_linear,
    std::uint32_t layer,
    ProfilePhase phase) = 0;
```

`inputs` is row-major with exactly `batch_size * expert.gate.cols` values. The result contains `batch_size` vectors in input order, each with `expert.down.rows` values. Validation occurs before allocation, profiling events, transfers, or kernel launches. It rejects zero batch size, multiplication overflow, wrong packed or scale sizes, non-group-32 weights, incompatible gate/up/down dimensions, and invalid SiTU parameters.

The CPU implementation is the portable oracle and evaluates the existing exact scalar MXFP4 path for each row. The CUDA implementation is available only for `cuda-custom + ffn-block`. It must not silently fall back to CPU.

## CUDA execution and memory

The new `launch_mxfp4_matvec_batch` launcher uses a grid of `(rows, batch_size)` and indexes input/output by `blockIdx.y`. Within a row, the reduction order, E2M1 decode, E8M0 exponent application, and FP32 `fmaf` sequence remain the same as the scalar kernel. Batch size one must match the scalar launcher.

For one expert group of size `B`, the backend reserves or allocates these contiguous buffers.

- input: `B * latent_input_cols * sizeof(float)`
- gate and up: each `B * expert_hidden_rows * sizeof(float)`
- activation: `B * expert_hidden_rows * sizeof(float)`
- output: `B * latent_output_rows * sizeof(float)`
- transient packed and scale scratch: maximum of the three expert matrices, reused sequentially

The backend copies the three weight matrices exactly once each. It records their packed-plus-scale byte count once in `weight_h2d_bytes`, records the flat input batch in `activation_h2d_bytes`, and records the flat result in D2H profiling. The design intentionally starts with synchronous transfer and transient weights so B-0016 can attribute bytes without residency or prefetch ambiguity.

## Runtime integration

`forward_expert_major_block` keeps its current route planning, state snapshots, payload loading, and CPU router-slot accumulation. For each expert group it gathers assigned latent vectors into one flat buffer in stable assignment order, calls `mxfp4_situ_mlp_batch` once, validates the returned batch shape, and scatters each output back to its original token and router slot. The CPU reference may retain its existing scalar inner loop so B-0015 remains behaviorally and numerically stable; the CUDA path uses the batch primitive.

CUDA expert-major verification is accepted only when all of the following hold.

- backend is `cuda-custom`
- CUDA boundary is `ffn-block`
- allocation mode is `reused`
- weight mode is `transient`
- transfer mode is `synchronous`
- MoE fusion is `none`
- L1 expert cache is disabled
- L2 scheduling is blocking
- routing is natural
- runtime profile observation is disabled
- execution is incremental on the four-layer synthetic graph

CPU expert-major keeps its existing accepted combination. Unsupported combinations fail before Reader, output, recurrent state, or CUDA telemetry mutation.

## Telemetry

Add backend runtime counters for batched expert FFN calls and tokens. Existing `weight_h2d_bytes`, `activation_h2d_bytes`, D2H profiler events, `ffn_block_calls`, and `ffn_block_experts` remain authoritative byte and call counters. Generation retains M14 unique-expert, assignment, evaluated-position, and discarded-position counters.

B-0016 must verify, per row, that CUDA expert-major weight H2D equals the sum of one native gate/up/down payload per reported unique expert load. The token-major comparison may upload the same expert more than once and must report its measured bytes rather than a derived estimate.

## Tests

1. CPU backend contract tests cover batch sizes one and two, input order, scalar parity, empty batch, malformed flat length, invalid weight shape, and invalid SiTU parameters.
2. CUDA literal tests compare batch sizes one, two, and three with the CPU oracle and require batch-one scalar parity.
3. CUDA telemetry tests prove one expert payload upload for multiple tokens and no profiler or counter mutation on validation failure.
4. Runtime tests compare greedy, token-major, CPU expert-major, and CUDA expert-major generated tokens, final KDA/MLA state, committed routes, and accepted-prefix behavior for perfect and mixed scripts.
5. CLI tests require the exact capability combination and reject unsupported combinations before side effects.
6. Compute Sanitizer covers the literal batch kernel and perfect/mixed CUDA expert-major CLI paths.

## Benchmark gate

B-0016 uses the synthetic executable model with three warmups and twenty measured samples for exactly five end-to-end identities: greedy CUDA, token-major perfect, CUDA expert-major perfect, token-major mixed, and CUDA expert-major mixed. It records decode/prefill/TTFT, exact token/state/route parity, weight and activation H2D, D2H, kernel time, VRAM, Reader calls/bytes, unique experts, assignments, acceptance, evaluated/discarded positions, and batched-call/token counters.

B-0016 also runs a bounded released-dimension microbenchmark with the existing non-executable expert slice. It compares repeated scalar execution with batch sizes two and four, records measured weight/activation H2D, D2H, kernel time, latency, and VRAM, and labels the result as kernel/H2D evidence rather than token throughput. No favorable direction is required. CUDA expert-major remains non-default unless representative native-Linux and RTX 5080 evidence, physical traffic attribution, and quality measurements justify changing it.

## Out of scope

- learned DSpark or self-speculative drafting
- acceptance-aware dynamic block sizing
- EcoSpec, MoE-Spec, or AcceptMoE expert budgeting
- asynchronous prepared transfer, persistent expert residency, L1 caching, or deadline L2 scheduling
- reduced/adaptive Top-K
- CUDA-side router-slot accumulation across experts
- full Kimi K3 checkpoint execution or paid cloud resources
