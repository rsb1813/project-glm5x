# K3X Resident Multi-Token Multi-Expert CUDA Grid Design

## Status

Accepted design for Milestone 20. Nothing in this document is implemented or measured until the corresponding tests, runtime path, and B-0021 evidence pass.

## Objective

Reduce the host launch and activation-transfer granularity left after B-0020 by adding one exact, opt-in CUDA execution identity that evaluates a rectangular batch of native MXFP4 experts and tokens with resident weights. Preserve natural routing, router-slot accumulation order, AURORA proposal and rollback behavior, KDA/MLA state, and strict target verification.

## Evidence motivating the boundary

B-0020 resident rows reduce draft weight H2D by 88.81% to 89.78%, but retain 410 to 451 stream synchronizations and mixed decode results. A post-publication Nsight Systems trace of the exact 8 MiB resident fixed-token path at public head `01eac162` observed the following across five context-prefill and five incremental draft forwards.

- 1,040 CUDA kernel launches.
- 1,346 `cudaMemcpyAsync` calls.
- 410 `cudaStreamSynchronize` calls.
- Approximately 1.13 ms of aggregate GPU kernel duration.
- Approximately 35.03 ms in host `cudaLaunchKernel` API calls and 71.55 ms in host `cudaMemcpyAsync` API calls under instrumentation.

The trace is diagnostic, not a throughput benchmark. It shows that merely eliminating more stream-wait duration is insufficient; the next boundary must reduce the number of launches and activation round trips.

## Alternatives

### Selected: resident rectangular expert grid

Add a K3-shaped backend contract for `expert_count × token_count` native MXFP4 SiTU experts. Stable resident gate, up, and down pointers are supplied through a device descriptor table. Gate, up, SiTU, and down each launch once over a three-dimensional logical grid. The result remains separated by expert and token so the existing CPU router-slot accumulation order is unchanged.

This is the smallest dependency-closed boundary that attacks launch count while retaining the current CPU KDA, MLA, router, residual, and target implementation.

### Deferred: CUDA Graph cache by routed expert set

A graph cache could replay existing kernels, but its key contains the ordered expert set and shapes. Routing-set reuse is not yet measured, graph count needs a hard bound, and graph update/capture failure semantics would be a separate policy. It is deferred until B-0021 establishes whether a direct grid is insufficient.

### Deferred: complete device-resident draft graph

Keeping KDA/MLA state, routing, argmax, and hidden activations on the GPU could remove most remaining transfers and synchronizations. It also changes every major correctness boundary at once and is too broad for one measured milestone. It remains the likely later architecture if dependency-closed grids are still host-bound.

## Public execution identity

Extend `CudaBatchingMode` with `resident_grid`. The existing `scalar` and `grouped` identities and defaults remain unchanged.

`resident_grid` is valid only when all of the following are true.

- Backend is `cuda-custom`.
- Boundary is `ffn-block`.
- Allocation is `reused`.
- Weight mode is `resident` with a positive hard byte capacity.
- Transfer mode is `synchronous`.
- MoE fusion is `none`.
- Every expert in the requested grid has equal gate, up, and down dimensions and native MXFP4 group size 32.

For AURORA, only `aurora-persistent + cuda-custom` may select the identity. Replay remains CPU-only, CPU remains the draft default, and ordinary greedy execution retains literal zero grid telemetry.

## Backend contract

Add a backend method with this logical contract.

```cpp
Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_grid(
    std::span<const float> inputs,
    std::size_t token_count,
    std::span<const Mxfp4MlpView> experts,
    float situ_beta,
    std::optional<float> situ_linear,
    std::uint32_t layer,
    ProfilePhase phase);
```

`inputs` is token-major with `token_count * input_width` values. The result has one entry per expert; each entry is token-major with `token_count * output_width` values. The method rejects zero tokens, zero experts, overflow, unequal shapes, invalid native payload sizes, duplicate or zero tensor identifiers, non-finite SiTU parameters, and unsupported backend identities before CUDA mutation.

The CPU backend implements the same contract as an oracle by iterating expert first and token second through the existing exact MXFP4 SiTU path. This is a reference implementation, not a CPU performance path.

## CUDA data flow

1. Validate the complete grid and acquire all six native extents per expert from `ResidentWeightTable`.
2. If every acquisition is resident, build the host descriptor array from stable device pointers.
3. Upload one token-major activation block and one bounded descriptor array.
4. Launch gate once across expert, token, and row.
5. Launch up once across expert, token, and row.
6. Launch SiTU once across all expert-token intermediate elements.
7. Launch down once across expert, token, and row.
8. Copy one expert-major output block to the host and synchronize once.
9. Split the flat block into the public per-expert token-major result.

Scratch storage is grow-only and sized with checked multiplication for `expert_count * token_count * intermediate_or_output_width`. Descriptor capacity grows with the largest accepted grid and is included in memory accounting.

## Exact fallback

The grid is all-or-nothing. If any resident acquisition reports hard-cap bypass, the backend does not mix resident-grid and transient scratch pointers. It records one grid fallback and executes the complete request through the existing serial `mxfp4_situ_mlp_group` behavior once per token. This preserves exact output and bounded capacity without partial launch ambiguity.

Acquisition or CUDA errors remain failures; they are not converted into CPU fallback. Malformed input fails before resident-table or profiler mutation wherever the current backend contract permits.

## Runtime integration

The Stable LatentMoE path continues to load experts and compute routing exactly as today. With `resident_grid`, it calls the grid contract with `token_count=1`, receives one output per selected expert, and accumulates those vectors using the existing router-slot loop. It must not use atomic routed accumulation or reorder expert contributions.

The multi-token dimension is exercised by direct backend tests and a kernel benchmark. AURORA draft generation does not batch causally dependent candidate tokens; the next token still comes from the previous draft logit's argmax. Documentation must not claim concurrent autoregressive draft-token generation.

## Telemetry

Add target and draft fields for the following exact counters.

- `resident_grid_calls`.
- `resident_grid_experts`.
- `resident_grid_tokens`.
- `resident_grid_expert_tokens`.
- `resident_grid_kernel_launches`.
- `resident_grid_fallbacks`.
- `resident_grid_descriptor_h2d_bytes`.

A successful grid call increments kernel launches by four. Exact fallback increments only `resident_grid_fallbacks`; it must not increment successful grid calls or expert-token totals. Existing activation, weight, D2H, resident-cache, allocation, synchronization, FFN, and kernel-time counters remain authoritative and are not duplicated.

## CLI and failure boundary

The target CLI accepts `--cuda-batching resident-grid` only for the closed backend identity above. Add `--aurora-draft-batching grouped|resident-grid`; it is owned only by `aurora-persistent + cuda-custom`. Selecting `resident-grid` with zero resident capacity, transient weights, CPU build, replay, speculation `none`, prefetch transfer, routed accumulation, or non-reused allocation fails before Reader access or output-file creation.

The default remains `grouped` for CUDA AURORA and remains unchanged everywhere else.

## B-0021 evidence

The canonical end-to-end matrix contains natural greedy plus four matched resident-grouped/resident-grid pairs.

- Fixed block-2 token-major target.
- Adaptive token-major target.
- Fixed block-2 expert-major target.
- Adaptive expert-major target.

Each pair uses CPU natural Top-16 target, CUDA persistent fixed Top-4 draft, 8 MiB resident capacity, FP32, reused allocation, `ffn-block`, synchronous transfer, fusion `none`, four prompt tokens, six generated tokens, three warmups, and twenty measured samples.

Before writing summaries, the runner requires identical proposals, acceptance, committed tokens, final KDA/MLA state, committed routing, Reader bytes, selected K, and residency capacity semantics. Grid rows must have positive successful grid counters, exactly four launches per grid call, zero fallback at 8 MiB, and lower MoE launch count than their grouped pairs. Throughput direction is recorded and never forced.

A direct CUDA benchmark additionally compares the CPU oracle and CUDA grid for token counts 1, 2, and 4 and expert counts 1, 2, and 4 on the executable synthetic dimensions. It is kernel-contract evidence, not full-model throughput.

## Verification gates

- CPU oracle tests for shape, order, overflow, duplicate IDs, payload validation, and exact values.
- CUDA kernel tests for 1×1, 1×4, 2×2, and 4×4 grids against the CPU oracle.
- Full-resident and one-byte fallback tests with exact output and counter separation.
- AURORA CPU/grouped/grid proposal, acceptance, token, state, route, and cursor parity.
- CLI ownership, invalid-combination, CPU-build unavailable, and no-output-on-failure tests.
- JSON/CSV schema and deterministic-counter tests.
- B-0021 committed-evidence digest and headline recomputation.
- Complete CPU, liburing/direct, ASan/UBSan, CUDA, and Compute Sanitizer matrices.

## Non-goals

- No CUDA Graph or graph cache.
- No persistent cooperative kernel.
- No device-resident KDA, MLA, router, argmax, or whole-token state.
- No concurrent generation of causally dependent draft tokens.
- No reduced precision, dynamic eviction, ORBIT prediction, EcoSpec, MoE-Spec, AcceptMoE, proxy, or pruning.
- No default change without measured end-to-end and quality evidence.
- No full checkpoint download or paid cloud resource.

