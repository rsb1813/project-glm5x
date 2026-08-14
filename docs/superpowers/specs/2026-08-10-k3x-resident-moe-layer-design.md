# K3X Resident MoE-Layer CUDA Boundary Design

## Status

Accepted Milestone 21 design under the user's standing approval to continue autonomously before paid Cloud Run work. Nothing in this document is implemented or measured until the corresponding RED/GREEN tests, runtime path, and B-0022 evidence pass.

## Objective

Remove the repeated activation round trips and synchronizations that remain around the Milestone 20 resident expert grid. Keep routing, expert choice, contribution order, AURORA proposal/rollback behavior, KDA/MLA state, and strict target verification unchanged. Return one hidden-width MoE result to the CPU rather than expert-major latent outputs and separately computed shared/routed projections.

## Evidence motivating the boundary

B-0021 fixed grid rows execute 30 resident grids and report 470 draft stream synchronizations, 108,800 activation-H2D bytes, and 102,880 D2H bytes. Adaptive grid rows execute 33 grids and report 517 synchronizations. Weight H2D is already bounded at 644,160 to 647,424 bytes, so another weight-only change does not address the observed host-driven boundary.

The current token-major MoE path separately performs routed-down projection, shared SiTU MLP, resident expert grid, CPU contribution accumulation and RMSNorm, and routed-up projection. Each CUDA subcall owns its own host upload, result download, and stream synchronization. Milestone 21 joins only these dependency-closed MoE operations.

## Alternatives

### Selected: exact resident MoE-layer boundary

Keep router logits, full expert order, selected K, and normalized contributions on the CPU. After selection, submit the hidden input, contributions, dense projection/norm/shared weights, and native MXFP4 expert views to one backend call. The CUDA backend executes the complete MoE feed-forward branch on one stream and returns one hidden-width vector.

This removes three synchronization boundaries per successful synthetic MoE call without changing KDA, MLA, Attention Residual, router selection, greedy argmax, or speculative target ownership.

### Deferred: routed-set CUDA Graph cache

CUDA Graphs can reduce repeated host launch setup when a topology is instantiated and relaunched. NVIDIA documents separate definition, instantiation, execution, and parameter-update phases; topology changes require re-instantiation, while pointer or kernel-parameter changes require graph or node updates. K3X has not measured ordered routed-set reuse or defined a bounded graph-cache eviction policy. Adding that policy now would combine execution and caching changes.

Reference: [NVIDIA CUDA Programming Guide — CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

### Deferred: complete device-resident token graph

A whole-token graph could retain KDA/MLA state, routing, residuals, logits, and argmax on the GPU. It would replace every major draft correctness boundary at once and would make rollback and failure attribution too broad for one milestone. It remains the likely direction if the measured MoE-layer boundary is still host-bound.

## Public execution identity

Extend `CudaBoundaryMode` with `moe_layer`, serialized as `moe-layer`. The existing `operation` and `ffn-block` identities and defaults remain unchanged.

The new boundary is valid only when all of the following are true.

- Backend is `cuda-custom`.
- Dense precision is FP32.
- Allocation is `reused`.
- Weight mode is `resident` with a positive hard byte capacity.
- Batching is `resident-grid`.
- Transfer mode is `synchronous`.
- MoE fusion is `none`.
- Native MXFP4 group size is 32.

For AURORA, only `aurora-persistent + cuda-custom` may select `--aurora-draft-boundary moe-layer`. Replay remains CPU-only, CPU remains the draft default, and `ffn-block` remains the CUDA draft default.

## Backend contract

Add explicit dense-vector, MoE-layer-weight, and execution-result views.

```cpp
struct DenseVectorView {
    std::uint64_t tensor_id;
    std::span<const float> values;
};

struct ResidentMoeLayerView {
    DenseWeightView routed_down;
    DenseVectorView routed_norm;
    DenseWeightView routed_up;
    DenseMlpView shared;
};

struct ResidentMoeLayerResult {
    bool executed{};
    std::vector<float> output;
};

virtual Result<ResidentMoeLayerResult> resident_mxfp4_moe_layer(
    std::span<const float> input,
    ResidentMoeLayerView layer_weights,
    std::span<const Mxfp4MlpView> experts,
    std::span<const float> contributions,
    float epsilon,
    float situ_beta,
    std::optional<float> situ_linear,
    std::uint32_t layer,
    ProfilePhase phase);
```

`executed=true` returns exactly one hidden-width MoE branch result. `executed=false` is a successful hard-cap bypass signal with an empty output; the runtime then executes the existing Milestone 20 path under the same CUDA backend. CUDA, validation, or acquisition errors remain failures and never become CPU fallback.

The CPU backend implements the complete contract as an oracle and always returns `executed=true`. It evaluates routed-down, each exact expert, router-slot ordered weighting, double-accumulated RMSNorm, routed-up, shared SiTU MLP, and final addition with the existing portable operations.

## Validation and residency

Before residency mutation, validate all of the following.

- Nonempty input, expert list, and matching finite contribution list.
- Positive finite epsilon and SiTU parameters.
- Routed-down shape `latent × hidden` and routed-up shape `hidden × latent`.
- Routed norm length `latent`.
- Shared gate/up shape `intermediate × hidden` and shared down shape `hidden × intermediate`.
- Equal native expert dimensions `latent → expert intermediate → latent`.
- Checked products and CUDA grid limits.
- Unique nonzero tensor IDs across all six dense/vector weights and all three matrices of every expert.

Then acquire the six dense/vector weights and every native expert matrix from the existing `ResidentWeightTable`. The norm vector uses the dense-FP32 representation with shape `1 × latent`. If any acquisition bypasses the hard cap, record one layer fallback, preserve any successful admission/H2D accounting, launch nothing, and return `executed=false`.

## CUDA data flow

For a fully resident request, use grow-only layer-specific scratch so device pointers remain stable throughout the call.

1. Upload the hidden input, contribution vector, and expert descriptor table.
2. Run resident FP32 routed-down projection into the latent buffer.
3. Launch native MXFP4 gate and up grids over all selected experts.
4. Launch grid-wide SiTU.
5. Launch native MXFP4 down grid into expert-major latent outputs.
6. Launch one ordered weighted-sum kernel. Each latent row loops selected slots in router order.
7. Launch RMSNorm using double-precision square accumulation before FP32 scaling by the resident norm vector.
8. Run resident FP32 routed-up projection.
9. Run resident shared gate and up projections from the original hidden input.
10. Launch shared SiTU and resident shared-down projection.
11. Launch one hidden-width routed-plus-shared addition.
12. Copy one hidden-width result to the host and synchronize once.

The logical operation count is thirteen launches: five routed-expert operations, weighted sum, RMSNorm, routed-up, four shared operations, and final add. This may exceed the ten logical operations in the split path; the hypothesis is that eliminating three synchronization boundaries and intermediate transfers matters more on the tiny host-driven graph. B-0022 records throughput without forcing its direction.

## Exact fallback

The layer boundary is all-or-nothing. A residency bypass launches no partial layer work and returns `executed=false`. The runtime then performs the existing routed-down, shared MLP, resident-grid/serial expert path, ordered CPU accumulation, RMSNorm, routed-up, and addition. Existing component gates accept `moe-layer` only for this internal exact fallback.

Malformed requests fail before acquisition. Acquisition errors fail before kernels. CUDA errors after launch remain errors and do not retry through the split path because partially executed device work cannot be treated as an atomic bypass.

## Runtime integration

The model computes router scores and `RoutingDecision` exactly once. It loads the selected expert payloads in the existing order and builds the existing contribution vector. When the backend boundary is `moe_layer`, it calls the new contract before computing split-path latents or shared outputs.

On `executed=true`, the returned hidden vector is the MoE branch output and the model applies the existing residual addition. On `executed=false`, the same payload handles, decision, and contributions feed the unchanged split reference path. Routing traces, cold-rescue counts, expert-store access sets, profile observation, proposal state, and target verification remain unchanged.

Expert-major target verification remains outside this first boundary because it batches multiple positions and has a different gather/scatter lifecycle. Direct backend tests cover the contract, while B-0022 exercises persistent AURORA token-major and CPU expert-major target modes with a token-major draft provider.

## Telemetry

Add target and draft fields for these exact counters.

- `resident_moe_layer_calls`.
- `resident_moe_layer_experts`.
- `resident_moe_layer_kernel_launches`.
- `resident_moe_layer_fallbacks`.
- `resident_moe_layer_contribution_h2d_bytes`.

A successful layer call increments calls once, experts by selected K, and logical kernel launches by thirteen. A hard-cap bypass increments only layer fallbacks among the new success counters. Successful layer execution also increments the existing resident-grid call/expert/token/expert-token counters and four resident-grid kernel launches because the native expert grid still executes.

Existing weight/activation H2D, D2H, resident-cache, scratch, allocation, synchronization, FFN, grid, and kernel-time counters remain authoritative. Descriptor and contribution uploads are included in activation H2D and also exposed by their narrow counters; they are not counted twice in total H2D. Unlike the split path's CPU RMSNorm, the layer path must admit the routed norm vector to L0. B-0022 therefore reports this one-time cold weight/resident-byte delta explicitly instead of incorrectly requiring equal weight H2D.

## B-0022 evidence

The canonical end-to-end matrix contains natural greedy plus four matched Milestone 20 grid/Milestone 21 MoE-layer pairs.

- Fixed block-2 token-major target.
- Adaptive token-major target.
- Fixed block-2 CPU expert-major target.
- Adaptive CPU expert-major target.

Every pair uses CPU natural Top-16 target, CUDA persistent fixed Top-4 draft, 8 MiB exact residency, FP32, reused allocation, synchronous transfer, fusion `none`, four prompt tokens, six generated tokens, three warmups, and twenty measured samples. Only the draft boundary changes from `ffn-block + resident-grid` to `moe-layer + resident-grid`.

Before summary publication, the runner requires identical proposals, acceptance, target tokens, final KDA/MLA state, committed routing, Reader bytes, and selected K. Fully resident layer rows require positive layer calls, zero layer fallbacks, thirteen logical launches per call, three fewer stream synchronizations per successful MoE-layer call than the matched split row, and lower activation H2D plus D2H. The layer-minus-split weight-H2D delta must equal the corresponding resident-weight-byte delta and be positive because it is the routed norm vector's cold admission; total H2D must still decrease. Decode direction is measured and never forced.

## Verification gates

- CPU oracle value, shape, ID, dimension, contribution, epsilon, and SiTU validation tests.
- CUDA literal parity for one and four experts against the CPU oracle.
- Full-resident success and one-byte all-or-nothing bypass with counter separation.
- CUDA error propagation without split or CPU fallback.
- Runtime token/state/route parity for target and persistent AURORA draft selection.
- CLI ownership, invalid-combination, CPU-build unavailable, and no-output-on-failure tests.
- Target/draft JSON and benchmark JSON/CSV schema tests.
- B-0022 raw digest, paired invariant, traffic, synchronization, and headline recomputation tests.
- Complete CPU, liburing/direct, ASan/UBSan, CUDA, and Compute Sanitizer matrices.

## Non-goals

- No CUDA Graph or graph cache.
- No KDA, MLA, Attention Residual, router, logits, or argmax on the GPU.
- No expert-major draft batching inside the new layer call.
- No reduced precision, dynamic eviction, ORBIT prediction, EcoSpec, MoE-Spec, AcceptMoE, proxy, or pruning.
- No default change without measured end-to-end and quality evidence.
- No full checkpoint download or paid cloud resource.
