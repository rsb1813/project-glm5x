# K3X Admission-Time Immutable Validation and Attribution Design

## Status and scope

This design implements D-048 as Milestone 23. It removes repeated scans of immutable dense MoE-layer weights from the exact CUDA hot path while preserving a runtime-selectable per-call reference mode. It also adds direct validation telemetry and a released-dimension B-0024 attribution matrix.

The milestone does not add CUDA Graphs, a larger device-resident token graph, dynamic L0 eviction, reduced precision, full-checkpoint execution, or token-throughput claims. It does not change natural routing or the default execution boundary.

## Evidence motivating the change

Corrected B-0023 measures the complete released-dimension MoE-layer boundary at 20.488, 20.954, and 24.422 ms for 1, 4, and 16 experts, versus 1.228, 2.371, and 5.681 ms for the split path. The complete method currently scans six immutable FP32 tensors totaling 469,776,384 bytes on every call before any resident acquisition.

The aggregate kernel-time increase is much smaller than the wall-time increase. This makes repeated host validation the next code-backed attribution target, but not yet a measured causal decomposition.

## Alternatives

### Validate on every `ResidentWeightTable::acquire` miss

This is the smallest implementation, but it can upload earlier tensors before discovering a later non-finite tensor. It therefore weakens the existing fail-before-CUDA-mutation contract for a complete layer. It is rejected.

### Add an opaque prepared-layer public API

An explicit prepare/execute token gives the strongest lifetime boundary, but it changes the public backend interface, model call sites, and benchmark ownership for one isolated bottleneck. Static residency already keeps layer tensors alive for the backend lifetime, so this is deferred until a larger prepared token graph is justified.

### Cache complete-layer validation inside the CUDA backend

This is the accepted approach. The backend validates every new immutable tensor identity in a complete host-only preflight, commits identities only after the whole preflight succeeds, and begins resident acquisition afterward. Exact identity hits skip the scan. The implementation remains private to the CUDA backend and preserves current callers.

## Runtime modes and defaults

Add `CudaWeightValidationMode` with two values.

- `per_call` reproduces the current behavior. Every complete-layer invocation scans all six immutable FP32 tensors.
- `admission` scans an immutable identity once per CUDA backend lifetime and uses constant-time identity hits afterward.

`BackendOptions::cuda_weight_validation` defaults to `per_call`. The public CLI exposes `--cuda-weight-validation per-call|admission` and emits the selected mode in JSON. CPU execution and CUDA boundaries other than `moe-layer` reject a non-default selection through the existing option-ownership checks.

No existing default changes in Milestone 23. A later decision may promote `admission` only after B-0024 and full correctness validation.

## Immutable identity and lifetime contract

The admission registry is owned by one `CudaBackend` and destroyed with it. Each entry contains the following exact identity.

- nonzero tensor ID;
- host data pointer;
- element count and byte count;
- rows and columns.

For an unseen tensor ID, the backend scans all values with `std::isfinite`. For an existing tensor ID, every identity field must match exactly. A different pointer, shape, or length fails with `invalid_mxfp4` before resident acquisition or CUDA mutation.

The caller guarantees that an admitted host allocation remains alive and immutable for the CUDA backend lifetime. In-place modification behind the same pointer is outside the contract and is not detected by a repeated O(bytes) checksum. This matches the existing stable tensor-ID residency contract and is made explicit rather than silently depending on it.

There is no independent invalidation API because current L0 residency is grow-only and has no eviction. Backend destruction invalidates the complete registry. Future eviction-capable residency must couple validation generation and resident generation before it may reuse this optimization.

## Validation algorithm

`resident_mxfp4_moe_layer` keeps all existing dynamic and structural checks. Input, contributions, epsilon, SiTU parameters, tensor shapes, native MXFP4 layout, and duplicate or zero IDs remain validated on every call.

The six immutable dense/vector views are assembled only after shape and ID validation.

In `per_call` mode, the backend scans all six views and fails immediately if any value is non-finite.

In `admission` mode, the backend performs two host-only phases.

1. Compare every view with the registry. Exact matches are validation hits; new identities are staged; identity conflicts fail.
2. Scan every staged view. If any scan fails, discard the complete staged set. If all pass, commit every staged identity together.

Only after this succeeds may resident acquisition, allocation, H2D upload, scratch reservation, event creation, or kernel launch begin.

## Telemetry

Extend `BackendRuntimeStats` with four monotonic counters.

- `immutable_validation_scans` counts individual dense tensor scans.
- `immutable_validation_hits` counts exact admission-registry identity hits.
- `immutable_validation_bytes` counts bytes actually scanned.
- `immutable_validation_nanoseconds` measures host wall time spent in identity lookup and finite scans.

The counters exclude dynamic input and contribution validation. They are zero for split execution. The released complete layer has exactly six immutable views and 469,776,384 validation bytes per full scan.

The benchmark reports cold deltas separately from measured warm deltas. This prevents one-time admission work from being mislabeled as steady-state work.

## B-0024 attribution benchmark

Extend `k3x_cuda_moe_layer_bench` with strict `--validation per-call|admission` and `--profiler on|off` arguments. The independent split oracle remains scoped and destroyed before the selected backend is created.

B-0024 contains eighteen rows.

- split boundary at 1, 4, and 16 experts with profiler off and on;
- complete layer with per-call validation at 1, 4, and 16 experts with profiler off and on;
- complete layer with admission validation at 1, 4, and 16 experts with profiler off and on.

Every row uses the same released expert artifact, deterministic released-size dense fixture, 1 GiB hard capacity, three warmups, and twenty measured iterations. `kernel_nanoseconds` is JSON `null` when the profiler is off; it is never fabricated as zero.

The runner enforces the following physical gates.

- maximum absolute error at most `1e-5`;
- zero resident bypass and fallback;
- zero measured warm weight H2D;
- exact split/layer synchronization, launch, traffic, and 14,336-byte routed-norm deltas retained from B-0023;
- per-call layer warm scans equal `iterations * 6` and warm validation bytes equal `iterations * 469,776,384`;
- admission layer cold scans equal six and cold validation bytes equal 469,776,384;
- admission layer warm scans and bytes equal zero and warm hits equal `iterations * 6`;
- profiler on/off does not change numerical output or non-profiler physical counters.

Latency direction is recorded but not used as a pass gate. The summary stores raw JSON SHA-256 values, runner and artifact hashes, a canonical aggregate hash, LF-only CSV, and a summary CSV hash. No token, prefill, or TTFT field is emitted.

## Correctness tests

CUDA tests first witness failure for the following missing behavior.

- admission mode scans six tensors once and records six later hits;
- per-call mode scans six tensors on every invocation;
- a non-finite tensor in the last view fails before resident bytes, allocations, uploads, scratch, events, launches, and synchronizations change;
- failed staged admission does not cache earlier identities;
- reusing a tensor ID with a different host pointer or shape fails before CUDA mutation;
- exact repeated execution matches the CPU oracle;
- default options and CLI JSON remain `per-call`.

The benchmark and committed-evidence tests independently recompute every counter identity and digest.

## Failure behavior

Dynamic or structural invalidity, non-finite immutable weights, identity conflicts, and malformed native expert views return `invalid_mxfp4`. Capacity insufficiency remains an exact successful bypass with `executed=false`. CUDA allocation, transfer, event, plan, kernel, and synchronization errors remain `backend_unavailable` and are never converted into a quality-changing fallback.

## Acceptance

Milestone 23 is accepted only after focused RED/GREEN evidence, B-0024 measurement on the RTX 5080 under WSL2, Compute Sanitizer, applicable CPU/liburing/ASan/CUDA matrices, independent raw-summary digest validation, final read-only review, synchronized README and TITAN Ledger, public PR integration, and post-merge correctness and CodeQL.
