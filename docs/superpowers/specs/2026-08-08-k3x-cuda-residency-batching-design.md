# K3X CUDA Residency and Projection Batching Design

## 1. Goal

Milestone 2 removes the avoidable per-operation CUDA costs measured in B-0002 without changing the synthetic K3 graph, natural routing, expert bytes, recurrent state, or greedy-token semantics. It adds independently switchable device-allocation reuse, exact immutable weight residency, and same-input projection batching. The Milestone 1 per-operation path remains executable as the reference ablation.

This milestone is not the three-tier cache, asynchronous storage pipeline, or a full-layer GPU executor. It establishes the L0 primitives and measurements those later systems require.

## 2. Evidence motivating the work

B-0002 measured the deterministic synthetic graph on the RTX 5080 as follows.

| Backend | Dense precision | Decode tok/s | H2D bytes/run | CUDA kernel ms/run |
|---|---|---:|---:|---:|
| `cpu` | FP32 | 19.4858 | 0 | 0 |
| `cuda-dense` | FP32 | 11.6682 | 4,999,104 | 11.561 |
| `cuda-custom` | FP32 | 10.1118 | 5,107,968 | 14.521 |

The CUDA implementation currently creates device buffers, cuBLASLt descriptors, heuristics, and timing events inside each matrix operation. It copies each input and weight, copies every result back to the host, synchronizes the stream, and frees the buffers before returning. Kernel time is only a small part of end-to-end time. The immediate measured hypothesis is therefore boundary overhead and redundant weight traffic, not insufficient kernel arithmetic throughput.

## 3. Considered approaches

### Scratch reuse only

Reuse input, weight-staging, and output buffers while retaining all transfers and scalar calls. This is low risk and isolates allocation cost, but it cannot remove the dominant repeated immutable-weight traffic or synchronization count.

### Selected staged residency design

Add three orthogonal switches: reusable scratch allocation, bounded static immutable-weight residency, and grouped same-input projections. This preserves the current graph and makes allocation, transfer, and synchronization savings independently measurable. It also creates explicit tensor identity and L0 residency primitives without prematurely choosing an expert eviction policy.

### Whole-layer GPU executor

Keep activations, KDA/MLA state, routing, and residual computation on device for a complete layer or token. This is the eventual direction for eliminating activation round trips, but it changes too many numerical boundaries at once and would make the source of parity failures difficult to isolate. It is deferred until the selected design quantifies the remaining cost.

## 4. Runtime switches and reference contract

The CUDA CLI adds these explicit options.

| Option | Values | Reference value | Meaning |
|---|---|---|---|
| `--cuda-allocation` | `per-operation`, `reused` | `per-operation` | Recreate buffers/descriptors/events each call or reuse backend-owned grow-only scratch and matching CUDA resources |
| `--cuda-weights` | `transient`, `resident` | `transient` | Copy weights for every use or retain admitted immutable weights in VRAM |
| `--cuda-batching` | `scalar`, `grouped` | `scalar` | Execute each projection separately or group independent projections sharing one input |
| `--cuda-resident-bytes` | non-negative integer | `0` | Hard upper bound for resident immutable weight bytes; required to be positive with `resident` |

CPU rejects CUDA-only switches that differ from their reference values. A CUDA-disabled build rejects a requested CUDA backend before interpreting residency options. Unknown values and invalid combinations return a CLI validation error. No runtime error silently changes the requested backend.

All three optimizations remain off by default until their end-to-end and correctness measurements are accepted. The exact B-0002 behavior is reproduced by `per-operation + transient + scalar`.

## 5. Backend contracts

### 5.1 Stable tensor identity

Every dense or MXFP4 weight request carries its K3X tensor ID in addition to shape and byte spans. A cache key contains the tensor ID, storage representation, dense precision, dimensions, and MXFP4 group size where applicable. Reusing one tensor ID with incompatible metadata is an error, not a cache hit.

Tensor identity is passed explicitly from `Engine`; host pointers are never used as persistent identities. This is required because reader-owned expert buffers are temporary while dense vectors have unrelated allocation lifetimes.

### 5.2 Reusable scratch allocation

`CudaBackend` owns typed scratch slots for input, transient weight data, output, BF16 staging, and grouped-operation metadata. A slot grows only when a request exceeds its capacity and otherwise reuses the existing allocation. Growing a slot allocates the replacement first, switches ownership only after success, then frees the old allocation. Allocation failure leaves the previous valid slot intact and returns a typed backend error.

The `per-operation` reference path retains the Milestone 1 local `DeviceBuffer` behavior. Reuse must not alter H2D/D2H byte accounting or numerical results.

### 5.3 Bounded static weight residency

Resident mode stores exact immutable device copies keyed by the stable tensor identity.

- FP32 and BF16 dense representations are distinct cache entries.
- MXFP4 packed E2M1 bytes and E8M0/32 scale bytes remain separate exact device extents under one logical entry.
- A hit performs no weight H2D copy and records the saved bytes.
- An admitted miss uploads once, verifies launch-visible metadata, and remains resident until backend destruction.
- If the configured byte bound cannot admit an entry, that operation uses transient staging and increments an admission-bypass counter.
- This milestone performs no eviction and therefore does not introduce FIFO, LRU, LFU, Least-Stale, or a predictive cache policy.
- Cache admission, hit, miss, bypass, and resident-byte accounting cannot affect routing or output.

The bound applies only to resident immutable weights. Scratch capacity is measured separately, and total backend-owned peak VRAM remains reported.

### 5.4 Same-input grouped projections

`ComputeBackend` gains grouped dense and grouped MXFP4 operations. A group is an ordered list of independent projections that consume the same input. The CPU implementation is the reference and executes the existing scalar operations in list order. CUDA may upload the shared input once, enqueue all projections on the same nonblocking stream, copy ordered outputs, and synchronize once at the group boundary.

Initial graph call sites are limited to dependency-free groups already present in the synthetic graph.

- KDA `q_proj`, `k_proj`, and `v_proj`.
- Dense and shared-expert `gate` and `up` projections.
- Routed expert MXFP4 `gate` and `up` projections.

Dependent projections such as `f_a_proj` followed by `f_b_proj`, expert `down`, and attention output projections remain scalar. MLA grouping and larger MoE groups are deferred until the first grouped implementation is measured.

A grouped failure identifies the failing member and returns no partial result to the model. Output ordering follows request ordering exactly.

## 6. Resource lifetime and data flow

For the fully enabled path, one operation group follows this sequence.

1. `Engine` provides a shared activation and stable tensor-keyed weight requests.
2. The backend obtains or grows one input scratch slot and copies the activation once.
3. Each weight lookup either returns a resident device extent, admits and uploads an exact extent, or selects transient staging because the bound is exhausted.
4. Cached cuBLASLt descriptors and algorithms keyed by shape and precision are reused; custom MXFP4 launches use the existing native-byte kernel.
5. Independent operations are enqueued in request order.
6. Results are copied into ordered host vectors and the stream synchronizes once for the group.
7. CPU graph logic continues with unchanged activation, routing, state, residual, and token-selection code.

Backend destruction releases descriptors, events, resident entries, scratch slots, the cuBLASLt handle, and the stream. Normal destruction must leave no CUDA allocation visible to Compute Sanitizer.

## 7. Profiling and benchmark schema

The existing timing and traffic fields remain unchanged. The runtime adds measured counters.

- `device_allocation_count` and `device_free_count`.
- `stream_synchronization_count`.
- `weight_cache_hits`, `weight_cache_misses`, and `weight_cache_bypasses`.
- `resident_weight_bytes` and `peak_resident_weight_bytes`.
- `scratch_bytes` and `peak_scratch_bytes`.
- `weight_h2d_bytes` and `activation_h2d_bytes`, whose sum equals existing H2D accounting.
- `grouped_projection_calls` and `grouped_projection_members`.

Counters are serialized by `k3x_run`, preserved in JSON/CSV, and aggregated by median for timing values and exact equality for deterministic byte/count values. A benchmark aborts if backend identity, option values, or deterministic counters change unexpectedly across measured samples.

Runtime JSON is sampled before backend destruction. Allocation/free counters therefore cover inference-time operations and explicit resource replacement, not final teardown; Compute Sanitizer independently verifies that teardown releases every remaining allocation.

GPU utilization, memory bandwidth, NVMe traffic, and I/O stall time remain explicitly not measured in this milestone unless a validated collector is added separately.

## 8. Numerical and error contracts

- The CPU/PyTorch contracts remain unchanged.
- FP32 CUDA uses the existing dense `1e-5` and native MXFP4 `1e-4` tolerances.
- BF16 uses the existing `2e-2` operation tolerance and reports diagnostic maximum error.
- Every tested switch combination must generate `[43, 32, 28, 49, 9, 28]` exactly.
- Cache hit, miss, capacity bypass, scratch growth, and grouped execution must not alter routing or state.
- A key/metadata collision, invalid group, capacity overflow, CUDA allocation/copy/launch/synchronization failure, or descriptor-cache failure returns a typed error without CPU fallback.
- Cache capacity arithmetic uses checked size operations.

## 9. Test strategy

Implementation follows TDD and preserves the Milestone 1 reference mode.

### Resource unit tests

- Reused scratch performs one allocation for repeated equal/smaller requests and a controlled grow for a larger request.
- Failed growth preserves the old allocation and correct live-byte accounting.
- Resident dense and MXFP4 entries upload once and hit thereafter.
- Precision or metadata changes do not alias one cache entry.
- Capacity exhaustion uses transient staging and increments bypass without changing output.
- Backend destruction is leak-free under Compute Sanitizer.

### Grouped-operation tests

- Literal heterogeneous-row dense groups match scalar CPU results and preserve order.
- MXFP4 gate/up groups cover independent packed/scales extents and match scalar results.
- Shared input is counted once per group.
- One final synchronization is counted per successful group.
- Invalid member metadata rejects the complete group.

### End-to-end matrix

The synthetic graph runs FP32 `cuda-dense` and `cuda-custom` across all eight allocation/weight/batching combinations. Selected fully enabled BF16 paths also run to retain BF16 coverage. Every case compares layer outputs, logits, canonical state, and exact tokens against the appropriate reference.

CPU-only CTest and pytest remain green, and the CUDA build retains native `sm_120` cubins.

## 10. Ablation and acceptance gates

Three warmups and twenty measured process runs are collected sequentially for at least these FP32 `cuda-custom` modes.

1. `per-operation + transient + scalar`, reproducing B-0002.
2. `reused + transient + scalar`, isolating allocation reuse.
3. `reused + resident + scalar`, isolating immutable-weight residency after allocation reuse.
4. `reused + resident + grouped`, isolating grouping after residency.

The same fully enabled comparison is measured for `cuda-dense`; BF16 fully enabled is measured separately and never substituted for FP32 correctness.

Milestone 2 is accepted when all of the following are true.

- Reference-mode B-0002 counters and numerical behavior remain reproducible within ordinary timing variance.
- Reused allocation reduces device allocation/free counts versus reference mode.
- Resident mode reduces weight H2D bytes and records real hits without exceeding the configured bound.
- Grouped mode reduces activation H2D or synchronization count at its enabled call sites.
- Every mode preserves exact tokens and declared numerical tolerances.
- End-to-end tok/s, TTFT, kernel time, transfers, RSS, VRAM, and new counters are recorded without requiring a prescribed winner.
- No optimization becomes the CUDA default unless the measured end-to-end result improves without a correctness regression.

Beating the CPU's tiny-graph throughput is desirable but is not an acceptance requirement. Synthetic CPU-versus-GPU ordering is not a proxy for full Kimi K3 performance.

## 11. Explicit exclusions and next boundary

This milestone does not implement pinned host memory, asynchronous storage, expert eviction, L1/L2 tiers, prefetch prediction, CUDA Graphs, persistent kernels, adaptive Top-K, speculative decoding, or full Kimi K3 dimensions.

After this milestone, the remaining host round trips and measured resident-set behavior determine whether the next change is a wider layer GPU executor or the first L0/L1 asynchronous transfer pipeline. The decision is recorded from ablation evidence rather than assumed in advance.
