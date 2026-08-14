# K3X Exact Asynchronous L0/L1 Transfer Design

## Status and scope

This document defines Milestone 4, the first exact system-RAM-to-VRAM transfer pipeline in K3X. It follows B-0004, which showed that a dependency-closed CUDA FFN boundary reduces activation traffic and synchronization but leaves the graph CPU-driven and does not overlap cold expert weight movement with useful work.

Milestone 4 moves only selected native MXFP4 expert payloads from bounded page-locked L1 staging to ephemeral L0 device staging. K3X routing, expert order, MXFP4 bytes, router scores, output mixing, recurrent state, and greedy selection remain exact. The existing synchronous path remains the default reference.

This milestone does not implement asynchronous NVMe reads, an L1 expert cache, L0 eviction, cache scoring, task/session profiles, expert prediction, adaptive Top-K, cold rescue, or speculative decoding. `Reader` continues to perform synchronous K3X extent reads, and no NVMe-overlap claim is permitted.

## Evidence and constraints

The CUDA 13.3 Best Practices Guide states that page-locked host memory is required for reliable asynchronous host/device transfers and that non-default streams are required to overlap transfers with kernel execution. It also warns that pinned memory is scarce and expensive to allocate, so K3X uses one bounded reusable allocation rather than per-request pinning.

CUDA stream ordering supplies the dependency contract. The transfer stream records a readiness event after the expert slab upload. The compute stream uses `cudaStreamWaitEvent` immediately before the prepared expert block, so the host does not synchronize merely to establish the dependency. Separate timing events measure transfer duration and dependency wait duration.

Primary references:

- <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
- <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>
- <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html>
- <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html>

## Alternatives

### Selected: two-phase exact prefetch token

After natural routing selects the expert set, the runtime loads the exact payloads into ordinary system RAM, prepares one bounded pinned slab, and enqueues one H2D transfer on a dedicated stream. While that transfer proceeds, the existing routed-down projection runs on the compute stream. An opaque single-use token later identifies the prepared expert group.

This is the smallest boundary that exposes an actual use deadline and useful overlap without introducing a predictor or L2 scheduler.

### Rejected for this milestone: expert-internal pipelining only

Uploading expert `i+1` while expert `i` computes avoids a public prepare/consume contract, but it cannot hide the first expert's transfer and provides no earlier deadline boundary. It remains a possible later optimization inside the prepared block.

### Rejected for this milestone: general worker-thread scheduler

A multi-request deadline queue and worker thread would prematurely combine L2 reads, cache admission, eviction, and prediction. Those concerns require separate measurements and belong after the exact L0/L1 primitive.

## Runtime modes and capability gate

K3X adds `CudaTransferMode { synchronous, prefetch }` and CLI `--cuda-transfer synchronous|prefetch`. `synchronous` is the default and preserves the Milestone 3 path.

The first `prefetch` implementation requires all of the following:

- backend `cuda-custom`;
- CUDA boundary `ffn-block`;
- CUDA allocation mode `reused`;
- CUDA weight mode `transient`;
- positive `--cuda-pinned-bytes` capacity;
- native MXFP4 group size 32.

Unsupported combinations fail before backend construction. There is no silent synchronous fallback in `prefetch` mode because it would make benchmark provenance ambiguous. Static resident admission remains independently selectable only with synchronous transfer until a later cache-policy milestone integrates readiness events with residency.

CPU and CUDA-disabled builds retain zero-valued transfer metadata and reject explicit prefetch requests.

## Public contracts

`BackendOptions` gains:

```cpp
enum class CudaTransferMode { synchronous, prefetch };

CudaTransferMode cuda_transfer{CudaTransferMode::synchronous};
std::uint64_t cuda_pinned_bytes{};
```

The backend exposes one opaque logical identifier rather than device pointers. The backend, not C++ copy semantics, enforces single-flight and single-use behavior:

```cpp
struct Mxfp4PrefetchToken {
    std::uint64_t value{};
    std::uint64_t use_sequence{};
};

virtual Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
    std::span<const Mxfp4MlpView> experts,
    std::uint64_t use_sequence,
    std::uint32_t layer,
    ProfilePhase phase) = 0;

virtual Result<std::vector<std::vector<float>>>
mxfp4_situ_mlp_group_prepared(
    std::span<const float> input,
    Mxfp4PrefetchToken token,
    float situ_beta,
    std::optional<float> situ_linear,
    std::uint32_t layer,
    ProfilePhase phase) = 0;
```

The synchronous `mxfp4_situ_mlp_group` API remains unchanged. A token is backend-local, single-use, and valid only for the layer, phase, sequence, expert order, tensor identities, representations, and dimensions captured by prepare. Only one token may be outstanding because the current decoder is single-request and single-token. A second prepare or a stale, foreign, repeated, or mismatched consume returns `INVALID_STATE` without discarding the valid pending request.

The backend copies every source byte into owned pinned staging before prepare returns. Later mutation or destruction of the reader-owned vectors cannot change the prepared result.

## Pinned and device staging

Two focused CUDA components keep the existing backend from absorbing more resource logic.

`PinnedBuffer` owns exactly one `cudaHostAlloc` allocation whose size is `cuda_pinned_bytes`. It is created with the backend and never grows or re-registers memory at runtime. Requests larger than the fixed slab fail before copying. Construction and destruction update current/peak pinned-byte counters, and no operation can temporarily exceed the configured capacity.

`AsyncMxfp4Pipeline` owns:

- one nonblocking transfer stream;
- one fixed-capacity pinned slab;
- one grow-only device slab tracked by existing VRAM counters;
- transfer-start and transfer-end timing events;
- one readiness event created with `cudaEventDisableTiming`;
- compute-stream wait-start and wait-end timing events;
- one optional pending request and a monotonically increasing token ID.

For each expert in router order, the slab stores gate packed bytes, gate scales, up packed bytes, up scales, down packed bytes, and down scales in that order. No padding, repacking, dequantization, or requantization is allowed. Device views are reconstructed as base-plus-offset byte spans.

Prepare validates the complete group before allocation, pinned copies, counter mutation, or CUDA submission. It then copies the exact bytes into the pinned slab, records the transfer-start event, submits one H2D slab copy, records transfer-end and readiness events, and returns the token without synchronizing either stream.

Consume validates the token before touching CUDA state, records wait-start on the compute stream, enqueues `cudaStreamWaitEvent`, records wait-end, and runs the existing ordered exact expert FFN kernels against the staged device views. The existing final block synchronization makes output and event timings observable. The pending request is released only after successful completion or explicit backend teardown.

## Graph scheduling

The synchronous graph order is unchanged. In prefetch mode the MoE path performs these steps:

1. Compute exact router scores and stable natural Top-K order.
2. Load the selected exact K3X expert payloads synchronously into system RAM.
3. Prepare the expert group with a monotonically increasing use sequence.
4. Compute the routed-down projection on the existing compute stream while expert H2D is eligible to run on the transfer stream.
5. Consume the prepared group at its exact use point.
6. Mix expert outputs with the unchanged scores and order.
7. Continue routed normalization, routed-up, shared expert, residual, and state work unchanged.

The use sequence is an explicit deadline identity, not a queue priority. A real multi-request deadline scheduler is deferred until K3X can have more than one outstanding request.

## Failure and teardown semantics

- Invalid shapes, group sizes, reserved scales, overflow, or empty groups fail before side effects.
- A payload larger than `cuda_pinned_bytes` fails with `INVALID_EXTENT`; it never falls back synchronously.
- Duplicate prepare and invalid token use fail with `INVALID_STATE`.
- Host allocation, device allocation, stream, event, copy, wait, or synchronization errors return `BACKEND_UNAVAILABLE` with a specific message.
- Backend destruction synchronizes the transfer stream before freeing a pending pinned or device slab, then destroys events and streams in dependency-safe order.
- Failed prepare does not increment successful-prefetch, H2D, wait, or ready counters.
- Failed consume does not report a successful FFN block.

## Measurement contract

`BackendRuntimeStats` and benchmark JSON/CSV gain:

- `pinned_host_bytes` and `peak_pinned_host_bytes`;
- `async_prefetch_calls` and `async_prefetch_bytes`;
- `async_prefetch_ready_before_use` and `async_prefetch_late_at_use`;
- `transfer_stream_wait_count`;
- `pinned_staging_nanoseconds`;
- `transfer_device_nanoseconds`;
- `transfer_stall_nanoseconds`;
- device `async_engine_count` and `device_overlap` capability metadata.

`weight_h2d_bytes` still includes the prepared payload exactly once. The pinned slab copy is a system-RAM copy and is not counted as PCIe traffic; its CPU wall time is recorded as `pinned_staging_nanoseconds`. `transfer_device_nanoseconds` is CUDA-event time around the transfer-stream H2D. `transfer_stall_nanoseconds` is CUDA-event time between compute-stream wait markers. Ready/late classification uses `cudaEventQuery` immediately before the dependency is enqueued.

The benchmark must label synchronous K3X file reads as synchronous and must not populate NVMe GB/token or I/O-stall fields from these H2D counters.

## Tests

### Portable and CLI tests

- Defaults serialize `synchronous` and zero pinned/async counters.
- CPU and unsupported CUDA option combinations are rejected before backend construction.
- Benchmark JSON/CSV preserves the requested transfer identity and all counters.

### CUDA resource tests

- `PinnedBuffer` allocates the fixed configured capacity once, reuses it without further host allocation, rejects oversized requests, and returns all current bytes to zero on destruction.
- Prepare with native literal bytes returns without incrementing host synchronization count.
- Mutating the original host vectors after prepare does not change the prepared output.
- Consume returns the CPU oracle result and exact expert order.
- Group-size, scale, extent, capacity, duplicate-prepare, stale-token, and repeated-consume failures preserve required side-effect invariants.
- Compute Sanitizer reports zero errors for the pinned-buffer and prepared-FFN tests.

### Graph parity

For FP32 and BF16, scalar and grouped scheduling, synchronous and prefetch modes must preserve:

- generated token IDs exactly;
- selected expert trace exactly;
- layer-output, logits, and KDA/MLA state tolerances;
- logical expert byte order and count;
- no additional host stream synchronization before prepared consume.

## Ablation and acceptance

`tools/ablate_cuda_transfer.py` runs four matched transient FFN-block cases per precision:

1. synchronous scalar;
2. prefetch scalar;
3. synchronous grouped;
4. prefetch grouped.

B-0005 uses the regenerated deterministic synthetic artifact, three warmups, and 20 measured processes for FP32 and BF16. It records decode/prefill throughput, TTFT, RSS, VRAM, H2D/D2H, pinned bytes, transfer time, dependency stall, ready/late counts, synchronizations, kernel time, exact tokens, routing parity, and numeric error.

Prefetch remains experimental unless all correctness checks pass and measurements show explainable behavior. A throughput regression does not invalidate the primitive, but it prevents default activation. Synthetic WSL2 results are not full-model or native-Linux claims.

## Follow-on boundary

After this milestone, the next storage work may add an L1 exact expert bank and L2 asynchronous extent reader. Those components may submit multiple deadline-bearing requests to a scheduler, but they must reuse this pinned/event dependency contract and preserve the synchronous reference path.
