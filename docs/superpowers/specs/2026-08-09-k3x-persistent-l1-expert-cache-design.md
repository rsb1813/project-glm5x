# K3X Milestone 5 Persistent L1 Expert Cache Design

## Outcome and boundary

Milestone 5 implements a bounded, runtime-session-lifetime L1 system-RAM store for exact native MXFP4 expert triplets. It removes repeated K3X file reads and temporary pageable-vector construction for admitted experts while preserving natural routing and the existing synchronous and prepared L1-to-L0 paths.

This milestone is a storage primitive, not the chartered cache-policy milestone. It supports only `disabled` and no-eviction `static` admission. LRU, LFU, Least-Stale, task/session priors, transition prediction, eviction, asynchronous NVMe reads, and cold rescue remain unimplemented.

## Evidence from the current runtime

`Model::load_expert` currently performs six synchronous reader calls for every selected expert: packed and E8M0 scale extents for gate, up, and down. The returned vectors live only through the current expert invocation or FFN block. Milestone 4 begins pinned staging only after these reads and allocations complete, so it cannot hide or avoid this repeated L2-to-pageable-host work.

The existing `Reader` counters already measure successful tensor extent calls and bytes after artifact verification. They can therefore prove whether L1 admission avoids file reads without labeling logical reads as physical NVMe traffic.

## Alternatives

### A. Model-adjacent immutable expert store

Cache one complete gate/up/down payload under `(layer, expert)` and return a shared immutable handle. Admission is all-or-nothing and capacity is the sum of the six exact payload extents.

Benefits are expert-atomic accounting, stable view lifetime, CPU/CUDA independence, and a clean future policy boundary. The cost is introducing an explicit runtime option and shared payload type.

### B. Reader-level tensor byte cache

Cache individual tensor and auxiliary extents inside `Reader`.

This is superficially smaller but loses expert atomicity, mixes trunk and expert admission, produces six independent decisions per expert, and makes later expert policy signals difficult to express. It is rejected.

### C. CUDA backend-owned host cache

Cache payloads beside the prepared transfer pipeline.

This couples storage residency to one compute backend, cannot serve the CPU reference path, and duplicates model metadata. It is rejected.

Milestone 5 selects A.

## Public runtime contract

The runtime adds an independent host-storage identity.

```cpp
enum class L1ExpertCacheMode { disabled, static_admission };

struct RuntimeOptions {
    bool incremental{true};
    bool diagnostics{};
    L1ExpertCacheMode l1_expert_cache{L1ExpertCacheMode::disabled};
    std::uint64_t l1_expert_cache_bytes{};
};

struct L1ExpertCacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t bypasses{};
    std::uint64_t resident_bytes{};
    std::uint64_t peak_resident_bytes{};
};
```

The CLI exposes `--l1-expert-cache disabled|static` and `--l1-expert-cache-bytes`. Disabled requires zero bytes. Static requires a positive capacity. CPU and CUDA backends accept the same host-cache options because this boundary precedes compute selection.

Existing `generate_greedy` overloads remain source-compatible and delegate to disabled-cache `RuntimeOptions`. A new options overload carries the explicit runtime configuration. `GenerationResult` returns final L1 statistics for serialization.

## Payload and key

One entry contains the exact native bytes and metadata for all three projections.

```text
ExpertKey = layer index + expert index
ExpertPayload = gate(packed, scales, id, rows, cols)
              + up(packed, scales, id, rows, cols)
              + down(packed, scales, id, rows, cols)
charged bytes = sum of six vector sizes
```

The cache never dequantizes, repacks, pads, or alters E2M1/E8M0 bytes. The model derives the same tensor IDs and validates the same directory dimensions as the disabled path. Entries are immutable after admission and owned by stable shared handles, so unordered index growth cannot invalidate spans held through CUDA preparation or CPU execution.

## Lookup and admission

For each naturally selected expert, execution follows this order.

1. If disabled, load the complete payload from `Reader` and return a transient handle with zero cache counters.
2. If static, look up `(layer, expert)`.
3. On hit, return the existing immutable handle and increment `hits` without a reader call.
4. On miss, load and validate the complete payload into temporary ownership.
5. If its charged bytes fit the remaining hard capacity, insert once, increment `misses`, and update current/peak bytes.
6. If it does not fit, increment `misses` and `bypasses`, return the exact transient handle, and leave residency unchanged.

A failed read or validation never inserts a partial entry and never changes resident bytes. No eviction occurs. First-observation admission is explicitly experimental and is not described as a policy suitable for full Kimi K3.

## Lifetime and integration

The model retains a vector of shared payload handles for the complete routed group. It derives `Mxfp4MlpView` objects only after every handle is acquired. Those handles outlive synchronous expert execution or Milestone 4 prepare, routed-down overlap, and prepared consume. The pinned pipeline still copies exact bytes into its owned slab before returning its token.

The operation boundary and CPU path use the same handles, so parity tests can compare disabled and static modes without changing routing, compute backend, precision, batching, or transfer mode.

## Accounting

Runtime JSON and benchmark JSON/CSV add the following measured fields.

- `l1_expert_cache_mode`.
- `l1_expert_cache_bytes` configured capacity.
- `l1_expert_cache_hits`.
- `l1_expert_cache_misses`.
- `l1_expert_cache_bypasses`.
- `l1_expert_cache_resident_bytes`.
- `peak_l1_expert_cache_resident_bytes`.

Reader `calls`, `requested_bytes`, and `completed_bytes` remain actual logical file-read counters. A hit must not increment them. These are not OS block-device or physical NVMe measurements. Host allocator overhead is not included in charged bytes and must be stated in benchmark documentation.

## Failure and exactness invariants

- Disabled mode reproduces current reader calls, bytes, tokens, routing, numerical outputs, and zero L1 counters.
- An admitted hit performs no reader call and returns byte-identical packed/scales payloads.
- Capacity is never exceeded, and an oversized or no-room expert uses exact transient bytes.
- Admission is complete-expert atomic; no subset of gate/up/down becomes resident.
- Failed load or validation leaves the index, counters, and resident bytes unchanged except that no successful result is returned.
- Cache mode changes neither router order nor selected experts.
- Synchronous and prefetch L1-to-L0 paths consume the same payload representation.

## B-0006 acceptance matrix

B-0006 fixes `cuda-custom + ffn-block + reused + transient + scalar` and crosses two independent axes.

| Case | L1 cache | L1-to-L0 transfer |
|---|---|---|
| `disabled-synchronous` | disabled, 0 bytes | synchronous |
| `static-synchronous` | static, 65,536 bytes | synchronous |
| `disabled-prefetch` | disabled, 0 bytes | prefetch, 1 MiB pinned |
| `static-prefetch` | static, 65,536 bytes | prefetch, 1 MiB pinned |

FP32 is the primary mechanism measurement. BF16 is repeated as a quality/traffic guard because dense precision changes non-expert H2D but not native expert payload bytes.

The runner requires identical tokens, routing, backend identity, boundary, batching, allocation, GPU weight mode, precision, and sample counts. Static cases must have positive hits and misses, zero bypass at the selected synthetic capacity, positive bounded residency, and fewer reader calls and bytes than their disabled match. Matched cache cases must preserve H2D, D2H, FFN counts, and synchronization; prefetch matching retains the Milestone 4 transfer invariants.

Three warmups and 20 samples are recorded per row. Throughput, TTFT, RSS, logical file bytes/token, H2D, D2H, VRAM, pinned memory, cache counters, transfer timing, kernel time, exact tokens/routing, and numerical error are preserved in raw JSON/CSV and a compact cross-checked manifest. GPU utilization, memory bandwidth, and physical NVMe bytes remain not measured unless a real counter is added.

## Default and next boundary

Disabled remains the public default until B-0006 passes. Even if static admission wins on the synthetic graph, it is only an exact L1 primitive. The next milestone adds an independently switchable L2 read path and native-Linux buffered versus `io_uring`/`O_DIRECT` measurement. Least-Stale and other policies begin only after the primitive exposes representative hit, miss, bypass, size, and latency evidence.
