# Packed Sidecar Traffic Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate packed-sidecar file, decode, host-to-device, and transfer timing counters so the next residency/prefetch decision is based on measured traffic rather than logical K3X reads.

**Architecture:** Keep telemetry opt-in and additive. `GLM5XPackedExpertCache` records per-request counters while preserving the exact synchronous and pinned reference behavior; the reference benchmark snapshots those counters per prefill/decode phase and writes them to JSON. No router, quantization, cache admission, or output semantics change.

**Tech Stack:** Python 3, PyTorch CUDA events/streams when available, existing `GLM5XPackedExpertCache`, pytest, JSON benchmark schema.

**Spec:** `PROJECT_CHARTER.md`, `ARCHITECTURE.md`, `BENCHMARKS.md`

## Global Constraints

- Preserve exact BF16 routing and generated-token parity in the default path.
- Keep pinned staging and all packed precisions opt-in; zero-capacity behavior remains unchanged.
- Do not label logical artifact bytes as physical NVMe bytes.
- Do not run or claim a full-model 10--20 tok/s result from a bounded probe.
- Do not modify C++ runtime or cloud infrastructure in this plan.

---

### Task 1: Add cache-level traffic counters

**Files:**
- Modify: `reference/glm5x_ref/packed_cache.py`
- Test: `tests/python/test_glm5x_packed_cache.py`

**Interfaces:**
- Produces additive cache stats for sidecar read calls/bytes, decoded payload bytes, H2D bytes, and H2D elapsed nanoseconds.
- Existing `GLM5XPackedExpertCacheStats` fields and `get/get_many` return values remain compatible.

- [ ] **Step 1: Write the failing test** that admits one sidecar and asserts the new counters are present and zero-capacity/default behavior remains valid.
- [ ] **Step 2: Run the focused test** with `PYTHONPATH=reference:converter ... -m pytest tests/python/test_glm5x_packed_cache.py -q` and confirm the missing fields fail.
- [ ] **Step 3: Add counters at the exact boundaries**: bytes read from the sidecar file, bytes decoded from validated sections, bytes submitted to `.to(device=...)`, and CUDA-event or wall-clock duration for the transfer. Do not count a host-cache hit as a file read.
- [ ] **Step 4: Run the focused cache tests** and confirm CPU-safe tests pass while CUDA-only timing tests skip cleanly without CUDA.
- [ ] **Step 5: Run `py_compile`** on the modified cache and test modules.

### Task 2: Expose phase-separated benchmark telemetry

**Files:**
- Modify: `reference/glm5x_ref/layer10_moe.py`
- Modify: `tools/benchmark_glm5x_reference.py`
- Test: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Adds prefill/decode JSON fields for packed sidecar read calls/bytes, decoded bytes, H2D bytes, and H2D nanoseconds.
- Existing JSON/CSV fields and default values remain stable when the packed cache is disabled.

- [ ] **Step 1: Write the schema regression** requiring the new keys with numeric zero defaults when no packed cache is configured.
- [ ] **Step 2: Run the schema test** and confirm it fails only for the missing fields.
- [ ] **Step 3: Snapshot cache stats** before and after prefill/decode, compute non-negative deltas, and emit the new fields without conflating them with `storage_read_*` logical K3X counters.
- [ ] **Step 4: Run the focused benchmark/schema suite** and the existing packed-cache suite.
- [ ] **Step 5: Run the complete WSL Python suite** and `git diff --check`.

### Task 3: Record the bounded measurement and decide the next scheduler boundary

**Files:**
- Modify: `BENCHMARKS.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PROJECT_STATE.md`
- Modify: `DECISIONS.md`

- [ ] **Step 1: Run one bounded real layer-10 sidecar comparison** with the exact synchronous path and the opt-in pinned path, using the same worker count and sidecar set.
- [ ] **Step 2: Record sidecar bytes, decoded bytes, H2D bytes/time, cache hits, route IDs, and output parity**; explicitly mark physical NVMe as unmeasured unless an independent device counter exists.
- [ ] **Step 3: Accept or reject the next prefetch/residency change** from the measured bottleneck, without projecting full-model TPS.
- [ ] **Step 4: Update the persistent documents last** and record the commit/test state.
