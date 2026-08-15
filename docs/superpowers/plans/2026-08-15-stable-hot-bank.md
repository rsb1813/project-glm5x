# Stable Per-Layer Hot-Bank Plan

**Goal:** Reduce repeated packed-expert H2D and device-cache churn by preserving a small exact expert bank per layer while bypassing low-value one-off admissions. Natural routing and expert values remain unchanged.

**Initial design boundary:** Extend the existing byte-bounded device expert cache with an opt-in policy that learns per-layer access frequency, reserves only a bounded number of stable entries per layer, and does not let transient admissions evict the retained bank. The first gate uses a repeated 16-layer trace of digest-matched real sidecars; no full-model claim follows from this structured trace.

**Alternatives to compare:** Existing LRU, current layer-balanced protection, and stable per-layer hot-bank admission. Larger plain LRU is already rejected by full-model zero-hit evidence. Predicted/learned routing and lossy proxy experts are outside this milestone.

## Checklist

- [x] Inspect current cache admission, protection, and layer call sites.
- [x] Add RED tests for stable retention, transient bypass, byte bounds, and runtime parity.
- [x] Implement the smallest opt-in stable hot-bank policy.
- [x] Run focused Python correctness tests against the latest CPU runner.
- [x] Measure a repeated 16-layer real-sidecar trace on RTX 5080 against LRU.
- [x] Update architecture, decisions, benchmarks, and project state last.
