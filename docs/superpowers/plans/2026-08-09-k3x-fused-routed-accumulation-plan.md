# K3X Milestone 12 Fused Routed Accumulation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to execute each task and superpowers:verification-before-completion before milestone claims.

**Goal:** Add and measure an opt-in exact CUDA path that fuses routed-expert scaling and ordered accumulation into the native MXFP4 down projection, returning one mixed latent vector.

**Architecture:** Preserve the existing `ffn-block` implementation as the reference. Add one lower-level accumulating MXFP4 launch contract and narrow backend methods for synchronous and prepared expert groups. The model passes immutable routing contribution weights only after selection, and the CUDA backend copies a single final mixed vector.

**Tech Stack:** C++20, CUDA 13.3, CMake/CTest, Python/pytest, Compute Sanitizer, RTX 5080 `sm_120`.

---

### Task 1: RED tests for the accumulating native MXFP4 kernel

**Files:**
- Modify: `tests/cuda/test_cuda_mxfp4.cu`
- Modify: `runtime/cuda/mxfp4.cuh`
- Modify: `runtime/cuda/mxfp4.cu`

1. Add literal CPU-oracle cases covering first-store, later accumulate, positive/negative/zero scale, high nibbles, distinct E8M0 groups, and more than 256 columns.
2. Build `test_cuda_mxfp4` and record the expected compile failure before adding the launcher.
3. Implement the smallest store-or-FMA mode in the existing one-block-per-row kernel without changing the reference launcher.
4. Run the focused test and Compute Sanitizer.
5. Commit the kernel contract as one semantic change.

### Task 2: RED tests for synchronous fused expert groups

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `runtime/src/backend_cuda_stub.cpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_ffn.cu`
- Modify: `tests/cpp/test_backend.cpp`

1. Add tests for two-expert ordered mixing, single expert, shape/count/finite validation, no side effects on rejection, one final output, and exact runtime statistics.
2. Record the missing-interface RED failure.
3. Add the narrow synchronous `mxfp4_situ_moe` contract to CPU, stub, and CUDA backends.
4. Reuse current gate/up/SiTU scheduling and weight residency, but launch accumulating down projections and perform one D2H copy.
5. Run focused CPU/CUDA tests and commit.

### Task 3: RED tests for prepared prefetch and runtime integration

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/model.cpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tests/cuda/test_cuda_async_ffn.cu`
- Modify: `tests/python/test_cpp_parity.py`

1. Add prepared-token tests for ordered weights, single use, foreign/wrong-shape token rejection before side effects, and parity with synchronous fusion.
2. Add CLI RED tests for the explicit fusion option, invalid combinations, default `none`, and diagnostic identity.
3. Implement the prepared fused backend path and pass routing contribution weights from the model without consulting residency.
4. Add fused call/expert counters and ensure existing benchmark JSON consumers receive stable defaults.
5. Run focused CTest/pytest and commit.

### Task 4: B-0013 measurement tooling

**Files:**
- Create: `runtime/src/cuda_expert_bench.cpp`
- Modify: `CMakeLists.txt`
- Create: `tools/ablate_cuda_fusion.py`
- Create: `tests/python/test_cuda_fusion_ablation.py`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_benchmark_schema.py`

1. Add failing schema and ablation-parity tests before implementation.
2. Implement a bounded released-dimension repeated-view benchmark using the existing storage fixture and explicit labeling.
3. Implement the matched synthetic natural-Top-16 ablation with raw JSON/CSV, checksums, correctness gates, D2H/H2D/kernel/runtime counters, and a dry smoke mode.
4. Run focused tests, then commit measurement tooling.

### Task 5: Full verification and RTX 5080 measurement

**Files:**
- Create: `results/b0013-fused-routed-accumulation/*`

1. Run CPU, liburing/direct, CUDA, and ASan/UBSan suites.
2. Run all applicable CUDA binaries under Compute Sanitizer.
3. Run B-0013 with declared warmup/sample counts and cross-check raw JSON, CSV, summary, artifact identity, and checksums.
4. Compare matched traffic, kernel time, decode/prefill/TTFT, memory, routing, token, logits, and recurrent state. Do not promote by theoretical reduction alone.
5. Commit immutable measurement evidence.

### Task 6: TITAN Ledger, self-review, and publication

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `README.md` only if the public feature surface changed materially.

1. Self-review every backend caller, serialization consumer, malformed-input path, and option combination.
2. Record accepted/rejected/default status from measurement in `DECISIONS.md` and measured values only in `BENCHMARKS.md`.
3. Update `ARCHITECTURE.md`, then `PROJECT_STATE.md` last with the last known-good commit and next bottleneck.
4. Run final verification from clean builds and `git diff --check`.
5. Commit ledger changes, push the public branch, open a PR, verify CI, merge only after success, and verify post-merge CI.
