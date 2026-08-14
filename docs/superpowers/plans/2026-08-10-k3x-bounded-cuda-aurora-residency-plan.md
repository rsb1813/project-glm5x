# K3X Bounded CUDA AURORA Residency Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The primary agent owns the connected implementation because the active workspace policy does not authorize subagent delegation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact, opt-in, hard-bounded CUDA resident-weight mode to persistent AURORA drafting and measure its H2D, VRAM, synchronization, and end-to-end effect against the transient CUDA baseline.

**Architecture:** Reuse the existing backend-owned `ResidentWeightTable`; a new draft-only byte-capacity option selects transient weights at zero or resident weights at a positive capacity. Keep the CPU target, replay oracle, scheduler, routing, precision, and kernels unchanged. Serialize capacity and occupancy through independent draft telemetry and measure nine canonical B-0020 rows.

**Tech Stack:** C++20, CUDA 13.3 native `sm_120`, CMake/Ninja, Python 3.12, pytest, WSL2, JSON/CSV evidence, GitHub Actions.

## Global Constraints

- Correctness and exact target ownership precede throughput.
- CPU persistent drafting remains the default; `aurora-replay` remains CPU-only.
- Draft arithmetic remains FP32 with reduced fixed Top-K; no precision or scheduler change is allowed.
- CUDA identity remains reused allocation, grouped execution, `ffn-block`, synchronous transfer, fusion `none`, and zero pinned capacity.
- Resident capacity is a hard unsigned 64-bit byte bound; capacity misses use exact transient bypass and never change routing.
- Target and draft Reader/backend/profiler/runtime counters remain independent.
- B-0020 uses an 8,388,608-byte canonical capacity, 3 warmups, and 20 samples.
- No full Kimi K3 checkpoint is downloaded and no paid cloud resource is provisioned.
- Every production change follows a witnessed RED before GREEN and ends in a semantic commit.
- Every new source file begins with a one-line Korean role comment.

## File structure

- Modify `runtime/src/aurora.cpp` for the persistent-provider backend contract.
- Modify `runtime/src/main.cpp` for CLI parsing, preflight, draft backend options, and telemetry.
- Modify `tests/cuda/test_cuda_aurora_draft.cu` for provider-level resident and bypass parity.
- Modify `tests/python/test_cuda_aurora_draft.py` for CLI and end-to-end transient/resident parity.
- Modify `tests/python/test_cpp_parity.py` for preflight and default compatibility.
- Modify `tests/python/test_benchmark_schema.py` for zero/default JSON/CSV schema coverage.
- Modify `tools/benchmark_synthetic.py` for capacity selection and residency aggregation.
- Create `tools/ablate_cuda_aurora_residency.py` for B-0020.
- Create `tests/python/test_cuda_aurora_residency_ablation.py` for the matrix and committed evidence.
- Create `results/b0020-cuda-aurora-residency-wsl/` only after measurement code is committed.
- Modify README and all mutable TITAN Ledger documents after measurements exist.

---

### Task 1: Accept exact bounded resident backends in the persistent provider

**Files:**

- Modify `tests/cuda/test_cuda_aurora_draft.cu:20-145`.
- Modify `runtime/src/aurora.cpp:28-47`.

**Interfaces:**

- Consumes: existing `BackendOptions`, `ResidentWeightTable`, and `AuroraPersistentDraftProvider::create`.
- Produces: provider acceptance for transient-zero or resident-positive canonical CUDA options while replay remains CPU-only.

- [x] **Step 1: Extend the direct CUDA provider test before production code**

Change the option helper to accept a capacity.

```cpp
k3x::BackendOptions cuda_options(std::uint64_t resident_bytes = 0) {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::fp32;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = resident_bytes == 0
        ? k3x::CudaWeightMode::transient
        : k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::grouped;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_resident_bytes = resident_bytes;
    return options;
}
```

Create CPU, transient CUDA, 8 MiB resident CUDA, and one-byte resident CUDA providers against separate Readers. Drive the same initial proposal, full acceptance, second proposal, and partial rollback through all providers. Assert candidate and cursor-stat parity. At the end assert the full-fit backend has positive cache misses and hits, zero bypasses, and `0 < resident_weight_bytes <= 8 MiB`; assert the one-byte backend has positive bypasses, zero resident bytes, and exact proposal parity.

Retain an invalid backend case by setting `cuda_weights=resident` with `cuda_resident_bytes=0` after backend construction.

- [x] **Step 2: Build and witness provider RED**

Run.

```bash
cmake --build build-cuda --parallel 8
build-cuda/test_cuda_aurora_draft artifacts/m18-top16.k3x
```

Expected failure: resident provider creation returns `INVALID_STATE` because `supported_persistent_backend` accepts only transient zero-capacity CUDA.

- [x] **Step 3: Generalize the provider backend predicate minimally**

In `runtime/src/aurora.cpp`, retain every existing fixed option and replace only the weight clause with.

```cpp
const bool valid_weight_identity =
    (options.cuda_weights == CudaWeightMode::transient &&
     options.cuda_resident_bytes == 0) ||
    (options.cuda_weights == CudaWeightMode::resident &&
     options.cuda_resident_bytes > 0);
return fixed_common_options && valid_weight_identity &&
    options.cuda_pinned_bytes == 0;
```

Do not change replay validation, runtime options, cursor state, or scheduler behavior.

- [x] **Step 4: Run provider GREEN and CUDA regression**

Run.

```bash
cmake --build build-cuda --parallel 8
build-cuda/test_cuda_aurora_draft artifacts/m18-top16.k3x
ctest --test-dir build-cuda --output-on-failure
```

Expected: direct resident/transient/bypass parity passes and CUDA CTest remains 23/23.

- [x] **Step 5: Commit the provider contract**

```bash
git add runtime/src/aurora.cpp tests/cuda/test_cuda_aurora_draft.cu
git commit -m "feat: accept bounded resident AURORA drafts"
```

---

### Task 2: Add draft residency CLI ownership and capability gates

**Files:**

- Modify `tests/python/test_cpp_parity.py:70-330,540-630`.
- Modify `tests/python/test_cuda_aurora_draft.py:20-135`.
- Modify `runtime/src/main.cpp:120-320,840-900`.

**Interfaces:**

- Consumes: Task 1 provider predicate.
- Produces: `--aurora-draft-resident-bytes N` and backend construction that maps zero to transient and positive to resident.

- [x] **Step 1: Write CLI preflight RED assertions**

Add exact cases.

```python
(
    ["--speculative-mode", "none", "--aurora-draft-resident-bytes", "1"],
    "speculative mode none does not accept speculative options",
),
(
    ["--speculative-mode", "aurora-replay",
     "--aurora-draft-resident-bytes", "1"],
    "AURORA draft residency requires persistent cuda-custom draft backend",
),
(
    ["--speculative-mode", "aurora-persistent",
     "--aurora-draft-resident-bytes", "1"],
    "AURORA draft residency requires persistent cuda-custom draft backend",
),
(
    ["--aurora-draft-resident-bytes", "-1"],
    "invalid AURORA draft resident byte capacity: -1",
),
```

Extend the CPU-build no-fallback test so persistent CUDA with `--aurora-draft-resident-bytes 8388608` returns code 4, emits `BACKEND_UNAVAILABLE: CUDA backend is disabled at build time`, and creates no output.

- [x] **Step 2: Extend end-to-end CUDA parity RED assertions**

For each fixed/adaptive and token/expert target case, run three draft identities: CPU, transient CUDA, and resident CUDA with 8 MiB. Assert proposal, acceptance, target tokens, final state, committed routes, and cursor counters are equal. Assert resident JSON identifies `draft_cuda_weights == "resident"` while transient remains `transient`.

- [x] **Step 3: Run and witness CLI RED**

Run.

```bash
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_cpp_parity.py \
  tests/python/test_cuda_aurora_draft.py -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected failure: the runner reports `unknown argument: --aurora-draft-resident-bytes`.

- [x] **Step 4: Parse and validate the draft-only capacity**

In `main.cpp`, add a default string `"0"`, a supplied flag, and parse through `std::from_chars` into `std::uint64_t aurora_draft_resident_bytes`. Include the supplied flag in the existing `none` and `scripted-reference` rejection sets. Add this preflight after mode and backend parsing.

```cpp
if (aurora_draft_resident_bytes_supplied &&
    (speculative_mode_name != "aurora-persistent" ||
     aurora_draft_backend_name != "cuda-custom")) {
    std::cerr << "AURORA draft residency requires persistent cuda-custom draft backend\n";
    return 2;
}
```

For `cuda-custom`, set.

```cpp
draft_backend_options.cuda_weights = aurora_draft_resident_bytes == 0
    ? k3x::CudaWeightMode::transient
    : k3x::CudaWeightMode::resident;
draft_backend_options.cuda_resident_bytes = aurora_draft_resident_bytes;
```

- [x] **Step 5: Run CLI GREEN and regressions**

Run the Step 3 commands plus CPU/CUDA CTest. Expected: prior commands remain transient, resident commands are exact, and unavailable builds do not fall back.

- [x] **Step 6: Commit CLI ownership**

```bash
git add runtime/src/main.cpp tests/python/test_cpp_parity.py \
  tests/python/test_cuda_aurora_draft.py
git commit -m "feat: configure bounded CUDA draft residency"
```

---

### Task 3: Preserve capacity and occupancy in independent telemetry

**Files:**

- Modify `tests/python/test_benchmark_schema.py:85-235`.
- Modify `tests/python/test_cuda_aurora_draft.py:85-135`.
- Modify `runtime/src/main.cpp:960-1120`.
- Modify `tools/benchmark_synthetic.py:15-190,235-720,696-950,1150-1390`.

**Interfaces:**

- Consumes: Task 2 effective draft backend options and runtime stats.
- Produces: `draft_cuda_resident_bytes`, `draft_resident_weight_bytes`, and `draft_peak_resident_weight_bytes` in runtime and benchmark JSON/CSV.

- [x] **Step 1: Write schema RED assertions**

Add zero defaults to ordinary records and CSV.

```python
assert payload["draft_cuda_resident_bytes"] == 0
assert payload["draft_resident_weight_bytes"] == 0
assert payload["draft_peak_resident_weight_bytes"] == 0
```

In the CUDA integration test assert transient values are zero. For the resident row assert configured capacity equals 8,388,608, current and peak resident bytes are positive and bounded, cache misses and hits are positive, bypasses are zero, and resident weight H2D is lower than transient weight H2D. Target residency fields remain zero.

- [x] **Step 2: Run and witness telemetry RED**

Run.

```bash
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_benchmark_schema.py::test_benchmark_json_and_csv_preserve_schema -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected failure: the three new fields are absent.

- [x] **Step 3: Serialize effective capacity and occupancy**

In `main.cpp`, serialize.

```cpp
output << ",\"draft_cuda_resident_bytes\":"
       << effective_draft_options.cuda_resident_bytes
       << ",\"draft_resident_weight_bytes\":"
       << draft_runtime.resident_weight_bytes
       << ",\"draft_peak_resident_weight_bytes\":"
       << draft_runtime.peak_resident_weight_bytes;
```

Use the existing zero `default_draft_options` and zero `BackendRuntimeStats` outside AURORA.

- [x] **Step 4: Extend benchmark invocation and aggregation**

Add defaulted dataclass fields and `aurora_draft_resident_bytes: int = 0` to `_run_process` and `benchmark_once`. Append `--aurora-draft-resident-bytes` only when the value is positive. Propagate it through every process call, option consistency tuple, argparse, and `BenchmarkRecord` construction. Treat capacity/current/peak/cache counters and H2D bytes as deterministic fields; aggregate only draft kernel timing by median.

- [x] **Step 5: Run telemetry GREEN**

Run CPU benchmark schema, CUDA draft integration, persistent AURORA, and CTest suites. Expected: resident H2D reduction and bounded occupancy are observed without target counter contamination.

- [x] **Step 6: Commit telemetry**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py \
  tests/python/test_benchmark_schema.py tests/python/test_cuda_aurora_draft.py
git commit -m "feat: report CUDA draft residency telemetry"
```

---

### Task 4: Build and measure canonical B-0020

**Files:**

- Create `tools/ablate_cuda_aurora_residency.py`.
- Create `tests/python/test_cuda_aurora_residency_ablation.py`.
- Create `results/b0020-cuda-aurora-residency-wsl/*` after the tool commit.

**Interfaces:**

- Consumes: `benchmark_once(..., aurora_draft_resident_bytes=0|8388608)`.
- Produces: nine raw JSON/CSV pairs, summary JSON/CSV, pair deltas, and canonical SHA-256 evidence.

- [x] **Step 1: Write the missing-runner RED test**

Create the test with this first line.

```python
# B-0020 transient와 resident CUDA AURORA draft의 동등성과 증거를 검증합니다.
```

Import `CASES`, `PAIRS`, and `run_ablation` from the missing module. Assert exact case order.

```python
(
    "natural-greedy",
    "transient-fixed-2-token", "resident-fixed-2-token",
    "transient-adaptive-token", "resident-adaptive-token",
    "transient-fixed-2-expert", "resident-fixed-2-expert",
    "transient-adaptive-expert", "resident-adaptive-expert",
)
```

The live one-sample test runs only with `K3X_TEST_CUDA=1`. It checks natural target parity, pair proposal/acceptance parity, transient zero residency, resident 8 MiB capacity, positive hits/misses/occupancy, zero bypass, lower resident weight H2D, all 18 raw digests, LF-only CSV, and aggregate/summary digests.

- [x] **Step 2: Run and witness missing-module RED**

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_residency_ablation.py -q
```

Expected: `ModuleNotFoundError: tools.ablate_cuda_aurora_residency`.

- [x] **Step 3: Implement the B-0020 runner**

Create the tool with this first line.

```python
# B-0020 exact bounded CUDA AURORA draft residency의 성능과 트래픽을 측정합니다.
```

Reuse the B-0019 Top-16 fixture and diagnostic parity logic. Use only the nine cases above. Define each `PAIRS` entry as `(pair_name, transient_name, resident_name)`. Reject pair mismatch before summary creation. For resident rows require capacity `8 * 1024 * 1024`, positive hits/misses/occupancy, zero bypass, bounded current/peak occupancy, and lower weight H2D.

Record.

```python
resident["paired_decode_delta_percent"] = (
    resident["decode_tokens_per_second"]
    / transient["decode_tokens_per_second"] - 1.0
) * 100.0
resident["draft_weight_h2d_reduction_percent"] = (
    1.0 - resident["draft_weight_h2d_bytes"]
    / transient["draft_weight_h2d_bytes"]
) * 100.0
resident["draft_resident_hit_rate"] = (
    resident["draft_weight_cache_hits"]
    / (resident["draft_weight_cache_hits"]
       + resident["draft_weight_cache_misses"])
)
```

- [x] **Step 4: Run focused GREEN**

Run the Step 2 command. Expected: matrix and live one-sample evidence pass.

- [x] **Step 5: Commit measurement code before measuring**

```bash
git add tools/ablate_cuda_aurora_residency.py \
  tests/python/test_cuda_aurora_residency_ablation.py
git commit -m "bench: add CUDA AURORA residency ablation"
```

- [x] **Step 6: Run canonical B-0020**

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python \
  tools/ablate_cuda_aurora_residency.py \
  --runner build-cuda/k3x_run \
  --output results/b0020-cuda-aurora-residency-wsl \
  --warmups 3 --samples 20
```

Do not claim a speedup until the summary exists and every parity, capacity, bypass, and H2D gate passes.

- [x] **Step 7: Add committed-evidence validation and commit results**

Extend the test to load committed B-0020, require 10 JSON and 10 CSV files, recompute every raw digest, summary CSV digest, aggregate digest, pair delta, H2D reduction, and hit rate. Run it and commit.

```bash
git add results/b0020-cuda-aurora-residency-wsl \
  tests/python/test_cuda_aurora_residency_ablation.py
git commit -m "bench: measure bounded CUDA AURORA residency"
```

---

### Task 5: Full verification, TITAN Ledger, review, and publication

**Files:**

- Modify `README.md`.
- Modify `ARCHITECTURE.md`.
- Modify `PERFORMANCE_MODEL.md`.
- Modify `DECISIONS.md` with D-043.
- Modify `BENCHMARKS.md` with B-0020.
- Modify `checklist.md`.
- Modify `context-notes.md`.
- Modify `PROJECT_STATE.md` last.
- Modify this plan's checkboxes.

**Interfaces:**

- Consumes: committed implementation and B-0020 evidence.
- Produces: synchronized measured state and public integration.

- [x] **Step 1: Run the complete CPU matrix**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
K3X_BUILD_DIR=build python -m pytest -q
```

- [x] **Step 2: Run liburing/direct and ASan/UBSan matrices**

```bash
cmake -S . -B build-uring -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_IO_URING=ON
cmake --build build-uring --parallel 8
ctest --test-dir build-uring --output-on-failure
K3X_BUILD_DIR=build-uring K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 \
  python -m pytest -q

cmake -S . -B build-uring-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DK3X_ENABLE_IO_URING=ON \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-uring-asan --parallel 8
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-uring-asan --output-on-failure
```

- [x] **Step 3: Run the complete CUDA matrix**

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest -q
```

- [x] **Step 4: Run Compute Sanitizer on full-fit and exact-bypass paths**

Run persistent expert-major CUDA draft once with 8 MiB and once with one byte, using the generated Top-16 artifact. Require exit zero and `ERROR SUMMARY: 0 errors` for both.

- [x] **Step 5: Update measured documents and commit**

Record exact hardware, commits, B-0020 values, raw and summary hashes, tests, accepted/rejected decision, current bottleneck, and next isolated axis. Keep `PROJECT_CHARTER.md` unchanged and update `PROJECT_STATE.md` last.

```bash
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md \
  BENCHMARKS.md checklist.md context-notes.md PROJECT_STATE.md \
  docs/superpowers/plans/2026-08-10-k3x-bounded-cuda-aurora-residency-plan.md
git commit -m "docs: publish CUDA AURORA residency evidence"
```

- [x] **Step 6: Perform final self-review**

Review `origin/main...HEAD` for default compatibility, replay CPU-only behavior, hard capacity, exact bypass, target/draft counter separation, proposal/target parity, raw-summary digests, measured-versus-proposed language, and accidental cloud/full-checkpoint work. Apply at most one focused correction batch and rerun affected checks.

- [x] **Step 7: Publish and verify public main**

Push `codex/milestone-nineteen-aurora-residency`, open a ready public PR, wait for push and PR correctness, rebase-merge, and wait for the post-merge `main` run. If publication leaves stale branch status, use one documentation-only reconciliation PR and verify its post-merge run.
