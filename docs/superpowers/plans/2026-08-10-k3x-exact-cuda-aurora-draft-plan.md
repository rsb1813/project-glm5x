# Exact CUDA AURORA Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, separately measured `cuda-custom` backend for persistent AURORA drafting while retaining CPU target verification, CPU replay, and CPU persistent defaults.

**Architecture:** The CLI owns an independent draft Reader, profiler, backend, and RuntimeSession. `aurora-persistent` may select either the existing CPU backend or one fixed FP32 transient CUDA identity; replay remains CPU-only. Draft profiler, memory, and runtime counters are serialized separately from target counters and B-0019 compares matched CPU/CUDA draft rows without changing target execution.

**Tech Stack:** C++20, CUDA 13.3 native `sm_120`, Python 3.12, CMake/Ninja, pytest, Compute Sanitizer, K3X synthetic Top-16 fixtures.

## Global Constraints

- Correctness precedes performance; target tokens, final state, and committed routing remain authoritative.
- Default speculation remains `none`; default AURORA draft backend remains `cpu`.
- `aurora-replay` remains CPU-only and available as the candidate oracle.
- CUDA draft identity is exactly FP32, reused allocation, transient weights, grouped batching, FFN-block boundary, synchronous transfer, fusion none, and zero resident/pinned capacity.
- Target and draft Reader/backend/profiler counters never mix.
- No reduced precision, CUDA residency, new kernels, full checkpoint, paid cloud resource, or default-mode change belongs in this milestone.
- Every source change follows witnessed RED then minimal GREEN.
- New source files begin with a one-line Korean role comment.

---

## File structure

- Modify `runtime/src/main.cpp` for draft-backend parsing, preflight, backend ownership, and separated JSON telemetry.
- Modify `runtime/src/aurora.cpp` to keep replay CPU-only while accepting only the canonical CUDA identity for persistent drafting.
- Modify `tools/benchmark_synthetic.py` to carry draft-backend selection and draft CUDA metrics through `BenchmarkRecord`, process invocation, aggregation, JSON, and CSV.
- Modify `tests/python/test_cpp_parity.py` for CLI compatibility and preflight behavior.
- Modify `tests/python/test_benchmark_schema.py` for zero-default and explicit draft CUDA schema behavior.
- Create `tests/cuda/test_cuda_aurora_draft.cu` for direct CPU/CUDA provider proposal and lifecycle parity on one generated artifact.
- Modify `CMakeLists.txt` to build the CUDA provider parity executable only in CUDA builds.
- Create `tests/python/test_cuda_aurora_draft.py` to generate the Top-16 artifact and invoke the CUDA parity executable and CLI integration path.
- Create `tools/ablate_cuda_aurora_draft.py` for the nine-row B-0019 matrix and evidence publication.
- Create `tests/python/test_cuda_aurora_draft_ablation.py` for matrix, parity, digest, and committed-evidence validation.
- Update `README.md`, `ARCHITECTURE.md`, `PERFORMANCE_MODEL.md`, `DECISIONS.md`, `BENCHMARKS.md`, `checklist.md`, `context-notes.md`, `PROJECT_STATE.md`, and this plan after measurement.

---

### Task 1: Canonical CUDA persistent provider contract

**Files:**

- Create `tests/cuda/test_cuda_aurora_draft.cu`.
- Modify `CMakeLists.txt:105-180`.
- Modify `runtime/src/aurora.cpp:15-45,184-205`.

**Interfaces:**

- Consumes: `ComputeBackend`, `BackendOptions`, and `AuroraPersistentDraftProvider::create`.
- Produces: a persistent provider that accepts CPU or the exact fixed CUDA identity while replay remains CPU-only.

- [x] **Step 1: Write the direct provider parity test**

Create `tests/cuda/test_cuda_aurora_draft.cu` beginning with.

```cpp
// CPU와 CUDA persistent AURORA draft proposal 및 commit lifecycle의 동등성을 검증합니다.
```

The executable accepts one Top-16 artifact path, creates independent CPU and CUDA Readers/backends/sessions/providers, requests fixed block-2 candidates, and asserts literal lifecycle invariants.

```cpp
options.kind = k3x::BackendKind::cuda_custom;
options.dense_precision = k3x::DensePrecision::fp32;
options.cuda_allocation = k3x::CudaAllocationMode::reused;
options.cuda_weights = k3x::CudaWeightMode::transient;
options.cuda_batching = k3x::CudaBatchingMode::grouped;
options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
options.cuda_transfer = k3x::CudaTransferMode::synchronous;
options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
```

Compare the complete candidate vectors before any update, after an all-accepted update, and after a one-token-accepted rollback update. Assert CPU and CUDA provider stats have equal proposal/candidate/context/forward/rollback/crop counts.

Extend `tests/python/test_persistent_aurora_runtime.py` with a CUDA-build-only wrapper that creates the existing 24-expert natural-Top-16 artifact and invokes `test_cuda_aurora_draft`.

- [x] **Step 2: Build CUDA and witness RED**

Run.

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda -j 8
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_persistent_aurora_runtime.py -q
```

Expected failure: missing `test_cuda_aurora_draft` executable and CPU-only provider validation.

- [x] **Step 3: Add the CUDA test target and provider validation**

Add the CUDA-only executable to `CMakeLists.txt` without registering it as a no-argument CTest. Split the shared validator into replay CPU validation and persistent validation. The persistent CUDA predicate must compare every fixed option and both zero capacities; any mismatch returns `unsupported AURORA persistent runtime combination` before cursor creation.

- [x] **Step 4: Run GREEN and regression tests**

Run.

```bash
cmake --build build-cuda -j 8
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_persistent_aurora_runtime.py \
  tests/python/test_aurora_runtime.py -q
ctest --test-dir build-cuda --output-on-failure
```

Expected: proposal/lifecycle parity passes and CUDA CTest passes without changing CPU defaults.

- [x] **Step 5: Commit the provider contract**

```bash
git add CMakeLists.txt runtime/src/aurora.cpp \
  tests/cuda/test_cuda_aurora_draft.cu \
  tests/python/test_persistent_aurora_runtime.py
git commit -m "feat: accept canonical CUDA AURORA drafts"
```

---

### Task 2: Draft-backend CLI ownership and fail-closed execution

**Files:**

- Create `tests/python/test_cuda_aurora_draft.py`.
- Modify `tests/python/test_cpp_parity.py:879-1055`.
- Modify `runtime/src/main.cpp:100-330,820-990`.

**Interfaces:**

- Consumes: Task 1's canonical provider contract and `make_cuda_backend`.
- Produces: parsed `aurora_draft_backend_name`, independent backend ownership, and JSON field `aurora_draft_backend` with values `none|cpu|cuda-custom`.

- [x] **Step 1: Write the failing CLI and integration tests**

Extend `test_cpp_runner_rejects_invalid_speculative_options` with literal cases.

```python
(
    ["--aurora-draft-backend", "cuda-custom"],
    "speculative mode none does not accept speculative options",
),
(
    [
        "--speculative-mode", "aurora-replay",
        "--speculative-block-size", "2",
        "--aurora-draft-k", "4",
        "--aurora-draft-backend", "cuda-custom",
    ],
    "AURORA replay requires CPU draft backend",
),
(
    [
        "--speculative-mode", "aurora-persistent",
        "--speculative-block-size", "2",
        "--aurora-draft-k", "4",
        "--aurora-draft-backend", "warp",
    ],
    "unknown AURORA draft backend: warp",
),
```

Create `tests/python/test_cuda_aurora_draft.py` beginning with.

```python
# Top-16 합성 artifact에서 CUDA persistent AURORA draft CLI 경계를 검증합니다.
```

It skips unless `K3X_BUILD_DIR` is `build-cuda`, runs CPU-target persistent fixed and adaptive rows with CPU and CUDA draft backends, and asserts equal target tokens, state, routes, proposal/acceptance counts, and cursor lifecycle counters. Existing CPU persistent integration asserts omitted selection reports `cpu`; ordinary greedy reports `none`.

- [x] **Step 2: Build and witness RED**

```bash
cmake --build build -j 8
cmake --build build-cuda -j 8
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_cpp_parity.py::test_cpp_runner_rejects_invalid_speculative_options \
  tests/python/test_cpp_parity.py -k 'aurora and not cuda' -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected failure: unknown CLI option and missing `aurora_draft_backend` output.

- [x] **Step 3: Implement parser, preflight, and backend ownership**

Add state beside the existing AURORA options.

```cpp
std::string aurora_draft_backend_name = "cpu";
bool aurora_draft_backend_supplied = false;
```

Parse and validate `cpu|cuda-custom`, include explicit selection in non-AURORA rejection, and reject replay plus CUDA. When persistent CUDA is selected, construct the exact fixed options from the design and call `make_cuda_backend(options, &aurora_profiler)`. Propagate creation failure without CPU fallback. Serialize `none` outside AURORA and the effective value inside AURORA.

- [x] **Step 4: Run GREEN and compatibility tests**

```bash
cmake --build build -j 8
cmake --build build-cuda -j 8
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_cpp_parity.py::test_cpp_runner_rejects_invalid_speculative_options \
  tests/python/test_persistent_aurora_runtime.py -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py \
  tests/python/test_persistent_aurora_runtime.py -q
ctest --test-dir build --output-on-failure
ctest --test-dir build-cuda --output-on-failure
```

- [x] **Step 5: Commit CLI execution**

```bash
git add runtime/src/main.cpp tests/python/test_cpp_parity.py \
  tests/python/test_cuda_aurora_draft.py \
  tests/python/test_persistent_aurora_runtime.py
git commit -m "feat: execute persistent AURORA drafts on CUDA"
```

---

### Task 3: Separate draft CUDA telemetry and benchmark schema

**Files:**

- Modify `tests/python/test_benchmark_schema.py:1-240`.
- Modify `tests/python/test_cuda_aurora_draft.py`.
- Modify `runtime/src/main.cpp:900-1120`.
- Modify `tools/benchmark_synthetic.py:15-210,250-330,400-1100,1180-1280`.

**Interfaces:**

- Consumes: Task 2's independent `aurora_profiler` and `aurora_backend`.
- Produces: the eighteen draft identity/telemetry fields specified by the design in CLI JSON and `BenchmarkRecord` JSON/CSV.

- [x] **Step 1: Write schema and runtime RED assertions**

Extend `_record()` expectations so ordinary records contain literal defaults.

```python
assert payload["aurora_draft_backend"] == "none"
assert payload["draft_device"] == "CPU"
assert payload["draft_cuda_allocation"] == "per-operation"
assert payload["draft_cuda_weights"] == "transient"
assert payload["draft_cuda_batching"] == "scalar"
assert payload["draft_cuda_boundary"] == "operation"
assert payload["draft_cuda_transfer"] == "synchronous"
assert payload["draft_cuda_moe_fusion"] == "none"
assert payload["draft_kernel_nanoseconds"] == 0
assert payload["draft_weight_h2d_bytes"] == 0
assert payload["draft_peak_vram_bytes"] == 0
```

In the CUDA integration test assert draft kernel/H2D/VRAM/allocation/synchronization are positive while target `kernel_nanoseconds`, `host_to_device_bytes`, and `peak_vram_bytes` remain zero for a CPU target.

- [x] **Step 2: Run and witness RED**

Run.

```bash
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_benchmark_schema.py::test_benchmark_json_and_csv_preserve_schema -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected failure: missing draft identity and telemetry fields.

- [x] **Step 3: Serialize independent backend snapshots**

After generation, snapshot `aurora_profiler.summary()`, `aurora_backend->memory_stats()`, and `aurora_backend->runtime_stats()` when AURORA is active. Otherwise use zero structs. Serialize the exact effective draft options and profiler/runtime counters under the design field names. `draft_device` is `CPU` outside CUDA and `aurora_backend->device_name()` for AURORA.

- [x] **Step 4: Extend `BenchmarkRecord` and all process calls**

Add defaulted dataclass fields, add `aurora_draft_backend: str = "cpu"` to `_run_process` and `benchmark_once`, append the CLI option only for AURORA modes, include all new deterministic counters in the consistency gate, and aggregate timing fields by median while keeping byte/count fields exact. Add the argparse choice `cpu|cuda-custom`.

- [x] **Step 5: Run GREEN and the whole benchmark-schema suite**

Run.

```bash
cmake --build build -j 8
cmake --build build-cuda -j 8
K3X_BUILD_DIR=build python -m pytest tests/python/test_benchmark_schema.py -q
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft.py \
  tests/python/test_persistent_aurora_ablation.py \
  tests/python/test_aurora_replay_ablation.py -q
```

Expected: schema defaults, CPU compatibility, and separated CUDA counters pass.

- [x] **Step 6: Commit telemetry**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py \
  tests/python/test_benchmark_schema.py \
  tests/python/test_cuda_aurora_draft.py
git commit -m "feat: separate CUDA draft telemetry"
```

---

### Task 4: B-0019 exact CPU/CUDA draft ablation

**Files:**

- Create `tools/ablate_cuda_aurora_draft.py`.
- Create `tests/python/test_cuda_aurora_draft_ablation.py`.
- Create `results/b0019-cuda-aurora-draft-wsl/*` by running the committed tool.

**Interfaces:**

- Consumes: `benchmark_once(..., aurora_draft_backend="cpu|cuda-custom")` from Task 3.
- Produces: nine raw JSON/CSV pairs, `summary.json`, `summary.csv`, exact pair deltas, and canonical SHA-256 records.

- [x] **Step 1: Write the missing-runner RED tests**

Create `tests/python/test_cuda_aurora_draft_ablation.py` beginning with.

```python
# B-0019 CPU와 CUDA persistent AURORA draft 배치의 동등성과 증거를 검증합니다.
```

Assert the exact case order.

```python
[
    "natural-greedy",
    "cpu-fixed-2-token", "cuda-fixed-2-token",
    "cpu-adaptive-token", "cuda-adaptive-token",
    "cpu-fixed-2-expert", "cuda-fixed-2-expert",
    "cpu-adaptive-expert", "cuda-adaptive-expert",
]
```

For a zero-warmup/one-sample run, assert all rows preserve natural target tokens/state/routes, each CPU/CUDA pair has equal proposed/accepted/committed counts and acceptance, CPU draft CUDA counters are zero, CUDA draft counters are positive, target CUDA counters stay zero, all eighteen raw digests match, summary CSV is LF, and the canonical aggregate matches.

- [x] **Step 2: Run and witness RED**

Run.

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft_ablation.py -q
```

Expected failure: `ModuleNotFoundError: tools.ablate_cuda_aurora_draft`.

- [x] **Step 3: Implement the minimal runner**

Create `tools/ablate_cuda_aurora_draft.py` beginning with.

```python
# B-0019 exact CPU와 transient CUDA AURORA draft 실행을 쌍대로 측정합니다.
```

Reuse the B-0018 Top-16 fixture and diagnostic parity logic. Add only the nine specified cases. Reject any pair mismatch before writing summary files. Record paired decode delta, draft kernel delta, H2D bytes, and peak draft VRAM without interpreting logical Reader bytes as physical NVMe.

- [x] **Step 4: Run focused GREEN**

Run.

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft_ablation.py -q
```

Expected: matrix, live one-sample parity, and digest tests pass.

- [x] **Step 5: Commit the measurement code before measuring**

```bash
git add tools/ablate_cuda_aurora_draft.py \
  tests/python/test_cuda_aurora_draft_ablation.py
git commit -m "bench: add exact CUDA AURORA draft ablation"
```

- [x] **Step 6: Run canonical B-0019**

Run on WSL2 with the RTX 5080.

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python \
  tools/ablate_cuda_aurora_draft.py \
  --runner build-cuda/k3x_run \
  --output results/b0019-cuda-aurora-draft-wsl \
  --warmups 3 --samples 20
```

Do not claim a speedup until `summary.json` exists and all pair gates pass.

- [x] **Step 7: Add committed-evidence validation and commit results**

Extend the test to load `results/b0019-cuda-aurora-draft-wsl/`, recompute all raw/summary/aggregate digests and headline deltas, and assert 9 records plus 20 raw files. Run it, then commit.

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest \
  tests/python/test_cuda_aurora_draft_ablation.py -q
git add results/b0019-cuda-aurora-draft-wsl \
  tests/python/test_cuda_aurora_draft_ablation.py
git commit -m "bench: measure exact CUDA AURORA drafting"
```

---

### Task 5: Full verification, TITAN Ledger, review, and publication

**Files:**

- Modify `README.md`.
- Modify `ARCHITECTURE.md`.
- Modify `PERFORMANCE_MODEL.md`.
- Modify `DECISIONS.md` with D-042.
- Modify `BENCHMARKS.md` with B-0019.
- Modify `checklist.md`.
- Modify `context-notes.md`.
- Modify `PROJECT_STATE.md` last.
- Modify this plan's checkboxes.

**Interfaces:**

- Consumes: committed implementation and B-0019 artifacts.
- Produces: reproducible verification evidence, synchronized current state, and public integration.

- [x] **Step 1: Rebuild and run the full CPU matrix**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
ctest --test-dir build --output-on-failure
K3X_BUILD_DIR=build python -m pytest -q
```

- [x] **Step 2: Run liburing/direct and ASan/UBSan matrices**

```bash
cmake -S . -B build-uring -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_IO_URING=ON
cmake --build build-uring -j 8
ctest --test-dir build-uring --output-on-failure
K3X_BUILD_DIR=build-uring K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 \
  python -m pytest -q

cmake -S . -B build-uring-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DK3X_ENABLE_IO_URING=ON \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-uring-asan -j 8
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-uring-asan --output-on-failure
```

- [x] **Step 3: Run the full CUDA matrix**

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda -j 8
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest -q
```

- [x] **Step 4: Run Compute Sanitizer on the new draft path**

Create an ignored synthetic artifact with the existing Python converter, then run.

```bash
compute-sanitizer --tool memcheck --error-exitcode=99 \
  build-cuda/k3x_run \
  --model build-cuda/m18-top16.k3x \
  --prompt-ids 1,7,3,9 --generate 6 --mode incremental \
  --backend cpu --diagnostics true \
  --speculative-mode aurora-persistent \
  --speculative-verification expert-major \
  --speculative-block-size 2 --aurora-draft-k 4 \
  --aurora-block-policy fixed \
  --aurora-draft-backend cuda-custom \
  --json build-cuda/m18-sanitizer.json
```

Expected: `ERROR SUMMARY: 0 errors` and exit code zero.

- [x] **Step 5: Update measured documents and commit**

Record exact commits, hardware, B-0019 values, hashes, tests, caveats, accepted/rejected decision, and the next measured bottleneck. Keep `PROJECT_CHARTER.md` unchanged. Update `PROJECT_STATE.md` last.

```bash
git diff --check
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md \
  BENCHMARKS.md checklist.md context-notes.md PROJECT_STATE.md \
  docs/superpowers/plans/2026-08-10-k3x-exact-cuda-aurora-draft-plan.md
git commit -m "docs: publish exact CUDA AURORA evidence"
```

- [x] **Step 6: Perform final self-review**

Review `origin/main...HEAD` for CPU default compatibility, replay CPU-only behavior, canonical CUDA identity, target/draft counter separation, CUDA failure propagation, proposal/target parity, raw-summary digests, measured-versus-proposed language, and accidental full-checkpoint/cloud work. Apply at most one focused correction batch and rerun affected tests.

- [x] **Step 7: Publish and verify public main**

Push `codex/milestone-eighteen-cuda-aurora-draft`, open a ready public PR, wait for push and PR correctness, rebase-merge, and wait for the post-merge `main` run. If publication leaves a stale active-branch statement, use one small reconciliation PR and verify its post-merge `main` run.
