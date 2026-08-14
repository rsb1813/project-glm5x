# K3X Resident Multi-Token Multi-Expert CUDA Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an exact opt-in resident CUDA grid that evaluates equal-shaped native MXFP4 experts and token inputs with four grid-wide launches, integrates it into persistent AURORA, and measures B-0021 without changing defaults.

**Architecture:** Extend the backend with a rectangular expert/token oracle and a `resident_grid` CUDA batching identity. The CUDA path resolves all expert weights through the existing bounded resident table, launches gate/up/SiTU/down over expert-token grids, returns separate outputs for stable CPU accumulation, and uses one resolved serial fallback when any weight bypasses capacity. AURORA uses token count one; direct tests exercise token counts two and four.

**Tech Stack:** C++20, CUDA 13.3 native `sm_120`, CMake/Ninja, Python 3.12, pytest, Nsight Systems, Compute Sanitizer, JSON/CSV evidence.

## Global Constraints

- Correctness and natural target authority outrank throughput.
- No production code is written before its failing test is observed.
- CPU, `scalar`, and `grouped` defaults remain byte-for-byte compatible.
- `resident_grid` requires `cuda-custom + ffn-block + reused + resident + synchronous + fusion-none` and a positive resident byte cap.
- A hard-cap miss executes the complete request through the exact resolved serial path; CUDA failures never fall back to CPU.
- Router-slot accumulation order, proposals, acceptance, KDA/MLA state, and committed routing do not change.
- Causally dependent AURORA candidates are not claimed or implemented as concurrent token generation.
- Every new source file starts with a one-line Korean role comment.
- No full Kimi K3 checkpoint is downloaded and no paid cloud resource is provisioned.
- `PROJECT_CHARTER.md` remains unchanged; `PROJECT_STATE.md` is updated last at each ledger milestone.

---

## File structure

- Modify `runtime/include/k3x/backend.hpp` for `resident_grid`, seven counters, and the rectangular backend method.
- Modify `runtime/src/backend_cpu.cpp` for the exact expert-first/token-second oracle.
- Modify `runtime/cuda/mxfp4.cuh` and `runtime/cuda/mxfp4.cu` for device matrix descriptors and two input layouts.
- Modify `runtime/cuda/backend_cuda.cu` for full validation, resident resolution, grid execution, and resolved serial fallback.
- Modify `runtime/src/aurora.cpp` and `runtime/src/main.cpp` for closed AURORA ownership and CLI parsing.
- Modify `runtime/src/model.cpp` so Stable LatentMoE selects the grid only for the explicit identity and retains the existing accumulation loop.
- Modify `tools/benchmark_synthetic.py` for target/draft option and counter schema.
- Create `tests/cuda/test_cuda_expert_grid.cu` for the direct CUDA contract.
- Modify `tests/cpp/test_backend.cpp`, `tests/cuda/test_cuda_aurora_draft.cu`, `tests/python/test_cpp_parity.py`, `tests/python/test_cuda_aurora_draft.py`, and `tests/python/test_benchmark_schema.py`.
- Create `runtime/src/cuda_expert_grid_bench.cpp`, `tools/ablate_cuda_aurora_grid.py`, and `tests/python/test_cuda_aurora_grid_ablation.py` for B-0021.
- Modify `CMakeLists.txt` to register the CUDA grid test and benchmark.
- Modify README and TITAN Ledger documents only after measurement.

---

### Task 1: Define the rectangular backend contract and CPU oracle

**Files:**
- Modify: `runtime/include/k3x/backend.hpp:15-65,130-150`
- Modify: `runtime/src/backend_cpu.cpp:130-220`
- Modify: `tests/cpp/test_backend.cpp`

**Interfaces:**
- Produces `CudaBatchingMode::resident_grid`.
- Produces `ComputeBackend::mxfp4_situ_mlp_grid(inputs, token_count, experts, situ_beta, situ_linear, layer, phase)`.
- Produces seven zero-default `BackendRuntimeStats` fields named exactly as the design.

- [x] **Step 1: Write the CPU oracle failing tests**

Add hand-derived two-expert, two-token native payload fixtures. Assert expert-major output shape and literal values, then add independent zero-token, empty-expert, mismatched-shape, duplicate-tensor-ID, invalid-group, and multiplication-overflow cases.

```cpp
auto grid = backend->mxfp4_situ_mlp_grid(
    std::array<float, 4>{1.0F, -2.0F, 0.5F, 3.0F}, 2,
    experts, 1.0F, std::nullopt, 7, k3x::ProfilePhase::decode);
require(static_cast<bool>(grid));
require(grid.value().size() == 2);
require_close(grid.value()[0], std::array<float, 2>{
    0.8818111F, 0.26973557F});
require_close(grid.value()[1], std::array<float, 2>{
    1.7636222F, 0.53947115F});
require(!backend->mxfp4_situ_mlp_grid({}, 0, experts, 1.0F,
                                     std::nullopt, 7,
                                     k3x::ProfilePhase::decode));
```

The expected literals must be calculated from fixture codes and scales in the test, not by calling production MXFP4 helpers.

- [x] **Step 2: Run the focused test and witness RED**

Run:

```bash
cmake --build build --parallel 8
ctest --test-dir build -R '^backend$' --output-on-failure
```

Expected: compile failure because `mxfp4_situ_mlp_grid` and `resident_grid` do not exist.

- [x] **Step 3: Add the interface and minimal CPU oracle**

Add the pure virtual method and implement validation before profiler mutation. Iterate expert first, then token, and append each token output into that expert's flat vector.

```cpp
for (const auto& expert : experts) {
    std::vector<float> expert_outputs;
    expert_outputs.reserve(token_count * expert.down.rows);
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto input = inputs.subspan(token * input_width, input_width);
        auto one = mxfp4_situ_mlp_group(
            input, std::span<const Mxfp4MlpView>(&expert, 1),
            situ_beta, situ_linear, layer, phase);
        if (!one) {
            return Result<std::vector<std::vector<float>>>::failure(
                one.error(), one.message());
        }
        expert_outputs.insert(expert_outputs.end(),
                              one.value().front().begin(),
                              one.value().front().end());
    }
    outputs.push_back(std::move(expert_outputs));
}
```

- [x] **Step 4: Run focused and complete CPU tests**

Run:

```bash
cmake --build build --parallel 8
ctest --test-dir build -R '^backend$' --output-on-failure
ctest --test-dir build --output-on-failure
```

Expected: backend and all 14 CPU CTests pass.

- [x] **Step 5: Commit the contract**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp \
  tests/cpp/test_backend.cpp
git commit -m "feat: add exact expert grid oracle"
```

---

### Task 2: Add the native MXFP4 expert-grid kernel primitive

**Files:**
- Modify: `runtime/cuda/mxfp4.cuh`
- Modify: `runtime/cuda/mxfp4.cu`
- Create: `tests/cuda/test_cuda_expert_grid.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces `cuda::Mxfp4DeviceMatrix { const std::uint8_t* packed; const std::uint8_t* scales; }`.
- Produces `cuda::ExpertGridInputLayout::{shared_token_major,expert_token_major}`.
- Produces `launch_mxfp4_matvec_grid(...)` with checked expert and token counts.

- [x] **Step 1: Add a failing direct kernel test**

Create `tests/cuda/test_cuda_expert_grid.cu` with the required Korean first-line role comment. Upload two descriptor entries and literal packed/scales fixtures. Test 1×1, 1×4, 2×2, and 4×4 grids for both shared and expert-major input layouts against hand-derived literal output vectors. Also assert zero experts, zero tokens, and a grid dimension above 65,535 return `cudaErrorInvalidValue` without launching.

```cpp
require(k3x::cuda::launch_mxfp4_matvec_grid(
    d_inputs, d_descriptors, d_outputs, rows, cols,
    2, 2, k3x::cuda::ExpertGridInputLayout::shared_token_major,
    stream) == cudaSuccess);
require(cudaStreamSynchronize(stream) == cudaSuccess);
require_close(downloaded, std::array<float, 4>{
    1.0F, -2.0F, 2.0F, -4.0F});
```

- [x] **Step 2: Register the test and witness RED**

Add the executable and `add_test(NAME cuda_expert_grid ...)`, then run:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda --parallel 8
```

Expected: NVCC compile failure because the grid descriptor and launcher are missing.

- [x] **Step 3: Implement the minimal 3D grid kernel**

Use `blockIdx.z` for expert, `blockIdx.y` for token, and `blockIdx.x` for output row. Reuse the existing E2M1 decode and identical 256-thread reduction order.

```cpp
const auto expert = static_cast<std::size_t>(blockIdx.z);
const auto token = static_cast<std::size_t>(blockIdx.y);
const auto row = static_cast<std::size_t>(blockIdx.x);
const auto& matrix = descriptors[expert];
const auto input_base = layout == shared_token_major
    ? token * cols
    : (expert * token_count + token) * cols;
const auto output_base = (expert * token_count + token) * rows;
```

Reject rows, tokens, or experts that exceed CUDA grid limits before launch.

- [x] **Step 4: Run the direct CUDA test and Compute Sanitizer**

Run:

```bash
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda -R '^cuda_expert_grid$' --output-on-failure
/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 99 ./build-cuda/test_cuda_expert_grid
```

Expected: CTest passes and Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.

- [x] **Step 5: Commit the kernel primitive**

```bash
git add runtime/cuda/mxfp4.cuh runtime/cuda/mxfp4.cu \
  tests/cuda/test_cuda_expert_grid.cu CMakeLists.txt
git commit -m "feat: add native MXFP4 expert grid kernel"
```

---

### Task 3: Implement resident-grid backend execution and exact fallback

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu:1400-1990,2400-2470`
- Modify: `tests/cuda/test_cuda_expert_grid.cu`

**Interfaces:**
- Consumes the Task 1 backend contract and Task 2 launcher.
- Produces a successful four-launch resident path and one resolved serial fallback path.

- [x] **Step 1: Extend the CUDA test with backend-level RED cases**

Create two backends with identical `resident_grid` options and capacities 8 MiB and one byte. Use four distinct experts and four tokens. Require exact CPU-oracle parity, one synchronization, four grid launches, one successful grid call, sixteen expert-token pairs, positive resident hits/misses, and zero fallback for 8 MiB. For one byte, require exact parity, one fallback, zero successful grid calls, and positive bypasses.

```cpp
const auto full = full_backend->mxfp4_situ_mlp_grid(
    inputs, 4, experts, 1.0F, std::nullopt, 3,
    k3x::ProfilePhase::decode);
require_close_grid(full.value(), cpu.value());
const auto full_stats = full_backend->runtime_stats();
require(full_stats.resident_grid_calls == 1);
require(full_stats.resident_grid_kernel_launches == 4);
require(full_stats.resident_grid_expert_tokens == 16);
require(full_stats.resident_grid_fallbacks == 0);
```

Mutation caught: selecting the grid after a bypass, counting a fallback as success, reordering outputs, or launching once per expert fails this test.

- [x] **Step 2: Run and witness RED**

Run:

```bash
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda -R '^cuda_expert_grid$' --output-on-failure
```

Expected: test fails because CUDA backend returns unavailable for the new method or has zero grid counters.

- [x] **Step 3: Resolve one resident grid descriptor representation**

Inside `backend_cuda.cu`, build one private descriptor representation containing stable device packed/scales pointers and uploaded bytes. Resolve every gate/up/down acquisition before selecting the grid. Keep the descriptors alive through grid execution.

```cpp
struct ResolvedWeight {
    Mxfp4WeightView view;
    const std::uint8_t* packed{};
    const std::uint8_t* scales{};
    bool bypass{};
    std::uint64_t uploaded_bytes{};
};
```

Do not add this internal type to public headers.

- [x] **Step 4: Implement full-resident four-launch execution**

Reserve grow-only input, descriptor, gate, up, activation, and output buffers using checked products. Reuse eight backend-owned timing events. Upload descriptors once per call, launch gate/up/grid-wide SiTU/down, copy one expert-major block, synchronize once, and split outputs.

```cpp
runtime_stats_.resident_grid_calls += 1;
runtime_stats_.resident_grid_experts += expert_count;
runtime_stats_.resident_grid_tokens += token_count;
runtime_stats_.resident_grid_expert_tokens += expert_count * token_count;
runtime_stats_.resident_grid_kernel_launches += 4;
runtime_stats_.resident_grid_descriptor_h2d_bytes += descriptor_bytes;
```

- [x] **Step 5: Reuse the exact serial fallback**

If any acquisition bypasses, execute the complete request token by token through the existing exact serial FFN implementation. Preserve its kernel order, profiler behavior, and output layout rather than duplicating that machinery inside the grid method. Increment only `resident_grid_fallbacks` among grid counters, and separately account for bytes admitted during the all-resident probe before entering the serial path.

- [x] **Step 6: Run CUDA tests and sanitizer**

Run:

```bash
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda -R 'cuda_(expert_grid|residency|ffn)$' \
  --output-on-failure
/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 99 ./build-cuda/test_cuda_expert_grid
```

Expected: all focused tests pass and sanitizer reports zero errors.

- [x] **Step 7: Commit backend execution**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_expert_grid.cu
git commit -m "feat: execute resident expert grids"
```

---

### Task 4: Integrate Stable LatentMoE and AURORA ownership

**Files:**
- Modify: `runtime/src/model.cpp:780-1030`
- Modify: `runtime/src/aurora.cpp:1-60`
- Modify: `runtime/src/main.cpp:120-760`
- Modify: `tests/cuda/test_cuda_aurora_draft.cu`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_cuda_aurora_draft.py`

**Interfaces:**
- Produces `--cuda-batching resident-grid`.
- Produces `--aurora-draft-batching grouped|resident-grid`, default `grouped`.
- Preserves replay CPU-only and target/draft backend ownership.

- [x] **Step 1: Add failing CLI and provider tests**

Add cases proving `resident-grid` is rejected with zero capacity, transient weights, CPU backend, speculation `none` when supplied as a draft option, replay, prefetch, routed accumulation, or per-operation allocation. Assert return code 2 for invalid options, code 4 for CUDA unavailable in CPU builds, the exact error class, and absence of the requested output file.

Extend the CUDA provider test with CPU, resident-grouped, resident-grid, and one-byte grid fallback providers. Require equal candidates for full accept and partial rollback, equal cursor counters, positive grid calls only for full-fit, and positive fallbacks only for one-byte.

- [x] **Step 2: Run focused tests and witness RED**

Run:

```bash
K3X_BUILD_DIR=build /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cpp_parity.py tests/python/test_cuda_aurora_draft.py \
  -k 'resident_grid or draft_batching'
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_draft.py -k 'resident_grid'
```

Expected: unknown-argument or unsupported-provider failures because the new identity is not parsed or accepted.

- [x] **Step 3: Add parsing and closed capability gates**

Parse `resident-grid` into `CudaBatchingMode::resident_grid`. Track whether `--aurora-draft-batching` was explicitly supplied. Only AURORA persistent CUDA owns it. Build draft options with `grouped` by default and `resident_grid` only when requested.

Update `supported_persistent_backend()` to accept grouped transient/resident as before and resident-grid only with positive resident capacity and the exact fixed backend identity.

- [x] **Step 4: Select the grid in Stable LatentMoE**

After existing payload view construction, select:

```cpp
auto outputs = backend_.options().cuda_batching ==
        CudaBatchingMode::resident_grid
    ? backend_.mxfp4_situ_mlp_grid(
          latent, 1, expert_views, config_.situ_beta,
          config_.situ_linear, layer, phase)
    : backend_.mxfp4_situ_mlp_group(
          latent, expert_views, config_.situ_beta,
          config_.situ_linear, layer, phase);
```

Keep the existing router-slot accumulation loop unchanged.

- [x] **Step 5: Run provider, CLI, and full CPU/CUDA CTests**

Run:

```bash
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_draft.py tests/python/test_cpp_parity.py
```

Expected: exact provider/CLI parity and all CTests pass.

- [x] **Step 6: Commit runtime ownership**

```bash
git add runtime/src/model.cpp runtime/src/aurora.cpp runtime/src/main.cpp \
  tests/cuda/test_cuda_aurora_draft.cu \
  tests/python/test_cpp_parity.py tests/python/test_cuda_aurora_draft.py
git commit -m "feat: route AURORA through resident expert grids"
```

---

### Task 5: Export deterministic target and draft telemetry

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_benchmark_schema.py`
- Modify: `tests/python/test_cuda_aurora_draft.py`

**Interfaces:**
- Consumes the seven Task 1 runtime counters.
- Produces matching C++ JSON and Python JSON/CSV fields, including `draft_` copies.

- [x] **Step 1: Add failing schema tests**

Require ordinary CPU output to serialize all seven target and draft fields as zero. Require full-fit CUDA draft output to report positive draft calls/experts/tokens/expert-tokens, `kernel_launches == calls * 4`, zero draft fallback, and zero target values. Require one-byte output to report positive draft fallback and zero successful draft calls.

- [x] **Step 2: Run schema tests and witness RED**

```bash
K3X_BUILD_DIR=build /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_benchmark_schema.py tests/python/test_cuda_aurora_draft.py \
  -k 'resident_grid'
```

Expected: missing-key or dataclass-constructor failures.

- [x] **Step 3: Propagate fields without aggregation ambiguity**

Add zero-default dataclass fields. Treat all seven as deterministic identity counters: each measured sample must equal the first sample. Do not median, average, or sum across repetitions. Include `cuda_batching` and `aurora_draft_batching` in option consistency tuples.

- [x] **Step 4: Run focused and complete CPU schema tests**

```bash
K3X_BUILD_DIR=build /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_benchmark_schema.py tests/python/test_cpp_parity.py
```

Expected: pass with CUDA-only cases skipped under the CPU build.

- [x] **Step 5: Commit telemetry**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py \
  tests/python/test_benchmark_schema.py tests/python/test_cuda_aurora_draft.py
git commit -m "feat: report resident expert grid telemetry"
```

---

### Task 6: Build the direct grid benchmark and B-0021 ablation

**Files:**
- Create: `runtime/src/cuda_expert_grid_bench.cpp`
- Create: `tools/ablate_cuda_aurora_grid.py`
- Create: `tests/python/test_cuda_aurora_grid_ablation.py`
- Modify: `CMakeLists.txt`
- Modify: `tools/benchmark_synthetic.py`

**Interfaces:**
- Produces `k3x_cuda_expert_grid_bench` JSON for token/expert counts 1, 2, and 4.
- Produces nine canonical B-0021 graph rows and a checksummed summary.

- [x] **Step 1: Add Python runner and live-contract tests**

Test command construction, nine-case order, capacity and batching identity, exact-pair validation, four-launch invariant, zero-fallback full-fit gate, lower MoE launch count gate, raw digest generation, LF CSV, and committed-evidence replay. The test must execute a fake runner that returns complete real-schema records rather than asserting only source text.

- [x] **Step 2: Verify the canonical matrix before live execution**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_grid_ablation.py
```

Expected: import failure for `tools.ablate_cuda_aurora_grid`.

- [x] **Step 3: Implement the direct benchmark executable**

Follow `cuda_expert_batch_bench.cpp` option and JSON conventions. Generate or load executable synthetic expert views, invoke CPU oracle and CUDA grid for the requested rectangle, report median latency, kernel time, launch count, H2D/D2H, peak VRAM, and max absolute error. Reject unsupported counts and malformed artifacts before output creation.

- [x] **Step 4: Implement the canonical ablation runner**

Define natural greedy plus grouped/grid fixed/adaptive token/expert target pairs. Use 8 MiB, three warmups, twenty samples, and exact B-0020 graph settings. Validate pair invariants before writing raw and summary artifacts. Record speed direction without requiring improvement.

- [x] **Step 5: Run schema and one-sample live tests**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_grid_ablation.py
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_grid_ablation.py -k live
```

Expected: all fake tests and capability-enabled one-sample rows pass.

- [x] **Step 6: Commit measurement tooling**

```bash
git add runtime/src/cuda_expert_grid_bench.cpp \
  tools/ablate_cuda_aurora_grid.py \
  tests/python/test_cuda_aurora_grid_ablation.py \
  tools/benchmark_synthetic.py CMakeLists.txt
git commit -m "bench: add resident expert grid ablation"
```

---

### Task 7: Measure B-0021 and publish evidence locally

**Files:**
- Create: `results/b0021-cuda-aurora-grid-wsl/*.json`
- Create: `results/b0021-cuda-aurora-grid-wsl/*.csv`
- Modify: `tests/python/test_cuda_aurora_grid_ablation.py`

**Interfaces:**
- Consumes committed tooling and clean CUDA build.
- Produces content-addressed B-0021 raw and summary evidence.

- [x] **Step 1: Run canonical B-0021**

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python \
  tools/ablate_cuda_aurora_grid.py \
  --output-dir results/b0021-cuda-aurora-grid-wsl \
  --warmup 3 --iterations 20
```

Expected: nine graph JSON/CSV pairs plus summary JSON/CSV, exact pair parity, full-fit zero fallback, and no required throughput direction.

- [x] **Step 2: Add committed-evidence validation**

Require exact file count, recompute every raw JSON/CSV SHA-256, summary CSV digest, canonical aggregate, pair deltas, launch reductions, H2D, cache and grid counters, token/state/route parity, and LF bytes.

- [x] **Step 3: Run evidence tests**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest -q \
  tests/python/test_cuda_aurora_grid_ablation.py
```

Expected: fake, live-gated, and committed-evidence tests pass.

- [x] **Step 4: Commit measured evidence**

```bash
git add results/b0021-cuda-aurora-grid-wsl \
  tests/python/test_cuda_aurora_grid_ablation.py
git commit -m "bench: measure resident expert grid"
```

---

### Task 8: Full verification, TITAN Ledger, review, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `PROJECT_STATE.md` last
- Modify: this plan as steps complete

**Interfaces:**
- Consumes the committed implementation and B-0021 evidence.
- Produces synchronized measured state and public integration.

- [ ] **Step 1: Run the complete CPU matrix**

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
ctest --test-dir build --output-on-failure
K3X_BUILD_DIR=build /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q
```

- [ ] **Step 2: Run liburing/direct and ASan/UBSan matrices**

```bash
cmake -S . -B build-uring -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_IO_URING=ON
cmake --build build-uring --parallel 8
ctest --test-dir build-uring --output-on-failure
K3X_BUILD_DIR=build-uring K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q

cmake -S . -B build-uring-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DK3X_ENABLE_IO_URING=ON \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build build-uring-asan --parallel 8
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-uring-asan \
  --output-on-failure
```

- [ ] **Step 3: Run the complete CUDA matrix and sanitizer**

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DK3X_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda --parallel 8
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q
/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck \
  --error-exitcode 99 ./build-cuda/test_cuda_expert_grid
```

Also run full-fit AURORA grid and one-byte fallback CLI paths under Compute Sanitizer. Require exit zero and `ERROR SUMMARY: 0 errors`.

- [ ] **Step 4: Update measured documents and commit**

Record exact hardware, commits, B-0021 values, raw and summary hashes, tests, accepted/rejected default decision, measured bottleneck, and next isolated axis. Do not replace the accepted-design section until implementation and tests exist. Keep `PROJECT_CHARTER.md` unchanged and update `PROJECT_STATE.md` last.

- [ ] **Step 5: Perform final self-review**

Review `origin/main...HEAD` for default compatibility, exact fallback, stable accumulation, target/draft counter separation, causal-token wording, raw-summary parity, measured-versus-proposed language, and accidental cloud/full-checkpoint work. Apply at most one focused correction batch and rerun affected checks.

- [ ] **Step 6: Publish and verify public main**

Push `codex/milestone-twenty-resident-grid`, open a ready public PR, wait for push and PR correctness, rebase-merge, and wait for post-merge `main`. Use one documentation-only reconciliation PR if publication leaves stale branch status.
