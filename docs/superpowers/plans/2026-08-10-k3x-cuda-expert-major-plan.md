# K3X CUDA Expert-Major Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute each speculative-block expert union on RTX 5080 with one native MXFP4 weight upload per unique expert and exact token, state, and committed-route parity.

**Architecture:** Add a flat single-expert/multi-token backend primitive, implement its native MXFP4 projections with a `(row, token)` CUDA grid, and invoke it from the existing M14 stable expert-major plan. Keep CPU expert-major and token-major unchanged, restrict the first CUDA capability combination, and prove both logical Reader reuse and physical H2D reuse in B-0016.

**Tech Stack:** C++20, CUDA 13.3, native `sm_120`, CMake/Ninja, Python 3.12, pytest, Compute Sanitizer, K3X synthetic and released-dimension storage fixtures.

## Global Constraints

- Natural routing and exact native MXFP4 payloads remain unchanged.
- Token-major verification remains the default.
- The CUDA expert-major path is limited to `cuda-custom + ffn-block + reused + transient + synchronous + fusion none`, disabled L1, blocking L2, natural routing, no runtime profile observation, incremental execution, and the four-layer synthetic graph.
- Validation failures occur before Reader, output, recurrent state, profiler, runtime-counter, allocation, transfer, or kernel side effects.
- Every production behavior begins with a witnessed failing test.
- No full Kimi K3 checkpoint is downloaded and no paid cloud resource is provisioned.
- Every new C++, CUDA, or Python source file starts with a one-line Korean role comment.
- B-0016 records measurements without requiring a favorable performance direction.

## File map

- `runtime/include/k3x/backend.hpp` defines the flat batch API and CUDA batch counters.
- `runtime/src/backend_cpu.cpp` provides the exact portable batch oracle.
- `runtime/cuda/mxfp4.cuh` and `runtime/cuda/mxfp4.cu` expose and implement the two-dimensional native MXFP4 launcher.
- `runtime/cuda/backend_cuda.cu` owns validation, scratch, transfers, launches, profiling, and output reconstruction.
- `runtime/src/model.cpp` gathers and scatters M14 assignments and extends exact preflight.
- `runtime/src/main.cpp` validates CLI combinations and exports telemetry.
- `tests/cpp/test_backend.cpp`, `tests/cuda/test_cuda_mxfp4.cu`, and `tests/cuda/test_cuda_ffn.cu` cover the primitive.
- `tests/python/test_cpp_parity.py` covers CUDA runtime/CLI parity and preflight.
- `runtime/src/cuda_expert_batch_bench.cpp` isolates released-dimension scalar-versus-batch execution.
- `tools/ablate_cuda_expert_major.py` and `tests/python/test_cuda_expert_major_ablation.py` create and validate B-0016.
- `README.md`, `ARCHITECTURE.md`, `PERFORMANCE_MODEL.md`, `DECISIONS.md`, `BENCHMARKS.md`, `PROJECT_STATE.md`, `checklist.md`, and `context-notes.md` record the accepted, measured state.

---

### Task 1: Portable flat batch backend contract

**Files:**
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`

**Interfaces:**
- Consumes: `Mxfp4MlpView`, `ProfilePhase`, and the existing scalar native MXFP4 operations.
- Produces: `ComputeBackend::mxfp4_situ_mlp_batch(std::span<const float>, std::size_t, Mxfp4MlpView, float, std::optional<float>, std::uint32_t, ProfilePhase)` returning outputs in batch order.

- [ ] **Step 1: Write the failing CPU contract test**

Add calls that concatenate two known 32-element inputs, request `batch_size == 2`, and compare both returned rows with two scalar `mxfp4_situ_mlp_group` oracle calls. Add zero-batch, wrong-flat-length, and malformed-weight cases and capture the profiler event count before every rejected call.

```cpp
const auto batched = backend->mxfp4_situ_mlp_batch(
    flat_inputs, 2, expert_mlps[0], 2.0F, 1.5F, 14,
    k3x::ProfilePhase::decode);
if (!batched || batched.value().size() != 2 ||
    batched.value()[0] != scalar_first.value()[0] ||
    batched.value()[1] != scalar_second.value()[0]) return 69;
```

- [ ] **Step 2: Run the focused build and witness RED**

Run:

```bash
cmake --build build-cpu --target test_backend
```

Expected: compilation fails because `ComputeBackend` has no member named `mxfp4_situ_mlp_batch`.

- [ ] **Step 3: Add the interface and minimal CPU implementation**

Validate `batch_size != 0`, guard `batch_size * expert.gate.cols` overflow, require exact flat length, validate the entire MLP before any scalar call, then slice one row at a time and reuse the exact portable path. Do not add CUDA behavior in this task.

```cpp
for (std::size_t row = 0; row < batch_size; ++row) {
    const auto input = inputs.subspan(row * expert.gate.cols,
                                      expert.gate.cols);
    const std::array<Mxfp4MlpView, 1> one{expert};
    auto output = mxfp4_situ_mlp_group(
        input, one, situ_beta, situ_linear, layer, phase);
    if (!output) return Result<std::vector<std::vector<float>>>::failure(
        output.error(), output.message());
    outputs.push_back(std::move(output.value().front()));
}
```

- [ ] **Step 4: Run focused and full CPU GREEN**

Run:

```bash
cmake --build build-cpu --target test_backend
ctest --test-dir build-cpu -R '^backend$' --output-on-failure
ctest --test-dir build-cpu --output-on-failure
```

Expected: backend test and all CPU CTest targets pass.

- [ ] **Step 5: Commit the portable contract**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp tests/cpp/test_backend.cpp
git commit -m "feat: define batched expert FFN contract"
```

---

### Task 2: Native MXFP4 two-dimensional launcher

**Files:**
- Modify: `tests/cuda/test_cuda_mxfp4.cu`
- Modify: `runtime/cuda/mxfp4.cuh`
- Modify: `runtime/cuda/mxfp4.cu`

**Interfaces:**
- Consumes: contiguous FP32 input rows and one native E2M1 plus E8M0/32 matrix.
- Produces: `launch_mxfp4_matvec_batch(..., rows, cols, batch_size, stream)` with contiguous row-major outputs.

- [ ] **Step 1: Write the failing literal kernel test**

Reuse the existing high/low-nibble literal matrix and build three distinct input rows. Compare the batch launcher with three independent CPU decodes and require batch-size-one equality with the scalar launcher.

```cpp
require(k3x::cuda::launch_mxfp4_matvec_batch(
    device_inputs, device_packed, device_scales, device_outputs,
    rows, cols, 3, stream) == cudaSuccess);
```

- [ ] **Step 2: Build the focused CUDA target and witness RED**

Run:

```bash
cmake --build build-cuda --target test_cuda_mxfp4
```

Expected: NVCC fails because `launch_mxfp4_matvec_batch` is not declared.

- [ ] **Step 3: Generalize the scalar kernel to a batch grid**

Use `blockIdx.y` as the batch row, offset input by `token * cols`, offset output by `token * rows`, and preserve the existing per-row reduction body. Launch scalar with `dim3(rows, 1)` and batch with `dim3(rows, batch_size)`. Return `cudaErrorInvalidValue` for a zero batch before launch.

```cpp
const auto token = static_cast<std::size_t>(blockIdx.y);
input += token * cols;
output += token * rows;
```

- [ ] **Step 4: Run focused CUDA GREEN and Compute Sanitizer**

Run:

```bash
cmake --build build-cuda --target test_cuda_mxfp4
ctest --test-dir build-cuda -R '^cuda_mxfp4$' --output-on-failure
compute-sanitizer --error-exitcode=99 build-cuda/test_cuda_mxfp4
```

Expected: test passes and sanitizer reports `ERROR SUMMARY: 0 errors`.

- [ ] **Step 5: Commit the launcher**

```bash
git add runtime/cuda/mxfp4.cuh runtime/cuda/mxfp4.cu tests/cuda/test_cuda_mxfp4.cu
git commit -m "feat: batch native MXFP4 matvecs"
```

---

### Task 3: CUDA batched expert FFN and physical H2D accounting

**Files:**
- Modify: `tests/cuda/test_cuda_ffn.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/include/k3x/backend.hpp`

**Interfaces:**
- Consumes: Task 1 batch API and Task 2 batch launcher.
- Produces: exact CUDA gate/up/SiTU/down batch execution plus `batched_expert_ffn_calls` and `batched_expert_ffn_tokens` runtime counters.

- [ ] **Step 1: Write failing CUDA backend and telemetry tests**

Create one native literal expert and two distinct latent inputs. Compare CUDA batch output with a CPU backend oracle. Snapshot runtime stats and profiler size, then require exactly one batch call, two batch tokens, one expert payload worth of `weight_h2d_bytes`, one flat input H2D, and no mutation for malformed flat length.

```cpp
const auto delta = after.weight_h2d_bytes - before.weight_h2d_bytes;
require(delta == expert.gate.packed.size_bytes() +
                 expert.gate.scales.size_bytes() +
                 expert.up.packed.size_bytes() +
                 expert.up.scales.size_bytes() +
                 expert.down.packed.size_bytes() +
                 expert.down.scales.size_bytes());
```

- [ ] **Step 2: Build `test_cuda_ffn` and witness RED**

Run:

```bash
cmake --build build-cuda --target test_cuda_ffn
```

Expected: CUDA backend remains abstract or the new counters are absent.

- [ ] **Step 3: Implement the narrow synchronous transient path**

Reject every backend option outside the design combination before creating events or reserving scratch. Reserve batch-sized input, gate, up, activation, and output scratch; upload the flat activation once; upload gate, up, and down packed/scales once each on the same stream; call the batch launcher for all three projections; call `launch_situ_glu` with `batch_size * expert.gate.rows`; synchronize once; reconstruct nested outputs; and update profiler/counters only after success.

- [ ] **Step 4: Run focused CUDA GREEN and sanitizer**

Run:

```bash
cmake --build build-cuda --target test_cuda_ffn
ctest --test-dir build-cuda -R '^cuda_ffn$' --output-on-failure
compute-sanitizer --error-exitcode=99 build-cuda/test_cuda_ffn
```

Expected: oracle, byte accounting, side-effect rejection, and sanitizer gates pass.

- [ ] **Step 5: Commit the CUDA FFN primitive**

```bash
git add runtime/include/k3x/backend.hpp runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_ffn.cu
git commit -m "feat: execute batched CUDA expert FFNs"
```

---

### Task 4: Exact CUDA expert-major runtime and CLI boundary

**Files:**
- Modify: `runtime/src/model.cpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Consumes: M14 `ExpertMajorPlan` groups and Task 3 batch API.
- Produces: exact CUDA expert-major speculative generation with unchanged commit semantics and exported batch telemetry.

- [ ] **Step 1: Write failing CUDA runtime parity tests**

Add build-cuda-only tests for perfect and mixed block-2 scripts. Run greedy and token-major with the same CUDA backend identity, then require CUDA expert-major token IDs, final state SHA-256, committed routes, accepted/committed counts, and Reader union counts to match their exact references. Require positive batch counters and lower perfect-row weight H2D than token-major. Add one unsupported prefetch or resident case and assert nonzero exit plus no output artifact.

- [ ] **Step 2: Run focused pytest and witness RED**

Run:

```bash
K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python \
  -m pytest tests/python/test_cpp_parity.py -k 'cuda_expert_major' -q
```

Expected: CLI rejects expert-major because it still requires the CPU backend.

- [ ] **Step 3: Gather, batch, and scatter each expert group**

Keep the CPU inner loop unchanged. For `cuda-custom`, flatten `latents[assignment.token_index]` in stable assignment order, form one `Mxfp4MlpView` from the loaded payload, call the batch API once, validate result count and row width, and scatter by the original `token_index` and `router_slot`. Extend library and CLI preflight with the exact option conjunction from the design.

```cpp
for (const auto& assignment : group.assignments) {
    const auto& latent = latents[assignment.token_index];
    flat_inputs.insert(flat_inputs.end(), latent.begin(), latent.end());
}
```

- [ ] **Step 4: Export the new counters in JSON and CSV**

Emit `batched_expert_ffn_calls` and `batched_expert_ffn_tokens` next to existing FFN and H2D fields. Preserve field presence with zero values on CPU and token-major paths.

- [ ] **Step 5: Run focused and full CUDA GREEN**

Run:

```bash
cmake --build build-cuda --target k3x_run
K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python \
  -m pytest tests/python/test_cpp_parity.py -k 'cuda_expert_major' -q
ctest --test-dir build-cuda --output-on-failure
```

Expected: focused parity and all CUDA CTest targets pass.

- [ ] **Step 6: Commit runtime integration**

```bash
git add runtime/src/model.cpp runtime/src/main.cpp tests/python/test_cpp_parity.py
git commit -m "feat: run exact CUDA expert-major blocks"
```

---

### Task 5: B-0016 synthetic and released-dimension ablation

**Files:**
- Create: `runtime/src/cuda_expert_batch_bench.cpp`
- Create: `tools/ablate_cuda_expert_major.py`
- Create: `tests/python/test_cuda_expert_major_ablation.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: CUDA expert-major CLI telemetry, the released-dimension storage fixture, and Task 3 batch API.
- Produces: five synthetic records, scalar/batch released-dimension records, checksummed JSON/CSV summaries, and strict parity gates under `results/b0016-cuda-expert-major-wsl/`.

- [ ] **Step 1: Write failing ablation schema tests**

Require the five exact synthetic names, perfect/mixed pair parity, positive batch counters only in expert-major rows, exact raw JSON/CSV field parity, aggregate SHA-256, and released records named `scalar-2`, `batch-2`, `scalar-4`, and `batch-4`. Require batch weight H2D to equal one expert payload per iteration and scalar H2D to equal batch size times that payload.

- [ ] **Step 2: Run pytest and witness RED**

Run:

```bash
K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python \
  -m pytest tests/python/test_cuda_expert_major_ablation.py -q
```

Expected: collection fails because `tools.ablate_cuda_expert_major` does not exist.

- [ ] **Step 3: Implement the released-dimension benchmark**

Load one exact storage-slice expert, create deterministic latent rows, compute a CPU oracle, execute either repeated scalar calls or one batch call, and print one JSON record with mode, batch size, payload bytes, latency, kernel time, weight/activation H2D, D2H, peak VRAM, and maximum absolute error. Reject batch sizes outside 1 through 4 before opening the artifact.

- [ ] **Step 4: Implement the ablation runner and checksums**

Adapt the B-0015 runner without weakening its exact gates. Add the required CUDA backend flags, capture the two new counters and all H2D/D2H fields, run the four released cases, write raw JSON/CSV before summaries, store every raw SHA-256, and compute the canonical aggregate over sorted-key compact JSON records.

- [ ] **Step 5: Run focused GREEN and one-sample evidence check**

Run:

```bash
cmake --build build-cuda --target k3x_cuda_expert_batch_bench k3x_run
K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python \
  -m pytest tests/python/test_cuda_expert_major_ablation.py -q
/home/jolib/.venvs/k3x-m1/bin/python -m tools.ablate_cuda_expert_major \
  --model build-fixtures/synthetic.k3x \
  --storage-model artifacts/m12-bounded.k3x \
  --runner build-cuda/k3x_run \
  --expert-runner build-cuda/k3x_cuda_expert_batch_bench \
  --warmup 0 --iterations 1 --output /tmp/k3x-b0016-smoke
```

Expected: tests pass and the smoke summary reports exact parity plus checksum-valid records.

- [ ] **Step 6: Commit measurement tooling**

```bash
git add CMakeLists.txt runtime/src/cuda_expert_batch_bench.cpp \
  tools/ablate_cuda_expert_major.py tests/python/test_cuda_expert_major_ablation.py
git commit -m "bench: add CUDA expert-major ablation"
```

---

### Task 6: Measure, verify, document, review, and publish Milestone 15

**Files:**
- Create: `results/b0016-cuda-expert-major-wsl/*`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: Task 5 benchmark runner and every applicable build.
- Produces: measured B-0016 evidence, complete correctness/sanitizer evidence, synchronized TITAN Ledger, and public integration.

- [ ] **Step 1: Run the measured B-0016 matrix**

Run the Task 5 module with `--warmup 3 --iterations 20` and output `results/b0016-cuda-expert-major-wsl/`. Preserve unfavorable rows and never substitute theoretical values for measured fields.

- [ ] **Step 2: Cross-check raw and summary evidence independently**

Recompute every raw JSON/CSV SHA-256, canonical aggregate hash, exact token/state/route parity, weight-H2D union equations, and headline percentage deltas in a separate read-only Python invocation.

- [ ] **Step 3: Run the complete verification matrix**

Run:

```bash
ctest --test-dir build-cpu --output-on-failure
K3X_BUILD_DIR=build-cpu /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python -q
ctest --test-dir build-uring --output-on-failure
K3X_BUILD_DIR=build-uring K3X_TEST_IO_URING=1 K3X_TEST_DIRECT_IO=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python -q
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python -m pytest tests/python -q
ctest --test-dir build-uring-asan --output-on-failure
```

Run Compute Sanitizer with `--error-exitcode=99` over `test_cuda_mxfp4`, `test_cuda_ffn`, the released batch benchmark, and perfect/mixed CUDA expert-major CLI invocations. Record only executions that contain instrumented CUDA API calls.

- [ ] **Step 4: Update all milestone documents from measured evidence**

Record D-038, B-0016, the exact accepted CUDA boundary, all measured fields, caveats, hashes, current bottleneck, defaults, known failures, and next concrete cost-aware speculation task. Update `PROJECT_STATE.md` only after every other document is synchronized.

- [ ] **Step 5: Self-review the diff and fix Critical or Important issues once**

Check option preflight before side effects, overflow and shape validation, stable gather/scatter order, profiler and counter mutation, batch-one scalar parity, benchmark caller/schema parity, raw checksums, defaults, and documentation claim strength. Run `git diff --check` after fixes.

- [ ] **Step 6: Commit the measured milestone**

Split results/tooling and final ledger updates if they are independently reviewable; every commit must describe one logical change.

- [ ] **Step 7: Push, open a ready PR, merge after CI, and verify public main**

Push `codex/milestone-fifteen-cuda-expert-major`, open a ready PR against `main`, require branch and PR correctness, rebase-merge, fast-forward both local `main` and the next feature branch, and require post-merge `main` correctness before reporting publication.
