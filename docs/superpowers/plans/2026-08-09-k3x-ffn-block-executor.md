# K3X CUDA FFN Block Executor Implementation Plan

> **Execution mode:** The primary agent executes this connected plan task-by-task with review checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute complete dense/shared and native MXFP4 expert FFN chains inside one CUDA boundary so intermediate gate, up, and SiTU-GLU activations remain on device.

**Architecture:** Add an explicit `operation|ffn-block` boundary switch while preserving every Milestone 2 option and default. The `cuda-custom` block path reuses the existing cuBLASLt plans, native MXFP4 kernels, tensor-keyed residency table, tracked scratch memory, stream, and profiler. CPU block methods provide literal composition oracles, but runtime validation permits `ffn-block` only for `cuda-custom`.

**Tech Stack:** C++20, CUDA 13.3, cuBLASLt, native `sm_120`, Python 3.12, pytest, CTest, Compute Sanitizer, JSON/CSV benchmark artifacts.

## Global Constraints

- Correctness precedes throughput. Preserve exact tokens `[43, 32, 28, 49, 9, 28]` and declared layer, logit, and state tolerances.
- `operation + per-operation + transient + scalar` remains the default reference path.
- `ffn-block` is supported only by `cuda-custom`; unsupported combinations fail before model execution and never fall back to CPU.
- Native K3 MXFP4 remains low-nibble-first E2M1 with one E8M0 scale per 32 values and is never repacked or requantized.
- Existing tensor-ID, representation, shape, group-size keys and the hard resident-byte capacity remain authoritative.
- The SiTU CUDA translation unit must not use `--use_fast_math`.
- KDA, MLA, Attention Residual, RMSNorm, routing, routed mixing, recurrent state, residual addition, and token selection remain on CPU.
- CPU-only builds retain no CUDA or cuBLAS dependency.
- New source files start with a one-line Korean role comment.
- Every shell command shown below is executed through the required `rtk` prefix, even where a code block shows only the underlying command for readability.
- No full Kimi K3 weights, cloud resources, async storage, eviction, adaptive Top-K, speculation, proxy, or pruning enter this plan.
- Every task ends with tests and one semantic commit. Do not push until the final Milestone 3 review gate.

---

### Task 1: Boundary option and serialized runtime contract

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Produces: `CudaBoundaryMode`, `BackendOptions::cuda_boundary`, `BackendRuntimeStats::ffn_block_calls`, and `BackendRuntimeStats::ffn_block_experts`.
- Produces: CLI and benchmark fields `cuda_boundary`, `ffn_block_calls`, and `ffn_block_experts`.
- Consumes: all existing Milestone 2 options without changing defaults.

- [x] **Step 1: Write failing native and CLI contract tests**

Add to `tests/cpp/test_backend.cpp`.

```cpp
if (defaults.cuda_boundary != k3x::CudaBoundaryMode::operation) return 58;
if (runtime_stats.ffn_block_calls != 0) return 59;
if (runtime_stats.ffn_block_experts != 0) return 60;
```

Add invalid and unsupported cases to `tests/python/test_cpp_parity.py`.

```python
(["--cuda-boundary", "layer"], "unknown CUDA boundary mode: layer")
```

Require `--backend cpu --cuda-boundary ffn-block` and `--backend cuda-dense --cuda-boundary ffn-block` to exit 2 with `ffn-block boundary requires cuda-custom`.

Extend `_record()` and JSON/CSV assertions in `tests/python/test_benchmark_schema.py` with `cuda_boundary="operation"`, `ffn_block_calls=0`, and `ffn_block_experts=0`.

- [x] **Step 2: Run tests and verify RED**

```bash
cmake --build build-linux -j2
ctest --test-dir build-linux -R backend --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-linux python -m pytest -q \
  tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
```

Expected: compilation or collection fails because the enum and serialized fields do not exist.

- [x] **Step 3: Implement option parsing, validation, and serialization**

Add to `runtime/include/k3x/backend.hpp`.

```cpp
enum class CudaBoundaryMode { operation, ffn_block };
```

Add `cuda_boundary{CudaBoundaryMode::operation}` to `BackendOptions` and the two zero-initialized counters to `BackendRuntimeStats`.

Parse `--cuda-boundary operation|ffn-block` in `runtime/src/main.cpp` and validate before backend construction.

```cpp
if (backend_options.cuda_boundary == k3x::CudaBoundaryMode::ffn_block &&
    backend_options.kind != k3x::BackendKind::cuda_custom) {
    std::cerr << "ffn-block boundary requires cuda-custom\n";
    return 2;
}
```

Serialize the effective option and counters. Extend `BenchmarkRecord`, `_run_process()`, `benchmark_once()`, deterministic-field validation, and benchmark CLI forwarding.

- [x] **Step 4: Verify CPU and CUDA schema paths**

Run the Step 2 commands with `K3X_BUILD_DIR=build-linux` and `build-cuda`. Expected: all targeted tests pass and unsupported modes fail before inference.

- [x] **Step 5: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/main.cpp \
  tools/benchmark_synthetic.py tests/cpp/test_backend.cpp \
  tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
git commit -m "feat: define CUDA FFN block boundary"
```

### Task 2: Block views and portable CPU composition oracle

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `tests/cpp/test_backend.cpp`

**Interfaces:**
- Produces: `DenseMlpView`, `Mxfp4MlpView`, `ComputeBackend::dense_situ_mlp()`, and `ComputeBackend::mxfp4_situ_mlp_group()`.
- Consumes: current dense, native MXFP4, and `situ_glu()` oracles.

- [x] **Step 1: Write failing dense and MXFP4 block tests**

Create a literal dense triplet and compare the new method with scalar gate, up, portable `situ_glu()`, and scalar down.

```cpp
const k3x::DenseMlpView dense_mlp{
    {301, gate_weight, 2, 3},
    {302, up_weight, 2, 3},
    {303, down_weight, 2, 2},
};
const auto dense_block = backend->dense_situ_mlp(
    dense_input, dense_mlp, 2.0F, 1.5F, 13,
    k3x::ProfilePhase::decode);
```

Create two native MXFP4 triplets and require ordered outputs from `mxfp4_situ_mlp_group()`. Add an invalid second down view and assert rejection before profiler events are appended.

- [x] **Step 2: Run the CPU backend test and verify RED**

```bash
cmake --build build-linux -j2
ctest --test-dir build-linux -R backend --output-on-failure
```

Expected: compilation fails because the block types and methods are missing.

- [x] **Step 3: Define exact interfaces**

Add `<optional>` and these view types to `backend.hpp`.

```cpp
struct DenseMlpView {
    DenseWeightView gate;
    DenseWeightView up;
    DenseWeightView down;
};

struct Mxfp4MlpView {
    Mxfp4WeightView gate;
    Mxfp4WeightView up;
    Mxfp4WeightView down;
};
```

Add the following pure virtual methods.

```cpp
virtual Result<std::vector<float>> dense_situ_mlp(
    std::span<const float> input, DenseMlpView weights,
    float situ_beta, std::optional<float> situ_linear,
    std::uint32_t layer, ProfilePhase phase) = 0;

virtual Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
    std::span<const float> input, std::span<const Mxfp4MlpView> experts,
    float situ_beta, std::optional<float> situ_linear,
    std::uint32_t layer, ProfilePhase phase) = 0;
```

- [x] **Step 4: Implement deterministic CPU composition**

Preflight all shapes before work. Dense calls scalar gate/up, portable `situ_glu()`, then scalar down. MXFP4 preflights every triplet, then performs the same scalar sequence in request order. Return the first typed error and never retain a partial output group.

- [x] **Step 5: Verify the complete CPU suite**

```bash
cmake --build build-linux -j2
ctest --test-dir build-linux --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-linux python -m pytest -q
```

- [x] **Step 6: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp \
  tests/cpp/test_backend.cpp
git commit -m "feat: add portable FFN block oracle"
```

### Task 3: Strict CUDA SiTU-GLU primitive

**Files:**
- Create: `runtime/cuda/situ.cuh`
- Create: `runtime/cuda/situ.cu`
- Create: `tests/cuda/test_cuda_situ.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `cuda::launch_situ_glu()` with FP32 or BF16-rounded device output.
- Consumes: device-resident FP32 gate and up vectors.

- [x] **Step 1: Write the failing literal CUDA test**

Start the test with.

```cpp
// GPU SiTU-GLU의 엄격한 FP32 계산과 BF16 반올림 출력을 검증합니다.
```

Test positive, negative, zero, saturated, and non-integer values. Compute FP32 expected output with portable `situ_glu()`. For BF16 output, independently round each expected FP32 value with the RNE bit oracle from `test_cuda_dense.cu`. Place guards after the output extent.

- [x] **Step 2: Add CMake targets and verify RED**

Add `runtime/cuda/situ.cu` to `k3x_runtime`, add `test_cuda_situ`, then run.

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_situ --output-on-failure
```

Expected: build fails because the launch contract is missing.

- [x] **Step 3: Implement the strict kernel**

`runtime/cuda/situ.cuh` starts with.

```cpp
// CUDA FFN block에서 사용하는 SiTU-GLU launch 계약을 선언합니다.
#pragma once
```

Declare.

```cpp
cudaError_t launch_situ_glu(
    const float* gate, const float* up, void* output,
    std::size_t count, float beta, bool has_linear_beta,
    float linear_beta, bool output_bf16, cudaStream_t stream);
```

`runtime/cuda/situ.cu` starts with.

```cpp
// gate와 up을 device에서 결합해 strict SiTU-GLU activation을 계산합니다.
```

Use one thread per element. Preserve the expression order from `ops.cpp`, using `expf` and `tanhf`. Compute FP32 first, then write FP32 or `__float2bfloat16_rn`. Reject null pointers, zero count, non-finite or non-positive beta, and invalid optional linear beta.

- [x] **Step 4: Verify tolerance, guards, sanitizer, and flags**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_situ --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_situ
cmake --build build-cuda --verbose 2>&1 | tee /tmp/k3x-m3-build.log
! grep -F -- "--use_fast_math" /tmp/k3x-m3-build.log
```

Expected: FP32 is within `1e-6`, BF16 bits match the oracle, guards stay unchanged, and sanitizer reports zero errors.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/cuda/situ.cuh runtime/cuda/situ.cu \
  tests/cuda/test_cuda_situ.cu
git commit -m "feat: add strict CUDA SiTU activation"
```

### Task 4: Dense and shared FFN CUDA block

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Create: `tests/cuda/test_cuda_ffn.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Implements: `CudaBackend::dense_situ_mlp()`.
- Produces: one activation upload, one final output download, one synchronization, and one successful block count.

- [x] **Step 1: Write failing FP32, BF16, residency, and validation tests**

Start the new test with.

```cpp
// cuBLASLt와 strict SiTU를 연결한 dense FFN block의 전송과 출력을 검증합니다.
```

Use a `3 → 4 → 2` literal block. Compare FP32 with the CPU block oracle within `1e-5`. Compare BF16 with independently rounded inputs and weights within `2e-2`.

Warm all three resident weights and require one input H2D, zero new weight H2D, one final D2H, one synchronization, `ffn_block_calls + 1`, and unchanged expert count. Reject mismatched gate/up input widths, unequal intermediate rows, invalid down columns, zero dimensions, and metadata collisions without successful counter increments.

- [x] **Step 2: Add target and verify RED**

Add `test_cuda_ffn` to `CMakeLists.txt` and run.

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_ffn --output-on-failure
```

- [x] **Step 3: Implement one-stream dense block execution**

Preflight the triplet. Convert input and weights to BF16 only when requested. Resolve all three weights through the existing resident table before activation upload. Reuse one transient weight slot safely through same-stream ordering.

Reserve checked arena offsets for gate FP32, up FP32, activated FP32 or BF16, and final FP32 output. Enqueue gate, up, strict SiTU, down, and final D2H, then synchronize exactly once. Read all CUDA events after that sync. Record actual split H2D, one D2H, three matvec events, and one SiTU event. Increment block counters only after timing extraction succeeds.

- [x] **Step 4: Verify block behavior and sanitizer**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_(dense|ffn)" --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_ffn
```

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_ffn.cu
git commit -m "feat: execute dense FFN blocks on CUDA"
```

### Task 5: Exact native MXFP4 routed-expert block group

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_ffn.cu`

**Interfaces:**
- Implements: `CudaBackend::mxfp4_situ_mlp_group()`.
- Consumes: an ordered list of exact gate/up/down MXFP4 triplets and one shared latent input.
- Produces: ordered expert outputs without changing routing or mixing.

- [x] **Step 1: Write failing ordered-group, residency, and atomic-validation tests**

Construct two independently encoded native MXFP4 expert triplets. Require output order to match the request order and compare each output with the CPU block oracle.

Warm all six resident weights, then require six resident hits, zero new weight H2D, one activation H2D, one final synchronization, one successful block call, and two successful block experts. Add a second-expert reserved-scale failure and require rejection before any copy, resident-table mutation, or successful counter increment. Add a hard-capacity case proving that the group cannot bypass the existing resident-byte limit.

- [x] **Step 2: Verify RED**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_ffn --output-on-failure
```

Expected: the group contract is absent or the new assertions fail.

- [x] **Step 3: Implement preflight-complete exact group execution**

Validate every expert triplet, shape, tensor identity, representation, and E8M0 scale before enqueuing any copy or mutating counters. Resolve all triplet weights through the existing resident table and its hard capacity.

Upload the shared latent once. For each expert, execute native MXFP4 gate and up projections, strict SiTU, then native MXFP4 down projection on the same stream. Reuse checked gate/up/activation/output arena regions only after their prior same-stream use. Copy ordered final outputs back, synchronize once after the complete group, read events, then commit profiler counters.

- [x] **Step 4: Verify exactness, accounting, and sanitizer**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_(mxfp4|ffn)" --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_ffn
```

- [x] **Step 5: Commit**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_ffn.cu
git commit -m "feat: execute exact expert FFN blocks on CUDA"
```

### Task 6: Connect the FFN boundary to the synthetic K3 graph

**Files:**
- Modify: `runtime/model.hpp`
- Modify: `runtime/model.cpp`
- Modify: `runtime/main.cpp`
- Modify: `tests/test_cpp_parity.py`

**Behavior:**
- Keeps `operation` structurally unchanged.
- Uses dense blocks for activated/shared FFNs and one ordered expert group for each routed MoE invocation when `cuda-boundary=ffn-block`.
- Keeps router selection and score-weighted mixing on CPU.

- [x] **Step 1: Write failing CLI and parity matrix tests**

Add `cuda-custom` parity cases for FP32 and BF16, scalar and grouped scheduling, and reused resident weights. Require exact generated tokens, layer outputs within the existing representation-specific tolerance, and exact recurrent-state evolution. Record the prefill routed-expert identities and require them to equal the operation reference so the test proves that the execution boundary did not alter routing.

Reject `ffn-block` with CPU or `cuda-dense`, and reject unsupported option combinations with a direct capability error rather than silently falling back.

- [x] **Step 2: Verify RED**

```bash
pytest -q tests/test_cpp_parity.py -k "ffn_block or cuda_boundary"
```

- [x] **Step 3: Refactor owned expert payloads without changing operation behavior**

Move the existing projection descriptor into the engine-private model surface and define an owned expert MLP payload containing exact gate/up/down projections. Update all loaders and operation call sites mechanically, then run the complete CPU parity suite before connecting the new branch.

- [x] **Step 4: Connect dense/shared and routed block branches**

When the selected boundary is `ffn-block`, call `dense_situ_mlp()` for activated and shared FFNs. For routed MoE, collect the already selected expert triplets in router order, call `mxfp4_situ_mlp_group()` once, and apply the unchanged CPU score-weighted mixing. Do not move router, Top-K, mixing, residual, attention, or recurrent state logic.

- [x] **Step 5: Verify graph parity**

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
pytest -q tests/test_cpp_parity.py
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
pytest -q tests/test_cpp_parity.py
```

- [x] **Step 6: Commit**

```bash
git add runtime/model.hpp runtime/model.cpp runtime/main.cpp tests/test_cpp_parity.py
git commit -m "feat: connect FFN blocks to the synthetic graph"
```

### Task 7: Extend profiler schema and add the FFN ablation runner

**Files:**
- Modify: `runtime/profile.hpp`
- Modify: `runtime/profile.cpp`
- Modify: `tests/test_profile.py`
- Modify: `tests/test_benchmark_schema.py`
- Create: `tools/ablate_cuda_ffn.py`

**Matrix:**
- `operation-scalar`.
- `operation-grouped`.
- `ffn-block-scalar`.
- `ffn-block-grouped`.

All four cases use `cuda-custom`, reused execution, resident weights, identical prompts, and identical generation length.

- [x] **Step 1: Write failing schema and ablation tests**

Require a `situ_glu` operation bucket and serialized `ffn_block_calls` and `ffn_block_experts` counters. Require the ablation runner to emit all four exact case names, common provenance fields, split weight/activation H2D, D2H, synchronization count, resident hits/misses, tokens, and parity status.

Assert that operation cases report zero FFN block counters. Assert that FFN cases report nonzero counters and less D2H and synchronization traffic than their matching operation cases. Preserve resident-capacity invariants and exact output parity in every case.

- [x] **Step 2: Verify RED**

```bash
pytest -q tests/test_profile.py tests/test_benchmark_schema.py -k "ffn or situ"
```

- [x] **Step 3: Implement the smallest schema extension**

Add the `situ_glu` timing bucket and the two counters without renaming existing fields. Serialize them in the same stable order as related CUDA metrics. Keep old benchmark consumers valid by adding fields rather than changing existing meanings.

- [x] **Step 4: Implement the exact four-case runner**

Start `tools/ablate_cuda_ffn.py` with a one-line Korean role comment. Reuse the established benchmark invocation and aggregation helpers. Refuse mixed commits, checkpoint identities, prompts, or sample counts within one report.

- [x] **Step 5: Run a one-sample dry run**

```bash
python tools/ablate_cuda_ffn.py --binary build-cuda/k3x-runtime --artifact artifacts/m3-source.k3x --warmup 0 --samples 1 --output results/m3-ffn-dry-run.json
pytest -q tests/test_profile.py tests/test_benchmark_schema.py
```

- [x] **Step 6: Commit**

```bash
git add runtime/profile.hpp runtime/profile.cpp tests/test_profile.py \
  tests/test_benchmark_schema.py tools/ablate_cuda_ffn.py
git commit -m "feat: add FFN block ablation reporting"
```

### Task 8: Full verification, B-0004 measurement, and TITAN Ledger update

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `README.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`
- Create: `results/b0004-ffn-blocks.json`

- [x] **Step 1: Run the complete CPU and CUDA suites**

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
pytest -q
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
pytest -q
```

- [x] **Step 2: Run all six CUDA sanitizer targets**

Run memcheck for the existing four CUDA targets plus `test_cuda_situ` and `test_cuda_ffn`. Record the exact error summary for each target.

- [x] **Step 3: Regenerate bounded synthetic artifacts**

Regenerate `artifacts/m3-source.k3x` and `artifacts/m3-synthetic.k3x` through the checked converter path. Verify the whole-artifact checksum and model identity. Do not download any full Kimi K3 checkpoint.

- [x] **Step 4: Measure B-0004**

Run the four-case matrix with three warmups and 20 measured iterations on native WSL Linux and RTX 5080. Repeat the matrix with the fully enabled BF16 representation. Record decode tok/s, TTFT where available, split H2D, D2H, synchronization count, resident hit rate, peak VRAM, kernel time, I/O stall time, exact generated tokens, and representation-specific numeric error.

Do not choose a new default from throughput alone. The FFN block path may become the experimental recommendation only if correctness holds and the measured traffic reduction does not introduce an unexplained regression.

- [x] **Step 5: Validate ledger invariants**

Check that every measured row names the commit, hardware, model identity, mode, context length, warmup/sample counts, and enabled optimizations. Mark unavailable metrics as not measured. Never copy theoretical values into measured fields.

- [x] **Step 6: Update durable project documents in order**

Update `ARCHITECTURE.md` with the actual implemented boundary, `DECISIONS.md` with the measured accept/reject decision, `BENCHMARKS.md` with B-0004, then README, checklist, and context notes. Update `PROJECT_STATE.md` last with the last known-good commit/test state and the newly measured bottleneck.

- [x] **Step 7: Perform final read-only review and fix Critical or Important findings**

Capture `git status --short` and the relevant diff, request one Terra high read-only final review, verify that it made no filesystem changes, and apply any evidence-backed Critical or Important findings in one batch. Re-run affected tests after fixes.

- [x] **Step 8: Commit and publish the milestone branch**

```bash
git add ARCHITECTURE.md DECISIONS.md BENCHMARKS.md README.md checklist.md \
  context-notes.md PROJECT_STATE.md results/b0004-ffn-blocks.json
git commit -m "docs: record CUDA FFN block ablations"
git push -u origin codex/milestone-three-ffn-blocks
```

Open a draft PR, wait for Linux CI, mark it ready only after all required checks pass, then fast-forward public `main` only when ancestry is verified. Confirm the post-merge CI result and record it in `PROJECT_STATE.md`.
