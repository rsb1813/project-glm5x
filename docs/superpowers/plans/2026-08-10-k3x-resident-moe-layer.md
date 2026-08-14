# K3X Resident MoE-Layer CUDA Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one exact, opt-in resident CUDA MoE-layer boundary that replaces split routed/shared subcalls with one upload, one result download, and one synchronization, then measure it as B-0022.

**Architecture:** CPU routing remains authoritative. The backend receives one hidden vector, the immutable routed/shared layer weights, selected native MXFP4 experts, and ordered contributions. A fully resident CUDA request runs thirteen logical operations on one stream and returns one hidden vector; any hard-cap bypass returns `executed=false` before launch so the existing Milestone 20 split CUDA path executes unchanged.

**Tech Stack:** C++20, CUDA 13.3, native `sm_120`, cuBLASLt FP32, custom native-MXFP4 kernels, CMake/CTest, Python 3/pytest, deterministic JSON/CSV evidence.

## Global Constraints

- Correctness outranks throughput; natural routing, selected experts, ordered contributions, KDA/MLA state, AURORA lifecycle, and target verification must not change.
- CPU drafting, `operation`, `ffn-block`, and grouped CUDA behavior remain the defaults.
- The new identity is only FP32 `cuda-custom + reused + resident + resident-grid + synchronous + fusion-none` with positive capacity.
- A residency bypass is a non-error all-or-nothing signal before launch; CUDA and validation errors never become CPU fallback.
- Every production change follows witnessed RED, minimal GREEN, focused verification, and a semantic commit.
- B-0022 requires parity and lower activation H2D, D2H, and synchronization count; it records but does not force throughput direction.
- No CUDA Graph, KDA/MLA device state, reduced precision, dynamic eviction, learned predictor, full checkpoint, or paid cloud resource is part of this plan.

---

## File map

- `runtime/include/k3x/backend.hpp` owns the public boundary enum, views, result, virtual method, and counters.
- `runtime/src/backend_cpu.cpp` owns the portable whole-layer oracle.
- `runtime/cuda/moe_layer.cuh` and `runtime/cuda/moe_layer.cu` own ordered mixing, strict RMSNorm, and final addition launchers.
- `runtime/cuda/backend_cuda.cu` owns residency preflight, device scratch, cuBLASLt/native-MXFP4 orchestration, fallback, events, and profiler accounting.
- `runtime/src/model.cpp` owns one routing decision and split/layer dispatch.
- `runtime/src/aurora.cpp` and `runtime/src/main.cpp` own capability and CLI/output contracts.
- `tools/benchmark_synthetic.py` owns target/draft JSON/CSV schema propagation.
- `tools/ablate_cuda_aurora_moe_layer.py` owns the canonical B-0022 matrix and paired gates.
- `tests/cpp/test_backend.cpp` covers the CPU contract.
- `tests/cuda/test_cuda_moe_layer_ops.cu` covers low-level literal kernels.
- `tests/cuda/test_cuda_moe_layer.cu` covers the complete CUDA backend and bypass.
- `tests/python/test_cuda_aurora_draft.py`, `test_cpp_parity.py`, and `test_benchmark_schema.py` cover runtime ownership and schema.
- `tests/python/test_cuda_aurora_moe_layer_ablation.py` covers B-0022 tooling and committed evidence.

---

### Task 1: Portable MoE-layer contract and CPU oracle

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `tests/cpp/test_backend.cpp`

**Interfaces:**
- Produces: `CudaBoundaryMode::moe_layer`.
- Produces: `DenseVectorView`, `ResidentMoeLayerView`, and `ResidentMoeLayerResult`.
- Produces: `ComputeBackend::resident_mxfp4_moe_layer(...)`.
- Produces: five zero-default `BackendRuntimeStats::resident_moe_layer_*` counters.

- [x] **Step 1: Write the failing CPU value and validation tests**

Add a literal two-expert fixture to `test_backend.cpp`. Build the expected result only from existing public CPU calls and `rms_norm`, then invoke the wished-for whole-layer API.

```cpp
const k3x::ResidentMoeLayerView layer_weights{
    routed_down, routed_norm, routed_up,
    {shared_gate, shared_up, shared_down},
};
const std::array experts{expert_one, expert_two};
const std::array contributions{0.75F, -0.25F};
const auto layer = backend->resident_mxfp4_moe_layer(
    input, layer_weights, experts, contributions, 1.0e-5F,
    4.0F, std::nullopt, 2, k3x::ProfilePhase::decode);
assert(layer && layer.value().executed);
assert_vector_close(layer.value().output, expected, 1.0e-6F);
```

Add separate assertions for empty experts, mismatched contribution count, non-finite contribution, zero epsilon, malformed routed/shared dimensions, short norm vector, duplicate tensor ID, and zero tensor ID. Each must return `invalid_mxfp4` and leave all new counters zero.

- [x] **Step 2: Run RED and witness the missing API failure**

Run:

```bash
cmake --build build -j 12 --target test_backend
```

Expected: compile failure naming missing `ResidentMoeLayerView` or `resident_mxfp4_moe_layer`.

- [x] **Step 3: Add the public types, virtual contract, and zero-default counters**

Add these exact public shapes to `backend.hpp`.

```cpp
enum class CudaBoundaryMode { operation, ffn_block, moe_layer };

struct DenseVectorView {
    std::uint64_t tensor_id;
    std::span<const float> values;
};

struct ResidentMoeLayerView {
    DenseWeightView routed_down;
    DenseVectorView routed_norm;
    DenseWeightView routed_up;
    DenseMlpView shared;
};

struct ResidentMoeLayerResult {
    bool executed{};
    std::vector<float> output;
};
```

Append the counters in this order.

```cpp
std::uint64_t resident_moe_layer_calls{};
std::uint64_t resident_moe_layer_experts{};
std::uint64_t resident_moe_layer_kernel_launches{};
std::uint64_t resident_moe_layer_fallbacks{};
std::uint64_t resident_moe_layer_contribution_h2d_bytes{};
```

Give the virtual method a default `backend_unavailable` body so CPU-only CUDA stubs do not need duplicate boilerplate.

- [x] **Step 4: Implement the minimal CPU oracle**

In `backend_cpu.cpp`, prevalidate the full request and unique tensor IDs before calling any child operation. Then execute the operations in this order.

```cpp
auto latent = dense_matvec(input, weights.routed_down, layer, phase);
auto expert_outputs = mxfp4_situ_mlp_group(
    latent.value(), experts, situ_beta, situ_linear, layer, phase);
std::vector<float> mixed(weights.routed_norm.values.size(), 0.0F);
for (std::size_t slot = 0; slot < experts.size(); ++slot) {
    for (std::size_t row = 0; row < mixed.size(); ++row) {
        mixed[row] += contributions[slot] * expert_outputs.value()[slot][row];
    }
}
std::vector<float> normalized(mixed.size());
rms_norm(normalized, mixed, weights.routed_norm.values, epsilon);
auto routed = dense_matvec(normalized, weights.routed_up, layer, phase);
auto shared = dense_situ_mlp(
    input, weights.shared, situ_beta, situ_linear, layer, phase);
for (std::size_t row = 0; row < routed.value().size(); ++row) {
    routed.value()[row] += shared.value()[row];
}
return Result<ResidentMoeLayerResult>::success(
    {true, std::move(routed.value())});
```

The CPU oracle does not increment the CUDA-specific resident-layer counters.

- [x] **Step 5: Run GREEN and the portable suite**

Run:

```bash
cmake --build build -j 12 --target test_backend
ctest --test-dir build -R "backend|ops|profile" --output-on-failure
```

Expected: selected CTest cases pass and the new oracle validation is covered.

- [x] **Step 6: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp tests/cpp/test_backend.cpp
git commit -m "feat: add resident MoE layer oracle"
```

---

### Task 2: Ordered mix, strict RMSNorm, and final-add CUDA primitives

**Files:**
- Create: `runtime/cuda/moe_layer.cuh`
- Create: `runtime/cuda/moe_layer.cu`
- Create: `tests/cuda/test_cuda_moe_layer_ops.cu`
- Modify: `runtime/include/k3x/profile.hpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `launch_ordered_expert_mix(...)`.
- Produces: `launch_strict_rms_norm(...)`.
- Produces: `launch_vector_add(...)`.
- Produces: profile labels `moe_mix`, `rms_norm`, and `residual_add` whose device time participates in the unchanged profiler total.

- [x] **Step 1: Write literal low-level tests and wire the target**

The new test uploads this fixed expert-major matrix and contribution vector.

```cpp
const std::array expert_outputs{
    1.0F, 2.0F, 3.0F, 4.0F,
    -2.0F, 1.0F, 0.5F, 8.0F,
};
const std::array contributions{0.75F, -0.25F};
const std::array norm_weight{1.0F, 0.5F, 2.0F, 1.5F};
```

Assert ordered mix values, strict CPU `rms_norm` parity within `1e-6`, final add values, null-pointer rejection, zero-size rejection, zero/NaN epsilon rejection, zero experts, and non-finite contributions rejected by the public launcher.

Add `runtime/cuda/moe_layer.cu` to `k3x_runtime`, and add `test_cuda_moe_layer_ops` with `CUDA_ARCHITECTURES "120-real"` and `runtime/cuda` include access.

- [x] **Step 2: Run RED and witness missing launcher/header failure**

Run:

```bash
cmake -S . -B build-cuda -DK3X_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 12 --target test_cuda_moe_layer_ops
```

Expected: compile failure because `moe_layer.cuh` or its launch functions do not exist.

- [x] **Step 3: Implement ordered mix and final add**

Use one thread per output row and loop experts in slot order.

```cpp
__global__ void ordered_expert_mix_kernel(
    const float* outputs, const float* contributions, float* mixed,
    std::size_t experts, std::size_t width) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (row >= width) return;
    float value = 0.0F;
    for (std::size_t slot = 0; slot < experts; ++slot) {
        value += contributions[slot] * outputs[slot * width + row];
    }
    mixed[row] = value;
}
```

The add kernel writes `output[row] = routed[row] + shared[row]`. Launchers validate all pointers/counts and return `cudaGetLastError()`.

- [x] **Step 4: Implement strict RMSNorm**

Launch one block. Thread zero accumulates squares in `double` in increasing index order and publishes one FP32 inverse through shared memory; all threads scale independent rows.

```cpp
__shared__ float inverse;
if (threadIdx.x == 0) {
    double squares = 0.0;
    for (std::size_t row = 0; row < width; ++row) {
        squares += static_cast<double>(input[row]) * input[row];
    }
    inverse = 1.0F /
        sqrtf(static_cast<float>(squares / width) + epsilon);
}
__syncthreads();
for (std::size_t row = threadIdx.x; row < width; row += blockDim.x) {
    output[row] = input[row] * inverse * weight[row];
}
```

Use 256 threads and reject widths that do not fit the backend's checked products before this launcher is reached.

- [x] **Step 5: Run GREEN and Compute Sanitizer**

Run:

```bash
cmake --build build-cuda -j 12 --target test_cuda_moe_layer_ops
ctest --test-dir build-cuda -R cuda_moe_layer_ops --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build-cuda/test_cuda_moe_layer_ops
```

Expected: CTest passes and sanitizer reports `ERROR SUMMARY: 0 errors`.

- [x] **Step 6: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/profile.hpp runtime/cuda/moe_layer.cuh runtime/cuda/moe_layer.cu tests/cuda/test_cuda_moe_layer_ops.cu
git commit -m "feat: add resident MoE layer kernels"
```

---

### Task 3: Complete resident CUDA MoE-layer backend

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Create: `tests/cuda/test_cuda_moe_layer.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: Task 1's whole-layer views/result and Task 2's three launchers.
- Produces: fully resident `executed=true` result or launch-free hard-cap `executed=false`.
- Produces: exact new counters and existing grid/traffic/profile counters.

- [x] **Step 1: Write complete backend RED tests**

Build one literal layer from existing test helpers with one expert and four experts. Create a CPU oracle and a CUDA backend configured as follows.

```cpp
k3x::BackendOptions options;
options.kind = k3x::BackendKind::cuda_custom;
options.dense_precision = k3x::DensePrecision::fp32;
options.cuda_allocation = k3x::CudaAllocationMode::reused;
options.cuda_weights = k3x::CudaWeightMode::resident;
options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
options.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
options.cuda_transfer = k3x::CudaTransferMode::synchronous;
options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
options.cuda_resident_bytes = 8U * 1024U * 1024U;
```

Assert output parity within `1e-6`, calls `1`, experts equal fixture size, kernel launches `13`, fallbacks `0`, contribution bytes equal `experts * sizeof(float)`, grid calls `1`, grid kernel launches `4`, one synchronization, positive activation H2D, and one hidden-width D2H profile event.

Create a second backend with `cuda_resident_bytes=1`. Assert `executed=false`, empty output, layer fallbacks `1`, successful layer/grid counters `0`, and no synchronization. Add malformed-input cases proving counters and resident bytes do not change.

- [x] **Step 2: Run RED and witness backend-unavailable result**

Run:

```bash
cmake --build build-cuda -j 12 --target test_cuda_moe_layer
build-cuda/test_cuda_moe_layer
```

Expected: assertion failure because the inherited virtual method returns `backend_unavailable`.

- [x] **Step 3: Add layer-specific scratch, events, and resident acquisition helpers**

Include `moe_layer.cuh`. Add grow-only allocations for hidden input, routed latent, descriptors, expert gate/up/activation/output, contributions, mixed latent, normalized latent, routed hidden, shared gate/up/activation/hidden, and final hidden. Add `std::array<EventOwner, 26>` for thirteen timed operations.

Add a private dense acquisition helper that uses the existing `ResidentWeightTable` key and `DensePlan` cache.

```cpp
struct ResidentDenseMember {
    const void* device{};
    DensePlan* plan{};
    std::uint64_t uploaded_bytes{};
    bool bypass{};
};
```

The helper accepts only FP32, nonzero tensor IDs, exact row/column payloads, and returns the stable resident pointer plus cached cuBLASLt plan. Treat the norm as representation `dense_fp32`, rows `1`, columns `latent`.

- [x] **Step 4: Implement prevalidation and all-or-nothing acquisition**

Validate all dimensions, checked products, finite values, epsilon, group-32 payloads, and unique IDs before `ResidentWeightTable::acquire`. Acquire six dense/vector tensors followed by every expert gate/up/down tensor. Sum uploaded bytes exactly once.

If any disposition is `bypass`, increment only `resident_moe_layer_fallbacks` among new success counters, add successful admission bytes to existing weight H2D/profile accounting, and return `{false, {}}` without reserving scratch, uploading activation, recording an event, or launching a kernel.

- [x] **Step 5: Implement the thirteen-operation stream**

Upload hidden input, contributions, and expert descriptors. Use the existing cached FP32 cuBLASLt plans with device-to-device inputs/outputs for routed-down, routed-up, shared gate/up/down. Reuse the M20 native grid launcher for expert gate/up/down and Task 2 launchers for mix, norm, and add.

Record an event before and after each operation. After final add, copy only `hidden * sizeof(float)` to host and synchronize once. Increment:

```cpp
++runtime_stats_.resident_moe_layer_calls;
runtime_stats_.resident_moe_layer_experts += experts.size();
runtime_stats_.resident_moe_layer_kernel_launches += 13;
runtime_stats_.resident_moe_layer_contribution_h2d_bytes +=
    contributions.size_bytes();
++runtime_stats_.resident_grid_calls;
runtime_stats_.resident_grid_experts += experts.size();
++runtime_stats_.resident_grid_tokens;
runtime_stats_.resident_grid_expert_tokens += experts.size();
runtime_stats_.resident_grid_kernel_launches += 4;
```

Include descriptors and contributions in activation H2D total, weights in weight H2D, and exactly one hidden result in D2H. Record all thirteen event durations so `ProfileSummary::device_nanoseconds` remains complete.

- [x] **Step 6: Permit the internal exact split fallback**

Where `dense_situ_mlp`, `mxfp4_situ_mlp_grid`, and the serial group implementation currently require `ffn_block`, accept `moe_layer` as well. Do not loosen any other kind/allocation/weight/batching/transfer/fusion gate. This allows `executed=false` to reuse the M20 CUDA split path without changing the public identity.

- [x] **Step 7: Run GREEN, full CUDA backend tests, and sanitizer**

Run:

```bash
cmake --build build-cuda -j 12 --target test_cuda_moe_layer test_cuda_expert_grid test_cuda_ffn
ctest --test-dir build-cuda -R "cuda_moe_layer|cuda_expert_grid|cuda_ffn|cuda_residency" --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 99 build-cuda/test_cuda_moe_layer
```

Expected: selected CTest cases pass and sanitizer reports zero errors.

- [x] **Step 8: Commit**

```bash
git add CMakeLists.txt runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_moe_layer.cu
git commit -m "feat: execute resident CUDA MoE layers"
```

---

### Task 4: Runtime, AURORA, and CLI ownership

**Files:**
- Modify: `runtime/src/model.cpp`
- Modify: `runtime/src/aurora.cpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tests/python/test_cuda_aurora_draft.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_persistent_aurora_runtime.py`

**Interfaces:**
- Consumes: Task 3's successful/bypass result.
- Produces: `--cuda-boundary moe-layer`.
- Produces: `--aurora-draft-boundary ffn-block|moe-layer` with `ffn-block` default.
- Preserves: one router decision, one selected payload set, and unchanged split-path fallback.

- [x] **Step 1: Add failing CLI ownership and parity tests**

Add a CPU-build case requesting target `--cuda-boundary moe-layer`; expect typed `BACKEND_UNAVAILABLE`, no output file, and no Reader activity. Add invalid CUDA combinations for transient weights, zero capacity, grouped batching, prefetch, routed accumulation, BF16, replay, speculation none with draft option, and CPU draft backend.

Add live CUDA cases for full-fit and one-byte capacity. Full-fit must have positive layer calls and zero fallbacks. One-byte must have zero successful layer calls, positive layer fallback, exact target tokens/state/routes, and positive existing grid fallback after the split path runs.

- [x] **Step 2: Run RED and witness unknown argument/boundary failures**

Run:

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_aurora_draft.py \
  tests/python/test_cpp_parity.py \
  tests/python/test_persistent_aurora_runtime.py -q
```

Expected: failures naming unknown `--aurora-draft-boundary` or unknown `moe-layer`.

- [x] **Step 3: Add CLI parsing and fail-closed capability gates**

Parse target `moe-layer` into `CudaBoundaryMode::moe_layer`. Add `aurora_draft_boundary_name` and a supplied flag alongside draft batching. Reject ownership outside `aurora-persistent + cuda-custom` before artifact generation. Construct the draft backend with the selected boundary and serialize its effective value independently from the target.

Update `aurora.cpp` so the canonical persistent CUDA identity accepts either:

```cpp
options.cuda_boundary == CudaBoundaryMode::ffn_block
```

or the closed M21 combination:

```cpp
options.cuda_boundary == CudaBoundaryMode::moe_layer &&
options.cuda_batching == CudaBatchingMode::resident_grid &&
options.cuda_weights == CudaWeightMode::resident &&
options.cuda_resident_bytes > 0
```

- [x] **Step 4: Connect one model dispatch before split computations**

After payloads and contributions exist, build `ResidentMoeLayerView` from canonical tensor names. Call the backend before routed-down or shared execution.

```cpp
if (backend_.options().cuda_boundary == CudaBoundaryMode::moe_layer) {
    auto layer_result = backend_.resident_mxfp4_moe_layer(
        input, layer_view, expert_views, contributions, config_.epsilon,
        config_.situ_beta, config_.situ_linear,
        static_cast<std::uint32_t>(layer), phase);
    if (!layer_result) {
        throw std::runtime_error("resident MoE layer backend failure");
    }
    if (layer_result.value().executed) {
        return std::move(layer_result.value().output);
    }
}
```

On `executed=false`, fall through to the existing routed-down/shared/grid/mix/norm/up path using the same decision, payload handles, and contributions. Do not recompute routing or reload experts.

- [x] **Step 5: Run GREEN and CUDA parity cases**

Run the focused Python set once with the CPU build and once with:

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_aurora_draft.py \
  tests/python/test_cpp_parity.py \
  tests/python/test_persistent_aurora_runtime.py -q
```

Expected: CPU unavailable/ownership cases and live full-fit/bypass parity pass.

- [x] **Step 6: Commit**

```bash
git add runtime/src/model.cpp runtime/src/aurora.cpp runtime/src/main.cpp tests/python/test_cuda_aurora_draft.py tests/python/test_cpp_parity.py tests/python/test_persistent_aurora_runtime.py
git commit -m "feat: route AURORA through resident MoE layers"
```

---

### Task 5: Target/draft telemetry and benchmark schema

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_benchmark_schema.py`
- Modify: `tests/python/test_cuda_aurora_draft.py`

**Interfaces:**
- Consumes: five Task 1 runtime counters.
- Produces: five target JSON fields and five `draft_` JSON fields.
- Produces: `BenchmarkRecord` JSON/CSV fields with literal zero defaults.

- [x] **Step 1: Write failing zero-default and live-counter tests**

Extend the ordinary CPU schema case with exact zeros for all ten fields. Extend the live layer case with these identities.

```python
assert result["draft_resident_moe_layer_calls"] > 0
assert result["draft_resident_moe_layer_kernel_launches"] == (
    result["draft_resident_moe_layer_calls"] * 13
)
assert result["draft_resident_moe_layer_fallbacks"] == 0
assert result["resident_moe_layer_calls"] == 0
```

The one-byte case requires draft calls zero and fallback positive.

- [x] **Step 2: Run RED and witness missing JSON/record fields**

Run:

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_benchmark_schema.py \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected: key or dataclass-constructor failures naming `resident_moe_layer_calls`.

- [x] **Step 3: Propagate counters without aggregation reinterpretation**

Add the five target fields from the target backend snapshot and the five draft fields from the independent draft backend snapshot in `main.cpp`. Add matching integer dataclass fields in `BenchmarkRecord`, parsing in `run_case`, JSON serialization, and CSV field order in `benchmark_synthetic.py`.

For repeated samples, retain the existing first-sample deterministic-counter rule. Do not median, sum, or subtract these per-run values.

- [x] **Step 4: Run GREEN and schema regression**

Run:

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_benchmark_schema.py \
  tests/python/test_cuda_aurora_draft.py -q
```

Expected: schema and live-counter tests pass; existing records remain backward-compatible through zero defaults.

- [x] **Step 5: Commit**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py tests/python/test_benchmark_schema.py tests/python/test_cuda_aurora_draft.py
git commit -m "feat: report resident MoE layer telemetry"
```

---

### Task 6: B-0022 paired ablation tooling

**Files:**
- Create: `tools/ablate_cuda_aurora_moe_layer.py`
- Create: `tests/python/test_cuda_aurora_moe_layer_ablation.py`

**Interfaces:**
- Consumes: `run_benchmark(...)`, draft boundary option, and new telemetry.
- Produces: nine canonical rows, four named pairs, raw JSON/CSV, `summary.json`, and LF-stable `summary.csv`.

- [x] **Step 1: Write the canonical matrix test before the runner exists**

Define this exact case order in the test.

```python
EXPECTED = (
    "natural-greedy",
    "grid-fixed-2-token",
    "layer-fixed-2-token",
    "grid-adaptive-token",
    "layer-adaptive-token",
    "grid-fixed-2-expert",
    "layer-fixed-2-expert",
    "grid-adaptive-expert",
    "layer-adaptive-expert",
)
```

Assert four pair names, three warmups/twenty samples in committed evidence, and that the runner uses only the executable synthetic artifact.

- [x] **Step 2: Run RED and witness missing module**

Run:

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_aurora_moe_layer_ablation.py -q
```

Expected: import failure for `tools.ablate_cuda_aurora_moe_layer`.

- [x] **Step 3: Implement the minimum nine-row runner**

Reuse B-0021 fixed options. Split rows pass `aurora_draft_boundary="ffn-block"`; layer rows pass `aurora_draft_boundary="moe-layer"`. Both use `resident-grid`, FP32, reused allocation, synchronous transfer, fusion none, and 8 MiB residency.

Before writing summary artifacts, require for every pair:

```python
assert_same_target_and_draft_identity(split, layer)
if layer["draft_resident_moe_layer_calls"] <= 0:
    raise RuntimeError("resident MoE layer did not execute")
if layer["draft_resident_moe_layer_fallbacks"] != 0:
    raise RuntimeError("full-fit layer unexpectedly fell back")
if layer["draft_resident_moe_layer_kernel_launches"] != (
    layer["draft_resident_moe_layer_calls"] * 13
):
    raise RuntimeError("resident MoE layer launch accounting changed")
if split["draft_stream_synchronization_count"] - \
        layer["draft_stream_synchronization_count"] != \
        layer["draft_resident_moe_layer_calls"] * 3:
    raise RuntimeError("resident MoE layer synchronization reduction changed")
if layer["draft_activation_h2d_bytes"] >= split["draft_activation_h2d_bytes"]:
    raise RuntimeError("resident MoE layer did not reduce activation H2D")
if layer["draft_device_to_host_bytes"] >= split["draft_device_to_host_bytes"]:
    raise RuntimeError("resident MoE layer did not reduce D2H")
weight_delta = (
    layer["draft_weight_h2d_bytes"] - split["draft_weight_h2d_bytes"]
)
resident_delta = (
    layer["draft_resident_weight_bytes"] - split["draft_resident_weight_bytes"]
)
if weight_delta <= 0 or weight_delta != resident_delta:
    raise RuntimeError("resident norm cold-admission accounting changed")
if (
    layer["draft_weight_h2d_bytes"] + layer["draft_activation_h2d_bytes"]
    >= split["draft_weight_h2d_bytes"] + split["draft_activation_h2d_bytes"]
):
    raise RuntimeError("resident MoE layer did not reduce total H2D")
```

Compute paired decode, synchronization, activation-H2D, and D2H deltas from raw records. Use `csv.DictWriter(..., lineterminator="\n")` and SHA-256 every raw artifact plus canonical aggregate.

- [x] **Step 4: Run one live sample and schema tests**

Run:

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_aurora_moe_layer_ablation.py -q
```

Expected: matrix/schema tests and the marked live one-sample comparison pass.

- [x] **Step 5: Commit**

```bash
git add tools/ablate_cuda_aurora_moe_layer.py tests/python/test_cuda_aurora_moe_layer_ablation.py
git commit -m "bench: add resident MoE layer ablation"
```

---

### Task 7: RTX 5080 B-0022 evidence

**Files:**
- Create: `results/b0022-cuda-aurora-moe-layer-wsl/*.json`
- Create: `results/b0022-cuda-aurora-moe-layer-wsl/*.csv`
- Create: `results/b0022-cuda-aurora-moe-layer-wsl/summary.json`
- Create: `results/b0022-cuda-aurora-moe-layer-wsl/summary.csv`
- Modify: `tests/python/test_cuda_aurora_moe_layer_ablation.py`

**Interfaces:**
- Produces: committed measured B-0022 evidence with raw/summary parity.

- [x] **Step 1: Run the formal matrix**

Run under WSL2 on the local RTX 5080.

```bash
/home/jolib/.venvs/k3x-m1/bin/python \
  tools/ablate_cuda_aurora_moe_layer.py \
  --runner build-cuda/k3x_run \
  --output-dir results/b0022-cuda-aurora-moe-layer-wsl \
  --warmup 3 \
  --iterations 20
```

Expected: nine JSON, nine CSV, one summary JSON, and one summary CSV complete without a forced TPS assertion.

- [x] **Step 2: Add committed-evidence verification**

The evidence test must recompute all eighteen raw digests, summary CSV digest, canonical aggregate, pair identity, exact tokens/state/routes/acceptance, synchronization equation, activation/total-H2D and D2H direction, the norm cold-admission weight/resident-byte delta, thirteen launches per call, and zero fallback directly from committed bytes.

- [x] **Step 3: Run evidence verification**

Run:

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_aurora_moe_layer_ablation.py -q
```

Expected: committed evidence checks pass and reported percentages are recomputed, not copied constants.

- [x] **Step 4: Commit**

```bash
git add results/b0022-cuda-aurora-moe-layer-wsl tests/python/test_cuda_aurora_moe_layer_ablation.py
git commit -m "bench: measure resident MoE layers"
```

---

### Task 8: Complete verification, ledger, review, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: all implementation commits and B-0022 evidence.
- Produces: synchronized public evidence with no projected value presented as measured.

- [x] **Step 1: Run the complete local verification matrix**

Run:

```bash
ctest --test-dir build --output-on-failure
/home/jolib/.venvs/k3x-m1/bin/python -m pytest -q
ctest --test-dir build-uring --output-on-failure
K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 K3X_BUILD_DIR=build-uring \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q
ASAN_OPTIONS=detect_leaks=0 ctest --test-dir build-uring-asan --output-on-failure
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q
compute-sanitizer --tool memcheck --error-exitcode 99 build-cuda/test_cuda_moe_layer_ops
compute-sanitizer --tool memcheck --error-exitcode 99 build-cuda/test_cuda_moe_layer
```

Expected: every suite passes; sanitizer reports zero errors. If any count differs from the ledger, record the actual fresh count.

- [x] **Step 2: Cross-check evidence and defaults**

Run `git diff --check`, all B-0022 digest tests, and focused CLI ownership tests. Search every `CudaBoundaryMode` switch and every target/draft schema consumer. Confirm CPU, grouped, `ffn-block`, natural routing, and speculation-none defaults are unchanged.

- [x] **Step 3: Synchronize the TITAN Ledger**

Document actual code and measured B-0022 values. `PERFORMANCE_MODEL.md` must separate weight bytes from activation/descriptor/contribution traffic. `BENCHMARKS.md` must include missing metrics as not measured. `DECISIONS.md` must accept or reject default promotion from evidence. Update `PROJECT_STATE.md` last with the current public/local heads and next measured bottleneck.

- [x] **Step 4: Commit documentation**

```bash
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md BENCHMARKS.md PROJECT_STATE.md checklist.md context-notes.md
git commit -m "docs: synchronize milestone twenty one ledger"
```

- [x] **Step 5: Publish and merge**

Push `codex/milestone-twenty-one-resident-moe-layer`, create a ready public PR against `main`, wait for push and pull-request correctness runs, rebase-merge only after both pass, and wait for the post-merge `main` run. Then reconcile the README milestone row with the actual PR and public integration head in a small follow-up documentation PR.

---

## Plan self-review result

- Every design requirement maps to a task and an explicit verification command.
- Public types and counter names are consistent from backend through runtime JSON, `BenchmarkRecord`, B-0022, and ledger tasks.
- The only non-error fallback is `ResidentMoeLayerResult{executed=false}` before launch; all error paths remain fail-closed.
- No task changes routing, speculative target authority, precision, eviction policy, or defaults.
- No placeholder, implied implementation, forced TPS outcome, full-checkpoint action, or paid cloud action remains.
