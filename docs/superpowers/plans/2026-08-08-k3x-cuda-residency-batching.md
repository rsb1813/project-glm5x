# K3X CUDA Residency and Projection Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove measured per-operation CUDA allocation, repeated immutable-weight H2D, and avoidable same-input synchronization costs while preserving the complete Milestone 1 reference path and exact synthetic tokens.

**Architecture:** Extend `ComputeBackend` with stable tensor-keyed weight views, explicit CUDA execution switches, grouped projection calls, and structured runtime counters. Keep the CPU graph and scalar backend as the oracle; add tracked reusable device storage and a bounded no-eviction resident-weight table inside the CUDA backend; then opt selected dependency-free graph sites into grouped calls. Measure every optimization axis independently against B-0002 before changing defaults.

**Tech Stack:** C++20, CUDA 13.3 runtime, cuBLASLt, native SM 12.0 CUDA, CMake/CTest, Python 3.12, pytest, JSON/CSV benchmark artifacts, Compute Sanitizer.

## Global Constraints

- Work only in `codex/milestone-two-residency`; preserve public `main` at the last verified checkpoint until the milestone passes.
- Every new source file starts with a one-line Korean role comment.
- The default CUDA behavior remains `per-operation + transient + scalar` until measured evidence accepts another default.
- CPU and CUDA-disabled builds cannot acquire CUDA headers or libraries.
- No requested CUDA execution may silently fall back to CPU. The documented `cuda-dense` CPU MXFP4 oracle remains part of that backend's definition.
- Stable cache identity is the K3X tensor ID plus representation and shape metadata, never a host pointer.
- Resident capacity is a hard bound. Capacity bypass uses exact transient staging and cannot change routing or output.
- Milestone 2 has no eviction and does not claim LRU, LFU, Least-Stale, async storage, pinned transfer, CUDA Graphs, or full-layer GPU execution.
- Existing CPU/PyTorch, FP32 CUDA, MXFP4 CUDA, BF16, state, logit, and exact-token contracts remain unchanged.
- All code tasks use RED, minimal GREEN, full relevant regression, self-review, and one semantic commit.

---

### Task 1: CUDA option and runtime-stat contracts

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `runtime/src/backend_cuda_stub.cpp`
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `tests/cpp/test_backend_unavailable.cpp`

**Interfaces:**
- Produces: `CudaAllocationMode`, `CudaWeightMode`, `CudaBatchingMode`, expanded `BackendOptions`, `BackendRuntimeStats`, `ComputeBackend::options()`, and `ComputeBackend::runtime_stats()`.
- Preserves: existing `BackendKind`, `DensePrecision`, and `BackendMemoryStats` behavior.

- [ ] **Step 1: Write the failing option/stat tests**

Add compile-time and runtime checks equivalent to the following.

```cpp
k3x::BackendOptions options;
if (options.cuda_allocation != k3x::CudaAllocationMode::per_operation) return 30;
if (options.cuda_weights != k3x::CudaWeightMode::transient) return 31;
if (options.cuda_batching != k3x::CudaBatchingMode::scalar) return 32;
if (options.cuda_resident_bytes != 0) return 33;
auto backend = k3x::make_cpu_backend();
if (backend->options().kind != k3x::BackendKind::cpu) return 34;
const auto stats = backend->runtime_stats();
if (stats.device_allocation_count != 0 ||
    stats.stream_synchronization_count != 0 ||
    stats.resident_weight_bytes != 0) return 35;
```

Extend the CUDA-disabled factory test to pass non-default CUDA options and verify it still returns `backend_unavailable` rather than constructing CPU.

- [ ] **Step 2: Run the native CPU tests and verify RED**

Run:

```bash
cmake --build build-cpu -j2
```

Expected: compilation fails because the new enums, fields, and virtual accessors do not exist.

- [ ] **Step 3: Add the minimal public contracts**

Define the exact types in `backend.hpp`.

```cpp
enum class CudaAllocationMode { per_operation, reused };
enum class CudaWeightMode { transient, resident };
enum class CudaBatchingMode { scalar, grouped };

struct BackendOptions {
    BackendKind kind{BackendKind::cpu};
    DensePrecision dense_precision{DensePrecision::fp32};
    CudaAllocationMode cuda_allocation{CudaAllocationMode::per_operation};
    CudaWeightMode cuda_weights{CudaWeightMode::transient};
    CudaBatchingMode cuda_batching{CudaBatchingMode::scalar};
    std::uint64_t cuda_resident_bytes{};
};

struct BackendRuntimeStats {
    std::uint64_t device_allocation_count{};
    std::uint64_t device_free_count{};
    std::uint64_t stream_synchronization_count{};
    std::uint64_t weight_cache_hits{};
    std::uint64_t weight_cache_misses{};
    std::uint64_t weight_cache_bypasses{};
    std::uint64_t resident_weight_bytes{};
    std::uint64_t peak_resident_weight_bytes{};
    std::uint64_t scratch_bytes{};
    std::uint64_t peak_scratch_bytes{};
    std::uint64_t grouped_projection_calls{};
    std::uint64_t grouped_projection_members{};
};
```

Add pure virtual `options()` and `runtime_stats()` accessors. CPU returns canonical CPU options and zero stats. The CUDA-disabled stub continues returning failure from its factory and needs no fake backend.

- [ ] **Step 4: Run focused and full CPU verification**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu -R "backend|backend_unavailable" --output-on-failure
ctest --test-dir build-cpu --output-on-failure
```

Expected: focused tests and CTest 5/5 pass.

- [ ] **Step 5: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp runtime/src/backend_cuda_stub.cpp tests/cpp/test_backend.cpp tests/cpp/test_backend_unavailable.cpp
git commit -m "feat: define CUDA residency execution options"
```

### Task 2: CLI validation and benchmark schema

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Consumes: `BackendOptions` and `BackendRuntimeStats` from Task 1.
- Produces: four new CLI arguments and JSON/CSV fields for all Milestone 2 options and counters.

- [ ] **Step 1: Write failing CLI validation tests**

Add parametrized cases for unknown values and invalid combinations.

```python
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--cuda-allocation", "pool"], "unknown CUDA allocation mode: pool"),
        (["--cuda-weights", "lru"], "unknown CUDA weight mode: lru"),
        (["--cuda-batching", "graph"], "unknown CUDA batching mode: graph"),
        (["--cuda-resident-bytes", "-1"], "invalid CUDA resident byte capacity: -1"),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_execution_options(arguments, message):
    result = subprocess.run([str(cpp_binary("k3x_run")), *arguments], capture_output=True, text=True)
    assert result.returncode == 2
    assert result.stderr.strip() == message
```

Add CPU cases that reject `reused`, `resident`, `grouped`, or a nonzero resident capacity. Add a CUDA case that rejects `resident` with zero capacity and `transient` with nonzero capacity.

- [ ] **Step 2: Extend the schema test and verify RED**

Add these exact `BenchmarkRecord` fields and JSON/CSV assertions.

```python
cuda_allocation: str
cuda_weights: str
cuda_batching: str
cuda_resident_bytes: int
device_allocation_count: int
device_free_count: int
stream_synchronization_count: int
weight_cache_hits: int
weight_cache_misses: int
weight_cache_bypasses: int
resident_weight_bytes: int
peak_resident_weight_bytes: int
scratch_bytes: int
peak_scratch_bytes: int
weight_h2d_bytes: int
activation_h2d_bytes: int
grouped_projection_calls: int
grouped_projection_members: int
```

Run:

```bash
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q \
  tests/python/test_cpp_parity.py -k cuda_execution_options \
  tests/python/test_benchmark_schema.py::test_benchmark_json_and_csv_preserve_schema
```

Expected: tests fail because the CLI and dataclass do not expose the new contract.

- [ ] **Step 3: Implement parsing and serialization**

Parse exact enum spellings into `BackendOptions`. Validate the complete combination before constructing a backend. Serialize options and `runtime_stats()` with stable snake-case field names. CPU emits reference option values and zero counters.

Extend `_run_process()` and `benchmark_once()` with keyword-only parameters.

```python
def benchmark_once(
    artifact: Path,
    runner: Path,
    warmup: int,
    iterations: int,
    *,
    backend: str = "cpu",
    dense_precision: str = "fp32",
    cuda_allocation: str = "per-operation",
    cuda_weights: str = "transient",
    cuda_batching: str = "scalar",
    cuda_resident_bytes: int = 0,
) -> BenchmarkRecord:
```

Require exact equality across measured samples for option identity and deterministic counter fields. Keep medians for timing, RSS, and device timing.

- [ ] **Step 4: Run focused schema and CLI tests**

Run:

```bash
cmake --build build-cpu -j2
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
```

Expected: CPU-targeted cases pass and CUDA-only cases skip.

- [ ] **Step 5: Commit**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
git commit -m "feat: expose CUDA residency benchmark controls"
```

### Task 3: Stable tensor-keyed backend and CPU group oracle

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Produces: `DenseWeightView`, `Mxfp4WeightView`, scalar methods using those views, and ordered `dense_matvec_group()` / `mxfp4_matvec_group()` results.
- Preserves: exact CPU arithmetic, existing graph topology, layer/logit/state/token outputs.

- [ ] **Step 1: Write failing CPU identity and group tests**

Use literal weights with distinct IDs and heterogeneous row counts.

```cpp
const std::array<float, 2> input{2.0F, -1.0F};
const std::array<float, 4> first{1.0F, 0.0F, 0.0F, 1.0F};
const std::array<float, 2> second{3.0F, -2.0F};
const std::array<k3x::DenseWeightView, 2> group{{
    {101, first, 2, 2},
    {102, second, 1, 2},
}};
const auto output = backend->dense_matvec_group(
    input, group, 7, k3x::ProfilePhase::decode);
if (!output || output.value().size() != 2) return 40;
if (output.value()[0] != std::vector<float>{2.0F, -1.0F}) return 41;
if (output.value()[1] != std::vector<float>{8.0F}) return 42;
```

Add a two-member literal MXFP4 group and an invalid member whose failure rejects the entire group. Assert scalar and group results match exactly on CPU.

- [ ] **Step 2: Run CTest and verify RED**

Run:

```bash
cmake --build build-cpu -j2
```

Expected: compilation fails because the view types and grouped methods do not exist.

- [ ] **Step 3: Implement the view and group contracts**

Add exact public types.

```cpp
struct DenseWeightView {
    std::uint64_t tensor_id;
    std::span<const float> values;
    std::size_t rows;
    std::size_t cols;
};

struct Mxfp4WeightView {
    std::uint64_t tensor_id;
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    std::size_t rows;
    std::size_t cols;
    std::size_t group_size;
};
```

Change scalar methods to consume one view. Group methods return `Result<std::vector<std::vector<float>>>`. CPU validates every member before executing any member, then calls the same scalar arithmetic in request order. CPU `runtime_stats()` remains zero because CUDA grouping is disabled for CPU.

In `Engine`, add helpers that retain IDs explicitly.

```cpp
DenseWeightView dense_weight(const std::string& name,
                             std::size_t rows, std::size_t cols);
Vector matvec(const std::string& name, std::size_t rows, std::size_t cols,
              std::span<const float> input, std::uint32_t layer,
              ProfilePhase phase);
```

Convert every dense caller from `matvec(tensor(name), ...)` to `matvec(name, ...)`. Expert loading constructs owned byte vectors plus an `Mxfp4WeightView` whose spans live through the backend call. Do not cache expert host pointers.

- [ ] **Step 4: Verify CPU graph invariants**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q
```

Expected: CTest 5/5 and the complete CPU pytest suite pass with exact tokens unchanged.

- [ ] **Step 5: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/backend_cpu.cpp runtime/src/model.cpp tests/cpp/test_backend.cpp tests/python/test_cpp_parity.py
git commit -m "refactor: pass stable tensor identities to compute backends"
```

### Task 4: Tracked CUDA allocation and reusable scratch primitives

**Files:**
- Create: `runtime/cuda/device_memory.cuh`
- Create: `runtime/cuda/device_memory.cu`
- Create: `tests/cuda/test_cuda_memory.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: internal `cuda::DeviceAllocation` and `cuda::ScratchBuffer` with injected allocator functions, strong grow semantics, and exact allocation/free/live/peak accounting.
- Consumes: `BackendMemoryStats` and `BackendRuntimeStats` from Task 1.

- [ ] **Step 1: Write the failing resource test**

The test uses a fake allocator for failure semantics and the real allocator for device lifetime.

```cpp
k3x::BackendMemoryStats memory;
k3x::BackendRuntimeStats runtime;
k3x::cuda::ScratchBuffer scratch(&memory, &runtime);
if (scratch.reserve(64) != cudaSuccess) return 1;
void* first = scratch.get();
if (scratch.reserve(32) != cudaSuccess || scratch.get() != first) return 2;
if (runtime.device_allocation_count != 1 || runtime.device_free_count != 0) return 3;
if (scratch.reserve(128) != cudaSuccess) return 4;
if (runtime.device_allocation_count != 2 || runtime.device_free_count != 1) return 5;
if (runtime.scratch_bytes != 128 || runtime.peak_scratch_bytes != 128) return 6;
```

With an injected allocator that fails the second allocation, assert the old pointer, capacity, and live-byte counters remain unchanged.

- [ ] **Step 2: Add the CMake target and verify RED**

Add `device_memory.cu` to the CUDA runtime sources and `test_cuda_memory` under the CUDA-only block. Run:

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DK3X_ENABLE_CUDA=ON
cmake --build build-cuda -j2
```

Expected: compilation fails because the internal resource types are missing.

- [ ] **Step 3: Implement strong resource ownership**

`DeviceAllocation` owns one pointer and byte count and updates total backend memory plus runtime allocation/free counters. `ScratchBuffer::reserve()` allocates a temporary replacement, returns immediately on failure, then swaps and releases the old allocation only after success. Use checked `std::uint64_t` additions before updating counters.

The production allocator wrappers call `cudaMalloc` and `cudaFree`. The header exposes an internal function-table constructor only for the resource unit test; it is not part of `runtime/include/k3x`.

- [ ] **Step 4: Run CUDA unit verification and memcheck**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_memory --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_memory
```

Expected: the test passes and Compute Sanitizer reports 0 errors.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/cuda/device_memory.cuh runtime/cuda/device_memory.cu tests/cuda/test_cuda_memory.cu
git commit -m "feat: add tracked reusable CUDA memory"
```

### Task 5: Allocation-mode integration and reusable cuBLAS resources

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_dense.cu`
- Modify: `tests/cuda/test_cuda_mxfp4.cu`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Consumes: `CudaAllocationMode` and `cuda::ScratchBuffer`.
- Produces: reference per-operation allocation and independently switchable reused input/weight/output scratch for scalar dense and custom MXFP4 calls.

- [ ] **Step 1: Write failing allocation-ablation tests**

For each scalar CUDA operation, execute the same shape twice under both modes. Assert reference mode has equal allocation/free counts after each completed call. Assert reused mode performs no new allocation for the second call, keeps live scratch bytes, and reports the same result, H2D, D2H, and device timing semantics.

```cpp
const auto after_first = backend.value()->runtime_stats();
const auto second = backend.value()->dense_matvec(input, weight_view, 4,
                                                   k3x::ProfilePhase::decode);
const auto after_second = backend.value()->runtime_stats();
if (after_second.device_allocation_count != after_first.device_allocation_count) return 50;
if (after_second.stream_synchronization_count !=
    after_first.stream_synchronization_count + 1) return 51;
```

Add a larger third shape and assert only the affected scratch slots grow.

- [ ] **Step 2: Run CUDA tests and verify RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_dense|cuda_mxfp4" --output-on-failure
```

Expected: tests fail because `CudaBackend` still uses local buffers and reports no runtime counters.

- [ ] **Step 3: Integrate allocation modes**

Keep the Milestone 1 local buffers, descriptors, heuristic lookup, and timing events unchanged for `per_operation`. Under `reused`, add backend-owned scratch slots for dense input, dense transient weight, dense output, MXFP4 input, packed bytes, scales, and output. Reuse one start/end event pair for sequential scalar calls because every scalar call synchronizes before returning.

Only under `reused`, cache cuBLASLt descriptors and the zero-workspace algorithm by `(rows, cols, input_type, weight_type)` so repeated shapes do not recreate descriptors. Descriptor creation failure remains a typed backend error. This keeps `per-operation + transient + scalar` behavior comparable to B-0002 rather than partially optimizing the reference.

Increment `stream_synchronization_count` only after a successful `cudaStreamSynchronize`. Runtime JSON is sampled before destructor teardown, as specified.

- [ ] **Step 4: Verify both modes and memory safety**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_dense|cuda_mxfp4" --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_dense
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_benchmark_schema.py -k cuda
```

Expected: reference and reused outputs match their existing tolerances; both sanitizers report 0 errors.

- [ ] **Step 5: Commit**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_dense.cu tests/cuda/test_cuda_mxfp4.cu tests/python/test_benchmark_schema.py
git commit -m "feat: reuse CUDA operation resources"
```

### Task 6: Bounded dense-weight residency

**Files:**
- Create: `runtime/cuda/resident_weights.cuh`
- Create: `runtime/cuda/resident_weights.cu`
- Create: `tests/cuda/test_cuda_residency.cu`
- Modify: `CMakeLists.txt`
- Modify: `runtime/cuda/backend_cuda.cu`

**Interfaces:**
- Produces: internal `ResidentWeightTable` keyed by tensor ID, representation, rows, columns, and group size; dense FP32/BF16 admission, hit, and bypass behavior.
- Consumes: tracked allocations from Task 4 and stable tensor IDs from Task 3.

- [ ] **Step 1: Write failing dense residency tests**

Create two dense weight views and a capacity that fits exactly one representation. Verify:

- first use is one miss and one weight upload;
- second use is one hit and zero additional weight H2D;
- BF16 is a distinct entry from FP32;
- incompatible metadata for an existing tensor ID returns `invalid_extent`;
- the second weight bypasses when it does not fit and produces the scalar oracle result;
- `resident_weight_bytes` never exceeds the configured bound.

- [ ] **Step 2: Run the new target and verify RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_residency --output-on-failure
```

Expected: build or test fails because no resident table is implemented.

- [ ] **Step 3: Implement exact bounded admission**

Define an internal key with equality over every metadata field.

```cpp
enum class WeightRepresentation { dense_fp32, dense_bf16, mxfp4 };

struct ResidentWeightKey {
    std::uint64_t tensor_id;
    WeightRepresentation representation;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t group_size;
};
```

Store one tracked device allocation for dense data. Before admission, perform checked addition against the hard capacity. A non-fitting entry returns a transient decision, not an error. Upload succeeds before inserting into the map; failed upload releases the temporary allocation and leaves the map unchanged.

In `CudaBackend::dense_matvec`, resident mode asks the table for a device weight. Record `weight_h2d_bytes` only on admitted misses and transient bypass copies. Hits record no weight transfer. Activation H2D remains separate.

- [ ] **Step 4: Verify residency and sanitizer**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_residency|cuda_dense" --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_residency
```

Expected: exact counters and results pass; sanitizer reports 0 errors.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/cuda/resident_weights.cuh runtime/cuda/resident_weights.cu runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_residency.cu
git commit -m "feat: add bounded dense weight residency"
```

### Task 7: Native MXFP4 residency

**Files:**
- Modify: `runtime/cuda/resident_weights.cuh`
- Modify: `runtime/cuda/resident_weights.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_residency.cu`
- Modify: `tests/cuda/test_cuda_mxfp4.cu`

**Interfaces:**
- Extends: `ResidentWeightTable` with one logical MXFP4 entry owning exact packed and scale allocations.
- Preserves: `cuda-dense` CPU MXFP4 oracle and custom low-nibble-first E2M1/E8M0/32 kernel contract.

- [ ] **Step 1: Write failing MXFP4 residency tests**

Use the existing literal three-row MXFP4 fixture with a stable tensor ID. Assert first custom call is a miss, second is a hit, packed plus scale bytes upload only once, and exact output remains `[3.5, 1.0, -3.5]`. Add a capacity that is one byte too small and assert bypass plus exact output. Reuse the tensor ID with a changed group size or shape and assert rejection.

For `cuda-dense`, assert MXFP4 residency counters remain zero because its expert operation is the documented CPU oracle.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_residency|cuda_mxfp4" --output-on-failure
```

Expected: MXFP4 misses repeat and no hit is recorded.

- [ ] **Step 3: Implement paired exact extents**

An MXFP4 entry owns two device allocations under one logical key. Capacity arithmetic includes packed plus scale bytes. Insert only after both allocations and both copies succeed. A hit passes the native device pointers directly to `launch_mxfp4_matvec`; no repacking or dequantization is permitted.

- [ ] **Step 4: Verify native bytes and leaks**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R "cuda_residency|cuda_mxfp4" --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_residency
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
```

Expected: all tests pass and both sanitizer runs report 0 errors.

- [ ] **Step 5: Commit**

```bash
git add runtime/cuda/resident_weights.cuh runtime/cuda/resident_weights.cu runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_residency.cu tests/cuda/test_cuda_mxfp4.cu
git commit -m "feat: retain exact MXFP4 weights on device"
```

### Task 8: Grouped dense CUDA execution and graph sites

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cuda/test_cuda_dense.cu`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Implements: `dense_matvec_group()` for CUDA with one activation upload and one synchronization per group.
- Connects: KDA Q/K/V and dense/shared gate-up graph sites when `cuda_batching == grouped`.

- [ ] **Step 1: Write failing grouped dense CUDA tests**

Reuse Task 3's heterogeneous group. Under grouped mode, assert scalar-equivalent outputs, request ordering, `grouped_projection_calls == 1`, `grouped_projection_members == 2`, activation H2D equals one input, and synchronization count increments once. Add an invalid second member and assert no output is returned.

Add end-to-end FP32 `cuda-dense` and `cuda-custom` grouped cases that require exact token IDs and existing diagnostic tolerances.

- [ ] **Step 2: Run focused CUDA tests and verify RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_dense --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_cpp_parity.py -k grouped
```

Expected: grouped counters remain zero or graph tests fail because the model still issues scalar calls.

- [ ] **Step 3: Implement one-stream grouped dense execution**

Validate every member before any launch. Copy the shared input once. Resolve each weight through transient or resident mode, obtain its cuBLASLt plan, assign a disjoint output region in a grow-only grouped-output arena, enqueue ordered matmuls, and enqueue copies into ordered host vectors before one synchronization. A grow-only event pool supplies one start/end pair per member so individual kernel timing remains valid. Record individual logical weight/kernel events plus one group counter update. Do not count the shared activation once per member.

In `Engine`, add one helper returning ordered vectors from names and shapes. Use it only for KDA Q/K/V and gate-up pairs when `backend_.options().cuda_batching == grouped`; retain unchanged scalar code otherwise.

- [ ] **Step 4: Verify all dense modes**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_dense --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_cpp_parity.py
```

Expected: grouped and scalar CUDA graph cases pass with exact tokens.

- [ ] **Step 5: Commit**

```bash
git add runtime/cuda/backend_cuda.cu runtime/src/model.cpp tests/cuda/test_cuda_dense.cu tests/python/test_cpp_parity.py
git commit -m "feat: batch same-input CUDA dense projections"
```

### Task 9: Grouped native MXFP4 expert gate/up execution

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cuda/test_cuda_mxfp4.cu`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Implements: `mxfp4_matvec_group()` with one activation upload and one synchronization for `cuda-custom`.
- Connects: routed expert gate/up pair only; expert down remains scalar after CPU SiTU-GLU.

- [ ] **Step 1: Write failing grouped MXFP4 tests**

Create two literal MXFP4 requests sharing the same 64-column input and assert ordered scalar-equivalent outputs. Under resident mode, warm both entries, run the group again, and assert two cache hits, zero weight H2D, one activation H2D, and one synchronization.

Add end-to-end `cuda-dense` and `cuda-custom` cases across all eight FP32 allocation/weight/batching combinations. Add fully enabled BF16 cases for both backends. Every case must produce `[43, 32, 28, 49, 9, 28]` and satisfy layer/logit/state tolerance.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_mxfp4 --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_cpp_parity.py -k cuda_backends
```

Expected: grouped MXFP4 and combination cases fail before implementation.

- [ ] **Step 3: Implement grouped native launches**

Validate all views before copying. `cuda-dense` delegates the group to the CPU MXFP4 oracle in order and records no device traffic. `cuda-custom` copies the shared input once, resolves each exact packed/scale pair, enqueues native kernels and output copies in order, then synchronizes once. Increment group counters only after successful completion.

Refactor `Engine::expert()` to load gate and up owned byte payloads together and call the group only when enabled. The activation and down call remain unchanged.

- [ ] **Step 4: Verify the complete combination matrix and sanitizer**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_cpp_parity.py
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
```

Expected: CTest passes, all eight FP32 combinations retain exact tokens, and sanitizer reports 0 errors.

- [ ] **Step 5: Commit**

```bash
git add runtime/cuda/backend_cuda.cu runtime/src/model.cpp tests/cuda/test_cuda_mxfp4.cu tests/python/test_cpp_parity.py
git commit -m "feat: batch exact MXFP4 expert projections"
```

### Task 10: Deterministic ablation runner and full profiler export

**Files:**
- Modify: `runtime/include/k3x/profile.hpp`
- Modify: `runtime/src/profile.cpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Create: `tools/ablate_cuda_residency.py`
- Modify: `tests/cpp/test_profile.cpp`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Produces: split weight/activation H2D aggregation, stable runtime counter serialization, and a four-stage measured ablation driver.
- Consumes: all option and runtime statistics from Tasks 1--9.

- [ ] **Step 1: Write failing transfer-classification tests**

Add `ProfileOperation::weight_host_to_device` and `activation_host_to_device` fixtures. Assert both contribute to total `host_to_device_bytes` and separately to the new summary fields. Failed events contribute only to `failed_operations`.

Add schema assertions that `weight_h2d_bytes + activation_h2d_bytes == host_to_device_bytes` for every CUDA record.

- [ ] **Step 2: Write the failing ablation-runner test**

Expose a pure configuration function.

```python
def cuda_residency_matrix() -> tuple[dict[str, object], ...]:
    return (
        {"name": "reference", "cuda_allocation": "per-operation", "cuda_weights": "transient", "cuda_batching": "scalar"},
        {"name": "reuse", "cuda_allocation": "reused", "cuda_weights": "transient", "cuda_batching": "scalar"},
        {"name": "residency", "cuda_allocation": "reused", "cuda_weights": "resident", "cuda_batching": "scalar"},
        {"name": "grouped", "cuda_allocation": "reused", "cuda_weights": "resident", "cuda_batching": "grouped"},
    )
```

Test exact names, ordering, independent one-axis changes, and a required positive capacity for resident stages.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu -R profile --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q tests/python/test_benchmark_schema.py
```

Expected: new operations, fields, and ablation module are missing.

- [ ] **Step 4: Implement aggregation and runner**

The new Python source starts with:

```python
# CUDA allocation, residency, batching 단계를 동일 조건으로 순차 측정합니다.
```

The runner accepts artifact, runner, backend, dense precision, resident capacity, warmups, iterations, and output directory. It invokes `benchmark_once()` sequentially for the four configurations, writes one JSON/CSV pair per configuration, and writes `summary.json` containing the ordered records and pairwise counter deltas. It never runs GPU benchmark configurations concurrently.

- [ ] **Step 5: Verify schema, identities, and dry execution**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu -R profile --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q tests/python/test_benchmark_schema.py
cmake --build build-cuda -j2
K3X_BUILD_DIR=build-cuda python -m pytest -q tests/python/test_benchmark_schema.py
```

Expected: CPU and CUDA schema tests pass, and deterministic counters remain stable across one-sample integration runs.

- [ ] **Step 6: Commit**

```bash
git add runtime/include/k3x/profile.hpp runtime/src/profile.cpp runtime/cuda/backend_cuda.cu runtime/src/main.cpp tools/benchmark_synthetic.py tools/ablate_cuda_residency.py tests/cpp/test_profile.cpp tests/python/test_benchmark_schema.py
git commit -m "feat: add CUDA residency ablation reporting"
```

### Task 11: Full regression, measurement, Ledger, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify last: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Create: `results/m2-*.json`
- Create: `results/m2-*.csv`

**Interfaces:**
- Consumes: all passing Milestone 2 code and the deterministic artifact.
- Produces: B-0003 measurements, accepted/rejected default decision, synchronized TITAN Ledger, public branch, PR, and green Linux CI.

- [ ] **Step 1: Run final local correctness and memory verification**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
source /home/jolib/.venvs/k3x-m1/bin/activate
K3X_BUILD_DIR=build-cpu python -m pytest -q
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda python -m pytest -q
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_memory
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_residency
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_dense
/usr/local/cuda/bin/compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
git diff --check
```

Expected: every CPU/CUDA test passes, all sanitizer summaries report 0 errors, and no whitespace errors exist.

- [ ] **Step 2: Recreate the deterministic artifact**

Run:

```bash
python tools/generate_synthetic.py --output artifacts/m2-source
python -m k3x_converter.cli convert \
  artifacts/m2-source/source artifacts/m2-synthetic.k3x --chunk-bytes 257
```

Expected: bounded conversion succeeds without full Kimi K3 weights or cloud resources.

- [ ] **Step 3: Measure the required sequential ablations**

Choose a resident capacity larger than the synthetic artifact's total admissible device representation and record the exact value. Run:

```bash
python tools/ablate_cuda_residency.py \
  --artifact artifacts/m2-synthetic.k3x \
  --runner build-cuda/k3x_run \
  --backend cuda-custom \
  --dense-precision fp32 \
  --cuda-resident-bytes 8388608 \
  --warmup 3 \
  --iterations 20 \
  --output-dir results/m2-cuda-custom

python tools/ablate_cuda_residency.py \
  --artifact artifacts/m2-synthetic.k3x \
  --runner build-cuda/k3x_run \
  --backend cuda-dense \
  --dense-precision fp32 \
  --cuda-resident-bytes 8388608 \
  --warmup 3 \
  --iterations 20 \
  --output-dir results/m2-cuda-dense
```

Also measure fully enabled BF16 for both backends with three warmups and twenty samples. Do not prescribe which path wins.

- [ ] **Step 4: Validate measurement invariants before documentation**

Confirm from raw JSON:

- option identity matches each stage;
- reference total H2D remains comparable to B-0002;
- reused allocation count is below reference;
- resident weight H2D is below transient after warmup within each process run;
- grouped activation H2D or synchronization count is below scalar;
- resident bytes never exceed 8,388,608;
- all token sequences and numerical comparisons pass;
- missing utilization, bandwidth, NVMe, and I/O counters remain explicitly not measured.

If any invariant fails, diagnose the implementation or record the optimization as rejected. Never rewrite the acceptance criterion after seeing results.

- [ ] **Step 5: Update the TITAN Ledger in required order**

Update `ARCHITECTURE.md` statuses from actual tests only. Add the default decision and alternatives to `DECISIONS.md`. Add B-0003 raw values and unavailable fields to `BENCHMARKS.md`. Update `README.md`, `checklist.md`, and `context-notes.md`. Update `PROJECT_STATE.md` last with current milestone, remaining blockers, next task, latest measured bottleneck, public commit, and exact test counts.

- [ ] **Step 6: Self-review and commit measurements**

Search for stale implementation claims, fallback paths, debug output, untracked artifacts, and proposed TITAN components marked implemented. Verify all result files parse as JSON/CSV and all README local links resolve. Then run `git diff --check` and commit.

```bash
git add README.md ARCHITECTURE.md DECISIONS.md BENCHMARKS.md PROJECT_STATE.md checklist.md context-notes.md results
git commit -m "docs: record CUDA residency ablations"
```

- [ ] **Step 7: Publish only after green review gates**

Push `codex/milestone-two-residency`, open a draft PR to `main`, and require the repository Linux correctness workflow to pass. Mark the PR ready only after no Critical/Important review issue remains. Fast-forward public `main` only when the branch is an ancestor-compatible linear update and the post-merge `main` workflow succeeds.
