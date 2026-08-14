# K3X Exact Asynchronous L0/L1 Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a switchable exact MXFP4 expert prefetch path that copies a bounded pinned L1 slab to ephemeral L0 staging on a dedicated CUDA stream while the routed-down projection computes.

**Architecture:** Natural routing prepares one ordered expert group after synchronous K3X reads. A backend-owned `AsyncMxfp4Pipeline` copies exact native bytes into a fixed pinned slab, enqueues one H2D transfer, returns a single-use token, and later connects the transfer event to the compute stream before the existing expert FFN kernels. The synchronous Milestone 3 path remains the default and no L2, eviction, or prediction behavior enters this milestone.

**Tech Stack:** C++20, CUDA Runtime 13.3, cuBLASLt, native `sm_120`, CMake/CTest, Python 3.12, pytest, Compute Sanitizer.

## Global Constraints

- Correctness precedes speed. Selected experts and token IDs are exact; floating results retain the established FP32/BF16 tolerances.
- The default is `CudaTransferMode::synchronous`; prefetch is explicit and experimental.
- Prefetch supports only `cuda-custom + ffn-block + reused + transient + positive pinned capacity` in this milestone.
- The native expert representation remains low-nibble-first E2M1 with E8M0/32 scales. No repacking, dequantization, requantization, pruning, or proxy is allowed.
- `Reader` remains synchronous. Do not report NVMe overlap, NVMe GB/token, or storage I/O stall time.
- One backend may have at most one outstanding prefetch token. No silent synchronous fallback is allowed.
- Page-locked bytes are fixed at backend construction and never exceed `cuda_pinned_bytes`.
- Every new source file starts with a one-line Korean role comment.
- Every task follows RED, observed failure, minimal GREEN, relevant regression tests, and one semantic commit.
- Do not download full Kimi K3 weights or provision cloud resources.

---

### Task 1: Define transfer mode, token, state error, and CLI capability contract

**Files:**
- Modify: `runtime/include/k3x/status.hpp`
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cpu.cpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/main.cpp`
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Produces: `CudaTransferMode`, `Mxfp4PrefetchToken`, `BackendOptions::cuda_transfer`, `BackendOptions::cuda_pinned_bytes`, `ErrorCode::invalid_state`.
- Produces: abstract prepare/consume methods and asynchronous transfer counters used by Tasks 2 through 6.
- Preserves: existing synchronous `mxfp4_situ_mlp_group` signature and defaults.

- [x] **Step 1: Write failing C++ contract assertions**

Add to `tests/cpp/test_backend.cpp`:

```cpp
if (defaults.cuda_transfer != k3x::CudaTransferMode::synchronous) return 59;
if (defaults.cuda_pinned_bytes != 0) return 60;
if (k3x::error_code_name(k3x::ErrorCode::invalid_state) != "INVALID_STATE") {
    return 61;
}
const auto prefetch = backend->prefetch_mxfp4_situ_mlp_group(
    {}, 1, 0, k3x::ProfilePhase::decode);
if (prefetch || prefetch.error() != k3x::ErrorCode::backend_unavailable) {
    return 62;
}
const auto stats = backend->runtime_stats();
if (stats.pinned_host_bytes != 0 || stats.peak_pinned_host_bytes != 0 ||
    stats.async_prefetch_calls != 0 || stats.async_prefetch_bytes != 0 ||
    stats.async_prefetch_ready_before_use != 0 ||
    stats.async_prefetch_late_at_use != 0 ||
    stats.transfer_stream_wait_count != 0 ||
    stats.pinned_staging_nanoseconds != 0 ||
    stats.transfer_device_nanoseconds != 0 ||
    stats.transfer_stall_nanoseconds != 0 ||
    stats.async_engine_count != 0 || stats.device_overlap) {
    return 63;
}
```

- [x] **Step 2: Write failing CLI validation tests**

Extend `tests/python/test_cpp_parity.py` with literal cases for unknown transfer mode, malformed pinned capacity, CPU prefetch options, zero capacity, synchronous mode with positive pinned capacity, resident+prefetch, non-custom backend, non-FFN boundary, and per-operation allocation. The accepted prefetch argument vector is:

```python
[
    "--backend", "cuda-custom",
    "--cuda-boundary", "ffn-block",
    "--cuda-allocation", "reused",
    "--cuda-weights", "transient",
    "--cuda-transfer", "prefetch",
    "--cuda-pinned-bytes", "1048576",
]
```

- [x] **Step 3: Run RED checks**

Run:

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu -R 'backend|backend_unavailable' --output-on-failure
K3X_BUILD_DIR=$PWD/build-cpu python -m pytest -q \
  tests/python/test_cpp_parity.py -k 'cuda_transfer or cuda_execution_options'
```

Expected: compile failures for missing enum, fields, token, methods, and `invalid_state`, followed by unknown CLI option failures once compilation is restored minimally.

- [x] **Step 4: Implement the minimal public contract**

Add to `backend.hpp`:

```cpp
enum class CudaTransferMode { synchronous, prefetch };

struct Mxfp4PrefetchToken {
    std::uint64_t value{};
    std::uint64_t use_sequence{};
};
```

Add the two fields to `BackendOptions`, add all twelve zero-initialized asynchronous fields asserted above to `BackendRuntimeStats`, and add these virtual methods to `ComputeBackend`:

```cpp
virtual Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
    std::span<const Mxfp4MlpView> experts,
    std::uint64_t use_sequence,
    std::uint32_t layer,
    ProfilePhase phase) = 0;

virtual Result<std::vector<std::vector<float>>>
mxfp4_situ_mlp_group_prepared(
    std::span<const float> input,
    Mxfp4PrefetchToken token,
    float situ_beta,
    std::optional<float> situ_linear,
    std::uint32_t layer,
    ProfilePhase phase) = 0;
```

CPU and temporary CUDA methods return `BACKEND_UNAVAILABLE` without profiler mutation. Add `INVALID_STATE` to the stable error-name switch.

Parse `--cuda-transfer` and `--cuda-pinned-bytes`. Reject unsupported combinations before backend construction with exact messages asserted by the Python tests. Serialize `cuda_transfer` and `cuda_pinned_bytes` into runner JSON.

- [x] **Step 5: Run GREEN checks**

Run the Task 1 commands, then build both `build-cpu` and `build-cuda`. Expected: all selected tests pass and CUDA remains buildable with the temporary unavailable implementations in `CudaBackend`.

- [x] **Step 6: Commit**

```bash
git add runtime/include/k3x/status.hpp runtime/include/k3x/backend.hpp \
  runtime/src/backend_cpu.cpp runtime/cuda/backend_cuda.cu runtime/src/main.cpp \
  tests/cpp/test_backend.cpp \
  tests/python/test_cpp_parity.py
git commit -m "feat: define exact CUDA prefetch contract"
```

---

### Task 2: Implement fixed-capacity pinned host memory

**Files:**
- Create: `runtime/cuda/pinned_memory.cuh`
- Create: `runtime/cuda/pinned_memory.cu`
- Create: `tests/cuda/test_cuda_pinned_memory.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `BackendRuntimeStats` from Task 1.
- Produces: `k3x::cuda::PinnedBuffer`, used only by `AsyncMxfp4Pipeline`.

- [x] **Step 1: Write the failing real-CUDA resource test**

The test must construct a `PinnedBuffer`, allocate 64 bytes once, verify non-null storage and exact counters, verify a second allocation returns `cudaErrorInvalidValue`, and verify destruction returns current pinned bytes to zero while preserving the peak:

```cpp
k3x::BackendRuntimeStats stats;
{
    k3x::cuda::PinnedBuffer buffer(&stats);
    if (buffer.allocate(64) != cudaSuccess || !buffer.get()) return 1;
    if (buffer.size() != 64 || stats.pinned_host_bytes != 64 ||
        stats.peak_pinned_host_bytes != 64) return 2;
    if (buffer.allocate(32) != cudaErrorInvalidValue) return 3;
}
if (stats.pinned_host_bytes != 0 || stats.peak_pinned_host_bytes != 64) return 4;
```

- [x] **Step 2: Run RED**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_pinned_memory --output-on-failure
```

Expected: configure or compile failure because the files and type do not exist.

- [x] **Step 3: Implement `PinnedBuffer`**

The header contract is:

```cpp
class PinnedBuffer {
public:
    explicit PinnedBuffer(BackendRuntimeStats* runtime);
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    cudaError_t allocate(std::size_t bytes);
    void* get() const noexcept;
    std::size_t size() const noexcept;
private:
    BackendRuntimeStats* runtime_{};
    void* pointer_{};
    std::size_t bytes_{};
};
```

`allocate(0)` and repeated allocation return `cudaErrorInvalidValue`. Use `cudaHostAlloc(..., cudaHostAllocDefault)` and `cudaFreeHost`. Update counters only after successful allocation and during owned destruction.

- [x] **Step 4: Run GREEN and memcheck**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_pinned_memory --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build-cuda/test_cuda_pinned_memory
```

Expected: CTest passes and Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.

- [x] **Step 5: Commit**

```bash
git add runtime/cuda/pinned_memory.cuh runtime/cuda/pinned_memory.cu \
  tests/cuda/test_cuda_pinned_memory.cu CMakeLists.txt
git commit -m "feat: add bounded CUDA pinned staging"
```

---

### Task 3: Implement the single-flight asynchronous MXFP4 pipeline

**Files:**
- Create: `runtime/cuda/async_mxfp4_pipeline.cuh`
- Create: `runtime/cuda/async_mxfp4_pipeline.cu`
- Create: `tests/cuda/test_cuda_async_pipeline.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `PinnedBuffer`, `ScratchBuffer`, `Mxfp4MlpView`, `Mxfp4PrefetchToken`.
- Produces: `AsyncMxfp4Pipeline::prepare`, `consume`, and `complete` for Task 4.

- [x] **Step 1: Write failing preflight and lifecycle tests**

Use a literal two-expert fixture with valid 32-column native payloads. Tests must assert:

- prepare returns token `1` and does not increment `stream_synchronization_count`;
- successful prepare increments `async_prefetch_calls`, `async_prefetch_bytes`, `weight_h2d_bytes`, and `pinned_staging_nanoseconds` once;
- a second prepare returns `INVALID_STATE` without counter changes;
- a too-small pinned capacity returns `INVALID_EXTENT` before successful-prefetch counters;
- invalid group size, reserved scale, and malformed extent fail before allocations and counters;
- consuming token `0`, a stale token, or a mismatched layer/phase returns `INVALID_STATE` and retains the valid pending token;
- the valid consume returns device views in exact expert and gate/up/down order;
- repeated consume fails after `complete`.

- [x] **Step 2: Run RED**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_async_pipeline --output-on-failure
```

Expected: missing pipeline type and source failures.

- [x] **Step 3: Implement focused internal types**

Define:

```cpp
struct DeviceMxfp4WeightView {
    std::uint64_t tensor_id;
    const std::uint8_t* packed;
    const std::uint8_t* scales;
    std::size_t rows;
    std::size_t cols;
    std::size_t group_size;
};

struct DeviceMxfp4MlpView {
    DeviceMxfp4WeightView gate;
    DeviceMxfp4WeightView up;
    DeviceMxfp4WeightView down;
};

struct AsyncTransferMetrics {
    std::uint64_t bytes{};
    std::uint64_t staging_nanoseconds{};
    std::uint64_t transfer_nanoseconds{};
    std::uint64_t stall_nanoseconds{};
    bool ready_before_use{};
};
```

`AsyncMxfp4Pipeline` owns one transfer stream, fixed pinned buffer, device scratch, three transfer events, two compute-wait timing events, device views, and optional pending metadata. Create the transfer stream with `cudaStreamNonBlocking`. Create the readiness event with `cudaEventDisableTiming`; timing events remain timing-enabled.

- [x] **Step 4: Implement exact slab preparation**

Preflight the complete group with the same native E8M0/32 rules as the CUDA FFN block. Sum all six extents per expert with overflow guards. Reject empty groups and capacity overflow before allocation or counters.

Copy bytes into the pinned slab in router order without padding. Build device pointers from identical offsets. Enqueue one `cudaMemcpyAsync` on the transfer stream between timing events, then record readiness. After successful enqueue, update successful-prefetch calls/bytes, weight H2D bytes, and pinned staging time. Return without stream or event synchronization.

- [x] **Step 5: Implement consume and complete**

`consume` validates token, layer, and phase, queries readiness for ready/late classification, records wait-start, enqueues `cudaStreamWaitEvent`, records wait-end, and returns immutable device views.

After the backend's final compute-stream synchronization, `complete` obtains event durations, updates ready/late classification, wait count, transfer time, and stall time, clears the pending request, and returns `AsyncTransferMetrics`. Destruction synchronizes the transfer stream before resource teardown.

- [x] **Step 6: Run GREEN and sanitizer**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R 'cuda_pinned_memory|cuda_async_pipeline' \
  --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build-cuda/test_cuda_async_pipeline
```

- [x] **Step 7: Commit**

```bash
git add runtime/cuda/async_mxfp4_pipeline.cuh \
  runtime/cuda/async_mxfp4_pipeline.cu tests/cuda/test_cuda_async_pipeline.cu \
  CMakeLists.txt
git commit -m "feat: prepare exact MXFP4 transfers asynchronously"
```

---

### Task 4: Execute prepared expert FFN blocks

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_ffn.cu`
- Create: `tests/cuda/test_cuda_async_ffn.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: Task 3 pipeline and device views.
- Produces: concrete CUDA `prefetch_mxfp4_situ_mlp_group` and `mxfp4_situ_mlp_group_prepared`.

- [x] **Step 1: Write the failing prepared-block oracle test**

Create a `cuda-custom + ffn-block + reused + transient + prefetch` backend with a pinned capacity larger than the literal group. Prepare two experts, mutate every original packed and scale byte after prepare, then consume with the original latent input. Assert output equals the CPU oracle built from the pre-mutation payload, proving that the pinned slab owns the prepared bytes.

Assert one final host synchronization, one FFN block, two experts, one async prefetch, one transfer wait, exact weight H2D bytes, and no resident-cache activity.

- [x] **Step 2: Run RED**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R cuda_async_ffn --output-on-failure
```

Expected: CUDA backend returns the temporary unavailable result from Task 1.

- [x] **Step 3: Construct the pipeline only for valid prefetch options**

Add one transfer stream/pipeline owner to `CudaBackend`. Factory construction queries `cudaDeviceProp.asyncEngineCount` and `deviceOverlap`, allocates exactly `cuda_pinned_bytes`, and returns `BACKEND_UNAVAILABLE` if any resource setup fails. Synchronous options create no pinned allocation or transfer stream.

- [x] **Step 4: Implement prepare and prepared execution**

Delegate prepare to `AsyncMxfp4Pipeline`. Prepared execution obtains device views, reuses the current strict SiTU and native MXFP4 launch sequence, uploads the latent activation on the compute stream, downloads only final expert outputs, performs the existing single final compute-stream synchronization, obtains transfer metrics, and records the weight H2D profiler event exactly once.

Do not call `ResidentWeightTable`, transient packed/scales scratch copies, or the synchronous group function in this path.

- [x] **Step 5: Verify error atomicity**

Add invalid-beta, stale-token, repeated-token, insufficient-capacity, and prepared-kernel failure coverage. Invalid calls must not report successful FFN blocks. A valid pending token must survive a foreign-token attempt.

- [x] **Step 6: Run GREEN and sanitizer**

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda -R 'cuda_ffn|cuda_async_ffn' --output-on-failure
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  ./build-cuda/test_cuda_async_ffn
```

- [x] **Step 7: Commit**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_ffn.cu \
  tests/cuda/test_cuda_async_ffn.cu CMakeLists.txt
git commit -m "feat: execute prepared expert FFN blocks"
```

---

### Task 5: Schedule expert prefetch across routed-down computation

**Files:**
- Modify: `runtime/src/model.cpp`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Consumes: Task 4 prepare/consume API.
- Produces: exact graph-level overlap boundary and monotonically increasing use sequence.

- [x] **Step 1: Write failing complete-graph parity tests**

Extend the current FFN parity matrix with `cuda_transfer in {synchronous,prefetch}`. Synchronous uses transient weights and zero pinned capacity. Prefetch uses transient weights and a synthetic-capacity literal. For FP32/BF16 and scalar/grouped modes, require exact token IDs and routed experts plus established logit/layer/state tolerances.

Require synchronous rows to report zero async counters and prefetch rows to report positive prefetch, byte, wait, and pinned counters without increasing host synchronization count relative to the matched synchronous FFN-block row.

- [x] **Step 2: Run RED**

```bash
cmake --build build-cuda -j2
K3X_BUILD_DIR=$PWD/build-cuda python -m pytest -q \
  tests/python/test_cpp_parity.py -k 'async_transfer or ffn_block'
```

Expected: prefetch rows fail because the model never prepares a token.

- [x] **Step 3: Reorder only the prefetch MoE branch**

Add `next_prefetch_sequence_` to `Engine`. In prefetch mode:

```cpp
auto payloads = load_selected_experts(layer, order);
auto token = backend_.prefetch_mxfp4_situ_mlp_group(
    expert_views, next_prefetch_sequence_++, layer, phase);
auto latent = matvec(base + "routed_down_proj", ...);
auto expert_outputs = backend_.mxfp4_situ_mlp_group_prepared(
    latent, token.value(), config_.situ_beta, config_.situ_linear,
    layer, phase);
```

Keep the synchronous branch's order and calls unchanged. Routing trace collection remains before either expert execution path and uses the same `order` vector.

- [x] **Step 4: Run GREEN matrix**

Run the Task 5 pytest command, then all configured CUDA CTests. Every row must generate `[43, 32, 28, 49, 9, 28]`.

- [x] **Step 5: Commit**

```bash
git add runtime/src/model.cpp tests/python/test_cpp_parity.py
git commit -m "feat: overlap expert transfer with routed projection"
```

---

### Task 6: Serialize transfer capability and accounting

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Consumes: public runtime counters from Task 1 and device properties populated by Tasks 3 through 4.
- Produces: deterministic runtime JSON and `BenchmarkRecord` fields used by Task 7.

- [x] **Step 1: Write failing schema tests**

Assert all CPU/synchronous rows serialize zero for async counters and capability fields remain explicit. Assert a prefetch row serializes requested mode/capacity, nonzero pinned/prefetch/wait counters, `async_prefetch_bytes <= weight_h2d_bytes`, and nonnegative timing values.

Add these exact `BenchmarkRecord` fields:

```python
cuda_transfer: str
cuda_pinned_bytes: int
pinned_host_bytes: int
peak_pinned_host_bytes: int
async_prefetch_calls: int
async_prefetch_bytes: int
async_prefetch_ready_before_use: int
async_prefetch_late_at_use: int
transfer_stream_wait_count: int
pinned_staging_nanoseconds: int
transfer_device_nanoseconds: int
transfer_stall_nanoseconds: int
async_engine_count: int
device_overlap: bool
```

- [x] **Step 2: Run RED**

```bash
K3X_BUILD_DIR=$PWD/build-cuda python -m pytest -q \
  tests/python/test_benchmark_schema.py
```

Expected: missing JSON/dataclass fields.

- [x] **Step 3: Implement serialization and deterministic checks**

Expose the fields in runner JSON, benchmark dataclass, CSV, and deterministic-field tuple. Use runtime counters rather than derived estimates. CPU reports `async_engine_count=0` and `device_overlap=false`; CUDA reports actual `cudaDeviceProp` values.

- [x] **Step 4: Run GREEN**

Run CPU and CUDA benchmark-schema tests. Verify `write_results` round-trips JSON and CSV without list or boolean coercion errors.

- [x] **Step 5: Commit**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py \
  tests/python/test_benchmark_schema.py
git commit -m "feat: report asynchronous transfer accounting"
```

---

### Task 7: Add the B-0005 transfer ablation runner

**Files:**
- Create: `tools/ablate_cuda_transfer.py`
- Create: `tests/python/test_cuda_transfer_ablation.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: `benchmark_synthetic.py` fields from Task 6.
- Produces: four matched rows and `summary.json` for each precision.

- [x] **Step 1: Write failing ablation tests**

Use a fake benchmark callable returning literal records. Require this exact case order:

```python
(
    ("synchronous-scalar", "synchronous", "scalar", 0),
    ("prefetch-scalar", "prefetch", "scalar", PINNED_BYTES),
    ("synchronous-grouped", "synchronous", "grouped", 0),
    ("prefetch-grouped", "prefetch", "grouped", PINNED_BYTES),
)
```

The runner must reject token/routing mismatch, nonzero synchronous async counters, zero prefetch counters, extra host synchronization, H2D accounting mismatch, wrong backend/boundary/allocation/weight mode, and missing raw files.

- [x] **Step 2: Run RED**

```bash
python -m pytest -q tests/python/test_cuda_transfer_ablation.py
```

Expected: import failure because the runner does not exist.

- [x] **Step 3: Implement the runner**

Follow `ablate_cuda_ffn.py` structure but fix backend identity to `cuda-custom`, boundary to `ffn-block`, allocation to `reused`, and weights to `transient`. Accept artifact, runner, precision, pinned bytes, warmups, samples, and output directory. Store one JSON and CSV per case plus a summary containing measured deltas only.

- [x] **Step 4: Run GREEN and one-sample smoke**

```bash
python -m pytest -q tests/python/test_cuda_transfer_ablation.py
python tools/ablate_cuda_transfer.py \
  --artifact artifacts/m3-synthetic.k3x \
  --runner build-cuda/k3x_run \
  --dense-precision fp32 \
  --cuda-pinned-bytes 1048576 \
  --warmup 0 --iterations 1 \
  --output-dir /tmp/k3x-b0005-smoke
```

Expected: four exact-token records, prefetch traffic/wait counters positive, synchronous async counters zero, and no extra host sync.

- [x] **Step 5: Commit**

```bash
git add tools/ablate_cuda_transfer.py \
  tests/python/test_cuda_transfer_ablation.py checklist.md context-notes.md
git commit -m "feat: add asynchronous transfer ablation"
```

---

### Task 8: Full verification, B-0005, ledger, review, and publication

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `README.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`
- Create: `results/b0005-async-transfer.json`
- Create: `results/b0005-async-transfer-fp32/*`
- Create: `results/b0005-async-transfer-bf16/*`

**Interfaces:**
- Consumes: all milestone implementation and measurement artifacts.
- Produces: durable TITAN Ledger state and public Milestone 4 evidence.

- [x] **Step 1: Run complete CPU and CUDA suites**

```bash
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
K3X_BUILD_DIR=$PWD/build-cpu python -m pytest -q
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=$PWD/build-cuda python -m pytest -q
```

- [x] **Step 2: Run every CUDA sanitizer target**

Run Compute Sanitizer for all existing CUDA binaries plus `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, and `test_cuda_async_ffn`. Record each exact `ERROR SUMMARY`.

- [x] **Step 3: Regenerate and verify the bounded artifact**

Regenerate the deterministic synthetic source and K3X artifact through the streaming converter with `chunk_bytes=257`. Record converter maximum source read and artifact SHA-256. Do not download full weights.

- [x] **Step 4: Measure B-0005**

Run `ablate_cuda_transfer.py` with three warmups and 20 samples for FP32 and BF16. Preserve raw JSON/CSV. Do not select prefetch as default from throughput alone; analyze transfer duration, exposed stall, ready/late classification, pinned cost, and exactness together.

- [x] **Step 5: Validate accounting invariants**

For every row verify commit, hardware, environment, artifact identity, mode, context, warmup/sample counts, exact tokens, routing parity, precision error, and enabled switches. Mark unavailable NVMe, utilization, bandwidth, and I/O-stall fields as not measured. Cross-check compact manifest values against raw records programmatically.

- [x] **Step 6: Update durable documents in protocol order**

Update `ARCHITECTURE.md`, `DECISIONS.md`, `BENCHMARKS.md`, README, checklist, and context notes. Update `PROJECT_STATE.md` last with the latest measured bottleneck, current hardware, failures, next tasks, and last known-good commit/test state. Do not claim proposed L2/cache/predictor components as implemented.

- [x] **Step 7: Perform one final read-only review**

Capture `git status --short` and the full milestone diff. Request one Terra high reviewer to report only Critical or Important correctness, lifetime, event-ordering, accounting, and documentation issues. The reviewer must not modify files. Apply evidence-backed findings in one batch and rerun affected tests and sanitizer targets.

- [x] **Step 8: Commit and publish**

Commit the ledger/results, push `codex/milestone-four-async-l0-l1`, open a draft PR, wait for Linux CI, mark ready only after checks pass, verify `origin/main` ancestry, fast-forward public `main`, and confirm post-merge CI. Preserve the worktree for the next milestone.
