# K3X Exact Runtime, Profiler, and CUDA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a measured backend boundary, structured profiler, cuBLASLt dense baseline, and exact custom MXFP4 CUDA path while preserving the verified CPU synthetic graph.

**Architecture:** The existing CPU graph remains the numerical oracle. A narrow `ComputeBackend` owns dense and MXFP4 matrix-vector operations; profiling records CPU wall time, CUDA event time, and explicit byte traffic. `cuda-dense` changes only dense projection, while `cuda-custom` additionally replaces CPU MXFP4 with a native K3 E2M1/E8M0 custom kernel.

**Tech Stack:** C++20, CMake 3.25+, Python 3.12, pytest, CUDA 13.3, cuBLASLt, NVIDIA SM 12.0, JSON/CSV.

## Global Constraints

- Linux native is the final performance authority.
- Do not disable Windows Smart App Control or modify trust stores automatically.
- Do not download full Kimi K3 weights or provision paid cloud resources.
- CPU-only builds must not require a CUDA toolkit.
- Existing CPU layer/logit/state parity remains `atol=rtol=1e-6`; greedy tokens remain exact.
- CUDA FP32 dense parity is `atol=rtol=1e-5`.
- CUDA BF16-rounded-weight parity is `atol=rtol=2e-2` against the explicit FP32-accumulation oracle.
- Custom CUDA MXFP4 parity is `atol=rtol=1e-4` against native-byte CPU decode and FP32 accumulation.
- K3 MXFP4 E2M1/E8M0 group-32 bytes are never relabeled as cuBLASLt FP4, whose contract uses UE4M3/group-16 scaling.
- No throughput number is reported until the corresponding command runs successfully.
- Every new source file starts with a one-line Korean role comment.
- Production code follows a witnessed RED-GREEN-REFACTOR cycle.

---

### Task 0: Establish an executable Linux GPU verification environment

**Files:**
- Modify after verification: `PROJECT_STATE.md`

**Interfaces:**
- Consumes: RTX 5080, driver 591.86 or a compatible newer driver, CUDA-capable Linux native or WSL2 environment.
- Produces: a shell where newly built unsigned K3X test executables and `nvidia-smi` run successfully.

- [ ] **Step 1: Preserve the observed Windows failure evidence**

Run from Windows PowerShell:

```powershell
Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 100 |
  Format-List TimeCreated,Id,Message
```

Expected: events 3033 and 3077 identify `k3x_run.exe` and policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. Do not change the policy.

- [ ] **Step 2: Obtain explicit user authorization for the Linux environment**

If WSL2 is selected, the user explicitly runs or authorizes this system-changing command and any required reboot:

```powershell
wsl.exe --install -d Ubuntu-24.04
```

If Linux native is selected, record the distribution and mount/clone location instead. Do not execute this step without explicit authorization.

- [ ] **Step 3: Verify GPU passthrough and toolchain visibility**

Run inside Linux:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
nvcc --version
cmake --version
python3 --version
```

Expected: RTX 5080 is visible with compute capability 12.0; CUDA supports `sm_120`; Python is 3.12 or 3.13.

- [ ] **Step 4: Build and run the baseline tests that do not depend on the fixed `build` path**

Run:

```bash
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install -e '.[dev]'
.venv-linux/bin/cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
.venv-linux/bin/cmake --build build-linux
.venv-linux/bin/ctest --test-dir build-linux --output-on-failure
.venv-linux/bin/python -m pytest -q --ignore=tests/python/test_cpp_parity.py --ignore=tests/python/test_cpp_reader.py
```

Expected: CTest 2/2 and every non-cross-language Python test pass. The full 46-test baseline is completed in Task 1 immediately after the build-path fixture is added. If either command here fails, stop and diagnose before Task 1.

- [ ] **Step 5: Record the environment without claiming a new benchmark**

Append the distribution, driver, toolkit, and successful baseline commands to `PROJECT_STATE.md`. Do not add a `BENCHMARKS.md` entry because no new performance run occurred.

- [ ] **Step 6: Commit the environment record**

```bash
git add PROJECT_STATE.md
git commit -m "docs: record Linux GPU verification environment"
```

### Task 1: Make build-directory selection explicit in cross-language tests

**Files:**
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_cpp_reader.py`
- Modify: `tests/python/conftest.py`

**Interfaces:**
- Consumes: optional `K3X_BUILD_DIR` environment variable.
- Produces: `cpp_binary(name: str) -> Path`, resolving `${K3X_BUILD_DIR:-build}/<name>[.exe]` and failing with one diagnostic.

- [ ] **Step 1: Write the failing resolver test**

Add to `tests/python/test_cpp_reader.py`:

```python
def test_cpp_binary_uses_configured_build_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("K3X_BUILD_DIR", str(tmp_path / "native-build"))
    assert cpp_binary("test_reader").parent == tmp_path / "native-build"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
.venv-linux/bin/python -m pytest tests/python/test_cpp_reader.py::test_cpp_binary_uses_configured_build_directory -q
```

Expected: FAIL because `cpp_binary` does not exist.

- [ ] **Step 3: Add the shared fixture helper**

Add to `tests/python/conftest.py`:

```python
def cpp_binary(name: str) -> Path:
    build = Path(os.environ.get("K3X_BUILD_DIR", "build")).resolve()
    suffix = ".exe" if os.name == "nt" else ""
    return build / f"{name}{suffix}"
```

Import and use it from both cross-language test modules instead of duplicating `Path("build/...")` construction.

- [ ] **Step 4: Run targeted and full tests for GREEN**

Run:

```bash
K3X_BUILD_DIR=build-linux .venv-linux/bin/python -m pytest tests/python/test_cpp_reader.py tests/python/test_cpp_parity.py -q
K3X_BUILD_DIR=build-linux .venv-linux/bin/python -m pytest -q
```

Expected: targeted tests pass and the full Python suite passes 47/47 after adding the resolver test.

- [ ] **Step 5: Commit**

```bash
git add tests/python/conftest.py tests/python/test_cpp_reader.py tests/python/test_cpp_parity.py
git commit -m "test: support isolated native build directories"
```

### Task 2: Add deterministic runtime profiling primitives

**Files:**
- Create: `runtime/include/k3x/profile.hpp`
- Create: `runtime/src/profile.cpp`
- Create: `tests/cpp/test_profile.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: explicit `ProfileEvent` values recorded by CPU or CUDA code.
- Produces: `Profiler::record(ProfileEvent)`, `Profiler::events()`, and `Profiler::summary()`.

The public types are:

```cpp
enum class ProfilePhase { prefill, decode };
enum class ProfileOperation { tensor_read, dense_matvec, mxfp4_matvec, host_to_device, device_to_host };
enum class NumericPrecision { none, fp32, bf16_rounded, mxfp4_e2m1_e8m0 };
inline constexpr std::uint32_t profile_global_layer = UINT32_MAX;

struct ProfileEvent {
    ProfilePhase phase;
    ProfileOperation operation;
    NumericPrecision precision;
    std::uint32_t layer;
    std::uint64_t wall_nanoseconds;
    std::uint64_t device_nanoseconds;
    std::uint64_t logical_bytes;
    std::uint64_t transfer_bytes;
    bool success;
};

struct ProfileSummary {
    std::uint64_t wall_nanoseconds;
    std::uint64_t device_nanoseconds;
    std::uint64_t logical_bytes;
    std::uint64_t host_to_device_bytes;
    std::uint64_t device_to_host_bytes;
    std::size_t failed_operations;
};
```

- [ ] **Step 1: Write a failing aggregation test**

`tests/cpp/test_profile.cpp` records one successful FP32 dense event, one successful H2D event, and one failed MXFP4 event. Assert that precision is preserved, byte directions are separated, failed count is one, and the failed operation's duration is excluded from successful wall/device totals.

- [ ] **Step 2: Build and verify RED**

Run:

```bash
.venv-linux/bin/cmake --build build-linux --target test_profile
```

Expected: FAIL because `k3x/profile.hpp` and target `test_profile` do not exist.

- [ ] **Step 3: Implement the minimal profiler**

Implement `Profiler` as an owning `std::vector<ProfileEvent>` plus a single linear aggregation pass. Do not add clocks, threads, JSON, or CUDA dependencies to this class.

- [ ] **Step 4: Build and verify GREEN**

Run:

```bash
.venv-linux/bin/cmake --build build-linux --target test_profile
build-linux/test_profile
.venv-linux/bin/ctest --test-dir build-linux --output-on-failure
```

Expected: `test_profile` passes and CTest reports 3/3.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/profile.hpp runtime/src/profile.cpp tests/cpp/test_profile.cpp
git commit -m "feat: add deterministic runtime profiler"
```

### Task 3: Extract the exact CPU compute backend

**Files:**
- Create: `runtime/include/k3x/backend.hpp`
- Create: `runtime/src/backend_cpu.cpp`
- Create: `tests/cpp/test_backend.cpp`
- Modify: `runtime/src/model.cpp:56-67,82-168,331-354,400-460`
- Modify: `runtime/include/k3x/model.hpp:12-27`
- Modify: `runtime/src/main.cpp:11-40`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: row-major FP32 dense weights or native packed MXFP4 plus E8M0 scales.
- Produces: explicit CPU backend and backend-selected generation.

```cpp
enum class BackendKind { cpu, cuda_dense, cuda_custom };
enum class DensePrecision { fp32, bf16_rounded };

struct BackendOptions {
    BackendKind kind{BackendKind::cpu};
    DensePrecision dense_precision{DensePrecision::fp32};
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual BackendKind kind() const noexcept = 0;
    virtual Result<std::vector<float>> dense_matvec(
        std::span<const float> input, std::span<const float> weight,
        std::size_t rows, std::size_t cols, std::uint32_t layer,
        ProfilePhase phase) = 0;
    virtual Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, std::span<const std::byte> packed,
        std::span<const std::byte> scales, std::size_t rows,
        std::size_t cols, std::size_t group_size,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual BackendMemoryStats memory_stats() const noexcept = 0;
    virtual std::string_view device_name() const noexcept = 0;
};

std::unique_ptr<ComputeBackend> make_cpu_backend(Profiler* profiler);
Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions& options, Profiler* profiler);
```

`BackendMemoryStats` contains `current_device_bytes` and `peak_device_bytes`. The CPU backend returns zeros and device name `CPU`.

- [ ] **Step 1: Write failing CPU backend tests**

Test a literal 2x3 dense matrix and the existing 1x32 MXFP4 literal. Assert the exact existing CPU results and `BackendKind::cpu`.

- [ ] **Step 2: Build and verify RED**

Run:

```bash
.venv-linux/bin/cmake --build build-linux --target test_backend
```

Expected: FAIL because `backend.hpp`, `make_cpu_backend`, and `test_backend` do not exist.

- [ ] **Step 3: Move only matrix operations behind `CpuBackend`**

Move the current double-accumulation `matvec` body from `model.cpp` into `backend_cpu.cpp`. Delegate MXFP4 to the existing `k3x::mxfp4_matmul`. Record successful and failed events when a profiler is present.

Add this primary generation overload:

```cpp
Result<GenerationResult> generate_greedy(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt, std::size_t count,
    bool incremental, bool diagnostics);
```

Keep the existing overload as a convenience wrapper that constructs `CpuBackend`, preserving current callers and numerical behavior.

Change the internal `Engine::forward` signature to accept `ProfilePhase`. `generate_greedy` passes `prefill` for prompt execution and `decode` for later token forwards. Operations outside a decoder layer, such as the LM head, use `profile_global_layer`.

- [ ] **Step 4: Run unit and full parity tests for GREEN**

Run:

```bash
.venv-linux/bin/cmake --build build-linux
.venv-linux/bin/ctest --test-dir build-linux --output-on-failure
K3X_BUILD_DIR=build-linux .venv-linux/bin/python -m pytest tests/python/test_cpp_parity.py -q
K3X_BUILD_DIR=build-linux .venv-linux/bin/python -m pytest -q
```

Expected: CTest 4/4; all Python tests pass; token sequence remains `[43,32,28,49,9,28]`; diagnostic arrays retain `1e-6` parity.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/backend.hpp runtime/include/k3x/model.hpp runtime/src/backend_cpu.cpp runtime/src/model.cpp runtime/src/main.cpp tests/cpp/test_backend.cpp
git commit -m "refactor: isolate exact CPU compute backend"
```

### Task 4: Add an optional SM 12.0 CUDA backend shell

**Files:**
- Create: `runtime/src/backend_cuda_stub.cpp`
- Create: `runtime/cuda/backend_cuda.cu`
- Create: `tests/cpp/test_backend_unavailable.cpp`
- Create: `tests/cuda/test_cuda_device.cu`
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/include/k3x/status.hpp`
- Modify: `CMakeLists.txt:1-33`

**Interfaces:**
- Consumes: `BackendOptions`, optional CUDA 13.3 toolkit, and an SM 12.0 device.
- Produces: `make_cuda_backend(options, profiler)` returning either a CUDA backend or typed `backend_unavailable` / `unsupported_architecture` failure.

- [ ] **Step 1: Write the CPU-only unavailable test**

With `K3X_ENABLE_CUDA=OFF`, request `BackendKind::cuda_dense` and assert `ErrorCode::backend_unavailable`. This ensures CPU builds never silently fall back.

- [ ] **Step 2: Build CPU-only and verify RED**

Run:

```bash
.venv-linux/bin/cmake --fresh -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DK3X_ENABLE_CUDA=OFF
.venv-linux/bin/cmake --build build-cpu --target test_backend_unavailable
```

Expected: FAIL because the error code and factory do not exist.

- [ ] **Step 3: Implement the disabled stub and optional CMake branch**

Add `option(K3X_ENABLE_CUDA "Build the CUDA backend" OFF)`. The OFF build compiles `backend_cuda_stub.cpp`. The ON build enables CUDA, requires CUDAToolkit 13.3, links `CUDA::cudart` and `CUDA::cublasLt`, sets `CUDA_ARCHITECTURES 120-real`, and compiles `backend_cuda.cu` instead of the stub.

- [ ] **Step 4: Write and run the CUDA device RED test**

`tests/cuda/test_cuda_device.cu` constructs the backend, queries the selected device, and asserts compute capability 12.0 or newer and backend identity `cuda_dense`.

Run:

```bash
.venv-linux/bin/cmake --fresh -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DK3X_ENABLE_CUDA=ON
.venv-linux/bin/cmake --build build-cuda --target test_cuda_device
```

Expected before the real factory is implemented: FAIL at link or runtime with `backend_unavailable`.

- [ ] **Step 5: Implement CUDA resource ownership and capability validation**

Use RAII owners for one nonblocking CUDA stream and one cuBLASLt handle. Query `cudaDevAttrComputeCapabilityMajor` and `Minor`; reject devices below the accepted architecture before allocating operation buffers. Wrap every owned `cudaMalloc`/`cudaFree` pair so `BackendMemoryStats` tracks current and peak allocation. Store no global CUDA state.

- [ ] **Step 6: Verify GREEN and CPU isolation**

Run:

```bash
.venv-linux/bin/cmake --build build-cuda
.venv-linux/bin/ctest --test-dir build-cuda --output-on-failure
.venv-linux/bin/cmake --build build-cpu
.venv-linux/bin/ctest --test-dir build-cpu --output-on-failure
```

Expected: CUDA device test passes on RTX 5080; CPU-only suite passes without linking CUDA libraries.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/backend.hpp runtime/include/k3x/status.hpp runtime/src/backend_cuda_stub.cpp runtime/cuda/backend_cuda.cu tests/cpp/test_backend_unavailable.cpp tests/cuda/test_cuda_device.cu
git commit -m "feat: add optional RTX 5080 CUDA backend"
```

### Task 5: Implement cuBLASLt dense FP32 and BF16-rounded baselines

**Files:**
- Create: `tests/cuda/test_cuda_dense.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: row-major FP32 input and weights plus `DensePrecision`.
- Produces: FP32 output, CUDA event timing, and exact H2D/D2H byte records.

- [x] **Step 1: Write a failing FP32 literal test**

Use input `[1,2,3]` and row-major weights `[[1,0,-1],[0.5,2,1]]`. Assert output `[-2,7.5]` at `1e-5`, one dense device event, 36 H2D bytes, and 8 D2H bytes.

- [x] **Step 2: Run and verify RED**

Run:

```bash
.venv-linux/bin/cmake --build build-cuda --target test_cuda_dense
build-cuda/test_cuda_dense
```

Expected: FAIL because CUDA dense matvec returns `backend_unavailable`.

- [x] **Step 3: Implement the minimal FP32 cuBLASLt matmul**

Create row/column descriptors that preserve the public row-major result, use `CUBLAS_COMPUTE_32F`, copy input and weights on the backend stream, select one heuristic with zero workspace first, launch, record CUDA events, copy FP32 output, and synchronize only before returning the host result.

- [x] **Step 4: Verify FP32 GREEN**

Run the same target and expect the literal output and profiler byte counts to pass.

- [x] **Step 5: Write the failing BF16-rounded test**

Use weights containing values not exactly representable in BF16. Build the oracle by converting each weight FP32 → BF16 → FP32, then perform FP32 accumulation. Assert `2e-2` parity and confirm the profile labels the selected dense precision.

- [x] **Step 6: Implement BF16 operand staging**

Convert input and weights to `__nv_bfloat16` in bounded host buffers for this baseline, use the CUDA 13.3 documented shared BF16 A/B type with FP32 compute and FP32 output, and record the actual BF16 transfer byte count. Do not mutate or cache the K3X tensor.

- [x] **Step 7: Run CUDA and CPU regression suites**

Run:

```bash
.venv-linux/bin/cmake --build build-cuda
.venv-linux/bin/ctest --test-dir build-cuda --output-on-failure
.venv-linux/bin/ctest --test-dir build-cpu --output-on-failure
```

Expected: CUDA dense tests pass in both precisions and CPU tests remain unchanged.

- [x] **Step 8: Commit**

```bash
git add CMakeLists.txt runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_dense.cu
git commit -m "feat: add cuBLASLt dense baselines"
```

### Task 6: Implement exact custom K3 MXFP4 CUDA matvec

**Files:**
- Create: `runtime/cuda/mxfp4.cuh`
- Create: `runtime/cuda/mxfp4.cu`
- Create: `tests/cuda/test_cuda_mxfp4.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: K3X low-nibble-first E2M1 packed weights, one E8M0 scale byte per 32 flattened values, FP32 input, rows, and columns.
- Produces: FP32 accumulated output without runtime repacking or requantization.

- [x] **Step 1: Write the failing literal decode-and-matvec test**

Reuse the existing CPU literal with packed byte `0x10` and scale byte `127`, then add sign, exponent, high-nibble, two scale groups, and a non-multiple-of-warp row count. Compare the complete output vector with `k3x::mxfp4_matmul` at `1e-4`.

- [x] **Step 2: Run and verify RED**

Run:

```bash
.venv-linux/bin/cmake --build build-cuda --target test_cuda_mxfp4
build-cuda/test_cuda_mxfp4
```

Expected: FAIL because `BackendKind::cuda_custom` has no MXFP4 implementation.

- [x] **Step 3: Implement the minimal kernel**

Assign one CUDA block per output row. Threads stride columns, decode low nibble before high nibble, apply `2^(scale_byte-127)` to each 32-value group, multiply by FP32 input, reduce in FP32 shared memory, and write one FP32 output. Validate packed length, scale length, rows, columns, and group size before launch.

- [x] **Step 4: Record exact traffic and launch timing**

Record packed bytes, scale bytes, FP32 input H2D, FP32 output D2H, and CUDA-event kernel duration as separate events. On allocation, copy, launch, or synchronization failure, record one failed operation and return its typed status.

- [x] **Step 5: Verify GREEN and run sanitizer**

Run:

```bash
.venv-linux/bin/cmake --build build-cuda
build-cuda/test_cuda_mxfp4
compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
.venv-linux/bin/ctest --test-dir build-cuda --output-on-failure
```

Expected: numerical tests pass; compute-sanitizer reports zero errors; all CUDA and CPU tests pass.

- [x] **Step 6: Add the incompatible-library regression assertion**

In `test_cuda_mxfp4.cu`, assert the K3X request reports scale kind `e8m0_group32` and cannot be routed to a backend operation labeled cuBLASLt FP4. This is a contract test, not a runtime conversion.

- [x] **Step 7: Commit**

```bash
git add CMakeLists.txt runtime/cuda/mxfp4.cuh runtime/cuda/mxfp4.cu runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_mxfp4.cu
git commit -m "feat: add exact K3 MXFP4 CUDA baseline"
```

### Task 7: Integrate backend selection and profiler output end to end

**Files:**
- Modify: `runtime/include/k3x/model.hpp`
- Modify: `runtime/src/model.cpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`
- Modify: `tools/benchmark_synthetic.py`

**Interfaces:**
- Consumes: CLI `--backend cpu|cuda-dense|cuda-custom`, `--dense-precision fp32|bf16`, and existing generation arguments.
- Produces: one JSON execution record and one benchmark JSON/CSV row with backend, device, transfer, timing, memory, and numerical fields.

- [ ] **Step 1: Write failing CLI validation tests**

Add tests that unknown backend and precision values return exit code 2, CUDA requests on CPU builds return `backend_unavailable`, and the default invocation remains CPU FP32.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
K3X_BUILD_DIR=build-cpu .venv-linux/bin/python -m pytest tests/python/test_cpp_parity.py -q
```

Expected: new CLI tests fail because both arguments are unknown.

- [ ] **Step 3: Add explicit CLI backend construction**

Parse the two flags, create the requested backend, pass it to the primary generation overload, and emit backend/device/precision fields. Never catch `backend_unavailable` to create a CPU backend.

- [ ] **Step 4: Write failing profiler schema tests**

Extend `BenchmarkRecord` with these fields and assert exact JSON/CSV preservation:

```python
backend: str
device: str
dense_precision: str
kernel_nanoseconds: int
host_to_device_bytes: int
device_to_host_bytes: int
peak_vram_bytes: int | None
max_absolute_error: float | None
max_relative_error: float | None
```

Also change scope validation to accept `synthetic-milestone-zero` and `synthetic-milestone-one` explicitly.

- [ ] **Step 5: Run schema test and verify RED**

Run:

```bash
.venv-linux/bin/python -m pytest tests/python/test_benchmark_schema.py -q
```

Expected: FAIL because `BenchmarkRecord` lacks the new fields.

- [ ] **Step 6: Implement JSON/CSV plumbing**

Serialize `Profiler::summary()` and `ComputeBackend::memory_stats()` from `k3x_run`; aggregate medians in `benchmark_synthetic.py`; retain `Reader::read_bytes()` as `file_read_bytes_per_token`; never populate NVMe GB/token from that logical-read value.

- [ ] **Step 7: Add end-to-end CPU and CUDA parity tests**

For prompt `[1,7,3,9]`, run `cpu`, `cuda-dense`, and `cuda-custom`. CPU remains `1e-6` against PyTorch. CUDA outputs use their declared operation tolerances, while all three token sequences must equal `[43,32,28,49,9,28]`.

- [ ] **Step 8: Verify GREEN**

Run:

```bash
.venv-linux/bin/cmake --build build-cpu
.venv-linux/bin/ctest --test-dir build-cpu --output-on-failure
K3X_BUILD_DIR=build-cpu .venv-linux/bin/python -m pytest -q
.venv-linux/bin/cmake --build build-cuda
.venv-linux/bin/ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda .venv-linux/bin/python -m pytest -q
```

Expected: every CPU and CUDA test passes; CPU output is unchanged; CUDA modes produce exact token IDs and bounded numerical error.

- [ ] **Step 9: Commit**

```bash
git add runtime/include/k3x/model.hpp runtime/src/model.cpp runtime/src/main.cpp tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py tools/benchmark_synthetic.py
git commit -m "feat: profile backend-selected synthetic inference"
```

### Task 8: Measure, document, review, and publish Milestone 1

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify last: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: passing CPU/CUDA test suites and raw benchmark JSON/CSV.
- Produces: evidence-backed Milestone 1 report and updated TITAN LEDGER.

- [ ] **Step 1: Generate the deterministic artifact**

Run:

```bash
.venv-linux/bin/python tools/generate_synthetic.py --output artifacts/synthetic-source
.venv-linux/bin/python -m k3x_converter.cli convert artifacts/synthetic-source artifacts/synthetic.k3x --chunk-bytes 257
```

Expected: conversion succeeds, strict verification passes, and no full Kimi K3 weights are downloaded.

- [ ] **Step 2: Run three measured benchmark modes**

Run each with three warmups and twenty samples:

```bash
.venv-linux/bin/python tools/benchmark_synthetic.py --artifact artifacts/synthetic.k3x --runner build-cuda/k3x_run --backend cpu --warmup 3 --iterations 20 --json results/m1-cpu.json --csv results/m1-cpu.csv
.venv-linux/bin/python tools/benchmark_synthetic.py --artifact artifacts/synthetic.k3x --runner build-cuda/k3x_run --backend cuda-dense --warmup 3 --iterations 20 --json results/m1-cuda-dense.json --csv results/m1-cuda-dense.csv
.venv-linux/bin/python tools/benchmark_synthetic.py --artifact artifacts/synthetic.k3x --runner build-cuda/k3x_run --backend cuda-custom --warmup 3 --iterations 20 --json results/m1-cuda-custom.json --csv results/m1-cuda-custom.csv
```

Expected: three measured records with actual values. Do not prescribe which backend wins.

- [ ] **Step 3: Run final verification**

Run:

```bash
.venv-linux/bin/ctest --test-dir build-cpu --output-on-failure
K3X_BUILD_DIR=build-cpu .venv-linux/bin/python -m pytest -q
.venv-linux/bin/ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda .venv-linux/bin/python -m pytest -q
compute-sanitizer --tool memcheck build-cuda/test_cuda_mxfp4
git diff --check
```

Expected: all tests and sanitizer checks pass with no whitespace errors.

- [ ] **Step 4: Update the ledger in required order**

Update `ARCHITECTURE.md` statuses only for code that passed. Add accepted/rejected implementation decisions to `DECISIONS.md`. Add all three actual measurement records to `BENCHMARKS.md`, leaving unavailable counters explicitly not measured. Update `PROJECT_STATE.md` last with the current milestone, completed work, blockers, next task, bottleneck, and last known-good commit/tests.

- [ ] **Step 5: Update public documentation**

Add exact build commands, backend semantics, measured comparison, and limitations to `README.md`. Keep full-model targets labeled as targets and synthetic numbers labeled synthetic.

- [ ] **Step 6: Self-review the complete diff**

Check every changed line against the approved design, search for debug output and accidental fallback, verify every caller of `generate_greedy`, and ensure no proposed TITAN component is marked implemented.

- [ ] **Step 7: Commit the milestone**

```bash
git add README.md ARCHITECTURE.md DECISIONS.md BENCHMARKS.md PROJECT_STATE.md checklist.md context-notes.md
git commit -m "docs: record exact runtime CUDA baseline"
```

- [ ] **Step 8: Request one final review and publish after it passes**

Run the repository's review workflow once, address Critical/Important findings in one batch, rerun final verification, fast-forward `main`, push the public repository, and require the Linux correctness workflow to finish successfully before claiming completion.
