# K3X Exact Runtime, Profiler, and CUDA Baseline Design

## 1. Goal

Milestone 1 turns the verified synthetic K3X graph into a backend-driven runtime with reproducible CPU and RTX 5080 measurements. It preserves the current portable CPU path as the correctness oracle, adds structured profiling, and introduces an optional CUDA backend without changing routing, Top-K, state transitions, or generated-token semantics.

This milestone does not claim full-model Kimi K3 performance. It establishes the measurement and correctness foundation needed before tiered storage, expert caching, prefetch, or fused kernels can be evaluated.

## 2. Confirmed development environment

- GPU: NVIDIA GeForce RTX 5080 with 16,303 MiB reported memory.
- Runtime-reported compute capability: 12.0.
- Driver: 591.86.
- CUDA toolkit: 13.3, nvcc 13.3.73.
- Local nvcc supports `compute_120` and `sm_120` code generation.
- Installed CUDA 13.3 headers expose `CUDA_R_4F_E2M1` and `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`, but NVIDIA's cuBLAS contract assigns UE8M0/32 scaling to FP8. cuBLASLt FP4 requires UE4M3/16 scaling and is not a direct K3 MXFP4 path.
- Windows Smart App Control currently blocks the newly linked unsigned `k3x_run.exe` with Code Integrity events 3033 and 3077. K3X will not disable or weaken that policy automatically.

The production target remains Linux native. Windows is a development build target, not the final performance authority.

## 3. Considered approaches

### Custom CUDA only

This gives direct control over K3-specific layouts and future fusion, but it makes early numerical failures difficult to separate from layout, launch, and arithmetic errors.

### Library CUDA only

cuBLASLt provides a strong dense projection baseline. Its native FP4 path uses UE4M3 scales per 16 values, whereas K3 MXFP4 uses E8M0 scales per 32 values. Using it for exact K3 experts would require a format conversion and would no longer preserve the native expert path.

### Selected hybrid baseline

The runtime will use cuBLASLt as the independent dense FP32/BF16 baseline and a minimal custom CUDA MXFP4 path as the controlled exact expert implementation. The custom path consumes native K3X expert bytes and is compared against the CPU oracle. A direct cuBLASLt FP4 expert path is rejected for this milestone because its scaling contract is incompatible.

## 4. Scope

### Included

- Preserve the existing portable CPU graph and strict artifact verification.
- Add a narrow compute-backend boundary for projection and MXFP4 matrix multiplication.
- Keep KDA, MLA, router, Attention Residual, and recurrent-state semantics unchanged.
- Add optional CUDA 13.3 build support without making CUDA a CPU-build dependency.
- Generate an SM 12.0 native cubin for the RTX 5080 build.
- Add cuBLASLt FP32/BF16 projection baselines.
- Add a regression test that records the rejected cuBLASLt FP4 compatibility assumption: K3X uses E8M0/32 while cuBLASLt FP4 requires UE4M3/16.
- Add a minimal custom CUDA MXFP4 decode-and-matmul implementation.
- Add structured runtime profiling and JSON/CSV benchmark output.
- Compare CPU, CUDA library, and CUDA custom paths on the synthetic checkpoint.
- Record actual failures and unsupported configurations without silent CPU fallback.

### Excluded

- Full Kimi K3 checkpoint download or execution.
- NVMe/RAM/VRAM tiered caching and asynchronous prefetch.
- CUDA Graphs, persistent kernels, or multi-operation fusion.
- Adaptive Top-K, cold-expert rescue, speculative decoding, and proxy experts.
- Mixed-precision calibration or quality claims beyond the synthetic numerical tests.
- Automatic changes to Windows security policy, certificate stores, WSL, or paid cloud resources.

## 5. Architecture

### 5.1 Backend boundary

`ComputeBackend` owns only operations that can move intact between CPU and CUDA in this milestone.

- Dense matrix multiplication with FP32 input/output and FP32 or BF16 weights.
- Native MXFP4 matrix multiplication with packed E2M1 values and E8M0 scales.
- Explicit synchronization needed for result consumption and measurement.
- Backend identity and capability reporting.

The existing CPU implementation becomes `CpuBackend` without changing its arithmetic contract. `CudaBackend` owns CUDA streams, cuBLASLt resources, device allocations, and launch error propagation. Model code selects a backend explicitly; an unavailable CUDA backend returns an error rather than silently running on CPU.

This boundary is intentionally smaller than a general tensor library. Elementwise attention, routing, recurrent-state updates, and residual logic stay in the existing C++ model until projection and MXFP4 parity are proven.

### 5.2 CUDA paths

The library path uses cuBLASLt only for dense FP32/BF16 projection. It does not repack K3X MXFP4 into NVIDIA's distinct UE4M3/16 FP4 format.

The custom path performs K3X E2M1 decode and FP32 accumulation in a small CUDA kernel. It is a correctness and profiling baseline, not a claimed optimized kernel. It must not repack or requantize the expert payload during checkpoint conversion.

`cuda-dense` uses cuBLASLt for dense projection while retaining the CPU MXFP4 oracle. `cuda-custom` retains the same cuBLASLt dense path and replaces only MXFP4 matrix multiplication with the custom kernel. This makes the expert-path comparison single-variable.

The CMake CUDA option is disabled by default. Enabling it requires CUDA 13.3 or newer and emits real `sm_120` code for the target build. CPU-only Linux CI remains valid when no CUDA toolkit is present.

### 5.3 Profiling model

Profiling data is collected through a runtime-owned `Profiler` rather than embedded ad hoc in individual kernels. Each record carries a backend, phase, layer, operation, duration, logical bytes, host-to-device bytes, device-to-host bytes, and success state.

CPU durations use `steady_clock`. GPU kernel and copy durations use CUDA events recorded on the executing stream. End-to-end wall time remains separate from summed operation time so synchronization and launch overhead are visible rather than hidden.

The benchmark schema adds the following measured fields while retaining Milestone 0 fields.

- Backend and device metadata.
- Per-operation and per-layer CPU wall time.
- CUDA kernel, host-to-device, and device-to-host time.
- Logical K3X bytes read.
- RAM-to-GPU and GPU-to-RAM bytes.
- Peak process RSS and peak CUDA allocation owned by K3X.
- Prefill tokens per second, decode tokens per second, and TTFT.
- Numerical comparison target and observed maximum absolute/relative error.

GPU utilization and memory-bandwidth counters requiring Nsight or CUPTI are reported as unavailable in this milestone unless a supported collector is actually integrated and validated. They are never estimated from kernel time.

## 6. Data flow

1. The strict K3X reader verifies the complete artifact before execution.
2. Model execution requests a dense or MXFP4 operation from the selected backend.
3. `CpuBackend` reads the required extent and executes the existing portable arithmetic.
4. `CudaBackend` copies only the required synthetic tensor and input into owned device buffers, records transfer metrics, executes either cuBLASLt or the custom kernel, and copies the result back for the still-CPU graph.
5. The unchanged graph performs routing, state transitions, attention, residual, and greedy token selection.
6. The profiler emits one structured in-memory result that the CLI serializes to JSON; the benchmark driver produces CSV from the same schema.

Per-operation transfers are deliberately retained in the baseline because they expose the cost that future residency and fusion must remove. This milestone does not conceal that cost with an unmeasured tensor cache.

## 7. Numerical contracts

- Existing CPU layer outputs, logits, recurrent state, and tokens retain the `atol=rtol=1e-6` contract against PyTorch FP32.
- CUDA FP32 projection is compared with the CPU FP32 operation at `atol=rtol=1e-5`.
- CUDA native MXFP4 with FP32 accumulation is compared with the CPU MXFP4 oracle at `atol=rtol=1e-4`.
- CUDA BF16 projection is compared at `atol=rtol=2e-2` with an explicit PyTorch oracle that rounds weights through BF16, converts those rounded values back to FP32, and performs FP32 matrix multiplication.
- Greedy token IDs must match the corresponding reference path exactly.
- BF16 measurements are labeled as BF16 and cannot replace strict CPU correctness results.

If a tolerance fails, the test reports the operation, shape, element index, expected value, actual value, and maximum absolute/relative error. Tests are not weakened to accommodate unexplained drift.

## 8. Error handling

- CUDA disabled at build time returns `backend_unavailable` for an explicit CUDA request.
- Unsupported architecture, toolkit version, data type, group size, or layout returns a typed error before launch.
- CUDA allocation, copy, launch, synchronization, and cuBLASLt errors retain the originating status and operation name.
- A cuBLASLt dense heuristic miss does not silently switch to another backend. The caller selects the path, and benchmarks record it.
- Partial profiler records are marked failed and are not included in throughput medians.
- K3X checksum or directory failure remains fatal before either backend sees tensor bytes.

## 9. Testing strategy

Implementation follows test-driven development. Every production behavior begins with a focused failing test and an observed expected failure.

### CPU regression

- Run all existing CTest and Python tests unchanged.
- Confirm all CPU graph diagnostics and generated tokens retain their current contract.
- Verify profiling disabled mode does not change numerical output.

### Profiler

- Test deterministic aggregation and JSON/CSV schema using synthetic records.
- Test byte accounting independently from timing values.
- Test that failed operations are represented but excluded from throughput summaries.

### CUDA unit tests

- Test device capability reporting and explicit unavailable behavior.
- Test literal dense FP32 and BF16 matrices.
- Test literal MXFP4 nibble order, E2M1 values, E8M0 scales, and group boundaries.
- Compare the custom MXFP4 path against the CPU oracle and verify that no runtime path labels cuBLASLt's incompatible UE4M3/16 format as K3 MXFP4.
- Exercise invalid dimensions, invalid group size, allocation failure propagation where safely injectable, and launch error reporting.

### End-to-end synthetic test

- Convert the deterministic synthetic checkpoint once per test fixture.
- Run CPU, CUDA dense-only, and CUDA custom-MXFP4 modes with the same prompt.
- Compare layer outputs, logits, recurrent state, and generated token IDs under the declared numerical contracts.
- Produce JSON and CSV artifacts and validate every required field.

### Platform gates

- CPU-only Linux GitHub Actions must remain green.
- CUDA compilation must succeed with the installed CUDA 13.3 toolchain and `sm_120` target.
- CUDA execution is accepted only after running on Linux native or a user-approved WSL2 GPU environment. The current Windows Smart App Control block is not bypassed automatically.

## 10. Completion criteria

Milestone 1 is complete only when all of the following are true.

- Existing CPU correctness tests pass without relaxed assertions.
- The optional CUDA build does not affect CPU-only builds.
- Dense and MXFP4 CUDA operations pass their numerical contracts on the RTX 5080.
- The full synthetic graph produces matching greedy tokens for every enabled backend path.
- Reproducible JSON and CSV contain real wall, kernel, transfer, byte, memory, and error measurements.
- A CPU-versus-CUDA baseline report identifies the measured next bottleneck without extrapolating synthetic throughput to full Kimi K3.
- Documentation records the exact build, device, driver, toolkit, test commands, and limitations.

## 11. Next milestone boundary

After this baseline is measured, Milestone 2 may introduce persistent tensor residency and the L0/L1/L2 asynchronous pipeline. Its first hypothesis will be evaluated against the per-operation transfer and stall measurements produced here. No cache policy or fusion work begins merely from the theoretical performance model.
