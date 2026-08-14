# K3X Admission-Time Immutable Validation and Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve complete CUDA MoE-layer validation correctness while removing repeated immutable-weight scans from the admission-mode hot path and measure the effect with B-0024.

**Architecture:** `CudaBackend` retains a private tensor-identity registry and performs an atomic host-only preflight before resident acquisition. A new reference-preserving validation mode selects current per-call scans or exact-identity admission reuse, while monotonic telemetry and an eighteen-row released-dimension matrix attribute validation and profiler costs independently.

**Tech Stack:** C++20, CUDA 13.3 native `sm_120`, cuBLASLt, CMake/Ninja, Python 3.12, pytest 9.1.1, JSON/CSV/SHA-256, Compute Sanitizer.

## Global Constraints

- `BackendOptions::cuda_weight_validation` defaults to `per_call`; no existing default changes before measurement.
- Admission identity is tensor ID, host pointer, element count, byte count, rows, and columns, scoped to one CUDA backend lifetime.
- All new source files begin with a one-line Korean role comment.
- Dynamic input, contributions, scalars, shapes, IDs, and MXFP4 layouts remain per-call checks.
- Failed complete preflight performs no resident acquisition, CUDA allocation, upload, scratch reservation, event creation, launch, or synchronization.
- B-0024 emits no token, prefill, or TTFT metric and uses no full checkpoint or paid cloud resource.
- `kernel_nanoseconds` is JSON `null` when profiling is disabled.
- `PROJECT_STATE.md` is updated last after every meaningful milestone.

---

### Task 1: Validation option, CLI ownership, and telemetry schema

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Produces: `enum class CudaWeightValidationMode { per_call, admission }`.
- Produces: `BackendOptions::cuda_weight_validation` with `per_call` default.
- Produces: four `BackendRuntimeStats` fields named `immutable_validation_scans`, `immutable_validation_hits`, `immutable_validation_bytes`, and `immutable_validation_nanoseconds`.
- Produces: public CLI `--cuda-weight-validation per-call|admission` and target/draft JSON identity plus counters.

- [ ] **Step 1: Write failing option and schema tests**

Add an invalid CLI case expecting `unknown CUDA weight validation mode: cached`. Extend default-output assertions to require `cuda_weight_validation == "per-call"`, zero target counters, `draft_cuda_weight_validation == "per-call"`, and zero draft counters. Add an admission ownership case showing CPU rejects the non-default option.

- [ ] **Step 2: Run the focused tests and witness RED**

```bash
K3X_BUILD_DIR=build-wsl /home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
  -k 'weight_validation or default_runtime or benchmark_record' -q
```

Expected failure is a missing CLI option, missing JSON fields, or missing dataclass fields.

- [ ] **Step 3: Implement the minimal option and schema**

Add the enum, option, and counters. Parse only `per-call` and `admission`. Reject `admission` unless the target is `cuda-custom + fp32 + reused + resident + resident-grid + moe-layer + synchronous + fusion-none`. Apply the same mode to a CUDA AURORA draft backend only when its boundary is `moe-layer`; otherwise keep its default. Serialize target and draft mode/counters and add zero-default fields to `BenchmarkRecord` plus CSV field lists and sample aggregation.

- [ ] **Step 4: Rebuild CPU and verify GREEN**

```bash
cmake -S . -B build-wsl -G Ninja -DCMAKE_BUILD_TYPE=Release -DK3X_ENABLE_CUDA=OFF
cmake --build build-wsl -j2
K3X_BUILD_DIR=build-wsl /home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py -q
```

- [ ] **Step 5: Commit the public contract**

```bash
git add runtime/include/k3x/backend.hpp runtime/src/main.cpp
  tools/benchmark_synthetic.py tests/python/test_cpp_parity.py
  tests/python/test_benchmark_schema.py
git commit -m "runtime: add CUDA validation modes"
```

---

### Task 2: Atomic admission registry and correctness gates

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_moe_layer.cu`

**Interfaces:**
- Consumes: `CudaWeightValidationMode` and validation counters from Task 1.
- Produces: private `DenseValidationIdentity` and a backend-lifetime registry keyed by tensor ID.
- Produces: `validate_immutable_layer_weights(std::span<const DenseWeightView>)` returning `Result<bool>` before resident acquisition.

- [ ] **Step 1: Add failing CUDA admission tests**

Extend `test_cuda_moe_layer` with separate `per_call` and `admission` backends. Call each exact layer twice and assert literal counter values. Per-call must report twelve scans, zero hits, and twice the fixture dense bytes. Admission must report six scans, six hits, and one fixture dense-byte total.

Add a mutable fixture test that places `infinity` in `shared.down`, calls admission mode, and asserts `invalid_mxfp4` with all resident, allocation, H2D, scratch, layer, launch, fallback, and synchronization counters unchanged. Restore the value and call again; six scans must occur, proving the failed staged set was not cached.

Add an identity-conflict test that first admits a valid layer, then supplies the same routed-down tensor ID from a different host array. Require `invalid_mxfp4` and no delta in resident bytes, allocation count, H2D bytes, launches, or synchronization.

- [ ] **Step 2: Build the CUDA test and witness RED**

```bash
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DK3X_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda --target test_cuda_moe_layer -j2
ctest --test-dir build-cuda -R '^cuda_moe_layer$' --output-on-failure
```

Expected failure is missing counters or repeated admission scans.

- [ ] **Step 3: Implement atomic host preflight**

Define the identity with `const float* data`, `size`, `rows`, and `cols`. In per-call mode, scan every dense view and record six scans. In admission mode, first classify every view as exact hit, staged new identity, or conflict. Scan the complete staged set; commit all staged identities only if every scan succeeds. Record nanoseconds around only lookup and finite scanning. Remove the unconditional six `all_finite` predicates from the main validation condition while retaining input and contribution scans.

- [ ] **Step 4: Verify CUDA GREEN and sanitizer safety**

```bash
cmake --build build-cuda --target test_cuda_moe_layer test_cuda_residency -j2
ctest --test-dir build-cuda -R 'cuda_moe_layer|cuda_residency' --output-on-failure
/usr/local/cuda-13.3/compute-sanitizer/compute-sanitizer
  --tool memcheck --error-exitcode 99 build-cuda/test_cuda_moe_layer
```

- [ ] **Step 5: Commit the validation boundary**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_moe_layer.cu
git commit -m "cuda: cache immutable layer validation"
```

---

### Task 3: Released benchmark validation and profiler switches

**Files:**
- Modify: `runtime/src/cuda_moe_layer_bench.cpp`
- Modify: `tests/python/test_cuda_released_moe_layer.py`

**Interfaces:**
- Consumes: validation modes and counters from Tasks 1 and 2.
- Produces: strict benchmark arguments `--validation per-call|admission` and `--profiler on|off`.
- Produces: cold and warm validation deltas plus nullable `kernel_nanoseconds`.

- [ ] **Step 1: Add failing live CLI tests**

Parameterize the existing test over both validation modes and profiler states for one expert. Require identity fields, positive cold validation bytes for complete-layer rows, literal per-call/admission warm counter behavior, and `kernel_nanoseconds is None` only when the profiler is off. Add invalid values `cached` and `maybe` with exact error messages.

- [ ] **Step 2: Run the focused live test and witness RED**

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cuda_released_moe_layer.py -q
```

Expected failure is unknown arguments or missing validation telemetry.

- [ ] **Step 3: Implement strict benchmark ownership**

Pass `CudaWeightValidationMode` into selected backend options. Construct a `Profiler` only for `on`, snapshot validation counters before and after the cold call and measured loop, and emit `validation`, `profiler_enabled`, cold/warm validation fields, and nullable kernel time. The oracle remains per-call and is destroyed before the selected backend is created.

- [ ] **Step 4: Verify all benchmark CLI cases**

```bash
cmake --build build-cuda --target k3x_cuda_moe_layer_bench -j2
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cuda_released_moe_layer.py -q
```

- [ ] **Step 5: Commit the benchmark contract**

```bash
git add runtime/src/cuda_moe_layer_bench.cpp
  tests/python/test_cuda_released_moe_layer.py
git commit -m "bench: expose validation attribution"
```

---

### Task 4: B-0024 ablation runner and evidence gates

**Files:**
- Create: `tools/ablate_cuda_admission_validation.py`
- Create: `tests/python/test_cuda_admission_validation_ablation.py`

**Interfaces:**
- Consumes: `k3x_cuda_moe_layer_bench` JSON from Task 3.
- Produces: `run_ablation(artifact, runner, output_dir, warmup, iterations) -> dict`.
- Produces: eighteen canonical records and digest-backed JSON/CSV summary.

- [ ] **Step 1: Write failing runner unit tests**

Create literal fake rows for split, per-call layer, and admission layer with profiler off/on at 1, 4, and 16 experts. Assert the exact case order, pair names, validation gates, nullable kernel contract, profiler parity, physical traffic gates, LF-only CSV, raw hashes, and aggregate hash. Add mutations for one missing admission hit, one nonzero admission warm byte, one incorrect per-call scan count, profiler-dependent physical traffic, and a non-null profiler-off kernel value; each must raise a specific `RuntimeError`.

- [ ] **Step 2: Run the unit test and witness RED**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cuda_admission_validation_ablation.py -q
```

Expected failure is `ModuleNotFoundError: tools.ablate_cuda_admission_validation`.

- [ ] **Step 3: Implement the eighteen-row runner**

Start the new Python file with `# B-0024 admission validation과 profiler attribution matrix를 실행합니다.`. Use explicit `CASES` and `GROUPS` tuples. Validate released dimensions, exact counter formulas, traffic identities, error/fallback gates, profiler parity, and no token-like keys. Write raw JSON with LF, summary CSV with `lineterminator="n"`, and summary JSON containing artifact, runner, aggregate, and CSV SHA-256 values.

- [ ] **Step 4: Verify unit and one-sample live matrices**

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cuda_admission_validation_ablation.py -q
```

- [ ] **Step 5: Commit the ablation tooling**

```bash
git add tools/ablate_cuda_admission_validation.py
  tests/python/test_cuda_admission_validation_ablation.py
git commit -m "bench: add admission validation ablation"
```

---

### Task 5: Formal B-0024 measurement and committed evidence

**Files:**
- Create: `results/b0024-cuda-admission-validation-wsl/*.json`
- Create: `results/b0024-cuda-admission-validation-wsl/summary.csv`
- Modify: `tests/python/test_cuda_admission_validation_ablation.py`

**Interfaces:**
- Produces: measured RTX 5080 attribution evidence with no token interpretation.

- [ ] **Step 1: Reuse the checked released artifact**

Verify `build-fixtures/released-expert.k3x` has SHA-256 `e087ff78284e99760a7d113cf744562878537a6379e7a63be95585eec8b9f1be`. Regenerate it with the existing bounded source/converter only if the file is absent; do not download a checkpoint.

- [ ] **Step 2: Run the formal matrix**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m tools.ablate_cuda_admission_validation
  --artifact build-fixtures/released-expert.k3x
  --runner build-cuda/k3x_cuda_moe_layer_bench
  --output-dir results/b0024-cuda-admission-validation-wsl
  --warmup 3 --iterations 20
```

- [ ] **Step 3: Add committed-evidence verification**

Recompute all eighteen raw hashes, exact case order, canonical aggregate, summary CSV hash, validation formulas, profiler parity, physical gates, and reported percentage deltas directly from committed bytes. Require nineteen JSON files and one CSV.

- [ ] **Step 4: Run evidence verification and commit**

```bash
/home/jolib/.venvs/k3x-m1/bin/python -m pytest
  tests/python/test_cuda_admission_validation_ablation.py -q
git add results/b0024-cuda-admission-validation-wsl
  tests/python/test_cuda_admission_validation_ablation.py
git commit -m "bench: measure admission validation"
```

---

### Task 6: Verification, decision, ledger, review, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `PROJECT_STATE.md` last

**Interfaces:**
- Consumes: implementation and B-0024 evidence.
- Produces: evidence-based default decision and synchronized public project state.

- [ ] **Step 1: Run complete local verification**

Run CPU CTest/pytest, liburing/direct CTest/pytest, ASan/UBSan CTest, CUDA CTest/pytest, and Compute Sanitizer for `test_cuda_moe_layer` plus one released admission benchmark invocation. Record actual counts only.

- [ ] **Step 2: Cross-check evidence and defaults**

Run `git diff --check`, search every `CudaWeightValidationMode` consumer, recompute committed evidence, confirm default `per-call`, and confirm B-0024 contains no token/TPS/TTFT field.

- [ ] **Step 3: Synchronize TITAN Ledger and README**

Record validation scan/hit/bytes/time, profiler on/off wall changes, kernel time where measured, traffic, VRAM, exactness, missing full-model metrics, and the accepted or rejected default decision. Update `PROJECT_STATE.md` last.

- [ ] **Step 4: Request one read-only final review**

Review Critical/Important findings across the implementation diff, validation failure ordering, counter parity, evidence hashes, defaults, and documentation claims. Apply one fix batch and at most one re-review unless a Critical issue remains.

- [ ] **Step 5: Publish and reconcile public state**

Push a ready PR, wait for correctness and CodeQL, rebase-merge, verify post-merge `main`, then use one small follow-up PR to replace README branch placeholders and close the checklist/plan with public PR, head, and CI evidence.

---

## Plan self-review result

- Every design requirement maps to a task and an executable gate.
- All symbols are introduced before later tasks consume them.
- The plan preserves per-call reference mode and does not preselect a default result.
- There are no placeholder implementation steps or unbounded adjacent refactors.
