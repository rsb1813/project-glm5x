# Index-Backed CUDA MoE Implementation Plan

> **For agentic workers:** Execute inline in the current task. The user explicitly disabled subagents for this phase.

**Goal:** Run the existing exact learned layer-10 CUDA MoE boundary directly from the official 282-shard `.gxi` without Python bundle lookup or per-tensor shard scans.

**Architecture:** Extend `Glm5xRuntimeIndex` with a validated tensor metadata-plus-payload read and a metadata-only membership query. Keep the existing CUDA routing, expert-major execution, shared expert, GLM5XACT input/output, and artifact-directory control path unchanged; only the weight source becomes selectable.

**Tech Stack:** C++20, CUDA 13, existing K3X `Reader`, GLM5X runtime index, CMake/CTest, Pytest.

**Spec:** `ARCHITECTURE.md` official runtime-index boundary and `DECISIONS.md` D-0101.

## Global Constraints

- Preserve natural Top-8 routing, BF16 payload bytes, shared expert behavior, and existing CUDA numerical thresholds.
- Require exactly one of `--artifact-dir` or `--runtime-index`.
- Do not add a general tensor-source abstraction or JSON parser.
- Keep artifact-directory behavior as the control path.
- Report source reads/bytes and never label a layer-only result as token throughput.

---

### Task 1: Validated tensor metadata read

**Files:**
- Modify: `runtime/include/k3x/glm5x_runtime_index.hpp`
- Modify: `runtime/src/glm5x_runtime_index.cpp`
- Modify: `tests/cpp/test_glm5x_runtime_index.cpp`
- Modify: `tests/python/test_glm5x_runtime_index.py`

**Interfaces:**
- Produces: `Glm5xTensorLoad { TensorRecord record; std::vector<std::byte> payload; }`.
- Produces: `read_tensor_with_metadata(uint64_t)` and `contains_tensor(uint64_t)`.

- [x] Add metadata assertions to the C++ contract output and confirm compilation fails on the missing API.
- [x] Implement the membership and validated metadata-plus-payload read without duplicating payload validation.
- [x] Rebuild and run the focused C++/Python runtime-index regression.
- [x] Commit the runtime-index API change.

### Task 2: Runtime-index source for the existing CUDA gate

**Files:**
- Modify: `runtime/src/cuda_glm5x_real_expert_bench.cpp`

**Interfaces:**
- Consumes: `Glm5xRuntimeIndex::read_tensor_with_metadata`, `contains_tensor`, and `read_expert`.
- Produces: `--runtime-index FILE` as an alternative to `--artifact-dir DIR`.
- Reports: `source_kind`, `source_artifact_count`, `source_read_calls`, and `source_read_bytes`.

- [x] Add the new CLI/source branch while retaining the artifact-directory control path.
- [x] Build the CUDA benchmark and confirm invalid dual/missing source arguments are rejected.
- [x] Run the official layer-10 learned-MoE gate through `.gxi` and the existing GLM5XACT files.
- [x] Compare route IDs, expected-output error, host payload bytes, H2D bytes, and warm latency with the artifact-directory control.
- [x] Commit the benchmark integration.

### Task 3: Verification and measured evidence

**Files:**
- Create: `results/b0006-index-backed-cuda-moe-rtx5080/summary.json`
- Create: `results/b0006-index-backed-cuda-moe-rtx5080/summary.csv`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

- [x] Preserve raw control/index outputs and verify their SHA-256 digests.
- [x] Run focused regressions, applicable CTest/Pytest, and `git diff --check`.
- [x] Record B-0006 as a bounded MoE-layer result with no tok/s extrapolation.
- [x] Update `PROJECT_STATE.md` last, commit, push outside the sandbox, and open a stacked draft PR.
