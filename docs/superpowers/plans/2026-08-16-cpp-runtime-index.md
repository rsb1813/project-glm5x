# C++ GLM5X Runtime Index Implementation Plan

> **For agentic workers:** Execute inline in the current task. The user explicitly disabled subagents for this phase.

**Goal:** Give the C++ runtime a validated, constant-time `tensor_id -> K3X shard/record` index for the official 282-shard GLM-5.2 bundle without loading model payloads at startup.

**Architecture:** The converter emits one atomic `.gxi` file from the existing verified bundle. C++ reads its fixed records, verifies header CRC32C/body SHA-256, opens the referenced K3X metadata, validates root hashes and tensor locators, and exposes exact tensor/expert payload reads. The index is resident metadata; weight payloads remain out-of-core.

**Tech Stack:** Python 3.12, C++20, existing K3X `Reader`, SHA-256/CRC32C helpers, CMake/CTest, Pytest.

**Spec:** `ARCHITECTURE.md` current C++ full-model boundary, `DECISIONS.md` D-0100, and `PROJECT_STATE.md` adaptive hot-bank next tasks.

## Global Constraints

- Preserve the existing JSON expert bundle and Python reference behavior.
- Do not add a general JSON library to the C++ hot path.
- Keep natural routing, tensor bytes, precision, and CRC validation unchanged.
- Reject absolute paths, parent traversal, duplicate paths, duplicate tensor IDs, mismatched roots, counts, record positions, or CRC metadata.
- Build the index atomically through `.partial`, `fsync`, and replace.
- Keep the format little-endian, versioned, and deterministic.
- Do not claim full-model TPS from index-open or single-expert measurements.

---

### Task 1: Deterministic runtime-index writer

**Files:**
- Create: `converter/glm5x_converter/runtime_index.py`
- Modify: `converter/glm5x_converter/cli.py`
- Test: `tests/python/test_glm5x_runtime_index.py`
- Test: `tests/python/test_glm5x_cli.py`

**Interfaces:**
- Produces: `build_glm5x_runtime_index(bundle_path, output_path) -> GLM5XRuntimeIndexReport`.
- Binary layout: 128-byte header, 64-byte artifact records, 24-byte tensor locators sorted by tensor ID, concatenated UTF-8 relative paths.

- [x] Write a fixture test that independently decodes literal header fields and verifies deterministic bytes.
- [x] Run it and confirm RED on the missing module/API.
- [x] Implement atomic writer, CRC32C, SHA-256, duplicate checks, and CLI `build-runtime-index`.
- [x] Run focused writer/CLI tests and confirm GREEN.
- [x] Commit the writer and tests.

### Task 2: Validated C++ runtime-index reader

**Files:**
- Create: `runtime/include/k3x/glm5x_runtime_index.hpp`
- Create: `runtime/src/glm5x_runtime_index.cpp`
- Create: `tests/cpp/test_glm5x_runtime_index.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `Glm5xRuntimeIndex::open(path, ReaderOptions)`.
- Produces: `read_tensor(uint64_t tensor_id)` and `read_expert(layer_id, expert_id, hidden_size, intermediate_size)`.
- Exposes: artifact/tensor counts and aggregate reader counters.

- [x] Write the C++ executable contract test and add its build target.
- [x] Build first and confirm RED on the missing public interface.
- [x] Implement strict fixed-record parsing, range/overflow checks, reserved-zero checks, path containment, and sorted unique tensor lookup.
- [x] Validate each opened artifact's root SHA, tensor count, record index, tensor ID, and data CRC metadata before accepting the index.
- [x] Reuse the existing exact BF16 expert shape/dtype/CRC contract for three-role reads.
- [x] Run the C++ fixture through Pytest and confirm GREEN.
- [x] Commit the C++ reader and integration tests.

### Task 3: Official 282-shard bounded gate

**Files:**
- Modify: `tests/python/test_glm5x_runtime_index.py`
- Reuse: `glm5x-convert build-runtime-index` and the JSON-emitting `test_glm5x_runtime_index` executable; no redundant wrapper is required.
- Create: `results/b0005-cpp-runtime-index-rtx5080/summary.json`
- Create: `results/b0005-cpp-runtime-index-rtx5080/summary.csv`

**Interfaces:**
- Consumes the existing `build-glm5x-full-k3x/glm5x-experts-full.json`.
- Reports index bytes/build time/open time, exact payload bytes/digest, reader calls/bytes, and comparison with Python for one bounded expert.

- [x] Build the official `.gxi` without payload/root rescans and record its SHA-256.
- [x] Open all 282 metadata shards through C++ and load one real layer-10 expert.
- [x] Compare all three exact role payload digests/lengths with Python.
- [x] Record startup/load telemetry without extrapolating token throughput.
- [x] Run focused C++/Python regression and applicable full suites.

### Task 4: Persistent state and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

- [x] Record implemented/proposed boundaries and B-0005 measurements.
- [x] Verify raw hashes, JSON/CSV parity, compile/tests, and `git diff --check`.
- [ ] Commit evidence and update `PROJECT_STATE.md` last.
- [ ] Push the branch outside the sandbox and open a stacked draft PR.
