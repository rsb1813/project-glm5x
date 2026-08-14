# K3X Milestone 7 Full-Dimension Bounded Expert Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure one physically materialized, released-dimension Kimi K3 routed-expert storage slice without downloading or executing the full checkpoint.

**Architecture:** A bounded Python source writer streams one expert's six safetensors extents, the existing converter packs them into K3X v1 with an optional storage-fixture identity, and a dedicated C++ Reader benchmark loads the six ordered extents as one exact batch. The ordinary synthetic model remains the graph/token oracle; the storage fixture is explicitly rejected by model execution.

**Tech Stack:** Python 3.12, safetensors file contract, google-crc32c, C++20, optional liburing 2.5, CMake/CTest, pytest, WSL2 Ubuntu 24.04 capability measurement.

## Global Constraints

- Use released expert dimensions 3,072 x 3,584, native MXFP4 E2M1 packed bytes, and one E8M0 byte per 32 values.
- Materialize all 17,547,264 expert payload bytes; do not use sparse-file holes.
- Keep generator and converter memory bounded by an explicit chunk size, default 1 MiB.
- Preserve existing executable synthetic checkpoints and `pread + buffered` defaults.
- Never label B-0008 as token throughput, full-model performance, native P44 Pro traffic, or a default-selection benchmark.
- Do not download full Kimi K3 weights, drop global filesystem caches, provision Cloud Run, or create paid resources.
- Follow strict RED-GREEN-REFACTOR and commit each independently verified behavior.

---

### Task 1: Streaming full-dimension source fixture

**Files:**
- Create: `reference/k3x_ref/storage_fixture.py`
- Create: `tools/generate_bounded_slice.py`
- Create: `tests/python/test_storage_fixture.py`
- Modify: `reference/k3x_ref/__init__.py`

**Interfaces:**
- Produces: `StorageSliceReport(shard_path: Path, manifest_path: Path, maximum_chunk_bytes: int, payload_bytes: int, source_sha256: str)`.
- Produces: `write_bounded_expert_source(root: Path, *, seed: int = 20260809, chunk_bytes: int = 1 << 20, layer_id: int = 1, expert_id: int = 0) -> StorageSliceReport`.
- Produces: source manifest format `k3-storage-slice-v1` with `artifact_kind=storage_fixture`, released model config, weight map, packed shapes, per-tensor SHA-256, and source SHA-256.

- [x] **Step 1: Write the failing source-shape and bounded-memory test.**

Create a real fixture in `tmp_path`, inspect it with `inspect_shard`, and assert these hand-derived values.

```python
assert report.payload_bytes == 17_547_264
assert report.maximum_chunk_bytes <= 257 * 1024
assert tensors[gate_packed].shape == (5_505_024,)
assert tensors[gate_scale].shape == (344_064,)
assert len(tensors) == 6
```

Also assert that a second generation has identical manifest and shard SHA-256, that `chunk_bytes=0` fails before creating final files, and that the final shard contains no holes by comparing its size with header plus declared tensor lengths.

- [x] **Step 2: Run the test and verify RED.**

Run `K3X_BUILD_DIR=build-cpu python -m pytest tests/python/test_storage_fixture.py -q` in WSL. Expect import failure for `k3x_ref.storage_fixture`.

- [x] **Step 3: Implement the minimal streaming writer.**

Use a sibling `.partial`, an 8-byte safetensors header length, compact padded JSON metadata, and a bounded pattern writer. The packed byte at absolute tensor position `p` is `(seed + matrix_index * 37 + p) & 0xff`; scale tensors use literal bytes 120, 121, and 122 for gate, up, and down. Update each tensor digest while writing, fsync, verify final size, and atomically replace the final shard and manifest.

Every new Python source file starts with a one-line Korean role comment.

- [x] **Step 4: Verify GREEN and CLI behavior.**

Run the targeted pytest, then run `python tools/generate_bounded_slice.py --output /tmp/k3x-bounded-source --chunk-bytes 1048576` and independently inspect all six tensor ranges with `inspect_shard`.

- [x] **Step 5: Commit.**

Commit as `feat: stream full-dimension expert fixture`.

### Task 2: K3X storage-fixture identity and conversion

**Files:**
- Modify: `converter/k3x_converter/format.py`
- Modify: `converter/k3x_converter/writer.py`
- Modify: `converter/k3x_converter/reader.py`
- Modify: `K3X_FORMAT.md`
- Modify: `tests/python/test_converter_cli.py`
- Modify: `tests/python/test_converter_resume.py`
- Modify: `tests/python/test_k3x_format.py`

**Interfaces:**
- Produces: `OPTIONAL_STORAGE_FIXTURE = 1 << 0` in Python format code.
- Changes: `_load_plans(source)` accepts only `synthetic-k3-source-v1` and `k3-storage-slice-v1`; the latter must declare `artifact_kind=storage_fixture` and exactly one complete gate/up/down expert.
- Changes: finalized bounded artifacts set `Superblock.optional_features & OPTIONAL_STORAGE_FIXTURE` and preserve that identity through resume validation.

- [x] **Step 1: Write failing conversion and malformed-source tests.**

Convert the real bounded source with `chunk_bytes=193 * 1024` and assert one expert record, three MXFP4 tensor records, six extents, `sum(data_length + auxiliary_length) == 17_547_264`, exact released shapes, optional bit 0, and `maximum_source_read_bytes <= 193 * 1024`.

Create separate malformed manifests for a missing down tensor, wrong `artifact_kind`, wrong expert dimensions, and an unsupported source format. Assert stable K3X error codes and no finalized output.

- [x] **Step 2: Run the tests and verify RED.**

Run the three targeted Python test files. Expect `UNSUPPORTED_SOURCE_FORMAT` for the valid bounded source and missing optional-feature identity.

- [x] **Step 3: Implement the minimum converter support.**

Add source-kind validation before planning extents, derive the storage optional bit once, include it in the conversion configuration fingerprint, set it in the finalized superblock, and compare it when recovering an already-finalized conversion. Do not relax synthetic source validation or add a generic partial-model execution mode.

- [x] **Step 4: Add and verify real resume coverage.**

Stop after two extents, resume the same bounded source, assert those extent IDs are reused, verify the final root digest, then mutate source identity and assert resume fails closed.

- [x] **Step 5: Run targeted and full Python tests.**

Run the targeted files followed by `K3X_BUILD_DIR=build-cpu python -m pytest -q`.

- [x] **Step 6: Commit.**

Commit as `feat: convert bounded storage fixtures`.

### Task 3: C++ optional identity and execution guard

**Files:**
- Modify: `runtime/include/k3x/format.hpp`
- Modify: `runtime/include/k3x/status.hpp`
- Modify: `runtime/src/reader.cpp`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_reader.cpp`
- Modify: `tests/python/test_cpp_reader.py`
- Modify: `tests/python/test_cpp_parity.py`

**Interfaces:**
- Produces: `inline constexpr std::uint64_t optional_storage_fixture = 1ULL << 0`.
- Adds: `Superblock::optional_features` populated from superblock offset 32.
- Adds: `ErrorCode::non_executable_artifact`, rendered as `NON_EXECUTABLE_ARTIFACT`.
- Changes: every `generate_greedy` entry path rejects a Reader carrying the storage-fixture bit before backend work or tensor reads.

- [x] **Step 1: Write failing Reader identity and execution tests.**

Use the real converted bounded artifact. Assert `test_reader` can report optional bit 0, while `k3x_run --model <slice> ...` exits with the stable non-executable error and zero data-plane Reader calls. Assert the ordinary tiny synthetic artifact still generates `[43, 32, 28, 49, 9, 28]`.

- [x] **Step 2: Run and verify RED.**

Expect the C++ Reader to omit the optional bit and execution to fail later with a missing tensor rather than `NON_EXECUTABLE_ARTIFACT`.

- [x] **Step 3: Implement parsing and the earliest shared execution guard.**

Read offset 32 without treating optional bits as required features. Put one guard at the shared generation boundary so compatibility overloads cannot bypass it. Preserve existing required-feature rejection.

- [x] **Step 4: Build and verify GREEN.**

Configure/build `build-cpu`, run CTest, run the two targeted Python files, then run full CPU pytest with `K3X_BUILD_DIR=build-cpu`.

- [x] **Step 5: Commit.**

Commit as `feat: identify non-executable storage fixtures`.

### Task 4: Exact six-extent expert load executable

**Files:**
- Create: `runtime/include/k3x/storage_slice.hpp`
- Create: `runtime/src/storage_slice.cpp`
- Create: `runtime/src/storage_bench.cpp`
- Create: `tests/python/test_storage_bench.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `Result<StorageExpertLoad> load_storage_expert(Reader&, std::uint32_t layer_id, std::uint32_t expert_id)`.
- `StorageExpertLoad` contains `std::array<std::vector<std::byte>, 6> extents`, `std::array<std::byte, 32> ordered_sha256`, and `std::uint64_t logical_bytes`.
- Produces executable: `k3x_storage_bench --model PATH --layer 1 --expert 0 --warmup N --iterations N --l2-io pread|io-uring --l2-cache buffered|direct --l2-queue-depth N`.

- [x] **Step 1: Write failing real-boundary tests.**

Invoke the not-yet-existing executable against the bounded artifact. Assert exact layer/expert identity, six calls in one batch per iteration, 17,547,264 logical bytes per load, six completions, zero failures/short reads, a 64-hex-character ordered digest, and rejection of a tiny executable artifact, missing expert, zero iterations, and invalid queue depth.

- [x] **Step 2: Run and verify RED.**

Expect failure because `k3x_storage_bench` does not exist.

- [x] **Step 3: Implement exact record resolution and one-batch loading.**

Resolve canonical gate/up/down tensor IDs with `fnv1a64`, require native MXFP4 plus non-empty auxiliary extents, validate the three released shapes and total logical payload, then submit requests in gate-data, gate-scale, up-data, up-scale, down-data, down-scale order. Hash returned bytes in that same order.

- [x] **Step 4: Implement the minimal benchmark CLI.**

Warm up outside the measured Reader, open a fresh Reader for measurements, time each complete expert load, and emit one JSON object with median/p05/p95 latency, loads per second, Reader counters, storage bytes/time, direct alignments, digest, and Linux process-I/O deltas. Do not emit token fields.

- [x] **Step 5: Build CPU and liburing variants and verify GREEN.**

Run targeted pytest with `K3X_BUILD_DIR=build-cpu`, CTest in `build-cpu`, then rebuild `build-uring` and run the targeted test against it.

- [x] **Step 6: Commit.**

Commit as `feat: benchmark exact expert extent loads`.

### Task 5: B-0008 four-case ablation

**Files:**
- Create: `tools/ablate_bounded_slice.py`
- Create: `tests/python/test_bounded_slice_ablation.py`

**Interfaces:**
- Produces: `bounded_slice_matrix() -> tuple[dict[str, str], ...]` crossing the two independent Reader axes.
- Produces: `run_bounded_slice_ablation(artifact, runner, *, warmup, iterations, queue_depth, output_dir, environment_label) -> dict[str, object]`.
- Produces B-0008 raw JSON/CSV and `summary.json` with supported/skipped cases and exact parity status.

- [x] **Step 1: Write failing matrix, validation, and artifact-cross-check tests.**

Assert four unique cases, capability-only skips, identical ordered digest and 17,547,264 bytes/load across supported rows, buffered submitted/logical equality, direct submitted bytes not below logical bytes, raw file existence, and compact/raw equality.

- [x] **Step 2: Run and verify RED.**

Expect import failure for `tools.ablate_bounded_slice`.

- [x] **Step 3: Implement the minimum runner.**

Invoke the dedicated C++ executable directly, parse its single JSON object, validate identity and accounting before writing results, and never reuse `benchmark_synthetic.py` token schemas.

- [x] **Step 4: Verify all supported modes on WSL2 ext4.**

Copy the K3X artifact to `/tmp`, run one-sample smoke with the liburing build, and require exact digest/byte parity. Record unsupported modes only as `STORAGE_UNAVAILABLE`.

- [x] **Step 5: Run full CPU/liburing/CUDA test matrices.**

Run CTest and pytest for `build-cpu`, `build-uring`, and `build-cuda`; run ASan/UBSan for the new non-CUDA storage path. CUDA behavior must remain unchanged.

- [x] **Step 6: Commit.**

Commit as `feat: ablate bounded expert storage reads`.

### Task 6: Measurement, TITAN Ledger, review, and publication

**Files:**
- Create: `results/b0008-bounded-slice-wsl/*.json`
- Create: `results/b0008-bounded-slice-wsl/*.csv`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PERFORMANCE_MODEL.md` only if measured byte evidence changes a stated estimate.
- Modify: `README.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

**Interfaces:**
- Produces: B-0008 with actual commit, environment, filesystem, artifact SHA-256, warmups/samples, exact expert-load latency, Reader/storage counters, and explicit evidence limitations.
- Produces: one accepted/rejected decision for the bounded artifact identity and any Reader default decision supported by measurement.

- [x] **Step 1: Measure without fabricating cold-cache or NVMe claims.**

Run three warmups and 20 samples per supported case on the WSL2 ext4 artifact. Preserve raw outputs and independently cross-check the summary. If resource pressure makes this unsafe, reduce only after recording the reason; never substitute estimates.

- [x] **Step 2: Run final verification.**

Run CPU, liburing, CUDA, sanitizer, and source/converter targeted suites. Capture exact pass/skip counts and the artifact SHA-256.

- [x] **Step 3: Update the ledgers in protocol order.**

Update architecture and decisions, append B-0008, update README/checklist/context, and update `PROJECT_STATE.md` last. Keep proposed deadline scheduling, Least-Stale, and all named addenda labeled proposed.

- [x] **Step 4: Self-review the diff and commit results.**

   Run `git diff --check`, search for accidental tok/s or P44 Pro claims, inspect every changed file, and commit the result/ledger unit separately.

- [x] **Step 4a: Close final-review source and resume integrity findings.**

  Add failing regressions for storage-manifest hash validation, crash-safe fixture publication, and canonical source-matched resume extents. Apply the minimum fixes and rerun the affected converter suites before publication.

- [x] **Step 5: Publish and require public CI.**

Push `codex/milestone-seven-bounded-slice`, open a draft public PR, require push and PR Linux CI, verify `origin/main` ancestry, fast-forward public `main`, confirm GitHub marks the PR merged, and require post-merge main CI. Record run IDs in the ledger with a final documentation commit.

- [x] **Step 6: Preserve the worktree.**

Keep the isolated worktree available for the next deadline-aware cross-layer scheduling milestone.
