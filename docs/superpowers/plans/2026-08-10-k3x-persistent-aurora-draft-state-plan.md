# K3X Persistent AURORA Draft-State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace repeated complete-prefix AURORA replay with an exact persistent reduced-Top-K cursor that crops rejected KDA/MLA state, teacher-forces target commits, and is measured against replay without changing target defaults.

**Architecture:** An opaque `IncrementalDraftCursor` owns the internal `Engine`, mutable `ModelState`, next logits, and one proposal transaction. It snapshots fixed-size KDA state, records MLA logical size marks, restores the target-accepted prefix, and advances the target bonus token. `AuroraPersistentDraftProvider` wraps this cursor behind the existing `DraftProvider` contract; `aurora-replay` remains the exact oracle.

**Tech Stack:** C++20 runtime, Python 3.12 reference fixtures and benchmark tooling, CMake/CTest, pytest, optional liburing, CUDA 13.3 `sm_120`, ASan/UBSan, Compute Sanitizer, JSON/CSV evidence.

## Global Constraints

- Correctness precedes throughput; persistent candidates and state must match complete-prefix replay before performance is measured.
- Natural target routing, strict token-major/expert-major verification, scheduler thresholds, and all defaults remain unchanged.
- Persistent draft runtime is CPU-only, incremental, fixed K4/6/8/12 below natural K, disabled L1, blocking L2, and no profile observation.
- MLA rollback changes logical size and length without deep-copying context-proportional vectors.
- Every production behavior begins with a witnessed failing test and ends with focused plus broader green verification.
- Every new C++ or Python source file begins with a one-line Korean role comment.
- No paid cloud resource, full Kimi K3 checkpoint, reduced precision, resident-only draft bank, proxy, pruning, or learned drafter is introduced.
- Measured fields never substitute logical Reader bytes for physical NVMe traffic.

---

### Task 1: Incremental cursor creation and initial proposal

**Files:**
- Create: `runtime/include/k3x/incremental_cursor.hpp`
- Modify: `runtime/src/model.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/cpp/test_incremental_cursor.cpp`
- Create: `tests/python/test_persistent_aurora_runtime.py`

**Interfaces:**
- Consumes: `Reader`, `ComputeBackend`, `RuntimeSession`, internal `Engine`, internal `ModelState`, `Result<T>`.
- Produces: `IncrementalDraftCursor::create`, `propose`, `stats`, and `diagnostics` with the exact signatures in the design.

- [x] **Step 1: Add the artifact-backed failing cursor test**

Create `tests/cpp/test_incremental_cursor.cpp` with the Korean role header. Open the fixture path from `argv[1]`, create a CPU backend and fixed-K4 incremental diagnostic session, and require the following wished-for API.

```cpp
const std::vector<std::uint32_t> context{1, 7, 3, 9, 43};
auto cursor = k3x::IncrementalDraftCursor::create(
    reader.value(), *backend, context, session);
require(static_cast<bool>(cursor));
require(cursor.value()->diagnostics().mla_length == context.size());
auto proposal = cursor.value()->propose(2);
require(static_cast<bool>(proposal));
require(proposal.value().size() == 2);
require(cursor.value()->stats().context_prefill_tokens == context.size());
require(cursor.value()->stats().incremental_forward_calls == 1);
require(cursor.value()->diagnostics().mla_length == context.size() + 1);
```

Create `tests/python/test_persistent_aurora_runtime.py` using the existing synthetic Top-16 fixture pattern from `test_aurora_runtime.py`. Build and invoke `test_incremental_cursor` with the generated artifact.

- [x] **Step 2: Register the executable and verify RED**

Add the executable without a no-argument CTest registration because it requires an artifact.

```cmake
add_executable(test_incremental_cursor tests/cpp/test_incremental_cursor.cpp)
target_link_libraries(test_incremental_cursor PRIVATE k3x_runtime)
```

Run:

```bash
cmake --build build --target test_incremental_cursor -j2
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_persistent_aurora_runtime.py -q
```

Expected: compilation fails because `k3x/incremental_cursor.hpp` and `IncrementalDraftCursor` do not exist.

- [x] **Step 3: Add the minimal opaque cursor API**

Create `runtime/include/k3x/incremental_cursor.hpp` with an explicit destructor and move-disabled ownership so the incomplete `Impl` type is safe.

```cpp
// 증분 AURORA draft 상태와 proposal transaction 경계를 선언합니다.
#pragma once

namespace k3x {
struct IncrementalDraftCursorStats {
    std::uint64_t context_prefill_tokens{};
    std::uint64_t incremental_forward_calls{};
    std::uint64_t rollback_events{};
    std::uint64_t mla_positions_cropped{};
    std::uint64_t kda_checkpoint_bytes{};
};

struct IncrementalDraftCursorDiagnostics {
    std::vector<float> flattened_state;
    std::size_t mla_length{};
    std::size_t mla_key_elements{};
    std::size_t mla_value_elements{};
    std::size_t mla_shared_key_elements{};
};

class IncrementalDraftCursor {
public:
    ~IncrementalDraftCursor();
    static Result<std::unique_ptr<IncrementalDraftCursor>> create(
        Reader&, ComputeBackend&, std::span<const std::uint32_t>,
        RuntimeSession&);
    Result<std::vector<std::uint32_t>> propose(std::size_t count);
    Result<bool> commit(std::size_t accepted_prefix,
                        std::span<const std::uint32_t> committed_tokens);
    IncrementalDraftCursorStats stats() const noexcept;
    IncrementalDraftCursorDiagnostics diagnostics() const;
private:
    struct Impl;
    explicit IncrementalDraftCursor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
```

Implement `Impl` in `model.cpp` after `Engine`. `create` must acquire the draft session generation guard, reject storage fixtures, empty context, and nonincremental sessions, run the context exactly once with `ProfilePhase::prefill`, and retain state plus logits. `propose(2)` returns `argmax(logits)`, forwards the first candidate once with decode phase, and returns the second argmax. It latches one outstanding proposal.

- [x] **Step 4: Verify initial proposal GREEN**

Run:

```bash
cmake --build build --target test_incremental_cursor -j2
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_persistent_aurora_runtime.py -q
ctest --test-dir build --output-on-failure
```

Expected: focused artifact test passes and CPU CTest remains 14/14.

- [x] **Step 5: Commit the cursor foundation**

```bash
git add runtime/include/k3x/incremental_cursor.hpp runtime/src/model.cpp \
  tests/cpp/test_incremental_cursor.cpp \
  tests/python/test_persistent_aurora_runtime.py CMakeLists.txt
git commit -m "feat: add incremental draft cursor"
```

---

### Task 2: Transactional commit, KDA restore, and MLA crop

**Files:**
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_incremental_cursor.cpp`
- Modify: `tests/python/test_persistent_aurora_runtime.py`

**Interfaces:**
- Consumes: Task 1 cursor, pending proposal tokens, internal `ModelState`.
- Produces: exact `commit(accepted_prefix, committed_tokens)`, checkpoint restore, crop telemetry, and flattened-state diagnostics.

- [x] **Step 1: Add full-accept and rejected-prefix failing cases**

Extend the C++ test to compare the cursor against fresh fixed-K4 `generate_greedy` diagnostic runs.

```cpp
const auto first = proposal.value();
require(static_cast<bool>(cursor.value()->commit(
    2, std::array<std::uint32_t, 3>{first[0], first[1], 17})));
auto after_full = cursor.value()->propose(2);
require(static_cast<bool>(after_full));
require(after_full.value() == replay_tokens(
    {1, 7, 3, 9, 43, first[0], first[1], 17}, 2));
```

Create a second cursor, propose four tokens, accept only the first, commit a mismatching target bonus, and require its next proposal and flattened state to match a fresh teacher-forced run over `context + accepted + bonus`. Assert `rollback_events == 1` and `mla_positions_cropped > 0`.

- [x] **Step 2: Run the focused test and verify RED**

Run the same focused pytest command. Expected: `commit` returns `invalid_state` or diagnostics/crop counters do not match because Task 1 has no transaction restore implementation.

- [x] **Step 3: Implement bounded checkpoints and exact commit**

Inside `Impl`, add:

```cpp
struct MlaMark {
    std::size_t length{};
    std::size_t keys{};
    std::size_t values{};
    std::size_t shared_keys{};
};

struct Checkpoint {
    std::vector<KdaState> kda;
    MlaMark mla;
    Vector logits;
};
```

Record a base checkpoint before proposal and one checkpoint after each processed candidate. Restore by copying KDA, resizing MLA vectors, restoring `length`, and restoring logits. Count copied KDA scalar bytes with checked `uint64_t` arithmetic. For all-accepted commits, process the unconsumed final candidate; for shorter acceptance restore the matching checkpoint. Then process exactly one bonus token. Clear the transaction only after success.

- [x] **Step 4: Add lifecycle and failure-boundary RED/GREEN**

Add cases for zero proposal, zero accepted, outstanding proposal, accepted count beyond proposal, wrong accepted prefix, missing/multiple bonus tokens, second commit, and reuse after latched failure. Capture Reader counters before malformed commits and require no change.

Run focused pytest after adding the tests and observe the expected failure before implementation. Implement prevalidation before any restore or forward, latch failure on malformed input, then rerun until green.

- [x] **Step 5: Verify cursor and baseline tests**

```bash
cmake --build build -j2
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_persistent_aurora_runtime.py \
  tests/python/test_aurora_runtime.py -q
ctest --test-dir build --output-on-failure
```

- [x] **Step 6: Commit transaction semantics**

```bash
git add runtime/src/model.cpp tests/cpp/test_incremental_cursor.cpp \
  tests/python/test_persistent_aurora_runtime.py
git commit -m "feat: crop persistent draft state"
```

---

### Task 3: Persistent AURORA provider and replay-oracle parity

**Files:**
- Modify: `runtime/include/k3x/aurora.hpp`
- Modify: `runtime/src/aurora.cpp`
- Modify: `runtime/include/k3x/speculative.hpp`
- Modify: `runtime/include/k3x/model.hpp`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_aurora.cpp`
- Modify: `tests/cpp/test_model_session.cpp`
- Modify: `tests/python/test_aurora_runtime.py`

**Interfaces:**
- Consumes: `IncrementalDraftCursor`, `AdaptiveDraftScheduler`, existing replay provider and `DraftVerification`.
- Produces: `AuroraPersistentDraftProvider`, five new default-zero draft counters, and end-to-end provider stats.

- [x] **Step 1: Write the provider parity RED**

In `test_aurora.cpp`, construct replay and persistent providers with separate Readers/backends and identical fixed-K4 block-2 schedulers. Feed each the same `DraftRequest`, require equal candidates, apply the same full-accept verification, then require equal next candidates. Repeat with first-token rejection and a zero-candidate scheduler step.

Require persistent stats:

```cpp
require(stats.replayed_context_tokens == 0);
require(stats.context_prefill_tokens == prompt.size() + initial.size());
require(stats.incremental_forward_calls > 0);
require(stats.reader_bytes < replay_provider->stats().reader_bytes);
```

Run the focused artifact test and verify compilation fails because `AuroraPersistentDraftProvider` is missing.

- [x] **Step 2: Implement the persistent provider minimally**

Declare `AuroraPersistentConfig` and `AuroraPersistentDraftProvider` in `aurora.hpp`. Reuse the existing option validator through one private helper instead of duplicating conditions. Create the cursor lazily from `prompt_ + request.generated_tokens`, delegate selected proposal length, validate update before cursor commit, and update expected history and scheduler only after cursor success. Preserve the no-throw `update` method by latching errors.

- [x] **Step 3: Extend provider and generation telemetry through RED/GREEN**

Add these fields to `DraftProviderStats` and `GenerationResult`.

```cpp
std::uint64_t context_prefill_tokens{};
std::uint64_t incremental_forward_calls{};
std::uint64_t rollback_events{};
std::uint64_t mla_positions_cropped{};
std::uint64_t kda_checkpoint_bytes{};
```

First add assertions that ordinary greedy and replay keep the fields zero and persistent copies cursor values. Observe missing-field compilation failures, implement the copies in `generate_speculative`, and rerun.

- [x] **Step 4: Add end-to-end target parity cases**

In `test_model_session.cpp`, add persistent token-major and CPU expert-major runs using the same fixed/adaptive traces as replay. Require natural target token IDs, final state, committed routes, accepted counts, and proposal counts to match replay. Draft routing and bytes remain separate from target counters.

- [x] **Step 5: Verify provider GREEN and commit**

```bash
cmake --build build -j2
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_aurora_runtime.py \
  tests/python/test_persistent_aurora_runtime.py -q
ctest --test-dir build --output-on-failure
git add runtime/include/k3x/aurora.hpp runtime/src/aurora.cpp \
  runtime/include/k3x/speculative.hpp runtime/include/k3x/model.hpp \
  runtime/src/model.cpp tests/cpp/test_aurora.cpp \
  tests/cpp/test_model_session.cpp tests/python/test_aurora_runtime.py
git commit -m "feat: add persistent AURORA provider"
```

---

### Task 4: CLI identity and benchmark schema

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`
- Modify: `tests/python/test_aurora_runtime.py`

**Interfaces:**
- Consumes: persistent provider and five new counters.
- Produces: `--speculative-mode aurora-persistent`, JSON/CSV fields, and fail-closed option validation.

- [x] **Step 1: Add CLI and schema RED tests**

Extend `test_cpp_parity.py` so a Top-16 artifact runs `aurora-persistent` for fixed and adaptive token-major modes and matches natural token/final-state/routes. Add rejection cases for full mode, fixed target routing, draft K 0/16, block size 3, unknown policy, and AURORA options used with `none`.

Extend `test_benchmark_schema.py` with all five cursor keys and require zeros in nonpersistent fixtures.

Run:

```bash
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_cpp_parity.py \
  tests/python/test_benchmark_schema.py -q
```

Expected: unknown speculative mode or missing JSON/schema keys.

- [x] **Step 2: Wire the CLI with unchanged preflight order**

Accept `aurora-persistent` wherever `aurora-replay` currently accepts shared AURORA options. Keep provider construction after complete option/artifact validation and before output mutation. Open a separate draft Reader, create a CPU backend, and instantiate the persistent provider. Do not alter `none` or `scripted-reference` defaults.

- [x] **Step 3: Export five fields to JSON, CSV, and Python records**

Use these exact external names.

```text
draft_context_prefill_tokens
draft_incremental_forward_calls
draft_rollback_events
draft_mla_positions_cropped
draft_kda_checkpoint_bytes
```

Update `BenchmarkRecord`, sample parsing, equality checks for invariant counters, and `write_results`. Persistent mode must pass AURORA draft arguments exactly as replay does.

- [x] **Step 4: Verify focused and full CPU GREEN**

```bash
cmake --build build -j2
K3X_BUILD_DIR=build python -m pytest \
  tests/python/test_cpp_parity.py \
  tests/python/test_benchmark_schema.py \
  tests/python/test_aurora_runtime.py \
  tests/python/test_persistent_aurora_runtime.py -q
ctest --test-dir build --output-on-failure
```

- [x] **Step 5: Commit CLI and schema**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py \
  tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py \
  tests/python/test_aurora_runtime.py
git commit -m "feat: expose persistent AURORA runtime"
```

---

### Task 5: B-0018 replay-versus-persistent evidence

**Files:**
- Create: `tools/ablate_persistent_aurora.py`
- Create: `tests/python/test_persistent_aurora_ablation.py`
- Create: `results/b0018-persistent-aurora-wsl/*.json`
- Create: `results/b0018-persistent-aurora-wsl/*.csv`

**Interfaces:**
- Consumes: `benchmark_once`, `write_results`, B-0017 artifact construction and diagnostic parity helpers.
- Produces: nine-row B-0018 raw evidence, canonical summary, checksums, and exact matched-pair validation.

- [x] **Step 1: Add the B-0018 matrix and failing evidence test**

Create the runner with a Korean role header and this exact case matrix.

```python
CASES = (
    ("natural-greedy", "none", "token-major", 0, "none"),
    ("replay-fixed-2-token", "aurora-replay", "token-major", 2, "fixed"),
    ("persistent-fixed-2-token", "aurora-persistent", "token-major", 2, "fixed"),
    ("replay-adaptive-token", "aurora-replay", "token-major", 4, "adaptive"),
    ("persistent-adaptive-token", "aurora-persistent", "token-major", 4, "adaptive"),
    ("replay-fixed-2-expert", "aurora-replay", "expert-major", 2, "fixed"),
    ("persistent-fixed-2-expert", "aurora-persistent", "expert-major", 2, "fixed"),
    ("replay-adaptive-expert", "aurora-replay", "expert-major", 4, "adaptive"),
    ("persistent-adaptive-expert", "aurora-persistent", "expert-major", 4, "adaptive"),
)
```

The test must require matched proposal tokens, acceptance, target tokens, final state, committed routes, persistent replay-context zero, one context prefill, positive incremental forwards, and canonical raw JSON/CSV digests. Run it before results exist and verify RED due to missing summary.

- [x] **Step 2: Implement the runner and one-sample smoke**

Reuse B-0017 source generation and LF CSV writing. Add pair names to each summary record and reject any matched proposal/acceptance divergence. Run with zero warmups and one sample into an untracked build-results directory. Fix only runner/schema defects revealed by actual errors.

- [x] **Step 3: Run canonical B-0018**

```bash
python tools/ablate_persistent_aurora.py \
  --runner build/k3x_run \
  --output results/b0018-persistent-aurora-wsl \
  --warmups 3 --samples 20
```

Record the measured direction even if persistent state is slower. Never derive physical NVMe values from Reader counters.

- [x] **Step 4: Cross-check committed bytes independently**

The pytest must recompute every raw JSON/CSV SHA-256, summary CSV SHA-256, canonical sorted-record aggregate, exact diagnostic parity, and headline percentage deltas from committed bytes. Require nine raw pairs and fail on CRLF-normalized digest drift.

- [x] **Step 5: Commit B-0018 evidence**

```bash
git add tools/ablate_persistent_aurora.py \
  tests/python/test_persistent_aurora_ablation.py \
  results/b0018-persistent-aurora-wsl
git commit -m "bench: measure persistent AURORA state"
```

---

### Task 6: Full verification, TITAN Ledger, review, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `PROJECT_STATE.md` last
- Modify: this plan

**Interfaces:**
- Consumes: all implementation commits and canonical B-0018 evidence.
- Produces: verified Milestone 17 ledger, public PR integration, and the next measured bottleneck.

- [x] **Step 1: Run the complete verification matrix**

Configure missing variants from the current source, then run:

```bash
ctest --test-dir build --output-on-failure
K3X_BUILD_DIR=build python -m pytest -q
ctest --test-dir build-uring --output-on-failure
K3X_BUILD_DIR=build-uring K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 \
  python -m pytest -q
ASAN_OPTIONS=detect_leaks=0 \
  ctest --test-dir build-uring-asan --output-on-failure
ctest --test-dir build-cuda --output-on-failure
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 python -m pytest -q
```

Run Compute Sanitizer on one CUDA expert-major `aurora-persistent` CLI case. State explicitly that the CPU cursor itself is covered by ASan/UBSan, not CUDA instrumentation.

- [x] **Step 2: Synchronize measured documents**

Update README milestone status and reproduction command, ARCHITECTURE implementation status, D-041 result, PERFORMANCE_MODEL traffic comparison, complete B-0018 BENCHMARKS fields and caveats, checklist, and context notes. Update `PROJECT_STATE.md` last with exact commits, tests, hashes, public state, bottleneck, and next task. Never replace B-0017 or theoretical values with B-0018 measurements.

- [x] **Step 3: Verify documents and commit**

```bash
git diff --check
python -m pytest tests/python/test_persistent_aurora_ablation.py -q
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md \
  BENCHMARKS.md checklist.md context-notes.md PROJECT_STATE.md \
  docs/superpowers/plans/2026-08-10-k3x-persistent-aurora-draft-state-plan.md
git commit -m "docs: publish persistent AURORA evidence"
```

- [x] **Step 4: Perform final self-review**

Review `origin/main...HEAD` for default-path changes, replay compatibility, target ownership, malformed-update Reader access, deep MLA copies, telemetry parity, raw-summary digest parity, and proposed-versus-measured language. Apply at most one focused correction batch and rerun affected tests.

- [x] **Step 5: Publish and verify public main**

Push `codex/milestone-seventeen-persistent-aurora`, open a ready public PR against `main`, wait for push and PR correctness, rebase-merge, and wait for the post-merge `main` run. Record publication through a small reconciliation PR only if the ledger would otherwise retain a stale active-branch statement.

PR #23 was rebase-merged at public integration head `30bbf7a8` after branch and pull-request correctness runs `31340338639` and `31340340063` passed. Post-merge `main` correctness run `31340476396` also passed. This reconciliation change replaces the stale active-branch statement without changing measured evidence.
