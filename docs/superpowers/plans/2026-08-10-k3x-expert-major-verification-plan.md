# K3X Milestone 14 Exact Expert-Major Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in exact CPU expert-major speculative target verifier that preserves token-major commit semantics while loading each unique natural-routing expert payload once per layer and verification block.

**Architecture:** Keep ordinary greedy generation and token-major speculation unchanged. The new path first establishes pure stable expert grouping and pure block acceptance contracts, then evaluates a copied model state layer-major, commits only the accepted state prefix, and exposes physical work and Reader traffic separately from committed routing. The first executable boundary is CPU, incremental, blocking L2, disabled L1, natural routing, and disabled profile observation/output.

**Tech Stack:** C++20, CMake/CTest, existing K3X native-MXFP4 CPU backend, Python 3.12, pytest, JSON/CSV benchmark tooling.

## Global Constraints

- Correctness precedes throughput, and no projected speedup may be reported as measured.
- Do not download the full Kimi K3 checkpoint or provision paid cloud resources.
- `SpeculativeVerificationMode::token_major` remains the runtime default.
- Expert-major execution must preserve generated tokens, committed KDA/MLA state, committed natural routing, and router-slot FP32 accumulation order.
- Unsupported expert-major combinations fail before prefill or Reader activity.
- New source files begin with a one-line Korean role comment.
- Every production change follows a witnessed RED test, the focused GREEN test, and the complete applicable suite.

---

### Task 1: Stable expert-major scheduling plan

**Files:**
- Create: `runtime/include/k3x/expert_major.hpp`
- Create: `runtime/src/expert_major.cpp`
- Create: `tests/cpp/test_expert_major.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: token routes whose expert IDs and normalized contributions are already ordered by the immutable natural router decision.
- Produces: `Result<ExpertMajorPlan> build_expert_major_plan(std::span<const ExpertMajorTokenRoute> routes)`.

- [ ] **Step 1: Write the failing literal and validation tests**

```cpp
// expert-major 검증 블록의 안정적인 expert grouping 계약을 검사합니다.
#include "k3x/expert_major.hpp"

const std::vector<k3x::ExpertMajorTokenRoute> routes{
    {{2, 1}, {0.6F, 0.4F}},
    {{1, 3}, {0.7F, 0.3F}},
};
const auto result = k3x::build_expert_major_plan(routes);
require(result);
require(result.value().assignment_count == 4);
require(result.value().groups[0].expert_id == 2);
require(result.value().groups[1].expert_id == 1);
require(result.value().groups[1].assignments[0].token_index == 0);
require(result.value().groups[1].assignments[0].router_slot == 1);
require(result.value().groups[1].assignments[1].token_index == 1);
require(result.value().groups[1].assignments[1].router_slot == 0);
require(result.value().groups[2].expert_id == 3);
```

Add explicit failing cases for an empty route list, an empty token route, mismatched ID/contribution lengths, a duplicate expert within one token, `NaN`, and positive infinity.

- [ ] **Step 2: Run the focused build to verify RED**

Run: `cmake --build build-cpu --target test_expert_major -j 2`

Expected: FAIL because `k3x/expert_major.hpp` or target `test_expert_major` does not exist.

- [ ] **Step 3: Implement the smallest stable planner**

```cpp
struct ExpertMajorTokenRoute {
    std::vector<std::uint32_t> expert_ids;
    std::vector<float> contributions;
};

struct ExpertMajorAssignment {
    std::size_t token_index;
    std::size_t router_slot;
    float contribution;
};

struct ExpertMajorGroup {
    std::uint32_t expert_id;
    std::vector<ExpertMajorAssignment> assignments;
};

struct ExpertMajorPlan {
    std::vector<ExpertMajorGroup> groups;
    std::size_t assignment_count{};
};

Result<ExpertMajorPlan> build_expert_major_plan(
    std::span<const ExpertMajorTokenRoute> routes);
```

Scan token index then router slot, validate before returning success, create each group at first occurrence, and append assignments in scan order. Do not sort groups or contributions.

- [ ] **Step 4: Run focused and complete CPU tests**

Run: `cmake --build build-cpu --target test_expert_major -j 2 && ctest --test-dir build-cpu -R expert_major --output-on-failure`

Expected: `expert_major` passes.

Run: `ctest --test-dir build-cpu --output-on-failure`

Expected: all CPU tests pass.

- [ ] **Step 5: Commit the scheduling contract**

```bash
git add CMakeLists.txt runtime/include/k3x/expert_major.hpp runtime/src/expert_major.cpp tests/cpp/test_expert_major.cpp
git commit -m "feat: add stable expert-major scheduling plan"
```

### Task 2: Pure vector target verification

**Files:**
- Modify: `runtime/include/k3x/speculative.hpp`
- Modify: `runtime/src/speculative.cpp`
- Modify: `tests/cpp/test_speculative.cpp`

**Interfaces:**
- Consumes: `DraftProposal`, maximum draft length, vocabulary size, and exactly `candidate_tokens.size() + 1` target argmax tokens.
- Produces: `Result<DraftVerification> verify_greedy_target_block(const DraftProposal&, std::size_t, std::size_t, std::span<const std::uint32_t>)`.

- [ ] **Step 1: Write failing perfect, mismatch, empty, and validation tests**

```cpp
auto perfect = k3x::verify_greedy_target_block(
    DraftProposal{.anchor_token = 10, .candidate_tokens = {11, 12}},
    2, 128, std::vector<std::uint32_t>{11, 12, 13});
assert(perfect);
assert(perfect.value().accepted_draft_tokens == 2);
assert((perfect.value().committed_tokens ==
        std::vector<std::uint32_t>{11, 12, 13}));

auto middle = k3x::verify_greedy_target_block(
    DraftProposal{.anchor_token = 10, .candidate_tokens = {11, 99}},
    2, 128, std::vector<std::uint32_t>{11, 12, 100});
assert(middle);
assert(middle.value().accepted_draft_tokens == 1);
assert((middle.value().committed_tokens ==
        std::vector<std::uint32_t>{11, 12}));
```

Add first/final mismatch, empty proposal with one target token, too few/too many target tokens, invalid proposal token, and out-of-vocabulary target output.

- [ ] **Step 2: Run the focused test to verify RED**

Run: `cmake --build build-cpu --target test_speculative -j 2`

Expected: compile failure because `verify_greedy_target_block` is undeclared.

- [ ] **Step 3: Implement target-vector validation and strict commit calculation**

```cpp
Result<DraftVerification> verify_greedy_target_block(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size,
    std::span<const std::uint32_t> target_tokens);
```

Reuse a private proposal validator shared with `verify_greedy_draft`. Require `target_tokens.size() == proposal.candidate_tokens.size() + 1`, validate the complete target vector before building output, compare candidates in order, and commit exactly one mismatching target or the all-accepted bonus.

- [ ] **Step 4: Run speculative and complete CPU tests**

Run: `cmake --build build-cpu --target test_speculative -j 2 && ctest --test-dir build-cpu -R speculative --output-on-failure`

Expected: `speculative` passes with the callback and block APIs.

Run: `ctest --test-dir build-cpu --output-on-failure`

Expected: all CPU tests pass.

- [ ] **Step 5: Commit the vector verifier**

```bash
git add runtime/include/k3x/speculative.hpp runtime/src/speculative.cpp tests/cpp/test_speculative.cpp
git commit -m "feat: add exact block target verification"
```

### Task 3: Exact CPU layer-major block runtime

**Files:**
- Modify: `runtime/include/k3x/model.hpp`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_model_session.cpp`

**Interfaces:**
- Consumes: Task 1 `ExpertMajorPlan`, Task 2 block verifier, existing `Engine` tensor/attention/native-MXFP4 operations, and copied `ModelState`.
- Produces: `SpeculativeVerificationMode { token_major, expert_major }`, an internal `Engine::forward_expert_major_block`, accepted-prefix state selection, and physical expert-major counters in `GenerationResult`.

- [ ] **Step 1: Add failing mode-boundary and exact parity tests**

Extend the existing scripted-provider test harness with two sessions that differ only in `speculative_verification`. For perfect and mixed block-2 scripts, assert equality of `token_ids`, `final_state`, committed `routed_experts`, committed `routed_k`, and provider updates. Assert that expert-major reports block calls and evaluated positions, and that the perfect case performs fewer Reader bytes than token-major. Add unsupported CPU-boundary cases for non-disabled L1, deadline L2, non-natural routing, and profile observation.

- [ ] **Step 2: Run the focused runtime test to verify RED**

Run: `cmake --build build-cpu --target test_model_session -j 2`

Expected: compile failure because `SpeculativeVerificationMode` and expert-major telemetry fields are absent.

- [ ] **Step 3: Add the opt-in runtime identity and preflight validation**

```cpp
enum class SpeculativeVerificationMode { token_major, expert_major };

struct RuntimeOptions {
    // Existing fields remain unchanged.
    SpeculativeVerificationMode speculative_verification{
        SpeculativeVerificationMode::token_major};
};
```

Before creating `Engine` or executing prefill, reject expert-major unless backend kind is CPU, execution is incremental, L1 is disabled, L2 is blocking, routing mode is natural, profile observation is false, and no profile output mutation can occur.

- [ ] **Step 4: Implement copied-state layer-major evaluation**

Introduce private block-only structures in `model.cpp` for per-position hidden/source banks, per-layer KDA snapshots, MLA prefix boundaries, target logits, evaluated routes, and counters. For each layer, process attention positions causally, calculate all natural routes, call `build_expert_major_plan`, load each `(layer, expert)` once, reuse its exact payload for every assignment, store routed results by router slot, and accumulate each token in original slot order. Dense layer 0 and shared branches remain scalar per position.

- [ ] **Step 5: Select and commit only the accepted state prefix**

Call `verify_greedy_target_block` with all block argmax tokens. On mismatch index `i`, move the snapshots after input position `i` into the live state; on full acceptance, move the snapshots after input position `C`. Crop MLA keys, values, shared keys, and length to the selected prefix. Append only committed routes to canonical diagnostics and keep rejected-suffix routes in separate evaluated fields.

- [ ] **Step 6: Run focused parity and complete CPU tests**

Run: `cmake --build build-cpu --target test_model_session -j 2 && build-cpu/test_model_session.exe tests/fixtures/synthetic.k3x`

Expected: perfect and mixed token/state/routing/provider parity pass; perfect expert-major Reader bytes are lower; unsupported modes fail before Reader counters change.

Run: `ctest --test-dir build-cpu --output-on-failure`

Expected: all CPU tests pass.

- [ ] **Step 7: Commit the runtime boundary**

```bash
git add runtime/include/k3x/model.hpp runtime/src/model.cpp tests/cpp/test_model_session.cpp
git commit -m "feat: execute exact CPU expert-major blocks"
```

### Task 4: CLI and benchmark telemetry contract

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Consumes: `SpeculativeVerificationMode` and `GenerationResult` fields from Task 3.
- Produces: `--speculative-verification token-major|expert-major` and JSON/CSV fields for target block calls, evaluated/discarded positions, unique expert sum/max, assignments, reused assignments, payload loads, and evaluated routes.

- [ ] **Step 1: Add failing CLI and schema tests**

Add a successful expert-major perfect/mixed invocation and invalid invocations for expert-major without `scripted-reference`, CUDA backend, L1 cache, deadline L2, fixed/adaptive routing, profile observation, and profile output. Assert invalid commands return 2 and do not create the requested JSON file. Extend benchmark schema fixtures with exact zero defaults and expert-major values.

- [ ] **Step 2: Run targeted pytest to verify RED**

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python/test_cpp_parity.py -k expert_major tests/python/test_benchmark_schema.py`

Expected: failures for the missing CLI option and schema fields.

- [ ] **Step 3: Parse and validate the CLI identity**

Add `speculative_verification_name = "token-major"`, parse the new flag, reject unknown values and unsupported combinations before `Reader::open`, and set `runtime_options.speculative_verification`. Keep all existing commands valid without the new flag.

- [ ] **Step 4: Serialize physical and committed telemetry**

Add the new fields to the runtime JSON, `BenchmarkRecord`, command construction, consistency checks, `write_results`, and CSV field list. Keep existing non-expert-major defaults at zero and use `null` only for non-applicable rates, not counters.

- [ ] **Step 5: Run targeted and full Python tests**

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py`

Expected: all targeted tests pass.

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python`

Expected: all CPU-compatible Python tests pass with environment-specific CUDA/liburing tests skipped.

- [ ] **Step 6: Commit CLI and telemetry**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
git commit -m "feat: expose expert-major verification telemetry"
```

### Task 5: B-0015 ablation and evidence validation

**Files:**
- Create: `tools/ablate_expert_major_verification.py`
- Create: `tests/python/test_expert_major_verification_ablation.py`
- Create: `results/b0015-expert-major-verification-wsl/README.md`
- Create: `results/b0015-expert-major-verification-wsl/*.json`
- Create: `results/b0015-expert-major-verification-wsl/*.csv`

**Interfaces:**
- Consumes: benchmark runner and CLI telemetry from Task 4.
- Produces: five-case B-0015 raw and aggregate artifacts with raw-summary parity checks.

- [ ] **Step 1: Write a failing matrix and aggregation test**

Define exactly `greedy`, `token-major-perfect-2`, `expert-major-perfect-2`, `token-major-mixed-2`, and `expert-major-mixed-2`. Assert 3 warmups and 20 samples, exact token/state/committed-route parity, expected acceptance/block counters, summary/raw/CSV equality, and SHA-256 fields for aggregate artifacts.

- [ ] **Step 2: Run the ablation test to verify RED**

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python/test_expert_major_verification_ablation.py`

Expected: import or file-not-found failure because the B-0015 runner does not exist.

- [ ] **Step 3: Implement the five-case runner**

Reuse `benchmark_synthetic.run_benchmark` and `write_results`. Compare every speculative diagnostic against greedy for generated tokens, final state, and committed routes; compare token-major and expert-major provider semantics; reject a run if any raw/summary/CSV field diverges. Do not assert a favorable latency or byte direction for mixed blocks.

- [ ] **Step 4: Run the test and real B-0015 measurement**

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python/test_expert_major_verification_ablation.py`

Expected: pass.

Run: `$env:K3X_BUILD_DIR='build-cpu'; python tools/ablate_expert_major_verification.py --artifact tests/fixtures/synthetic.k3x --output results/b0015-expert-major-verification-wsl --warmup 3 --iterations 20`

Expected: five raw JSON/CSV pairs plus summary JSON/CSV and a successful exact-parity report.

- [ ] **Step 5: Independently cross-check artifacts and commit**

Run: `$env:K3X_BUILD_DIR='build-cpu'; python -m pytest -q tests/python/test_expert_major_verification_ablation.py tests/python/test_benchmark_schema.py`

Expected: raw-summary parity and schema tests pass.

```bash
git add tools/ablate_expert_major_verification.py tests/python/test_expert_major_verification_ablation.py results/b0015-expert-major-verification-wsl
git commit -m "bench: measure expert-major verification"
```

### Task 6: Full verification, TITAN Ledger synchronization, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify: `docs/references.md` only if primary-source metadata changed.

**Interfaces:**
- Consumes: all implementation commits and measured B-0015 artifacts.
- Produces: a verified public Milestone 14 PR and post-merge `main` state.

- [ ] **Step 1: Run the complete applicable verification matrix**

Run CPU CTest and Python suites, Linux liburing/direct tests under WSL, CUDA CTest/Python tests on the RTX 5080 build, ASan/UBSan CPU tests, and Compute Sanitizer for perfect and mixed expert-major CLI invocations. Record exact commands, pass counts, skips, and any corrected environment invocation.

- [ ] **Step 2: Self-review the complete milestone diff**

Run `git diff --check`, inspect every caller of changed options/results, search for placeholder text, verify new defaults against prior raw fixtures, and confirm failed preflight commands leave Reader/output state untouched.

- [ ] **Step 3: Update the persistent documents from measured evidence**

Document the actual CPU execution boundary, accepted/rejected alternatives, B-0015 measurements, precise limitations, latest bottleneck, verification matrix, commit IDs, and next CUDA boundary. Update `PROJECT_STATE.md` last. Do not replace earlier measured records.

- [ ] **Step 4: Commit the ledger synchronization**

```bash
git add README.md ARCHITECTURE.md DECISIONS.md BENCHMARKS.md PROJECT_STATE.md checklist.md context-notes.md docs/references.md
git commit -m "docs: record milestone fourteen verification"
```

- [ ] **Step 5: Publish only after all evidence gates pass**

Push `codex/milestone-fourteen-expert-major-verification`, open a PR, require branch and PR CI success, verify the PR head is a descendant of public `main`, fast-forward merge, then require post-merge `main` CI success. Record the final publication head in `PROJECT_STATE.md` and `context-notes.md`.
