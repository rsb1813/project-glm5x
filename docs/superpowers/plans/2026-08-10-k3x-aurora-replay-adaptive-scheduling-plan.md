# K3X AURORA Replay and Adaptive Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce real reduced-Top-K K3 self-draft candidates, verify them with the unchanged exact natural target, and adapt proposal length from measured acceptance and expert-union cost.

**Architecture:** A separate CPU Reader/backend/session replays the committed prefix and implements the existing `DraftProvider` lifecycle. A pure scheduler chooses among proposal lengths 1, 2, and 4. Target verification decorates its existing strict result with physical work feedback, while provider and target telemetry stay separate.

**Tech Stack:** C++20 runtime and tests, CMake/Ninja, Python 3.12 pytest tooling, deterministic K3X synthetic Top-16 artifact, optional CUDA 13.3 target verification.

## Global Constraints

- Natural target routing and strict greedy verification remain the correctness authority.
- The draft runtime is CPU FP32 with fixed K4/K6/K8/K12 strictly below the artifact natural K.
- Draft and target Reader, profiler, routing, and timing counters remain separate.
- AURORA replay supports maximum proposal lengths 1, 2, or 4 only.
- Ordinary greedy and `scripted-reference` behavior and defaults remain unchanged.
- Every new source file begins with a one-line Korean role comment.
- No full Kimi K3 checkpoint, paid cloud resource, reduced precision, proxy, pruning, EcoSpec, MoE-Spec, or AcceptMoE behavior enters this milestone.

---

### Task 1: Target feedback and provider telemetry contract

**Files:**
- Modify: `runtime/include/k3x/speculative.hpp`
- Modify: `runtime/src/speculative.cpp`
- Modify: `tests/cpp/test_speculative.cpp`

**Interfaces:**
- Produces: four default-zero physical feedback fields on `DraftVerification`.
- Produces: `DraftProviderStats` and `DraftProvider::stats() const noexcept` with a default-zero implementation.
- Preserves: the pure verifier determines committed tokens only and leaves all physical feedback zero.

- [x] **Step 1: Write the failing contract tests**

Append assertions to every representative perfect, mismatch, and empty result in `tests/cpp/test_speculative.cpp` and add a provider with no `stats()` override.

```cpp
class DefaultStatsProvider final : public k3x::DraftProvider {
public:
    k3x::Result<k3x::DraftProposal> propose(
        const k3x::DraftRequest& request) override {
        return k3x::Result<k3x::DraftProposal>::success(
            {.anchor_token = request.anchor_token});
    }
    void update(const k3x::DraftVerification&) override {}
};

assert(result.value().target_positions_evaluated == 0);
assert(result.value().target_positions_discarded == 0);
assert(result.value().expert_major_payload_loads == 0);
assert(result.value().expert_major_assignments == 0);
const auto stats = DefaultStatsProvider{}.stats();
assert(stats.proposal_calls == 0);
assert(stats.selected_length_4 == 0);
```

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
cmake --build build --target test_speculative -j 8
```

Expected: compilation fails because `DraftVerification` has no physical fields and `DraftProvider` has no `stats()` method.

- [x] **Step 3: Add the minimal contract**

Add these exact public fields in `runtime/include/k3x/speculative.hpp`.

```cpp
struct DraftProviderStats {
    std::uint64_t proposal_calls{};
    std::uint64_t candidate_tokens{};
    std::uint64_t replayed_context_tokens{};
    std::uint64_t generation_nanoseconds{};
    std::uint64_t reader_calls{};
    std::uint64_t reader_bytes{};
    std::uint64_t routing_decisions{};
    std::uint64_t routing_selected_experts{};
    std::uint64_t selected_length_1{};
    std::uint64_t selected_length_2{};
    std::uint64_t selected_length_4{};
    std::uint64_t scheduler_growths{};
    std::uint64_t scheduler_backoffs{};
};

struct DraftVerification {
    // existing fields stay in their current order
    std::size_t target_positions_evaluated{};
    std::size_t target_positions_discarded{};
    std::size_t expert_major_payload_loads{};
    std::size_t expert_major_assignments{};
};

class DraftProvider {
public:
    virtual ~DraftProvider() = default;
    virtual Result<DraftProposal> propose(const DraftRequest& request) = 0;
    virtual void update(const DraftVerification& verification) = 0;
    virtual DraftProviderStats stats() const noexcept { return {}; }
};
```

Do not write these fields in `verify_greedy_draft` or `verify_greedy_target_block`; aggregate initialization must leave them zero.

- [x] **Step 4: Run focused GREEN**

Run:

```bash
cmake --build build --target test_speculative -j 8
ctest --test-dir build -R '^speculative$' --output-on-failure
```

Expected: 1/1 speculative test passes.

- [x] **Step 5: Commit**

```bash
git add runtime/include/k3x/speculative.hpp runtime/src/speculative.cpp tests/cpp/test_speculative.cpp
git commit -m "feat: expose speculative target feedback"
```

### Task 2: Pure adaptive proposal-length scheduler

**Files:**
- Create: `runtime/include/k3x/aurora_scheduler.hpp`
- Create: `runtime/src/aurora_scheduler.cpp`
- Create: `tests/cpp/test_aurora_scheduler.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `DraftVerification` from Task 1.
- Produces: `AuroraBlockPolicy`, `AuroraSchedulerConfig`, and `AdaptiveDraftScheduler`.
- Guarantees: no rung skip, request cap, rejection backoff, prefix-survival gate, and expert-cost gate.

- [x] **Step 1: Add the failing executable and wished-for API tests**

Create `tests/cpp/test_aurora_scheduler.cpp` with a Korean first-line comment and literals equivalent to the following.

```cpp
const k3x::AuroraSchedulerConfig config{
    .policy = k3x::AuroraBlockPolicy::adaptive,
    .maximum_length = 4,
    .minimum_prefix_survival = 0.5,
    .maximum_unique_load_ratio = 0.9,
};
auto created = k3x::AdaptiveDraftScheduler::create(config);
require(created);
auto scheduler = std::move(created.value());
require(scheduler.select(4).value() == 1);

k3x::DraftVerification accepted_one{
    .proposed_draft_tokens = 1,
    .accepted_draft_tokens = 1,
    .expert_major_payload_loads = 8,
    .expert_major_assignments = 16,
};
require(scheduler.observe(accepted_one));
require(scheduler.select(4).value() == 2);

k3x::DraftVerification rejected_two{
    .proposed_draft_tokens = 2,
    .accepted_draft_tokens = 1,
    .expert_major_payload_loads = 20,
    .expert_major_assignments = 32,
};
require(scheduler.observe(rejected_two));
require(scheduler.select(4).value() == 1);
require(scheduler.stats().scheduler_backoffs == 1);
```

Add separate scopes for fixed policy, `select(0) == 0`, request caps, two fully accepted growth steps to length four, survival below 0.5, ratio above 0.9, loads without assignments, accepted greater than proposed, and unsupported maximum three.

Register `runtime/src/aurora_scheduler.cpp` in `k3x_runtime` and register `test_aurora_scheduler` as CTest `aurora_scheduler`.

- [x] **Step 2: Run and verify RED**

Run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_aurora_scheduler -j 8
```

Expected: compilation fails because `k3x/aurora_scheduler.hpp` does not exist.

- [x] **Step 3: Implement the validated scheduler**

Create `runtime/include/k3x/aurora_scheduler.hpp` with a Korean first-line comment and this public surface.

```cpp
enum class AuroraBlockPolicy { fixed, adaptive };

struct AuroraSchedulerConfig {
    AuroraBlockPolicy policy{AuroraBlockPolicy::fixed};
    std::size_t maximum_length{1};
    double minimum_prefix_survival{0.5};
    double maximum_unique_load_ratio{0.9};
};

class AdaptiveDraftScheduler {
public:
    static Result<AdaptiveDraftScheduler> create(
        AuroraSchedulerConfig config);
    Result<std::size_t> select(std::size_t request_maximum);
    Result<bool> observe(const DraftVerification& verification);
    DraftProviderStats stats() const noexcept;
private:
    explicit AdaptiveDraftScheduler(AuroraSchedulerConfig config);
    AuroraSchedulerConfig config_;
    std::array<std::uint64_t, 4> prefix_trials_{};
    std::array<std::uint64_t, 4> prefix_successes_{};
    std::array<std::uint64_t, 3> cost_loads_{};
    std::array<std::uint64_t, 3> cost_assignments_{};
    std::optional<std::size_t> largest_qualified_rung_;
    std::optional<std::size_t> rejection_cap_;
    DraftProviderStats stats_{};
};
```

Implement a fixed ladder `constexpr std::array<std::size_t, 3>{1, 2, 4}`. `create` rejects unsupported maxima and non-finite/out-of-range thresholds. `select` returns zero for request zero without counters, uses `min(request_maximum, maximum_length)`, and chooses the longest eligible rung. Adaptive eligibility is limited to length one before any observation, then through at most one rung past `largest_qualified_rung_`; observed rungs require final-position `(success+1)/(trials+2) >= minimum_prefix_survival`; observed expert feedback requires `loads/assignments <= maximum_unique_load_ratio`. If no rung is eligible, `select` returns zero so the target performs one ordinary step.

`observe` accepts an all-zero proposal as a state-preserving no-op. Otherwise it rejects proposed lengths outside the ladder, accepted greater than proposed, discarded greater than evaluated, and nonzero loads with zero assignments. It updates every proposed prefix position and the exact-length cost bucket. A rejection sets `rejection_cap_` to the previous rung, or a one-shot zero after a length-one rejection, and increments backoffs. Consuming the zero cap schedules one smallest-rung retry so cumulative survival cannot permanently disable drafting. A fully accepted and gate-qualified observation clears the cap, advances the qualified frontier, and increments growth only when it enables the next rung.

- [x] **Step 4: Run focused GREEN and existing verifier tests**

Run:

```bash
cmake --build build --target test_aurora_scheduler test_speculative -j 8
ctest --test-dir build -R '^(aurora_scheduler|speculative)$' --output-on-failure
```

Expected: 2/2 tests pass.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/aurora_scheduler.hpp runtime/src/aurora_scheduler.cpp tests/cpp/test_aurora_scheduler.cpp
git commit -m "feat: schedule adaptive AURORA blocks"
```

### Task 3: Replay-based self-draft provider

**Files:**
- Create: `runtime/include/k3x/aurora.hpp`
- Create: `runtime/src/aurora.cpp`
- Create: `tests/cpp/test_aurora.cpp`
- Create: `tests/python/test_aurora_runtime.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `AdaptiveDraftScheduler`, `Reader`, `ComputeBackend`, `RuntimeOptions`, `generate_greedy`.
- Produces: `AuroraReplayConfig` and `AuroraReplayDraftProvider::create`.
- Guarantees: one outstanding proposal, exact committed-history replay, separate draft telemetry, and latched lifecycle errors.

- [x] **Step 1: Write the failing integration test**

Create `tests/cpp/test_aurora.cpp` with a Korean first-line comment. Accept the synthetic artifact path as `argv[1]`, open a dedicated Reader and CPU backend, and construct this provider.

```cpp
k3x::RuntimeOptions draft_options;
draft_options.incremental = true;
draft_options.routing_policy.mode = k3x::RoutingMode::fixed;
draft_options.routing_policy.fixed_k = 4;
auto provider = k3x::AuroraReplayDraftProvider::create(
    draft_reader.value(), *draft_backend,
    std::vector<std::uint32_t>{1, 7, 3, 9}, draft_options,
    {.scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                   .maximum_length = 2}});
require(provider);
auto proposal = provider.value()->propose({
    .anchor_token = 43,
    .max_draft_tokens = 2,
    .generated_position = 1,
    .generated_tokens = std::vector<std::uint32_t>{43},
});
require(proposal);
require(proposal.value().anchor_token == 43);
require(proposal.value().candidate_tokens.size() == 2);
require(provider.value()->stats().proposal_calls == 1);
require(provider.value()->stats().candidate_tokens == 2);
require(provider.value()->stats().replayed_context_tokens == 5);
require(provider.value()->stats().reader_bytes > 0);
```

Add cases for a second `propose` before `update`, wrong position, wrong anchor, changed generated history, request maximum zero, a mismatch update followed by corrected target history, and a deliberately inconsistent update that makes the next proposal fail before Reader counters change.

Create `tests/python/test_aurora_runtime.py` with a Korean first-line comment. Its fixture builds the 24-expert Top-16 artifact with `SyntheticK3Config.default().replace(num_experts=24, top_k=16)` and invokes `test_aurora <artifact>` through `cpp_binary("test_aurora")`. Register the executable in CMake without adding it to CTest because it requires a generated artifact.

- [x] **Step 2: Run and verify RED**

Run:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_aurora -j 8
python -m pytest -q tests/python/test_aurora_runtime.py
```

Expected: compilation fails because `k3x/aurora.hpp` and `AuroraReplayDraftProvider` do not exist, so the Python wrapper cannot run.

- [x] **Step 3: Implement the provider**

Create `runtime/include/k3x/aurora.hpp` with a Korean first-line comment and this public surface. Move `runtime/src/model.cpp` into `k3x_runtime` and remove its duplicate direct compilation from `k3x_run` and `test_model_session`; the provider must call the public `generate_greedy` implementation through the library rather than copying model execution.

```cpp
struct AuroraReplayConfig {
    AuroraSchedulerConfig scheduler{};
};

class AuroraReplayDraftProvider final : public DraftProvider {
public:
    static Result<std::unique_ptr<AuroraReplayDraftProvider>> create(
        Reader& reader, ComputeBackend& backend,
        std::span<const std::uint32_t> prompt,
        RuntimeOptions draft_options, AuroraReplayConfig config);
    Result<DraftProposal> propose(const DraftRequest& request) override;
    void update(const DraftVerification& verification) override;
    DraftProviderStats stats() const noexcept override;
private:
    AuroraReplayDraftProvider(
        Reader& reader, ComputeBackend& backend,
        std::vector<std::uint32_t> prompt,
        RuntimeOptions draft_options,
        AdaptiveDraftScheduler scheduler);
    Reader& reader_;
    ComputeBackend& backend_;
    RuntimeSession session_;
    std::vector<std::uint32_t> prompt_;
    std::vector<std::uint32_t> expected_generated_;
    AdaptiveDraftScheduler scheduler_;
    DraftProviderStats stats_{};
    std::uint32_t pending_anchor_{};
    std::size_t pending_candidates_{};
    bool initialized_{};
    bool pending_{};
    bool lifecycle_error_{};
};
```

`create` rejects non-CPU backends, nonincremental options, non-fixed routing, fixed K outside 4/6/8/12, enabled cache/scheduler/profile features, empty prompt, and invalid scheduler configuration. `propose` validates lifecycle and request before calling the scheduler. It builds `sequence = prompt_ + request.generated_tokens`, snapshots Reader counters, calls `generate_greedy(reader_, backend_, sequence, selected, session_)`, and accumulates only the delta plus returned routing/timing fields. It records the pending anchor/count only after successful generation.

`update` accepts only a pending proposal with matching anchor, proposed count, accepted count, nonempty committed tokens, and scheduler-valid feedback. On inconsistency it sets `lifecycle_error_` and clears pending state. On success it appends committed tokens to `expected_generated_` and clears pending state.

- [x] **Step 4: Run focused GREEN**

Run:

```bash
cmake --build build --target test_aurora -j 8
python -m pytest -q tests/python/test_aurora_runtime.py
```

Expected: AURORA provider tests pass with real reduced-Top-K draft bytes and no target Reader dependency.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/aurora.hpp runtime/src/aurora.cpp tests/cpp/test_aurora.cpp tests/python/test_aurora_runtime.py
git commit -m "feat: add AURORA replay drafter"
```

### Task 4: Decorate target feedback and export provider stats

**Files:**
- Modify: `runtime/include/k3x/model.hpp`
- Modify: `runtime/src/model.cpp`
- Modify: `tests/cpp/test_model_session.cpp`

**Interfaces:**
- Consumes: Task 1 feedback fields and `DraftProvider::stats()`.
- Produces: `GenerationResult::draft_provider` counters flattened as explicit fields.
- Guarantees: feedback reflects actual target work and is visible to `update` before the provider chooses the next block.

- [x] **Step 1: Write failing runtime assertions**

Extend `ScriptedDraftProvider` in `test_model_session.cpp` to return sentinel stats and assert feedback received in `verifications`.

```cpp
k3x::DraftProviderStats stats() const noexcept override {
    return {.proposal_calls = 7, .candidate_tokens = 9,
            .selected_length_2 = 3};
}

require(perfect.verifications[0].target_positions_evaluated == 3);
require(perfect.verifications[0].target_positions_discarded == 0);
require(perfect_result.value().draft_proposal_calls == 7);
require(perfect_result.value().draft_candidate_tokens == 9);
require(perfect_result.value().draft_selected_length_2 == 3);
```

For mixed expert-major, assert evaluated positions equal the full block, discarded equals the rejected suffix, and feedback payload loads/assignments equal the block counters. For token-major mismatch, assert evaluated is accepted plus one and discarded is zero.

- [x] **Step 2: Run and verify RED**

Run:

```bash
cmake --build build --target test_model_session -j 8
```

Expected: compilation fails because `GenerationResult` lacks draft fields and runtime does not decorate feedback.

- [x] **Step 3: Implement runtime decoration**

Add explicit `draft_*` counters matching every `DraftProviderStats` field to `GenerationResult`. In each loop iteration capture target counters before verification. Before `draft_provider.update` set physical feedback.

```cpp
verification.value().target_positions_evaluated =
    expert_major ? block_inputs.size()
                 : result.target_decode_forward_calls - calls_before;
verification.value().target_positions_discarded =
    expert_major ? block_inputs.size() -
                       verification.value().committed_tokens.size()
                 : 0;
verification.value().expert_major_payload_loads =
    expert_major ? block.payload_loads : 0;
verification.value().expert_major_assignments =
    expert_major ? block.assignments : 0;
draft_provider.update(verification.value());
```

At successful function exit, copy `draft_provider.stats()` into `GenerationResult`. Do not copy provider stats on failure and do not alter ordinary greedy result fields.

- [x] **Step 4: Run focused and regression GREEN**

Run:

```bash
cmake --build build --target test_model_session test_speculative test_aurora -j 8
ctest --test-dir build -R '^speculative$' --output-on-failure
python -m pytest -q \
  tests/python/test_cpp_parity.py::test_runtime_session_reuses_l1_experts_across_generations \
  tests/python/test_aurora_runtime.py
```

Expected: all selected tests pass.

- [x] **Step 5: Commit**

```bash
git add runtime/include/k3x/model.hpp runtime/src/model.cpp tests/cpp/test_model_session.cpp
git commit -m "feat: feed target cost to AURORA"
```

### Task 5: CLI identity, separated JSON, and benchmark schema

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Consumes: `AuroraReplayDraftProvider` and Task 4 generation fields.
- Produces: `aurora-replay`, `--aurora-draft-k`, `--aurora-block-policy`, JSON/CSV draft telemetry.
- Preserves: existing CLI defaults and scripted schema values.

- [x] **Step 1: Write failing CLI and schema tests**

Add invalid cases to `test_cpp_parity.py` for unknown policy, K0/K16, block size 3, AURORA with reduced target routing, AURORA options in `none`, nonincremental mode, and draft K not below artifact natural K. Assert the output path remains absent.

Add one real Top-16 AURORA invocation that requires exact natural greedy token/final-state/committed-route parity and asserts nonzero draft proposal, replay, Reader, and routing counters.

Extend `_record()` and JSON/CSV assertions in `test_benchmark_schema.py` with the exact fields from `DraftProviderStats`, `aurora_draft_k`, and `aurora_block_policy`. Existing `none` and `scripted-reference` records must serialize zero or `not-applicable` values.

- [x] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest -q \
  tests/python/test_cpp_parity.py \
  tests/python/test_benchmark_schema.py
```

Expected: failures report unknown AURORA flags and missing benchmark fields.

- [x] **Step 3: Implement CLI preflight and construction**

In `main.cpp`, parse these defaults.

```cpp
std::string aurora_draft_k_text = "0";
std::string aurora_block_policy_name = "fixed";
```

Allow `speculative_mode_name` values `none`, `scripted-reference`, and `aurora-replay`. AURORA requires incremental mode, natural target routing, an empty script, block size 1/2/4, and draft K4/K6/K8/K12. After opening the target Reader, decode its natural Top-K and reject `draft_k >= natural_top_k` before creating providers or output files.

Open a second Reader with the same Reader options, create a separate unprofiled CPU backend, build fixed-reduced-Top-K `RuntimeOptions`, create `AuroraReplayDraftProvider`, and pass it to the existing `generate_speculative` entrypoint. Keep the draft Profiler separate from the target Profiler.

Serialize every new field with explicit names such as `draft_reader_completed_bytes`, never by adding draft bytes to `reader_completed_bytes`. Serialize `aurora_draft_k=0` and `aurora_block_policy="none"` for non-AURORA modes.

- [x] **Step 4: Extend Python schema and run GREEN**

Add dataclass fields and `_run_process` arguments in `tools/benchmark_synthetic.py`; pass the new flags only for AURORA mode and preserve LF CSV writing. Parse the new JSON values directly with no inferred byte totals.

Run:

```bash
cmake --build build --target k3x_run -j 8
python -m pytest -q \
  tests/python/test_cpp_parity.py \
  tests/python/test_benchmark_schema.py
```

Expected: all focused Python tests pass.

- [x] **Step 5: Commit**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
git commit -m "feat: expose AURORA replay runtime"
```

### Task 6: B-0017 ablation, full verification, and ledger publication

**Files:**
- Create: `tools/ablate_aurora_replay.py`
- Create: `tests/python/test_aurora_replay_ablation.py`
- Create: `results/b0017-aurora-replay-wsl/*`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

**Interfaces:**
- Consumes: public CLI and benchmark schema from Task 5.
- Produces: deterministic B-0017 raw JSON/CSV, summary digests, exact parity report, and measured fixed/adaptive comparison.
- Preserves: no default change without favorable representative evidence.

- [x] **Step 1: Write the failing ablation-schema test**

Create `tests/python/test_aurora_replay_ablation.py` with a Korean first-line comment. Require the matrix names below and reject any record that lacks exact target parity or separated draft bytes.

```python
assert [case["name"] for case in aurora_matrix()] == [
    "natural-greedy",
    "aurora-k4-fixed-1",
    "aurora-k4-fixed-2",
    "aurora-k4-fixed-4",
    "aurora-k4-adaptive-token",
    "aurora-k4-fixed-2-expert",
    "aurora-k4-adaptive-expert",
]
assert record["token_ids"] == natural["token_ids"]
assert record["final_state_max_abs_error"] <= 1e-6
assert record["committed_route_parity"] is True
assert record["draft_reader_completed_bytes"] > 0
assert record["reader_completed_bytes"] > 0
```

Add raw JSON/CSV SHA-256 and canonical aggregate checks patterned after B-0016, using LF CSV writers.

- [x] **Step 2: Run and verify RED**

Run:

```bash
python -m pytest -q tests/python/test_aurora_replay_ablation.py
```

Expected: import fails because `tools.ablate_aurora_replay` does not exist.

- [x] **Step 3: Implement and smoke the runner**

Create `tools/ablate_aurora_replay.py` with a Korean first-line comment. Materialize the 24-expert natural Top-16 fixture through existing converter helpers, run one warmup/one sample smoke, compare every AURORA row against natural diagnostics, and write raw artifacts plus checksummed summaries. Never infer physical NVMe, GPU utilization, or quality values that were not measured.

Run:

```bash
python tools/ablate_aurora_replay.py \
  --runner build/k3x_run \
  --output results/b0017-aurora-replay-wsl \
  --warmups 1 --samples 1
python -m pytest -q tests/python/test_aurora_replay_ablation.py
```

Expected: the smoke and focused cross-check pass.

- [x] **Step 4: Run the canonical measurement**

Run with three warmups and twenty samples.

```bash
python tools/ablate_aurora_replay.py \
  --runner build/k3x_run \
  --output results/b0017-aurora-replay-wsl \
  --warmups 3 --samples 20
```

Record the measured direction even if every replay row is slower. Recompute every raw JSON/CSV hash and the canonical aggregate from the committed bytes.

- [x] **Step 5: Run the full verification matrix**

Run CPU, liburing/direct, ASan/UBSan, and CUDA builds with the same capability flags used by B-0016. At minimum run:

```bash
ctest --test-dir build --output-on-failure
python -m pytest -q
ctest --test-dir build-uring --output-on-failure
K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1 python -m pytest -q
ctest --test-dir build-uring-asan --output-on-failure
ctest --test-dir build-cuda --output-on-failure
K3X_TEST_CUDA=1 python -m pytest -q
```

Run Compute Sanitizer on any new CUDA expert-major AURORA CLI row; do not claim sanitizer coverage for the CPU replay provider itself. If no new CUDA code exists, existing CUDA kernels retain their prior sanitizer evidence and B-0017 reports that scope explicitly.

- [x] **Step 6: Synchronize evidence documents and commit**

Update measured values only after the canonical run. Change AURORA registry status to `Experimental replay reference` only if code and all applicable tests pass. Record D-040's measured outcome, B-0017 fields and caveats, README milestone evidence, performance bottleneck, checklist, and context notes. Update `PROJECT_STATE.md` last with exact commits, tests, artifacts, bottleneck, and next persistent-state task.

```bash
git add tools/ablate_aurora_replay.py tests/python/test_aurora_replay_ablation.py \
  results/b0017-aurora-replay-wsl README.md ARCHITECTURE.md \
  PERFORMANCE_MODEL.md DECISIONS.md BENCHMARKS.md checklist.md \
  context-notes.md PROJECT_STATE.md
git commit -m "bench: measure AURORA replay scheduling"
```

- [x] **Step 7: Final review and public integration**

Run `git diff --check`, review every default and evidence boundary, request one read-only Terra high final review, apply at most one batch of valid Critical/Important fixes, rerun affected verification, push the branch, open a ready public PR, wait for push/PR correctness, rebase-merge, and verify post-merge `main` correctness before declaring Milestone 16 public.
