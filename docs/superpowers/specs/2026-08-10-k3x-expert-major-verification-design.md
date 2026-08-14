# K3X Milestone 14 Exact Expert-Major Verification Design

## Scope

Milestone 14 adds a non-default exact expert-major target verifier for the synthetic K3 graph. It evaluates one linear draft block layer by layer, forms the natural per-layer unique-expert union, loads each unique native-MXFP4 expert payload once, and reuses that payload for every verification token routed to it. The existing token-major verifier remains the reference and default.

The first executable boundary is intentionally `CPU + blocking L2 + disabled L1 + natural routing`. This isolates the new execution order, recurrent-state commit, and physical Reader traffic from cache eviction, profile observation, asynchronous loading, CUDA transfer, reduced Top-K, and lossy verifier budgeting. Unsupported combinations fail before target state or Reader counters are mutated.

This milestone does not implement a learned DSpark drafter, a draft tree, CUDA batched expert kernels, EcoSpec selection, MoE-Spec budgets, AcceptMoE eligibility, AURORA, or a default speedup.

## Primary-source evidence

- Kimi K3 vLLM commit `44351f81` accepts multi-token hidden-state matrices in `KimiMoE.forward`. Its KDA path separates speculative tokens, consumes `num_accepted_tokens`, and invokes multi-query causal-convolution and recurrent kernels. This establishes that K3 speculative verification needs an explicit accepted-state contract rather than treating a block as independent tokens.
- MoE-Spec arXiv `2602.16052`, Appendix C, describes expert-oriented batched MoE execution: gather assignments by active expert, compute each active expert for its assigned tokens, and scatter/accumulate token results. Its expert budgets and substitution/truncation policies are lossy and are not part of this milestone.
- SpecMoE arXiv `2604.10152` reports coalescing duplicate expert migrations during target verification. Its self-assisted drafter and hot-expert replacement policy are separate from K3X's exact target scheduler.
- EcoSpec arXiv `2607.12696` defines verification cost using the marginal growth of a candidate path's expert union while leaving the target verification rule unchanged. That supports measuring unique experts and reuse now, but cost-aware draft selection remains a later milestone.
- AcceptMoE arXiv `2608.02989` explicitly distinguishes token count, activated-expert union size, and transfer traffic. Its commitment-weighted constrained expert set changes the model distribution, so it remains outside strict natural routing.

The arXiv records inspected on 2026-08-10 expose no author-maintained implementation for SpecMoE, EcoSpec, MoE-Spec, or AcceptMoE. MoE-Spec identifies official EAGLE as its base but does not publish the described patch through the paper record. K3X therefore reproduces only source-described invariants and does not claim implementation compatibility.

## Alternatives

### Full CUDA block executor first

Rejected for this milestone. It would combine KDA speculative cache handling, MLA crop, layer-major hidden-state execution, expert assignment batching, CUDA kernels, H2D coalescing, and rollback in one correctness boundary. A failure would not identify which invariant broke.

### Route-trace replay or prefetch-only union

Rejected. Replaying previously recorded expert IDs can demonstrate an optimistic byte count but does not execute candidate hidden states, preserve target state, or prove that one fetched payload serves all naturally routed candidates.

### Exact CPU layer-major block reference

Selected. It executes the actual synthetic graph, changes real Reader traffic, and fixes the scheduling/state contract before CUDA specialization. Expert arithmetic may remain scalar inside each expert group, but payload loading and scheduling are genuinely expert-major.

## Runtime identity

`RuntimeOptions` gains `SpeculativeVerificationMode { token_major, expert_major }`, defaulting to `token_major`. The CLI exposes `--speculative-verification token-major|expert-major`; it is meaningful only with `--speculative-mode scripted-reference`.

`expert-major` requires all of the following.

- `backend=cpu`.
- Incremental execution.
- `l1-expert-cache=disabled`.
- `l2-schedule=blocking`.
- Natural routing.
- No runtime profile observation or profile output.

Validation occurs before prefill. Existing token-major combinations and ordinary greedy generation remain unchanged.

## Pure scheduling contract

`runtime/include/k3x/expert_major.hpp` defines the independently tested scheduling types.

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
    std::size_t assignment_count;
};
```

`build_expert_major_plan` rejects empty routes, length mismatches, non-finite contributions, and duplicate experts within one token route. Groups are ordered by first occurrence while scanning token index and then natural router slot. Assignments within a group retain that same stable scan order.

Group order controls payload fetch order only. Each token's expert outputs are stored by original router slot and accumulated in original router order, so grouping cannot change FP32 summation order.

## Layer-major block execution

For proposal candidates `c[0..C-1]`, the target input block is `[anchor, c[0], ..., c[C-1]]` and has `C+1` positions. Output position `i<C` predicts `c[i]`; output position `C` supplies the target bonus token.

The block executor starts from a copy of the committed `ModelState` and maintains hidden states and Attention Residual source banks per position. For each decoder layer it performs the following steps.

1. Process attention positions in causal order. KDA and MLA state updates therefore see exactly the same token order as scalar incremental execution.
2. Save the layer's persistent state after every position.
3. Produce every position's feed-forward input.
4. On the dense first layer, run the existing dense path per position.
5. On each MoE layer, compute full router scores and the immutable natural routing decision for every position.
6. Build the stable unique-expert plan from all position routes.
7. Compute every position's latent projection and shared expert branch.
8. Load each group's exact native-MXFP4 payload once and execute that payload for all assignments in the group.
9. Store results by `(position, router_slot)`, then accumulate each position in its original router order.
10. Finish routed norm/up projection, shared addition, residual addition, and advance every position to the next layer.

After the final layer, output residual, final norm, and LM head run for every block position. This yields `C+1` target argmax tokens and the per-position natural routing trace.

## Exact acceptance and state commit

The pure block verifier consumes the proposal plus exactly `C+1` target tokens. It applies the existing strict greedy rule and returns the same `DraftVerification` shape as token-major verification.

If the first mismatch is candidate position `i`, the committed target state contains the block inputs through position `i` inclusive. If all candidates match, it contains all `C+1` block inputs. This matches the invariant that state contains every committed output token except the final bonus token.

The block executor records each KDA layer's convolution and recurrent state after every input position. MLA appends the complete block to a temporary KV state and crops keys, values, shared keys, and length to the selected committed prefix. Only after verification succeeds does the runtime move the selected snapshots into the session state.

The original state is unchanged on proposal validation, target execution, scheduling, Reader, backend, or provider-update failure. The provider is updated only after state selection and the complete committed token block are known.

Snapshot memory is an exact synthetic reference, not the final full-size KDA strategy. B-0015 records host peak memory. A later CUDA/full-dimension implementation must replace per-position full KDA snapshots with accepted-state-aware kernels or compact rollback state before this mode can be considered production-capable.

## Routing and telemetry semantics

Canonical diagnostic `routed_experts` and `routed_k` continue to describe only the committed target history and therefore remain comparable with greedy generation. Expert-major execution additionally records the complete evaluated verification routes, including rejected suffix positions, in separate diagnostic fields.

The following counters are added.

- Target block forward calls.
- Target positions evaluated.
- Target positions discarded after the first mismatch.
- Sum and maximum of unique experts per verification layer.
- Total expert assignments.
- Reused assignments, defined as assignments minus unique expert payload loads.
- Exact expert payload loads performed by expert-major scheduling.

Existing routing-decision totals count physically evaluated decisions. Reader counters continue to report actual logical reads. No derived payload saving is substituted for measured Reader bytes.

## Failure and boundary rules

- Proposal anchor, candidate count, token range, and the complete `C+1` target-vector size are validated before state commit.
- Empty proposals execute one exact target input and form ordinary per-layer Top-K unions.
- Output counts zero and one retain current behavior and do not invoke a draft provider.
- A block cannot append more tokens than the requested output count.
- Any missing expert, invalid extent, non-finite contribution, duplicate route entry, or incomplete position snapshot fails explicitly.
- The default token-major verifier and all existing CLI commands retain their current output and schema values, with new fields zero or not applicable.

## Correctness tests

The milestone requires tests that would fail on wrong group order, duplicate payload loads, token-order accumulation, rejected-suffix state commit, MLA crop, KDA snapshot selection, or default-mode changes.

- Pure plan literals cover overlapping and disjoint routes, first-use order, assignments, contributions, duplicate rejection, mismatched lengths, and non-finite values.
- Pure block verification covers perfect, first/middle/final mismatch, empty proposal, wrong target-vector size, and out-of-vocabulary target output.
- C++ runtime tests compare token-major and expert-major perfect/mixed blocks for generated tokens, final flattened state, committed routing/K, and provider updates.
- Evaluated expert-major logits and routes are compared with scalar target execution over the same complete candidate input block on a copied state.
- Reader tests prove one load per unique `(layer, expert)` inside a block and exact payload reuse across assignments.
- CLI validation rejects every unsupported expert-major combination before prefill.
- Existing greedy and token-major suites remain unchanged and passing.

## B-0015 measurement

B-0015 runs greedy, token-major perfect block-2, expert-major perfect block-2, token-major mixed block-2, and expert-major mixed block-2 on the deterministic synthetic artifact with three warmups and twenty samples.

Every row records decode and prefill tok/s, TTFT, target calls/positions, acceptance, unique experts, assignments, reuse, Reader calls/bytes, cache counters, host RAM, routing K, tokens, logits/state diagnostics, and enabled runtime identity. The report separately states committed correctness and physically evaluated suffix work.

The expert-major mode is retained only if it preserves exact tokens, committed state, and committed natural routing and if raw counters prove unique-payload loading. A Reader-byte reduction on perfect proposals is expected but is not claimed until measured. Mixed blocks may cost more because rejected suffix positions are intentionally evaluated; that result must be reported rather than hidden.

## Default and follow-up

The public default remains ordinary greedy generation, and token-major remains the speculative reference. Milestone 14's CPU expert-major mode remains experimental regardless of B-0015 timing because it lacks production KDA state handling, CUDA multi-token kernels, H2D coalescing, and full-model evidence.

The next implementation boundary after B-0015 is a CUDA expert-major batch interface that consumes the same plan and state contract. EcoSpec, MoE-Spec, and AcceptMoE experiments remain later, separate, explicitly quality-measured modes.
