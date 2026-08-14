# K3X Milestone 13 Exact Speculative Block Verification Design

## Scope

Milestone 13 establishes a lossless speculative-decoding contract and a token-major target reference. It does not implement a trained DSpark drafter, parallel target execution, expert-major scheduling, confidence scheduling, EcoSpec, MoE-Spec, AcceptMoE, or a throughput default.

The reference remains ordinary incremental greedy generation. The speculative path must generate the same tokens and execute the target over the same committed token history for every proposal, including deliberately wrong proposals.

## Source evidence

- DeepSeek's DSpark paper, arXiv `2607.05147`, separates draft proposal from target verification, accepts the longest target-consistent prefix, appends a target bonus token, and treats confidence-based verification length as a separate scheduler.
- DeepSpec commit `005e03b81cec38b7da6399833d609ee89a2587f2` is the inspected implementation. `deepspec/eval/base_evaluator.py` defines `DraftProposal`, verifies an anchor-plus-draft block, commits the accepted prefix plus one target token, and crops target cache to the committed prefix. `deepspec/eval/dspark/evaluator.py` owns DSpark-specific context and updates it after target verification. `deepspec/eval/dspark/draft_ops.py` constructs proposals whose first verify token is the current accepted anchor.
- DeepSpec's public evaluator uses rejection sampling for nonzero temperature. K3X Milestone 13 is narrower: the existing runtime is greedy, so a candidate is accepted only when it equals the target argmax at that position. This is an intentional K3X contract, not a claim that DeepSpec itself uses greedy-only verification.

## Alternatives

### Add expert-major verification immediately

Rejected for this milestone. The current Engine exposes a one-token mutating forward. Adding expert unioning, multi-token state handling, and storage amortization before acceptance and rollback semantics are fixed would combine two independent correctness boundaries.

### Add a built-in reduced-Top-K self-drafter immediately

Rejected for this milestone. It would conflate AURORA quality policy, draft cost, and exact target verification. A scripted external proposal source is sufficient to test the verifier with perfect, partially correct, wrong, empty, and malformed blocks.

### Expose a semantic draft/target interface with token-major target execution

Selected. The proposal owns an accepted anchor plus zero or more candidate tokens. The verifier owns target execution, acceptance, bonus-token selection, state commit, and telemetry. The draft source receives the verification result so a future DSpark adapter can crop or advance its private cache without giving it authority over target correctness.

## Public contract

`DraftProposal` contains the current accepted anchor and a bounded candidate-token vector. `DraftRequest` contains the current anchor, maximum candidate count, absolute generated position, and already generated tokens. `DraftVerification` reports proposed and accepted candidate counts, the committed accepted prefix plus target bonus token, and whether the entire proposed prefix matched.

`DraftProvider` has two responsibilities.

1. `propose(request)` returns an anchor-matching bounded proposal.
2. `update(verification)` observes the exact commit so a stateful drafter can crop or advance its own cache.

The initial implementation is an in-process C++ interface. It is DSpark-compatible at the proposal/verification lifecycle boundary, not checkpoint- or tensor-ABI compatible with DeepSpec.

## Strict greedy verification

The runtime prefills normally and emits the first target-greedy token. Its recurrent state therefore contains the prompt but not that emitted token. Each speculative round then follows this state machine.

1. Limit the requested candidate count to `min(block_size, remaining_output_tokens - 1)` so one target bonus token always fits.
2. Require the proposal anchor to equal the last committed output token and its candidate count not to exceed the request.
3. Run the target on the anchor or last accepted candidate, obtaining the next target argmax and advancing KDA/MLA state by exactly that input token.
4. Accept a candidate only if it equals that argmax. On the first mismatch, commit the target argmax as the bonus token and stop without evaluating the mismatched candidate.
5. If every candidate matches, run one final target step on the last accepted candidate and commit its argmax as the bonus token.
6. Notify the draft provider only after the complete committed block is known.

The post-round state contains every committed token except the final bonus token, exactly matching ordinary incremental generation. No target-state rollback is needed in the token-major reference because rejected suffix tokens are never executed. Later parallel and expert-major verifiers must reproduce this state boundary, potentially through explicit snapshots or crop operations.

## Failure and boundary rules

- Speculation requires incremental execution and a positive block size.
- Empty proposals are valid and reduce to one ordinary target step.
- Wrong anchor, oversized block, out-of-vocabulary candidate, provider failure, or invalid configuration returns an explicit error.
- A proposal can never change target routing, Top-K, cache policy, precision, or expert residency selection.
- Output-count truncation is exact because the runtime never requests candidates that would displace the required target bonus token.
- Session generation serialization remains unchanged.

## Telemetry

`GenerationResult` records verification blocks, proposed draft tokens, accepted draft tokens, committed tokens produced through verification, and maximum proposed block length. Acceptance rate is derived as accepted divided by proposed and is undefined when no candidates were proposed.

Token-major verification does not claim unique-expert union or fetch amortization. Those fields remain absent or explicitly not applicable until Milestone 14 expert-major scheduling.

## Acceptance matrix

- Disabled/default generation is byte-for-byte behaviorally unchanged.
- Perfect, first-token mismatch, middle mismatch, final mismatch, empty, and short proposals all match ordinary greedy tokens.
- Final recurrent state, logits at the accepted prefix, natural routing trace, Reader bytes, cache outcomes, and cold-rescue counts match greedy execution under the same runtime options.
- The provider observes exactly the accepted candidate count and committed tokens.
- Count 0, count 1, block size larger than remaining output, and stop-at-count boundaries are covered.
- Invalid anchor, oversized proposal, invalid token, invalid configuration, and provider failure are rejected without fabricated output.
- CPU, liburing, and CUDA builds retain their applicable suites. The token-major verifier must pass ASan/UBSan; no speedup is required or claimed.

## Default and follow-up

The public default remains non-speculative greedy generation. Token-major speculation is a correctness reference and may be slower. The next milestone may add expert-major target scheduling only after this contract passes and is measured. DSpark confidence scheduling and learned drafting remain separate future work.
