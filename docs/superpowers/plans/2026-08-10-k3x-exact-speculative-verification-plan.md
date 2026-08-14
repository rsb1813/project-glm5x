# K3X Milestone 13 Exact Speculative Block Verification Plan

> **For Codex:** Use test-driven development for every implementation task and verification-before-completion before milestone claims.

## Goal

Implement a DSpark-lifecycle-compatible external draft interface and an exact greedy token-major target verifier without changing default generation or claiming expert-major acceleration.

## Task 1 — Pure verification contract

1. Add RED unit tests for proposal anchor validation, bounded candidate count, perfect and mismatching prefixes, empty proposals, and exact committed bonus tokens.
2. Add the smallest public proposal/request/verification/provider types.
3. Implement a pure greedy prefix-verification helper driven by a target-step callback.
4. Verify the focused native unit test and commit.

## Task 2 — Incremental runtime integration

1. Add RED cross-language tests using a scripted provider for perfect, wrong, partial, empty, and short proposals.
2. Add an explicit speculative generation entrypoint or options boundary while leaving all `generate_greedy` overloads unchanged.
3. Preserve the state invariant that target state contains every committed token except the final output token.
4. Verify token, prefill diagnostic, final state, routing, Reader, cache, and rescue parity against greedy execution.
5. Commit.

## Task 3 — Telemetry and CLI test harness

1. Add RED schema tests for verification blocks, proposed and accepted draft tokens, committed verification tokens, and maximum proposal length.
2. Add a deterministic scripted-draft CLI test mode only if it is necessary for end-to-end benchmarking; do not expose a fake production drafter identity.
3. Extend JSON/CSV benchmark schema without changing existing defaults.
4. Commit.

## Task 4 — B-0014 correctness and overhead measurement

1. Build a deterministic proposal trace containing perfect, early-mismatch, late-mismatch, and empty blocks.
2. Compare ordinary greedy and token-major speculative reference on the synthetic model.
3. Record decode, prefill, TTFT, target forwards, acceptance, Reader traffic, cache statistics, RAM, and applicable CUDA counters.
4. Label unique expert union and speculative speedup as not implemented or not claimed.
5. Cross-check raw JSON/CSV and commit the artifacts.

## Task 5 — Verification, ledger, and publication

1. Run CPU, liburing/direct, CUDA, ASan/UBSan, and applicable Compute Sanitizer checks.
2. Self-review the complete milestone diff for state, count, failure, and default-path errors.
3. Update ARCHITECTURE, DECISIONS, BENCHMARKS, README, context notes, checklist, and PROJECT_STATE last.
4. Push the public branch, require branch and PR CI, fast-forward main only after ancestry verification, and require post-merge main CI.
