# K3X Milestone 8 Deadline-Aware Expert Loader Plan

## Goal

Implement and measure an opt-in, exact current-layer deadline-aware L2 expert loader while retaining the blocking Reader path as the correctness reference.

## Task 1 — Scheduler contract and deterministic tests

- [x] Add failing unit tests for deadline ordering, stable ties, failure propagation, single ticket consumption, bounded queue counters, and destructor drain.
- [x] Implement the minimum single-worker deadline queue and metrics.
- [x] Run the scheduler unit tests and commit the logical unit.

## Task 2 — Reader and L1 concurrency safety

- [x] Add failing concurrency regressions for Reader counter/data-plane serialization and L1 hit/admission safety.
- [x] Add only the synchronization required by the scheduler.
- [x] Run C++ tests under ThreadSanitizer where supported, otherwise record the unavailable tool and run ASan/UBSan.

## Task 3 — Exact runtime integration

- [x] Add `blocking|deadline` runtime configuration with blocking as default.
- [x] Submit only the exact current-layer natural Top-K set after routing.
- [x] Overlap worker loads with independent routed/shared work and wait before expert use.
- [x] Preserve exact tokens, routing traces, Reader bytes, and reference-mode counters.

## Task 4 — CLI and telemetry

- [x] Add stable CLI validation and JSON schema fields for scheduler mode and counters.
- [x] Add Python cross-language parity and invalid-option tests.
- [x] Document that multi-layer lookahead remains unimplemented.

## Task 5 — B-0009 ablation

- [x] Add a blocking/deadline ablation crossed with supported Reader modes.
- [x] Require exact tokens, routing, logical bytes, and successful-load parity.
- [x] Run warm synthetic measurements and preserve raw JSON/CSV.
- [x] Do not claim native P44 Pro, cold-cache, or full-model performance.

## Task 6 — Verification, review, ledger, and publication

- [x] Run CPU, liburing/direct, CUDA, sanitizer, and applicable Compute Sanitizer suites.
- [x] Request one Terra high read-only final review and close Critical/Important findings.
- [x] Update architecture, decisions, benchmarks, README, checklist, context, and `PROJECT_STATE.md` last.
- [x] Publish through a public PR, require branch/PR/main CI, and preserve the worktree.
