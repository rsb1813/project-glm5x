# K3X Milestone 11 Adaptive Top-K and Exact Rescue Plan

## Goal

Add a deterministic runtime routing-policy boundary with natural Top-K reference, fixed and adaptive K, explicit quality escalation, and exact non-resident expert loading, then measure both quality divergence and traffic.

## Task 1 — Pure routing policy

- [x] Add RED tests for natural/fixed/adaptive selection, stable ties, invalid configuration, mass, entropy effective support, boundary confidence, and quality floors.
- [x] Implement the smallest standalone routing-policy type with bounded telemetry.
- [x] Verify the dedicated CPU unit test and unchanged natural router tests.

## Task 2 — Reference and synthetic Top-16 fixture

- [x] Extend fixture creation with an explicit configuration argument without changing the default artifact.
- [x] Add a deterministic 16-of-24 synthetic model and PyTorch routing-policy oracle.
- [x] Prove fixed K16 equals natural and reduced K modes run through the declared reference policy.

## Task 3 — Runtime, CLI, and exact rescue

- [x] Replace direct `config_.top_k` loop bounds with one immutable per-layer routing decision.
- [x] Add natural/fixed/adaptive CLI validation and explicit failure/critical quality-floor mapping.
- [x] Count exact selected cold loads only with an enabled L1 cache and prove residency never changes routing.
- [x] Export variable-K diagnostics and routing/rescue telemetry through JSON/CSV.

## Task 4 — B-0012 and quality evidence

- [x] Build an ablation across natural K16, fixed K4/8/12/16, adaptive thresholds, escalated adaptive, and bounded-cache rescue.
- [x] Record natural divergence, tokens, logits/state error, average K, cold rescues, cache/Reader traffic, timing, and memory.
- [x] Keep natural as default unless measured quality and end-to-end evidence justify a later decision.

## Task 5 — Verification, ledger, and publication

- [x] Run CPU, liburing/direct, CUDA, ASan/UBSan, and applicable Compute Sanitizer suites.
- [x] Self-review all callers and serialization consumers; close every Critical/Important issue.
- [x] Update ARCHITECTURE, DECISIONS, BENCHMARKS, and PROJECT_STATE last.
- [x] Publish through a public PR and verify branch/PR/main CI.
