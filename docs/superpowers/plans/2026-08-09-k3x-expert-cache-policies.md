# K3X Milestone 9 Expert Cache Policies Plan

## Goal

Reproduce SpecMD Least-Stale and compare it fairly with exact LRU/LFU policies while preserving disabled and static reference modes.

## Task 1 — Policy contract and trace oracle

- [x] Add failing deterministic trace tests for static, LRU, LFU, and Least-Stale victim selection.
- [x] Add selected-set protection and collision-miss accounting tests.
- [x] Implement the minimum policy metadata and eviction loop.

## Task 2 — Runtime integration

- [x] Add stable CLI/runtime identities without changing the disabled default.
- [x] Assign forward cycles and mark the natural Top-K set before admission.
- [x] Preserve exact handles, routing, tokens, and Reader error behavior.

## Task 3 — Telemetry and B-0010

- [x] Export evictions and collision misses through C++ JSON and Python benchmark schema.
- [x] Add a capacity/policy ablation with exact parity and capability-independent failures.
- [x] Record raw JSON/CSV and avoid native/full-model claims.

## Task 4 — Verification and publication

- [x] Run CPU, liburing/direct, CUDA, sanitizer, and applicable Compute Sanitizer suites.
- [x] Complete one read-only final review and close Critical/Important findings.
- [x] Update TITAN Ledger with `PROJECT_STATE.md` last.
- [x] Publish through a public PR and verify branch/PR/main CI.
