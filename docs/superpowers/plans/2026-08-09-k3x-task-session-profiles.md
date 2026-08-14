# K3X Milestone 10 Task and Session Profiles Plan

## Goal

Add bounded runtime-only metadata and deterministic persistent routing profiles that can bias an opt-in exact cache policy, while proving that prompt tokens and model outputs remain unchanged.

## Task 1 — Profile data model and persistence

- [x] Add RED tests for metadata validation, frequency/transition observation, hot-bank ordering, canonical round-trip, checksum failure, and load failure atomicity.
- [x] Implement the minimal bounded v1 profile parser/writer and atomic save path.
- [x] Verify the dedicated profile CTest.

## Task 2 — Profiled exact eviction

- [x] Add RED traces for helpful prior, conflicting prior, live-observation crossover, selected-set protection, and unchanged legacy policies.
- [x] Implement explicit `profiled` scoring with separate prior and live evidence.
- [x] Verify deterministic victim and counter behavior.

## Task 3 — Runtime and CLI integration

- [x] Add runtime metadata, task/session profile load, and session profile save options.
- [x] Prove metadata is not inserted into prompt tokens and exact tokens/routing/logits/state remain unchanged.
- [x] Export bounded profile telemetry without claiming prefix/KDA payload caching.

## Task 4 — Measurement, ledger, and publication

- [x] Run B-0011 with raw JSON/CSV and programmatic parity checks.
- [x] Run CPU, liburing/direct, CUDA, sanitizer, and applicable Compute Sanitizer suites.
- [x] Complete self-review and close Critical/Important findings.
- [x] Update TITAN Ledger with `PROJECT_STATE.md` last.
- [x] Publish through a public PR and verify branch/PR/main CI.
