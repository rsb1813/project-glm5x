# Adaptive Exact Hot-Bank Implementation Plan

> **For agentic workers:** Execute inline in the current task. The user explicitly disabled subagents for this phase.

**Goal:** Fill the 1.346 GB device-cache budget left idle by the stable bank with exact experts that have been observed at least twice, without changing natural routing or evicting the per-layer base bank.

**Architecture:** Add a separate opt-in `adaptive_hot_bank` policy to `GLM5XExpertTensorCache`. The configured protected count remains the mandatory base tier per layer. Candidates outside that tier bypass on first observation, enter a global exact extra tier on the second observation when capacity permits, and may replace only a colder extra entry when full.

**Tech Stack:** Python 3.12, PyTorch, existing GLM5X decoded expert cache and benchmark CLI.

**Spec:** `DECISIONS.md` D-0099 and B-0003 in `BENCHMARKS.md`.

## Global Constraints

- Preserve `stable_hot_bank`, `layer_balanced`, and `lru` behavior unchanged.
- Never evict a mandatory per-layer base entry to admit an adaptive extra.
- Admit only exact decoded expert payloads already selected by natural routing.
- First observation must bypass; second observation may admit.
- Equal-frequency candidates must not churn a full extra tier.
- Keep the new policy default-off.

## Checklist

- [ ] Write and run RED tests for first-observation bypass, second-observation admission, base protection, and colder-extra replacement.
- [ ] Implement the minimal adaptive policy inside `GLM5XExpertTensorCache`.
- [ ] Add model and CLI validation/telemetry wiring.
- [ ] Run focused cache/model/schema regression.
- [ ] Measure a deterministic repeated real-sidecar trace before any full-model rerun.
- [ ] Run the 78-layer full gate only if the bounded trace increases hits while preserving payload identity.
- [ ] Update persistent documents and project state last.
