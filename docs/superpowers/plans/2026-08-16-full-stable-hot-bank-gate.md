# Full Stable Hot-Bank Gate Implementation Plan

> **For agentic workers:** Execute inline in the current task. The user explicitly disabled subagents for this phase.

**Goal:** Measure whether the exact stable per-layer device hot bank improves the existing 78-layer natural-routing NVFP4 gate before changing the asynchronous runtime architecture.

**Architecture:** Reuse the existing full GLM-5.2 Python reference, verified K3X bundle, `.pgu` sidecars, 40 GiB trunk cache, and 4 GiB device cache. Change only the device-cache policy from the prior `layer_balanced` gate to `stable_hot_bank`; keep natural Top-8 routing, exact sidecar payloads, and two decode tokens.

**Tech Stack:** Python 3.12, PyTorch 2.13.0+cu130, WSL2 Ubuntu-24.04, RTX 5080, K3X bundle and packed expert sidecars.

**Spec:** `PROJECT_STATE.md` stable per-layer hot-bank milestone and `DECISIONS.md` D-0098.

## Global Constraints

- Do not report the structured B-0002 result as full-model TPS.
- Keep natural routing and generated-token evidence visible.
- Do not enable proxy experts, reduced Top-K, sparse attention, or pinned telemetry in this comparison.
- Stop before execution if host RAM or VRAM headroom is unsafe.
- Record the full raw JSON/log and compare against the prior `layer_balanced` gate.

## Checklist

- [x] Verify branch cleanliness, bundle/config identity, host RAM, and RTX 5080 headroom.
- [x] Run one prefill plus two decode tokens with the stable hot bank.
- [x] Extract token IDs, per-step seconds/TPS, cache hits/misses/bypasses/promotions, resident bytes, and VRAM.
- [x] Compare only like-for-like fields against a current-HEAD layer-balanced gate.
- [x] Update benchmarks, decisions, architecture if warranted, and project state last.
- [ ] Commit and publish the evidence before selecting the next code boundary.
