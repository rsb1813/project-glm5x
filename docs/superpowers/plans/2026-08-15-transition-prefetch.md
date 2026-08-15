# Exact N+1 Transition Prefetch Plan

**Goal:** Reduce selected-expert load stalls by predicting the next layer's candidate set from persisted/live routing transitions without changing natural routing or output.

**Design:** Add a deterministic `RuntimeProfile::predict_next()` ranking API first. It mixes normalized prior and live transition mass with the existing prior-strength schedule, returns only positive-score candidates, and resolves ties by `(layer, expert)`. Production scheduling remains opt-in and reuses a predicted ticket only when the next layer's exact router selects the same key; misses use the existing exact load path.

**Rejected first steps:** A tiny learned predictor has no real trace evidence yet. N+2 composition adds uncertainty before N+1 recall is measured. Predicted experts must not alter Top-K, routing weights, or permanent pruning.

## Checklist

- [x] Add RED tests for deterministic ranking, prior/live mixing, empty evidence, and candidate bounds.
- [x] Implement the minimal profile ranking API and pass focused C++ tests.
- [x] Add opt-in N+1 predicted ticket reuse to deadline scheduling with explicit hit/miss/overfetch telemetry.
- [x] Prove token, routing, logits, and state parity against deadline scheduling without prediction.
- [x] Run bounded synthetic recall/byte/stall measurements and reject or retain the production hook from evidence.
- [x] Update architecture, decisions, benchmarks, and project state last.
