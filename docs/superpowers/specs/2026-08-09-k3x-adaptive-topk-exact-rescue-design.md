# K3X Milestone 11 Adaptive Top-K and Exact Rescue Design

## Scope

Milestone 11 introduces a runtime routing-policy boundary while preserving the checkpoint's natural Top-K as the reference. Kimi K3 computes sigmoid scores for all 896 routed experts, applies correction bias only to stable expert selection, activates 16 experts, and normalizes the selected unbiased scores. The existing K3X synthetic graph implements the same rule at 2-of-8. The official architecture evidence is the [MoonshotAI Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3), and the current independent serving implementation is [vLLM's KimiLinear MoE path](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/kimi_linear.py).

This milestone adds fixed K4/6/8/12/16 and an adaptive policy for checkpoints whose natural Top-K is 16. A dedicated 16-of-24 synthetic fixture exercises the released K ladder without downloading full Kimi K3 weights. The existing 2-of-8 artifact remains unchanged and continues to validate the natural reference path.

Reduced K is explicitly lossy. It may alter layer outputs, logits, recurrent state, and generated tokens. Correctness means that each mode implements its declared routing rule exactly, while quality is measured against natural Top-16 rather than assumed equivalent.

## Alternatives and decision

Three integrations were considered.

1. Overwrite the decoded checkpoint `top_k`. This is small but destroys the distinction between model identity and runtime policy, makes reference mode harder to audit, and is rejected.
2. Keep the full score vector and stable natural order, then let a pure policy select a prefix. This is accepted because natural mode is unchanged, fixed/adaptive behavior is deterministic, and every selected expert retains its original ordering and unbiased weight.
3. Reorder or substitute experts using residency. This may reduce reads but changes the target routing rule independently of K and can silently replace a high-score cold expert with a lower-score resident expert. It is rejected for exact-rescue modes and reserved for explicitly lossy proxy/pruning experiments.

## Routing policy contract

The router always computes all expert logits, unbiased sigmoid scores, and the stable correction-biased order before policy selection. Let `N` be the checkpoint natural Top-K. Natural mode selects the first `N` entries and must remain bit-for-bit behaviorally identical to the existing runtime.

Fixed mode accepts only K4, K6, K8, K12, or K16, requires `K <= N`, and selects the first `K` entries of the natural order. Selected contribution weights are renormalized from the unbiased sigmoid scores over the chosen prefix and multiplied by the checkpoint routed scaling factor. Correction bias never enters contribution weights.

Adaptive mode evaluates the allowed ladder entries not exceeding `N`. For the natural prefix distribution `p`, it records normalized Shannon entropy, cumulative mass, and a boundary confidence gap. The smallest candidate is selected only when all configured predicates hold.

- Its cumulative unbiased-score mass within the natural prefix meets `mass_target`.
- It is at least the entropy effective support `ceil(exp(H(p)))` rounded up to the K ladder.
- Its normalized boundary gap `(p[K-1] - p[K]) / max(p[0], epsilon)` meets `minimum_boundary_gap`; the natural K has no boundary requirement.
- It is no smaller than the external quality floor.

The default routing mode remains `natural`. Adaptive thresholds have bounded validated ranges and no implicit hardware-dependent defaults. Ties retain smaller expert identifiers through the existing stable order.

## Quality escalation contract

The routing policy accepts an explicit quality floor rather than embedding compiler, test, or tool semantics into model math. The CLI may map an externally supplied consecutive agent-failure count to a floor of K8 after one failure, K12 after two failures, and natural K16 after three or more failures. A critical/recovery flag forces natural Top-K. The chosen floor and triggering signal are telemetry.

This boundary lets PHOENIX or a serving layer report failures later without coupling the core router to a particular agent harness. No background quality monitor or automatic tool-result parser is claimed in this milestone.

## Exact cold-expert rescue

Policy selection never consults residency. After the selected set is fixed, every selected non-resident expert is fetched through the existing exact K3X MXFP4 path before use. No proxy, pruning, zero fill, resident substitution, or score rewrite is permitted. With an enabled L1 cache, each selected miss increments `cold_rescue_count`; disabled caching remains an exact streaming baseline and is not labeled a cache rescue.

Repeated selected experts continue to update the session profile and can enter its derived hot bank. This is promotion by measured routing frequency, not permanent checkpoint pruning. Load failure remains a generation failure rather than a silent approximation.

## Telemetry

Generation and benchmark records add the declared routing mode, checkpoint natural Top-K, routing-decision count, total selected experts, average selected K, average normalized entropy, average selected cumulative mass, average boundary confidence, applied quality floor, escalation count, and exact cold-rescue count.

Variable-K routing diagnostics carry a per-decision K vector in addition to the flattened expert IDs. Existing natural-mode fields retain their values. B-0012 must store raw JSON/CSV and compare natural, fixed, adaptive, escalated, and cache-rescue cases one switch at a time.

## Correctness and quality gates

- Natural mode on the existing 2-of-8 artifact must preserve all prior tokens, routing, logits, recurrent state, cache counters, and Reader bytes.
- The pure policy must cover stable ties, invalid K, bounded thresholds, mass selection, entropy support, boundary rejection, and quality-floor escalation.
- Fixed K16 on the 16-of-24 fixture must equal natural Top-16 exactly.
- Reduced fixed/adaptive modes must match a PyTorch policy oracle for selected experts, normalized weights, layer outputs, logits, recurrent state, and tokens under that declared mode.
- A cold selected expert must be loaded exactly even when a lower-ranked resident expert exists; rescue must never change the selected IDs.
- B-0012 must report divergence from natural Top-16 alongside traffic and timing. No fast mode becomes a default without simultaneous quality evidence.

## Non-goals

This milestone does not implement permanent pruning, proxy experts, learned routing, ORBIT prefetch prediction, speculative verification budgets, mixed trunk quantization, full-model quality claims, or native-P44-Pro performance claims.
