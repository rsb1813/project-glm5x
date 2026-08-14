# K3X Milestone 9 Expert Cache Policies Design

## Scope

Milestone 9 replaces the no-eviction static L1 experiment with runtime-switchable exact `lru`, `lfu`, and `least-stale` policies. `disabled` remains the public default and `static` remains available as the no-eviction reference. Routing, Top-K, expert bytes, and miss handling remain exact.

This milestone does not implement task/session priors, transition prediction, ORBIT, prefetch-driven admission, mixed precision, proxy experts, or pruning.

## Source boundary

SpecMD arXiv 2602.03921 defines Least-Stale as a two-queue spatial-temporal eviction policy. Experts touched in a previous forward cycle are stale; experts touched or selected in the current cycle are current. Stale entries are evicted before current entries, and each queue uses FIFO ordering based on layer position. The paper reports its framework as open-sourcing underway and no author-linked public implementation was found during the 2026-08-09 review. K3X therefore labels this work a paper reproduction, not a code port.

## Runtime contract

Each cache access supplies `(forward_cycle, layer, access_sequence)`. A cycle is one token forward through all decoder layers. The store retains exact immutable whole-expert handles and charges the existing six native MXFP4 extents.

One `RuntimeSession` permits one active generation at a time. The session-level generation guard covers the complete graph call so concurrent callers cannot overwrite the store's active cycle, layer, or selected-set policy context. Independent sessions remain independent.

- `static` admits only when unused capacity is sufficient and otherwise bypasses exactly.
- `lru` evicts the least recently accessed entry, with stable insertion sequence as the tie-break.
- `lfu` evicts the lowest lifetime access count, then least recent access, then stable insertion sequence.
- `least-stale` evicts prior-cycle entries before current-cycle entries. Within a staleness class, already-processed left layers are victims before upcoming layers. Left layers use forward FIFO order; if no left victim exists and exact admission still requires space, the farthest future layer is evicted before a nearer upcoming layer. Stable access and insertion sequences break remaining ties. Current-cycle eviction is permitted only when exact admission otherwise cannot satisfy the hard capacity.

The selected Top-K set is marked current before any miss admission. This prevents sequential loading of one selected expert from evicting another expert needed later in the same layer. Evicted map entries may remain alive through outstanding shared handles; cache residency counters describe policy-owned entries, not total allocator lifetime.

## Metrics

Add evictions, collision misses, and policy identity. A collision miss occurs when an entry evicted during a forward cycle is requested again before that cycle ends. This includes a future-layer expert retained from the prior token, evicted by an earlier layer in the current token, and then requested when execution reaches its own layer. Preserve hits, misses, bypasses, current bytes, and peak bytes.

## Evaluation

First use a deterministic trace simulator with configurable capacity and expert sizes. It must contain a layer-sequential trace where LRU collides and Least-Stale protects upcoming-layer entries. Then integrate the same policies into the synthetic runtime and require exact token, routing, and Reader-byte accounting.

B-0010 will compare `disabled`, `static`, `lru`, `lfu`, and `least-stale` at 2-, 8-, and 16-expert synthetic capacities. It will report hit rate, misses, evictions, collision misses, logical Reader bytes, decode/prefill/TTFT, and exact output parity. No policy becomes default from WSL2 synthetic evidence.
