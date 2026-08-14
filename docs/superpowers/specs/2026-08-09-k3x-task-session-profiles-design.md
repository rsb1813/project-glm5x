# K3X Milestone 10 Task and Session Profiles Design

## Scope

Milestone 10 adds runtime-only task metadata and a small persistent expert-routing profile. Metadata and profile state are cache-policy inputs only; they are never appended to prompt token IDs and never alter natural router scores, selected Top-K experts, or expert weights.

The milestone records per-expert frequency, adjacent-layer expert transitions, a deterministic hot bank, and repository/task metadata. Prefix and KDA payload caching remains a later VAULT milestone; this profile may preserve opaque prefix/KDA identity metadata but does not claim that reusable model state exists.

## Alternatives and decision

Three integrations were considered.

1. Preload the task hot bank before inference. This can create large speculative L2 traffic before any observation and is rejected for this milestone.
2. Seed LFU counters directly from persisted counts. This hides whether an eviction came from prior or live evidence and prevents a clear prior-decay contract, so it is rejected.
3. Add an explicit experimental `profiled` eviction policy that keeps prior and live observations separate. This is accepted because it is runtime-switchable, auditable, and leaves all existing exact policies unchanged.

`disabled` remains the public default. `static`, `lru`, `lfu`, and `least-stale` retain their Milestone 9 behavior.

## Runtime metadata contract

Metadata is a bounded ordered map of UTF-8 key/value pairs such as `TASK=coding`, `LANG=cpp`, `PHASE=debug`, and `REPO=foo`. Keys use uppercase ASCII letters, digits, and underscore; values are non-empty UTF-8 strings without control characters. Duplicate records inside a persisted file or one CLI header are rejected. Explicit current-run metadata may replace a value loaded from an older session, allowing fields such as `PHASE` to advance. Limits are fixed and tested so an untrusted profile cannot allocate without bound.

The CLI accepts one comma-separated `--runtime-metadata` value. A test compares runs with and without metadata and requires identical prompt IDs, logits, routing, recurrent state, and generated tokens.

## Profile model

The session owns one profile with two evidence classes.

- Prior evidence is loaded from an optional task or repository profile before generation.
- Live evidence is collected from the exact natural Top-K sets observed during the current process.

For every routed layer, live frequency increments once per selected `(layer, expert)`. Adjacent routed layers add the Cartesian transition counts from the prior selected set to the current selected set. A generation boundary clears only the transient previous-layer set, preventing a transition from the final layer of one token or request to the first layer of the next.

The hot bank is the deterministic top-N list by merged prior-plus-live frequency, then smaller layer and expert identifiers. It is derived when saved and validated when loaded; it is not a separate source of truth.

## Profiled eviction score

At an admission decision, each resident entry receives normalized prior frequency `p` and normalized live-session frequency `s`. The prior weight is

`alpha = prior_strength / (prior_strength + live_route_observations)`.

The usefulness score is `alpha * p + (1 - alpha) * s`. The lowest usefulness is evicted first, followed by least recent access and stable insertion sequence. `prior_strength` is a bounded explicit runtime parameter and defaults to 64 route observations only when `profiled` is selected.

This gives a deterministic crossover: an empty session uses the prior, while sufficient contradictory live routing outweighs it. Selected-set protection and exact bypass remain unchanged. Transition data is persisted for the later predictor milestone but does not affect eviction yet.

## Persistence format

The v1 profile is deterministic, versioned, line-oriented UTF-8 with a fixed magic line, declared record counts, bounded integer fields, and a final CRC32C over the canonical body. Records are sorted before writing. Unknown versions, duplicate records, malformed UTF-8, count mismatches, overflow, and checksum mismatch fail without mutating the current session profile.

Save uses process-interruption-safe publication within one filesystem: write a sibling temporary file, flush and close it, then rename it over the destination. The temporary file is removed on failure. This milestone does not claim power-loss durability because it does not fsync the file and parent directory. Loading never assumes a complete-model or large-memory resident state.

The persisted aggregate becomes prior evidence when reopened. New live observations remain separate and gradually outweigh it. Metadata permits repository-specific files without baking a repository name into model tokens.

## Evaluation

Deterministic unit traces must prove parsing bounds, checksum rejection, idempotent canonical round-trip, frequency and transition accounting, hot-bank ordering, prior/live crossover, exact selected-set protection, and unchanged behavior of every existing cache mode.

B-0011 will compare `lfu`, `least-stale`, and `profiled` on a fixed synthetic routing trace under matched capacity, with a helpful prior, a conflicting prior, and enough live observations to cross over. It records exact output parity, cache traffic, prior weight, live observations, profile bytes, and load/save time. WSL2 synthetic timing cannot select a production default.
