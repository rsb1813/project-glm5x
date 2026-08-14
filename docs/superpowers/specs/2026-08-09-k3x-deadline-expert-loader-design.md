# K3X Milestone 8 Deadline-Aware Expert Loader Design

## Status and boundary

This milestone implements the first exact asynchronous L2-to-L1 boundary on the synthetic executable model. It does not implement routing prediction, multi-layer N/N+1/N+2 lookahead, eviction, Least-Stale, or native P44 Pro policy selection.

The blocking Reader and synchronous expert-load path remain the reference default. The new path is opt-in and must preserve exact tokens, routing traces, payload validation, and Reader byte accounting.

## Decision

Add a bounded single-worker expert loader with a deadline-priority queue. Each request carries an estimated use time, estimated fetch latency, payload bytes, current residency, and a stable submission sequence. Non-resident work is ordered by estimated slack.

```text
slack = estimated_use_time - now - estimated_fetch_latency
priority = smallest slack, then earliest use, then smallest sequence
```

The first runtime integration submits the exact natural Top-K expert set only after the current layer router has selected it. While the worker performs the blocking Reader load, the main thread computes independent routed/shared projections. It waits before the first expert use. Prediction errors cannot occur because no predicted routing is used.

## Alternatives

### Extend Reader with native asynchronous completion tokens

This could expose io_uring directly, but it would duplicate completion, direct-I/O bounce-buffer, failure-drain, Windows fallback, and counter-lifetime logic already proven in the blocking Reader. It is rejected for this milestone because it broadens the correctness surface before scheduling behavior is measured.

### Predict future-layer experts now

This could approach the chartered N/N+1/N+2 pipeline, but it requires ORBIT-style prediction policy and miss semantics that belong after the exact scheduler exists. It is deferred, not silently approximated.

### Launch one `std::async` task per expert

This is small but provides no global deadline order, bounded concurrency, deterministic teardown, or future multi-layer queue. It is rejected.

## Ownership and lifetime

`RuntimeSession` owns the scheduler so state and counters persist across generation calls. A request returns a move-only ticket backed by shared completion state. The worker owns the loader callable until completion. Session destruction drains or fails all queued work before referenced runtime objects can disappear.

Reader operations remain semantically blocking. A Reader-internal lock protects its shared data plane and counters because the new worker can read while the main thread lazily loads dense tensors. The lock may serialize concurrent storage calls, but CPU/GPU computation can still overlap with the worker's I/O.

`HostExpertStore` admission and lookup become synchronized. Resident hits complete inline and do not enter the worker queue. Misses use existing validation and capacity rules; failed work changes neither residency nor hit/miss statistics beyond the existing contract.

## Runtime modes

- `blocking` is the default and preserves the current call order.
- `deadline` enables exact current-layer asynchronous expert loads.

CLI configuration rejects unknown modes. Deadline mode does not require io_uring and must work with `pread|io_uring` and `buffered|direct` wherever the underlying Reader mode is supported.

## Metrics

Add scheduler counters for submissions, inline resident hits, completions, ready-before-use, late-at-use, queued high-water mark, requested expert bytes, estimated deadline misses, worker load time, and exposed wait time. These are runtime counters, not token-speed claims.

The ablation must record existing correctness and traffic metrics plus the new counters. A speedup is accepted only if measured; zero or negative benefit remains a valid result.

## Failure semantics

- Loader failures are delivered through the ticket and become the same generation failure as the blocking path.
- Queue insertion must be failure-atomic.
- A ticket may be consumed once.
- Session destruction cannot leave a worker touching Reader or store state.
- Unknown, duplicate, or stale tickets fail closed in tests and never substitute another payload.

## Verification

1. Unit-test deadline order with a blocked worker and controlled request deadlines.
2. Unit-test failure propagation, single consumption, bounded queue accounting, and clean destruction.
3. Run exact blocking/deadline token and routing parity on CPU.
4. Cross the deadline mode with supported Reader engines/cache modes.
5. Run CPU, liburing/direct, CUDA, sanitizers, and applicable Compute Sanitizer checks.
6. Measure a B-0009 synthetic ablation without labeling it full-model or native-NVMe evidence.

## Future extension

ORBIT or a transition-table predictor may later submit N+1/N+2 requests into the same queue. Those requests must remain prefetch-only: a miss or cancellation may add wait time but cannot alter natural routing or output. Multi-tier residency and MERCURY placement will consume the same request metadata only after separate correctness and measurement milestones.
