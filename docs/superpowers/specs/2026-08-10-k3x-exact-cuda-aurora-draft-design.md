# K3X Milestone 18 Exact CUDA AURORA Draft Design

## Status and objective

Milestone 18 measures one isolated AURORA axis: execute the existing persistent reduced-Top-K draft graph on the RTX 5080 through K3X's exact `cuda-custom` backend while leaving the target backend, target routing, target verification, scheduler, draft K, precision, and residency unchanged. CPU persistent drafting remains the default and `aurora-replay` remains the CPU candidate oracle.

The milestone is accepted only if a CPU target with a CUDA draft preserves the CPU-persistent proposal sequence, acceptance decisions, natural target tokens, final target state, and committed target routes on the deterministic Top-16 fixture. A speedup is not required. The result must report separate draft GPU traffic and memory counters so an unfavorable result is still useful evidence.

## Evidence and constraints

B-0018 removed complete-prefix replay and reduced logical draft Reader bytes by 45.96% to 63.08%, but persistent rows remain 38.08% to 52.26% slower than natural greedy. The recorded bottleneck is the additional per-token reduced-K draft graph. The runtime already has a tested `cuda-custom` backend, reusable allocations, grouped projections, FFN-block execution, native MXFP4 kernels, target-side CUDA telemetry, and `sm_120` builds. Milestone 18 reuses those contracts instead of adding a new kernel family.

The draft Reader, backend, profiler, and RuntimeSession remain separate from the target equivalents. This prevents draft transfers, allocations, residency, and kernel time from contaminating target counters. No full checkpoint, paid cloud resource, new checkpoint format, or quality-changing approximation is introduced.

## Alternatives

### Accepted: a separate fixed-identity CUDA draft backend

Add `--aurora-draft-backend cpu|cuda-custom`, defaulting to `cpu`. `cuda-custom` creates a second backend with the following fixed identity.

- Dense precision: FP32.
- Allocation: reused.
- Weights: transient.
- Batching: grouped.
- Boundary: FFN block.
- Transfer: synchronous.
- MoE fusion: none.
- Resident and pinned capacities: zero.

Only `aurora-persistent` accepts `cuda-custom`. Replay stays CPU so it remains a stable oracle. Fixing the CUDA identity makes backend placement the only experimental axis and avoids a combinatorial CLI surface before there is evidence to justify it.

### Rejected: reuse the target backend or target CUDA options

Sharing one backend would mix profiler, memory, weight-cache, and allocation state between draft and target. Copying target options would require moving the target to CUDA to move the draft, so B-0019 could not attribute a change to drafting alone.

### Deferred: resident CUDA drafting

Resident weights may reduce repeated H2D traffic, but enabling them together with the first GPU draft changes placement and residency simultaneously. Admission capacity, cache benefit, and VRAM pressure belong in a later matched experiment after transient GPU drafting establishes a baseline.

### Deferred: BF16 or other reduced-precision drafting

Reduced precision can change draft proposals and acceptance. It needs explicit quality and proposal-divergence evidence and must not be mixed into the exact execution-placement experiment.

## Runtime and ownership contract

The CLI continues to open a separate draft Reader. It constructs either the existing CPU backend or one `cuda-custom` backend with the fixed options above, then passes that backend to `AuroraPersistentDraftProvider`. Backend, profiler, Reader, and provider lifetimes extend through generation and JSON serialization.

`AuroraPersistentDraftProvider::create` accepts CPU and the exact fixed CUDA identity. It continues to require incremental fixed K4/6/8/12, disabled L1, blocking L2, no profile observation, and a nonempty prompt. Any other CUDA combination fails before cursor creation or Reader access. `AuroraReplayDraftProvider::create` remains CPU-only.

The opaque `IncrementalDraftCursor` does not gain CUDA-specific state. It already delegates every graph operation to `ComputeBackend`, owns only host KDA/MLA state and logits, and preserves its transaction, rollback, generation-guard, and failure-latch semantics.

## CLI and compatibility

Add `--aurora-draft-backend cpu|cuda-custom` with default `cpu`.

- `none` and `scripted-reference` reject an explicitly supplied draft-backend option.
- `aurora-replay` accepts only `cpu`.
- `aurora-persistent` accepts `cpu` or `cuda-custom`.
- Existing commands that omit the option produce the same CPU draft behavior and schema values as Milestone 17.
- The target `--backend` and all target CUDA flags remain independent.

The output records `aurora_draft_backend` and the effective fixed draft CUDA identity. Non-AURORA rows use `none`; CPU AURORA rows use `cpu` and zero CUDA counters.

## Separate draft CUDA telemetry

Serialize the draft profiler, memory, and runtime snapshots independently from target fields.

- `draft_device`.
- `draft_cuda_allocation`.
- `draft_cuda_weights`.
- `draft_cuda_batching`.
- `draft_cuda_boundary`.
- `draft_cuda_transfer`.
- `draft_cuda_moe_fusion`.
- `draft_kernel_nanoseconds`.
- `draft_host_to_device_bytes`.
- `draft_weight_h2d_bytes`.
- `draft_activation_h2d_bytes`.
- `draft_device_to_host_bytes`.
- `draft_peak_vram_bytes`.
- `draft_device_allocation_count`.
- `draft_stream_synchronization_count`.
- `draft_weight_cache_hits`.
- `draft_weight_cache_misses`.
- `draft_weight_cache_bypasses`.

These fields extend `BenchmarkRecord`, JSON, and CSV. They are deterministic where appropriate and default to zero or `none` outside the CUDA draft path. Existing target fields retain their meanings. Logical draft Reader bytes remain separate from physical NVMe traffic and are not relabeled.

## Failure behavior

Unsupported mode or backend strings fail during CLI preflight before opening a draft Reader. CUDA backend creation failure returns the backend error and does not fall back to CPU. Provider validation rejects a noncanonical CUDA identity. Cursor backend exceptions retain the existing failure latch and stop generation; target state remains independent and is never adopted from the draft.

## Correctness tests

- Existing commands without `--aurora-draft-backend` remain CPU and keep prior outputs.
- Non-AURORA explicit use, replay plus CUDA, and unknown draft backends fail with exact preflight errors.
- CPU builds reject requested CUDA drafting without silently falling back.
- A CUDA build runs persistent fixed and adaptive drafting with a CPU target and preserves CPU-persistent proposals, acceptance, target tokens, final state, and committed routes.
- CUDA draft rows have nonzero kernel, H2D, allocation, synchronization, and peak-VRAM evidence while CPU draft rows keep those counters zero.
- Target GPU counters remain zero when the target backend is CPU, proving separation from draft counters.
- Existing greedy, scripted, replay, CPU persistent, token-major, expert-major, and benchmark-schema tests remain passing.

## B-0019 measurement gate

B-0019 uses the deterministic Top-16 artifact, four prompt tokens, six generated tokens, three warmups, and twenty samples. The target remains CPU natural Top-16. It measures natural greedy plus matched CPU-draft and transient-CUDA-draft persistent rows for fixed block-2 and adaptive scheduling under token-major and CPU expert-major target verification.

Each matched CPU/CUDA draft pair must preserve proposed and accepted token counts, acceptance rate, target tokens, final state, and committed routes. The runner records decode, prefill, TTFT, RSS, target and draft Reader bytes, target and draft H2D, draft VRAM, draft kernel time, and all existing speculation metrics. Raw JSON/CSV and summary digests are independently recomputed.

The result determines only whether exact transient GPU execution is a useful draft baseline. It does not select resident drafting, reduced precision, a default AURORA mode, or a full-model performance claim. Native-Linux physical I/O, representative acceptance, full-model VRAM pressure, coding quality, and GPU utilization or memory-bandwidth sampling remain future gates.

## Follow-up boundary

If transient CUDA drafting is correct but transfer-bound, the next isolated experiment may add a bounded draft resident-weight capacity and compare H2D saved against VRAM consumed. If it is kernel- or synchronization-bound, the next step should measure a larger fused draft graph boundary before adding residency. Reduced precision remains a separate quality experiment in either case.
