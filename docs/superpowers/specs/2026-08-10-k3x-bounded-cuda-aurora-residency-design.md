# K3X Milestone 19 Bounded CUDA AURORA Residency Design

## Status and objective

Milestone 19 measures one isolated change after B-0019: retain exact immutable draft weights in a bounded RTX 5080 resident table across persistent AURORA forwards. The target backend, target natural routing, target verifier, proposal scheduler, draft Top-K, FP32 arithmetic, grouped FFN-block execution, and synchronous transfer identity remain unchanged. CPU drafting stays the default and replay stays CPU-only.

The milestone is accepted only if resident and transient CUDA draft pairs preserve proposal counts, acceptance, strict target tokens, final target KDA/MLA state, and committed target routes. Resident rows must report an exact hard capacity, positive admissions and hits, zero bypass at the canonical full-fit capacity, and less draft weight H2D than their transient matches. Decode improvement is measured rather than required.

## Evidence and constraints

B-0019 shows that exact transient CUDA drafting is 96.22% to 97.00% slower than CPU drafting on the Top-16 synthetic fixture. Fixed rows transfer 5,756,160 weight bytes and adaptive rows transfer 6,331,776 weight bytes while performing 410 to 451 synchronizations. Peak draft VRAM is only 44,448 bytes, so repeated immutable-weight upload rather than allocation capacity is the next isolated axis.

K3X already has a tested tensor-ID-keyed `ResidentWeightTable`. It preserves FP32, BF16, and native MXFP4 bytes, enforces a hard byte capacity, admits complete immutable tensors, counts misses and hits, and falls back to exact transient staging when an entry does not fit. Milestone 19 reuses this primitive without adding eviction, prediction, precision changes, a new kernel, or a second representation.

No full checkpoint, paid cloud resource, new storage format, or quality-changing approximation is introduced. Logical Reader bytes and CUDA H2D remain distinct from physical NVMe and physical PCIe measurements.

## Alternatives

### Accepted: reuse the bounded static CUDA resident table

Add `--aurora-draft-resident-bytes N`, defaulting to zero. For persistent `cuda-custom` drafting, zero preserves the B-0019 transient identity and a positive value selects resident weights with the requested hard capacity. All other CUDA draft options remain fixed to FP32, reused allocation, grouped execution, `ffn-block`, synchronous transfer, fusion `none`, and zero pinned capacity.

This changes only immutable-weight residency. It also reuses existing allocation, validation, exact-bypass, and telemetry contracts instead of creating a draft-specific cache implementation.

### Deferred: dynamic L0 eviction or promotion

A dynamic L0 expert bank is necessary when the useful set exceeds VRAM, but its eviction score, admission policy, prediction inputs, and interaction with L1 profiles are independent design decisions. Adding them now would prevent B-0020 from attributing H2D and timing changes to residency alone.

### Deferred: persistent multi-token or multi-expert CUDA kernels

A larger persistent execution boundary could reduce the synchronization count that residency does not address. It is a separate kernel/scheduling axis and follows only after B-0020 quantifies the residual synchronization and kernel cost with weight H2D removed.

### Deferred: reduced precision

BF16 or mixed precision can change proposals and acceptance. It remains a separate quality-measured experiment after the exact FP32 residency boundary is established.

## Runtime and ownership contract

The CLI continues to create an independent draft Reader, profiler, backend, and provider. With `--aurora-draft-backend cuda-custom`, it constructs the fixed CUDA identity and chooses the weight mode from the draft resident capacity.

- Capacity zero creates `CudaWeightMode::transient` with zero resident bytes.
- Positive capacity creates `CudaWeightMode::resident` with that exact byte limit.
- CPU draft and replay do not create a CUDA resident table.

`AuroraPersistentDraftProvider::create` accepts CPU or either canonical CUDA weight identity above. It continues to require incremental fixed K4/6/8/12, disabled L1, blocking L2, no profile observation, and a nonempty prompt. `AuroraReplayDraftProvider::create` remains CPU-only.

The resident table belongs to the draft backend, so entries persist through initial context prefill, candidate forwards, rollback recovery, and target-bonus teacher forcing for the life of one generation. It is not shared with the target backend or another session.

## CLI and compatibility

Add `--aurora-draft-resident-bytes N` with default zero.

- `none` and `scripted-reference` reject an explicitly supplied draft residency option.
- `aurora-replay` rejects it because replay remains CPU-only.
- `aurora-persistent` with CPU rejects a positive or explicitly supplied residency option.
- `aurora-persistent` with `cuda-custom` accepts zero or a positive unsigned 64-bit capacity.
- Existing commands that omit the option retain the exact B-0019 transient behavior.
- Target `--cuda-resident-bytes` and target weight mode remain independent.

The option never selects experts, changes router scores, or substitutes an approximation. A capacity miss uses the existing exact transient path and increments a bypass counter.

## Separate residency telemetry

Extend draft JSON/CSV telemetry with the fields needed to audit configured capacity and actual occupancy.

- `draft_cuda_resident_bytes` for the configured hard capacity.
- `draft_resident_weight_bytes` for current admitted immutable bytes.
- `draft_peak_resident_weight_bytes` for peak admitted immutable bytes.

Existing draft fields retain their meanings, including weight H2D, total H2D, peak VRAM, allocation count, synchronization count, cache hits, cache misses, and cache bypasses. CPU and non-AURORA rows emit zero for all three new fields. Target residency counters remain separate.

## Admission and failure behavior

The existing resident table validates tensor identity, representation, shape, group size, payload lengths, and capacity before admission. An entry is published only after allocations and H2D submissions succeed. A miss that cannot fit returns an exact bypass rather than evicting an existing entry or changing routing.

Invalid numeric input, incompatible modes, replay residency, and CPU residency fail during CLI preflight before opening the draft Reader. CUDA backend creation or resident allocation/upload failure propagates its typed backend error and never falls back to CPU. Provider validation rejects noncanonical CUDA combinations before cursor creation.

## Correctness and TDD gates

- A direct provider test first fails because resident CUDA options are rejected, then proves resident proposals and full/partial commit behavior equal the CPU and transient CUDA providers.
- The direct test verifies positive misses and hits, positive bounded resident bytes, and no bypass with an 8 MiB full-fit capacity.
- A tiny-capacity provider test proves exact proposal parity with positive bypass and occupancy never above capacity.
- CLI tests prove the new option is rejected outside persistent CUDA drafting and that a CPU build never silently falls back.
- Fixed and adaptive token-major and CPU expert-major runs preserve transient/resident proposal, acceptance, target token, final-state, and committed-route parity.
- Benchmark schema tests prove zero defaults and preserve the new capacity and occupancy fields through JSON and CSV.
- Existing greedy, scripted, replay, CPU persistent, transient CUDA persistent, target CUDA, and benchmark evidence tests remain passing.

## B-0020 measurement gate

B-0020 uses the deterministic Top-16 artifact, four prompt tokens, six generated tokens, three warmups, and twenty samples. The target remains CPU natural Top-16. The canonical resident capacity is 8,388,608 bytes, chosen as a bounded full-fit capacity for the synthetic graph and verified by zero bypass rather than assumed from configuration.

The nine rows are natural greedy plus matched transient/resident CUDA draft pairs for fixed block-2 and adaptive scheduling under token-major and CPU expert-major target verification. Every pair must preserve proposal and accepted-token counts, acceptance rate, target tokens, final state, and committed routes. Resident rows must have positive misses and hits, zero bypass, occupancy no greater than capacity, and lower draft weight H2D than transient rows.

The runner records paired decode delta, weight-H2D reduction, resident hit ratio, capacity, occupancy, peak draft VRAM, allocations, synchronizations, target and draft Reader bytes, and all existing speculation metrics. Raw JSON/CSV and summary digests are recomputed from committed bytes.

B-0020 determines whether static full-fit exact residency removes enough transfer cost to justify the next GPU draft boundary. It does not select a production eviction policy, change a default, establish full-model VRAM feasibility, or measure physical NVMe/PCIe traffic, coding quality, GPU utilization, or memory bandwidth.

## Follow-up boundary

If residency materially reduces H2D but decode remains far below CPU, the next isolated experiment is a persistent multi-token or multi-expert CUDA execution boundary that reduces synchronizations and launch overhead. If the full-fit capacity itself dominates representative VRAM, the next step is a measured dynamic L0 admission/eviction policy. Reduced precision remains separate in either case.
