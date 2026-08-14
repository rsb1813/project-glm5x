# GLM5X Benchmarks

No end-to-end GLM-5.2 throughput or quality benchmark has been run yet. The bounded CUDA records below are kernel/layer evidence only and must not be reported as model tok/s.

The first benchmark record must include the commit, hardware, model/checkpoint identity, mode, context length, decode and prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality result, and enabled optimizations.

The current focused correctness smoke run is recorded in `PROJECT_STATE.md` as 22 passing tests. It is not a performance measurement.

## 2026-08-14 — TurboQuant reference smoke

- Commit: `f50c37e`.
- Hardware: Windows host CPU reference path; no CUDA.
- Model/checkpoint: synthetic GLM5X tensors, no GLM-5.2 weights.
- Mode: TurboQuant reference, Hadamard rotation, 4-bit symmetric cache; separate K6/V4 capacity estimate.
- Context length: 6 tokens for incremental attention smoke; 1,000,000 tokens for formula-only storage estimate.
- Decode tok/s: not measured.
- Prefill tok/s: not measured.
- TTFT: not measured.
- VRAM: not applicable.
- System RAM: not recorded.
- NVMe GB/token: not applicable.
- H2D GB/token: not applicable.
- Cache hit rate: not applicable.
- Average Top-K: not applicable.
- Speculative acceptance: not applicable.
- Quality result: six tests passed, including lossless round-trip, compressed shape/size, fractional schedule, incremental attention parity, invalid configuration, and 1M-token capacity arithmetic.
- Enabled optimizations: CPU reference quantization only.
- Caveat: this record validates a contract and arithmetic, not full-model quality or throughput.

## 2026-08-14 -- GLM-5.2 DSA/indexer reference state smoke

- Commit: `02b9916`.
- Hardware: Windows CPU reference path; no CUDA and no checkpoint.
- Model/checkpoint: descriptor-only synthetic GLM-5.2 metadata with index width `2 x 2 = 4`, `index_topk=3`, and `index_topk_freq=2`.
- Mode: `GLM5XDSAState` with lossless 16-bit KV for selection parity; reference mode refreshes top-k for every query, while fast mode reuses the selection until two new tokens arrive.
- Correctness result: 4 DSA tests passed, covering descriptor wiring, explicit query/key projection, exact top-k/attention selection, refresh cadence, and capacity arithmetic.
- Formula-only capacity: with BF16 index keys, K6/V4 compressed KV, index width 4096, and key/value widths 256, the state estimate is 201,637,504 bytes at 600,000 tokens and 336,062,512 bytes at 1,000,000 tokens.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality score: not measured.
- Caveat: the projection API is exercised with synthetic matrices; official GLM tensor shapes/values, DSA CUDA kernels, MLA latent projections, and full-model quality remain unimplemented.

## 2026-08-14 -- Official GLM-5.2 manifest role probe

- Commit: `9def853`.
- Hardware: Windows metadata-only read; no shard opened and no CUDA.
- Model/checkpoint: `zai-org/GLM-5.2` config and `model.safetensors.index.json` from the local HF metadata cache; no weight payload downloaded.
- Result: 59,585 tensor entries across 282 safetensors shards; descriptor `indexer_types` has 78 entries. The role resolver maps shared layers to the nearest previous full indexer source, for example layer 3 -> layer 2, layer 7 -> layer 6, and layer 77 -> layer 74.
- Verified roles: `model.layers.{source}.self_attn.indexer.wk.weight` resolves to the expected shard for each probe layer.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality score: not measured.
- Caveat: the index contains names and shard placement only. Tensor shapes, dtypes, learned projection values, and full-model parity still require opening bounded real shards.

## 2026-08-14 -- GLM-5.2-shaped resident expert grid

- Commit: `31678d1`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL; host CPU/RAM not used for a full-model claim.
- Model/checkpoint: no checkpoint; deterministic zero-weight synthetic GLM-5.2 expert tensors.
- Mode: resident MXFP4 E2M1/E8M0 expert grid, admission validation, synchronous activation transfer, no proxy, no pruning, no CUDA graph.
- Shape: hidden size 6144, expert intermediate size 2048, group size 32, 256 routed experts represented by a selected expert set.
- Cold selected-weight payload: 160,432,128 bytes for 8 experts; resident weight bytes 160,432,128; peak VRAM 160,850,304 bytes.
- Warm 8-expert, 1-token block: median wall latency 2,662,772 ns over 100 iterations after 20 warmups; kernel time 1,092,040 ns per 100 calls; maximum absolute error 0.
- Warm 8-expert, 4-token expert-major block: median block latency 5,379,264 ns; 1,344,816 ns/token within the block; maximum absolute error 0; peak VRAM 162,103,680 bytes.
- Warm 8-expert, 8-token expert-major block: median block latency 8,791,638 ns; 1,098,955 ns/token within the block; maximum absolute error 0; peak VRAM 163,774,848 bytes.
- Warm 16-expert, 1-token block: median wall latency 5,458,462 ns; cold selected-weight payload 320,864,256 bytes; maximum absolute error 0.
- Decode tok/s: not measured; the records are one MoE expert block, not 78-layer generation.
- Prefill tok/s: not measured.
- TTFT: not measured.
- System RAM: not recorded for this CUDA-only fixture.
- NVMe GB/token: not applicable; weights were host-resident synthetic buffers.
- H2D GB/token: 24,576 bytes/token activation traffic in the 1-token case; cold weight upload is a one-time 160,432,128 bytes for 8 experts.
- Expert-cache hit rate: residency is verified by zero warm weight H2D bytes; a normalized cache-hit percentage was not emitted.
- Average Top-K: 8 selected experts in the expert-major case; natural router scores were not run.
- Speculative acceptance: not measured; 4/8 tokens are candidate batching only.
- Quality result: synthetic zero-weight reference parity, maximum absolute error 0; no GLM quality score.
- Enabled optimizations: resident expert grid, expert-major candidate-token batching, admission validation.
- Rejected micro-optimizations in this milestone: shared-input row tiling and E2M1 lookup/bit-scale decode both preserved output parity but regressed the 8-expert 1-token median, so neither is in the accepted path.
- Caveat: these measurements bound the current expert FFN cost. Attention, router, dense trunk, DSA indexer, KV state, storage traffic, and end-to-end scheduling are not included.

## 2026-08-14 -- Resident expert-major batch path

- Commit: `d204fb1`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: no checkpoint; the same deterministic GLM-5.2-shaped zero-weight tensors.
- Mode: `expert-batch`, resident MXFP4 E2M1/E8M0, synchronous transfer, 8 independent expert groups, 4 candidate tokens per group.
- Warm median block latency: 6,566,362 ns over 100 iterations after 20 warmups, or 1,641,591 ns per candidate token inside the block.
- Maximum absolute error: 0.
- Cold weight H2D: 160,432,128 bytes for the 8 selected experts; warm sample weight H2D: 0 bytes; resident weight bytes: 160,432,128; peak VRAM: 167,411,712 bytes.
- Batch telemetry: 800 batched expert calls and 3,200 batched expert tokens over the 100-iteration sample; activation H2D 78,643,200 bytes and device-to-host 78,643,200 bytes.
- Decode tok/s, prefill tok/s, TTFT, NVMe GB/token, quality score, and speculative acceptance: not measured; this is a per-expert-group layer record.
- Decision: resident exact batching is enabled for the expert-major path, but it is not claimed faster than the all-expert resident grid. Its immediate value is eliminating repeated weight movement; variable-union grouping and tensor-core execution remain open bottlenecks.

## 2026-08-14 -- Resident BF16 dequantized expert grid experiment

- Commit: `5d1c636`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: no checkpoint; deterministic zero-weight synthetic GLM-5.2-shaped expert tensors.
- Mode: resident expert grid with one-time host MXFP4 E2M1/E8M0 to BF16 dequantization, cublasLt BF16-input/FP32-output projections, synchronous transfer, no proxy, no pruning, no CUDA graph.
- Shape: hidden size 6144, expert intermediate size 2048, group size 32, 8 selected experts, 4 candidate tokens.
- Native reference: median wall latency 5,394,131 ns/block; kernel time 354,917,120 ns over 100 measured calls; resident weight bytes 160,432,128; peak VRAM 162,103,680 bytes; maximum absolute error 0.
- BF16 grid: median wall latency 2,582,527 ns/block; kernel time 67,868,032 ns over 100 measured calls; cold BF16 weight H2D 603,979,776 bytes; resident weight bytes 603,979,776; warm weight H2D 0; activation H2D 4,915,200 bytes; device-to-host 78,643,200 bytes; peak VRAM 630,636,544 bytes; maximum absolute error 0 for the zero-weight fixture.
- Relative result: BF16 resident grid was 2.09x faster in this bounded layer record, at 3.77x resident-weight memory. This is not an end-to-end tok/s measurement and does not establish GLM quality parity.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, average Top-K, speculative acceptance, and quality score: not measured.
- Reference switch: `CudaMxfp4Execution::native` remains the default; `dequantized_bf16` is opt-in and requires sufficient VRAM residency.
- Caveat: the benchmark still uses zero weights, so the reported absolute error is a kernel contract check rather than a model-quality result. Full GLM-5.2 weights and DSA/routing are absent.

## 2026-08-14 -- Nonzero synthetic native-reference comparison

- Commit: `bc5bb4c`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: no checkpoint; deterministic nonzero synthetic GLM-5.2-shaped MXFP4 tensors with the same packed E2M1/E8M0 representation used by both paths.
- Mode: resident BF16 dequantized expert grid compared against a separate native resident MXFP4 GPU reference; 8 experts, 4 tokens, 10 warmups, 30 measured iterations.
- BF16 median warm block latency: 2,563,496 ns; cold BF16 weight H2D 603,979,776 bytes; resident BF16 weight bytes 603,979,776; peak VRAM 630,636,544 bytes.
- Native-reference maximum absolute difference: 1,732.3086; maximum relative difference to the native reference maximum magnitude: 0.00950465 (0.95%).
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, average Top-K, speculative acceptance, and task quality: not measured.
- Interpretation: this is a numerical dequantization/accumulation check, not a GLM quality benchmark. The deterministic pattern is not a calibrated GLM shard.

## 2026-08-14 -- Released-dimension capacity fallback check

- Commit: `29b6fde`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: `build-glm5x-cuda-fixtures/released/released.k3x`, a bounded released-dimension synthetic fixture; not GLM-5.2 weights.
- Mode: `k3x_cuda_moe_layer_bench --boundary ffn-block --execution dequantized-bf16`, 16 selected experts, 1 GiB resident budget.
- Result: BF16 preflight detected that dense trunk residency plus all selected BF16 experts exceeded the budget and returned to the exact native MXFP4 group path. Warm median was 14,319,240 ns, `resident_grid_fallbacks=10`, maximum absolute error 0 versus the native oracle.
- Native comparison from the same fixture: 6,333,866 ns warm median at the same 16-expert shape and budget.
- Interpretation: this is a guardrail result, not a BF16 speed claim. A BF16 mode must use a VRAM budget that includes dense trunk and expert residency; otherwise native fallback is intentionally selected.

## 2026-08-14 -- Latest RTX 5080 BF16/native rerun

- Commit: `ad4b1c6` (benchmark binary built from the BF16-capacity-guarded code at `29b6fde`).
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: no checkpoint; deterministic GLM-5.2-shaped synthetic tensors, 8 selected experts, hidden size 6144, expert intermediate size 2048, 4 candidate tokens.
- Native zero-pattern grid: median 5,510,632 ns/block over 100 iterations after 20 warmups; cold weight H2D 160,432,128 bytes; resident weight bytes 160,432,128; activation H2D 9,830,400 bytes; device-to-host 78,643,200 bytes; peak VRAM 162,103,680 bytes; maximum absolute error 0.
- BF16 zero-pattern grid: median 4,386,083 ns/block over 100 iterations after 20 warmups; cold BF16 weight H2D 603,979,776 bytes; resident BF16 weight bytes 603,979,776; activation H2D 4,915,200 bytes; device-to-host 78,643,200 bytes; peak VRAM 630,636,544 bytes; maximum absolute error 0.
- Nonzero BF16 grid: median 4,044,675 ns/block over 30 iterations after 10 warmups; maximum absolute difference versus the native GPU reference 1,732.3086; maximum relative difference 0.0095046479255 (0.9505%).
- Relative result: this rerun was 1.26x faster for warm BF16 grid latency, while resident selected-weight memory was 3.76x larger. The cold BF16 admission was roughly 3.77x the native payload and is not on the per-token hot path after residency.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, average Top-K, speculative acceptance, and task quality: not measured.
- Interpretation: this is a bounded kernel/layer rerun only. The run-to-run latency differs from the earlier sample, so the repository records both samples instead of replacing history; no end-to-end GLM throughput or quality claim follows.
