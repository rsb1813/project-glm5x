# GLM5X Benchmarks

No end-to-end GLM-5.2 throughput or quality benchmark has been run yet. The bounded CUDA records below are kernel/layer evidence only and must not be reported as model tok/s.

The first benchmark record must include the commit, hardware, model/checkpoint identity, mode, context length, decode and prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality result, and enabled optimizations.

The current focused correctness smoke run is recorded in `PROJECT_STATE.md` as 13 passing tests. It is not a performance measurement.

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
