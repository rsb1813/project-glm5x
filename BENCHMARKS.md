# GLM5X Benchmarks

No end-to-end GLM-5.2 throughput or quality benchmark has been run yet. The bounded CUDA records below are kernel/layer evidence only and must not be reported as model tok/s.

The first benchmark record must include the commit, hardware, model/checkpoint identity, mode, context length, decode and prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality result, and enabled optimizations.

The current focused correctness smoke run is recorded in `PROJECT_STATE.md` as 31 passing tests. It is not a performance measurement.

## 2026-08-14 -- Cross-shard expert bundle index

- Commit: working tree after `4d596f5`; code change pending commit.
- Hardware: Windows host Python converter/indexer; no CUDA execution and no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, the two bounded probe artifacts only.
- Mode: `glm5x-convert assemble-experts`, sidecar/source digest validation, Python K3X directory and payload checks, copy-free role index emission.
- Result: 2 artifacts and 247 tensors indexed in approximately 11.9 seconds; 70 complete experts and 0 incomplete groups. The output JSON is 72,668 bytes and contains artifact-relative paths, tensor IDs, offsets, lengths, dtypes, quantization tags, and CRC32C values.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this measures metadata/index construction only. It does not copy payloads, load BF16 experts into CUDA, or establish end-to-end throughput.

## 2026-08-14 -- Real raw-BF16 bundle payload parity

- Commit: working tree after `a42425a`; code change pending commit.
- Hardware: Windows Python reference loader; no CUDA execution and no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, `model-00002-of-00282.safetensors` and its converted `second-shard.k3x` artifact.
- Mode: `GLM5XExpertBundle.open(...).read_expert(layer=10, expert=0)` with artifact identity and extent metadata checks.
- Result: `gate_proj`, `up_proj`, and `down_proj` each returned 25,165,824 bytes and matched the source safetensors bytes exactly. The loader rejected a tampered offset in the regression test. Focused GLM tests: 35/35 passed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this validates exact cross-shard raw-BF16 payload retrieval only. It is not a CUDA expert-kernel or model-throughput result.

## 2026-08-14 -- C++ cross-shard BF16 host loader

- Commit: working tree after `ebe2f50`; code change pending commit.
- Hardware: WSL on the target RTX 5080 host; this run used host readers only and did not launch CUDA.
- Model/checkpoint: `zai-org/GLM-5.2`, the two bounded probe artifacts, layer 10 expert 0.
- Mode: `test_glm5x_bf16_bundle` with metadata-only readers, canonical tensor-ID lookup across both shards, BF16 shape/length checks, and per-role CRC32C validation.
- Result: 3 roles, 75,497,472 payload bytes, `load_nanoseconds=465087758` (approximately 465.1 ms) and exit 0.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this is the first C++ exact host payload gate. It includes filesystem read and CRC cost, but no CUDA H2D, dequantization, MoE projection, attention, routing, or token generation.

## 2026-08-14 -- First real GLM expert CUDA bridge, FP32 resident reference

- Commit: working tree after `b2c10f4`; code change pending commit.
- Hardware: NVIDIA GeForce RTX 5080, CUDA 13.3, WSL; host CPU/RAM used for payload decode and CPU reference only.
- Model/checkpoint: `zai-org/GLM-5.2`, two bounded probe artifacts, layer 10 expert 0, real nonzero BF16 role bytes.
- Mode: `k3x_cuda_glm5x_real_expert_bench`, cross-shard C++ loader, BF16-to-FP32 host decode, resident dense SiTU FFN, synchronous transfer, 5 warm samples after 2 warmups in the smoke and 20 samples after 5 warmups in the recorded rerun.
- Shape: gate/up `2048 x 6144`, down `6144 x 2048`, one token.
- Rerun result: warm latency median 275,473 ns; cold latency 146,123,666 ns; host payload load 479,973,878 ns; cold weight H2D 150,994,944 bytes; warm weight H2D 0; resident weight bytes 150,994,944; GPU-vs-CPU maximum absolute error `8.38190317154e-09` and relative error `3.67445380789e-07`.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this is one real expert FFN execution only. It excludes router, DSA/MLA, dense trunk, residuals, other experts, and token generation.

## 2026-08-14 -- First real GLM expert CUDA bridge, BF16-rounded experiment

- Commit: same working tree after `b2c10f4`; code change pending commit.
- Hardware/model/shape: same RTX 5080 WSL and layer 10 expert 0 real shard as the FP32 record.
- Mode: resident `bf16-rounded` cublasLt dense SiTU path, 5 warm samples after 2 warmups.
- Result: warm latency median 28,154,650 ns; cold latency 202,057,408 ns; host payload load 443,554,804 ns; cold weight H2D 75,497,472 bytes; warm weight H2D 0; resident weight bytes 75,497,472; GPU-vs-CPU maximum relative error 0.00182774465 (0.1828%).
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this lower-memory path is materially slower in the current plan and remains experimental. It is not a model quality result.

## 2026-08-14 -- Cached BF16 host conversion rerun

- Commit: working tree after `4c4b444`; code change pending commit.
- Hardware/model/shape: same RTX 5080 WSL and real layer 10 expert 0 as the preceding records.
- Mode: resident `bf16-rounded` dense SiTU path with tensor-identity keyed host BF16 conversion cache, 5 warmups, and 20 measured iterations.
- Result: warm latency median 236,593 ns; cold latency 197,436,559 ns; host payload load 486,771,614 ns; cold weight H2D 75,497,472 bytes; warm weight H2D 0; resident weight bytes 75,497,472; GPU-vs-CPU maximum relative error 0.00182774465 (0.1828%).
- FP32 comparison rerun: 271,493 ns warm median, 150,994,944 resident bytes, and maximum absolute error `8.38190317154e-09`.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: caching removed the repeated 75 MiB host conversion and made the bounded BF16 path faster than FP32 at half resident weight bytes. It remains experimental pending full-layer/model quality.

## 2026-08-14 -- Eight-real-expert resident pressure, BF16

- Commit: working tree after `718cc1e`; code change pending commit.
- Hardware/model: RTX 5080, CUDA 13.3 in WSL; GLM-5.2 layer 10, the first 8 available real experts from the two probe artifacts.
- Mode: `k3x_cuda_glm5x_real_expert_bench --experts 8`, cached BF16-rounded resident dense SiTU, sequential expert calls, 5 warmups and 20 measured iterations.
- Result: host payload 603,979,776 bytes; host load 4,859,331,588 ns; cold latency 777,923,116 ns; warm median 1,854,140 ns; cold H2D 603,979,776 bytes; warm H2D 0; resident bytes 603,979,776; last-expert CPU relative error 0.00174141617.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this is sequential multi-expert pressure evidence, not expert-major batching or a full GLM layer.

## 2026-08-14 -- Eight-real-expert resident pressure, FP32 reference

- Commit: same working tree after `718cc1e`; code change pending commit.
- Hardware/model/mode: same RTX 5080 WSL and layer-10 8-expert set, FP32 resident dense SiTU, 5 warmups and 20 measured iterations.
- Result: host payload 603,979,776 bytes; host load 4,849,276,671 ns; cold latency 241,241,333 ns; warm median 13,153,048 ns; cold H2D 1,207,959,552 bytes; warm H2D 3,019,898,880 bytes; resident bytes 1,056,964,608; last-expert CPU relative error 3.8448615669e-07.
- Decode tok/s, prefill tok/s, TTFT, quality result, and cache hit rate: not measured.
- Interpretation: the configured 1 GiB resident budget forces FP32 bypass/eviction for this bank. This is the numerical reference, not the recommended multi-expert placement.

## 2026-08-14 -- Raw-BF16 expert directory C++ reader gate

- Commit: working tree after `1b22bcb`; code change pending commit.
- Hardware: Windows host conversion plus WSL C++ reader; no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, two bounded probe artifacts from `model-00001-of-00282.safetensors` and `model-00002-of-00282.safetensors` only.
- Mode: C++ `test_reader <artifact> metadata`, which validates superblock, directories, tensor metadata, layer/expert links, extents, and root/directory hashes without rereading every payload checksum.
- Result: both 5.3 GB artifacts accepted. The second artifact contains 212 tensors and 70 complete raw-BF16 expert records; the first contains 35 tensors. Full CTest result was 26/26 passed and focused GLM Python result was 31/31 passed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this proves the portable directory contract for raw-BF16 staging experts. It does not prove BF16 CUDA payload execution or end-to-end GLM throughput.

## 2026-08-14 -- Official GLM DSA indexer reference and real-shard payload gate

- Commit: `8824307`.
- Hardware: Windows CPU reference path; no CUDA execution and no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, only the already downloaded `model-00001-of-00282.safetensors` payload for the manual gate; synthetic tensors for automated parity.
- Mode: official-shaped `wq_b/wk/k_norm/weights_proj` reference, interleaved indexer RoPE, ReLU score aggregation, causal mask, and Top-K. The bounded loader materialized only five indexer tensors.
- Real payload shapes: `wq_b=(4096,2048)`, `wk=(128,6144)`, `weights_proj=(32,6144)`, `k_norm.weight=(128,)`, and `k_norm.bias=(128,)`.
- Correctness result: 3 focused official-indexer tests passed. The manual real-payload smoke produced causal Top-K `[[[0,1],[0,1]]]` for two zero activation positions with `index_topk=4`; this checks loading and execution boundaries only.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality benchmark: not measured.
- Interpretation: this is real weight shape/loading evidence plus synthetic formula parity. It is not a GLM quality or throughput result.

## 2026-08-14 -- Resumable GLM shard conversion gate

- Commit: `3400c35`.
- Hardware: Windows Python reference/converter path; no CUDA execution and no full-model load.
- Model/checkpoint: synthetic two-tensor/two-shard fixtures for the restart gate; the previously downloaded single GLM-5.2 shard remains the only real payload.
- Mode: source/config-fingerprinted `.partial` + `.resume.json` ledger, canonical aligned BF16 extents, source/partial CRC validation, complete same-shard expert-role directory records, and independent `convert-shards` orchestration.
- Correctness result: 28 focused `test_glm5x_*.py` tests passed. The resume test reused the first completed tensor extent; the multi-shard test verified two independent outputs and skipped both finalized artifacts on retry.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality benchmark: not measured.
- Interpretation: this is a crash-safety and storage-boundary milestone, not a conversion-throughput or model-throughput benchmark. No GLM tok/s claim follows.

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

## 2026-08-14 -- First official GLM-5.2 shard header parity

- Commit: `1cff340`.
- Hardware: Windows CPU metadata/header path; no CUDA and no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, `model-00001-of-00282.safetensors`, 5,342,821,416 bytes; this is the only downloaded shard.
- Mode: `safetensors.safe_open(...).get_slice()` header inspection plus manifest name parity; no tensor payload was materialized.
- Result: all 35 index-listed tensor names matched the shard header. Representative BF16 shapes were `embed/lm_head=(154880,6144)`, `indexer.wk=(128,6144)`, `indexer.wq_b=(4096,2048)`, `indexer.weights_proj=(32,6144)`, and `indexer.k_norm=(128,)` for layers 0 and 1.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality score: not measured.
- Caveat: this validates the first shard's header and names only. It does not run GLM projections, load the 1.5 TB checkpoint, or establish model quality/throughput.

## 2026-08-14 -- First bounded GLM5X shard artifact round-trip

- Commit: `b4e9b19`.
- Hardware: Windows Python converter plus WSL C++ reader; no full-model load.
- Model/checkpoint: `zai-org/GLM-5.2`, only `model-00001-of-00282.safetensors`; source SHA-256 `004bf9404964da8ea71ea2d3ebf02148fa766b956bd4fca3f54b093e58a6a74c`.
- Mode: `glm5x-convert convert-shard`, aligned raw BF16 extents, CRC32C per tensor, directory SHA-256/root SHA-256, and JSON sidecar name map; chunk size 8,388,608 bytes.
- Result: 35 tensors converted to `build-glm5x-hf-probe/first-shard.k3x` (5,342,863,616 bytes), 78 layer records, 0 expert records, and `maximum_source_read_bytes=8,388,608`. Python `K3XReader.open` passed all checksums; WSL C++ `test_reader` returned exit 0.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality score: not measured.
- Caveat: this is an experimental single-shard storage artifact. It does not yet provide resumable multi-shard conversion, expert bundle directories for raw BF16, quantization, DSA/MLA execution, or full-model throughput.

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
