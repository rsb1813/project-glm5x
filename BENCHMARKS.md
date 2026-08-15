# GLM5X Benchmarks

## 2026-08-15 -- Experimental MXFP4 sidecar quality/bytes gate

- Commit: working tree based on public `ee186c7` with the FP4 extension uncommitted at measurement time.
- Hardware: RTX 5080 16 GB, WSL2 CUDA 13.0/PyTorch 2.13.0; model/checkpoint: official GLM-5.2 layer-10 real activation and full expert bundle; context: one candidate token, natural Top-8.
- Precision path: reference MXFP4 E2M1 values with E8M0 group scales, `.pm4` fingerprinted sidecars, BF16 decode for the current reference MLP. Eight routed experts were selected and route IDs matched BF16.
- Storage: `160,440,156` sidecar bytes for the eight selected experts versus `603,979,776` corresponding raw BF16 role bytes (`26.5638291835785%`).
- Timing: first population including CPU pack/decode and atomic writes `27.440241339994827 s`; fresh sidecar reuse including CPU MXFP4 decode `17.867729659978068 s`; paired BF16 reference `2.79652249100036 s`. The current reference decode is therefore slower and is not a throughput optimization.
- Quality: relative L2 error `0.16359105706214905`, maximum absolute error `0.001750946044921875` against BF16; route equality `true`.
- Decision: keep `.pm4` and `expert_precision=mxfp4` experimental/default-off. Do not promote uncalibrated MXFP4, and do not interpret this bounded result as full-model tok/s. Native NVFP4/MXFP4 CUDA execution and calibrated residual metadata are the next FP4 gates. FP8 work is comparison-only and its abandoned full-model gate produced no result.

## 2026-08-15 -- Reduced-routing and shared-proxy quality gate

- Hardware: RTX 5080 16 GB, WSL2 CUDA 13.0/PyTorch 2.13.0; model/checkpoint: official GLM-5.2 layer-10 real activation and five-shard expert bundle; context: four candidate tokens.
- Exact reference: natural Top-8, BF16 expert weights, 31 unique routed experts, `12.43729756900575 s` for the measured four-token layer-10 block.
- Experimental proxy: natural route metadata retained, exact Top-4 routed experts plus shared-expert dropped-mass approximation, 16 unique experts, `5.043440291978186 s`; relative L2 error `0.8120684623718262`; maximum absolute error `0.01171150803565979` against the natural Top-8 output.
- Decode/prefill tok/s, TTFT, physical NVMe GB/token, H2D GB/token, and full-model quality are not applicable to this bounded layer experiment. No speculative acceptance or adaptive policy was enabled.
- Decision: keep the proxy and reduced-routing controls experimental and default-off. The result is not a full-model throughput claim.

## 2026-08-15 -- Fingerprinted FP8 sidecar reuse

- Hardware: RTX 5080 16 GB, WSL2 CUDA 13.0/PyTorch 2.13.0; model/checkpoint: official GLM-5.2 layer-10 real activation and five-shard expert bundle; context: four candidate tokens.
- Exact BF16 reference: natural Top-8, 31 unique routed experts, `11.759381022013258 s` for the measured block.
- First FP8 sidecar population: row-scaled E4M3 FP8 roles, 31 `.pf8` entries totaling `1,171,511,902` bytes (`50.05%` of the corresponding `2,340,421,632` raw-BF16 role bytes), `21.40642180899158 s`. This includes source reads, quantization, and atomic sidecar writes.
- Fresh-process FP8 sidecar reuse: `4.820426017016871 s`, identical route IDs, relative L2 drift `0.05696592479944229`, maximum absolute error `0.0007408261299133301` against BF16, and 31 unique experts.
- Decode/prefill tok/s, TTFT, physical NVMe GB/token, H2D GB/token, and full-model quality are not applicable to this bounded layer experiment. The sidecar is opt-in and no adaptive Top-K or proxy was enabled.
- Decision: retain the FP8 sidecar as an experimental warm-path candidate. It is not a 10--20 tok/s or quality-mode result.

No end-to-end GLM-5.2 throughput or quality benchmark has been run yet. The bounded CUDA records below are kernel/layer evidence only and must not be reported as model tok/s.

The first benchmark record must include the commit, hardware, model/checkpoint identity, mode, context length, decode and prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality result, and enabled optimizations.

The current focused correctness smoke run is recorded in `PROJECT_STATE.md` as 26 passing WSL CTest cases plus 35 focused GLM Python tests. It is not a performance measurement.

## 2026-08-15 -- Three-worker local conversion snapshot

- Date: 2026-08-15 05:03 KST.
- Commit: `5c5a2eb`.
- Hardware: RTX 5080 PC, WSL2 Ubuntu-24.04, C: NVMe workspace, 1 GbE network.
- Model/checkpoint: official `zai-org/GLM-5.2`; 282 source shards, no model weights committed to Git.
- Mode: three independent resumable stream workers, half-open ranges `10..100`, `101..191`, and `192..281`, `--no-assemble`.
- Measurement: 13 newly finalized artifacts from 04:39:53 to 05:02:30, 22m37s wall time, approximately `34.5 shards/hour` aggregate. 23/282 artifacts and source-deleted markers exist including the ten completed before launch.
- Projection: 259 artifacts remain; at the observed sample rate this is approximately 7.5 hours. This is a conditional conversion estimate, not a completion promise.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality result: not measured.
- Interpretation: this measures storage conversion overlap only. Full bundle indexing, all-layer logits, and end-to-end RTX 5080 execution remain open.

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

## 2026-08-14 -- Real GLM BF16 expert-major grid

- Commit: `399556d`.
- Hardware: NVIDIA GeForce RTX 5080, CUDA 13.3, WSL; host CPU/RAM used for exact shard loading and CPU reference only.
- Model/checkpoint: `zai-org/GLM-5.2`, two bounded probe artifacts, layer 10, first 8 available real experts (IDs 0, 1, 2, 3, 4, 5, 6, 15).
- Mode: `k3x_cuda_glm5x_real_expert_bench --experts 8 --tokens 4 --precision bf16-rounded`, resident dense BF16 expert-major grid, cublasLt gate/up/down projections, synchronous transfer, 5 warmups and 20 measured iterations. No proxy, pruning, router, or speculative acceptance was used.
- Context length: 4 candidate tokens in the FFN block; this is not a prefill or decode context benchmark.
- Result: warm block latency median 1,758,739 ns (approximately 439,685 ns per candidate token); cold latency 759,804,032 ns; host payload load 4,803,323,065 ns; cold weight H2D 603,979,776 bytes; warm weight H2D 0; resident weight bytes 603,979,776; last-expert CPU maximum relative error 0.00135118968 (0.135%).
- Decode tok/s, prefill tok/s, TTFT, GPU utilization, VRAM peak, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality benchmark: not measured.
- Interpretation: this is the first nonzero real-shard expert-major candidate-block gate. It proves payload-to-grid parity for one FFN block and removes repeated weight movement, but it is not a full GLM layer, routing result, or end-to-end tok/s claim. The command's `--tokens` argument is now bounded to 1..65535 and uses the exact scalar FP32 path when BF16 grid mode is not selected.

## 2026-08-14 -- Direct raw-BF16 real expert-major grid

- Commit: `29e4c61`.
- Hardware: NVIDIA GeForce RTX 5080, CUDA 13.3, WSL; same two bounded GLM probe artifacts.
- Model/checkpoint: `zai-org/GLM-5.2`, layer 10, first 8 available real experts, gate/up `2048 x 6144`, down `6144 x 2048`.
- Mode: `k3x_cuda_glm5x_real_expert_bench --experts 8 --tokens 4 --precision bf16-rounded` using `RawBf16MlpView` and direct resident-table admission. Non-reference experts never materialize FP32 vectors; only the last expert is decoded for CPU comparison. Five warmups and 20 measured iterations.
- Context length: 4 candidate tokens in one expert FFN block; not a prefill/decode context benchmark.
- Result: warm block latency median 1,648,927 ns (approximately 412,232 ns per candidate token); cold latency 135,877,327 ns; host payload/load setup 3,864,059,647 ns; cold weight H2D 603,979,776 bytes; warm weight H2D 0; resident weight bytes 603,979,776; last-expert CPU maximum relative error 0.00135118968 (0.135%).
- Single-expert check: 1 expert x 4 tokens measured 270,243 ns warm block median, 57,380,545 ns cold, and the same bounded CPU error gate.
- Decode tok/s, prefill tok/s, TTFT, GPU utilization, VRAM peak, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality benchmark: not measured.
- Interpretation: direct raw-byte admission reduced the 8-expert warm block by approximately 6.2% and cold execution by approximately 5.6x relative to the preceding dense-wrapper probe. The cold comparison also excludes seven experts' FP32 reference materialization, so it is a storage/setup result, not a pure kernel speedup or end-to-end tok/s claim.

## 2026-08-14 -- Pointer-array batched real BF16 expert grid

- Commit: `d36ad21` (pointer-array implementation `36ab952`).
- Hardware: NVIDIA GeForce RTX 5080, CUDA 13.3, WSL; same two bounded GLM probe artifacts.
- Model/checkpoint: `zai-org/GLM-5.2`, layer 10, first 8 available real experts, 4 candidate tokens.
- Mode: direct raw-BF16 resident grid with cublasLt pointer-array layouts. Gate, up, and down projections are each submitted as one multi-expert batch; the SiTU activation remains one fused launch. Five warmups and 20 measured iterations.
- Result: warm block latency median 1,065,026 ns (approximately 266,257 ns per candidate token); cold latency 153,395,924 ns; host payload/load setup 4,285,733,935 ns; cold weight H2D 603,979,776 bytes; warm weight H2D 0; resident weight bytes 603,979,776; maximum relative CPU difference 0.00135860045 (0.136%). The cumulative 26 calls reported 104 resident-grid kernel launches and 14,976 descriptor bytes, equal to 4 launches and 576 descriptor bytes per call.
- Decode tok/s, prefill tok/s, TTFT, GPU utilization, VRAM peak, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality benchmark: not measured.
- Interpretation: the pointer-array path reduced the latest warm block from 1,648,927 ns to 1,065,026 ns (approximately 35.4%). Cold execution is slightly higher because pointer-plan creation and descriptor setup are included; single-expert calls remain on the scalar plan after measuring pointer setup overhead.

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

## 2026-08-14 -- Opt-in BF16-output real expert grid

- Commit: `95f596d`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: two bounded `zai-org/GLM-5.2` probe artifacts; layer 10, experts 0-7, exact raw BF16 role bytes. No full checkpoint.
- Mode: raw-BF16 resident pointer-array grid, 4 candidate tokens, 10 warmups, 30 measured iterations. Paired runs changed only `--output fp32` versus `--output bf16`; native routing, proxy, pruning, and speculation were not involved.
- FP32-output result: 1,091,122 ns warm median per 8-expert/4-token block, 1,065,026 ns in the earlier pointer-array sample, 603,979,776 resident weight bytes, zero warm weight H2D, and 0.00135860045 maximum CPU-relative difference in the paired command.
- BF16-output result: 1,034,950 ns warm median per block, 603,979,776 resident weight bytes, zero warm weight H2D, and 0.00316690677 maximum CPU-relative difference. Gate/up/down intermediate and final device output buffers use BF16; the public result is converted to float after D2H.
- Relative result: BF16 output was approximately 5.1% faster than the paired FP32-output run and halves the physical final D2H bytes. The higher numerical difference keeps the mode experimental and opt-in.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, cache hit rate, natural average Top-K, speculative acceptance, and task quality: not measured.
- Caveat: this is one FFN block over bounded real shards. It does not establish full-layer, end-to-end, or 10+ tok/s performance.

## 2026-08-14 -- cublasLt workspace sweep on real expert grid

- Commit: `716967c`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: the same two bounded GLM-5.2 probe artifacts, layer 10, experts 0-7, 4 candidate tokens. No full checkpoint.
- Mode: raw-BF16 resident pointer-array grid, FP32 output unless noted, 10 warmups, 30 measured iterations, synchronous transfer, zero warm weight H2D.
- Warm medians: 0 bytes `994,529 ns`, 8 MiB `986,393 ns`, 16 MiB `1,073,612 ns`, and 64 MiB `967,790 ns` per block. Maximum CPU-relative difference stayed `0.00135860045`.
- BF16-output cross-check: 64 MiB workspace `1,080,469 ns` versus `1,034,950 ns` with zero workspace; maximum CPU-relative difference stayed `0.00316690677`.
- Interpretation: workspace can improve FP32-output selection by about 2.7% in this sample, but the effect is not monotonic and can regress BF16 output. Keep it runtime-selectable and default-off.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, cache hit rate, natural Top-K, speculative acceptance, and task quality: not measured.
- Caveat: this is a bounded expert FFN block, not full-layer or end-to-end throughput.

## 2026-08-14 -- Sparse-packed real-shard probe

- Commit: `2d22579`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: two bounded GLM-5.2 probe artifacts, layer 10, experts 0-7, exact raw BF16 payloads; no full checkpoint.
- Mode: 10 warmups, 30 measured iterations, FP32 output, synchronous transfer. `common` broadcasts a 2-token block to every expert; `sparse-packed` alternates two logical tokens across experts and sends one packed token slab per expert. The assignment pattern is deterministic and not learned GLM routing.
- Common result: 1,040,559 ns warm median/block, 0.00177719456 maximum CPU-relative difference.
- Sparse-packed result: 965,550 ns warm median/block, 0.00166274444 maximum CPU-relative difference; zero warm weight H2D and the same 603,979,776 resident bytes.
- BF16-output sparse-packed cross-check: 995,611 ns warm median/block and 0.00396688282 maximum CPU-relative difference.
- Relative result: sparse-packed was approximately 7.2% lower latency in this paired sample, but it evaluates one candidate per expert instead of two and therefore is not a direct quality or tokens/s claim.
- Decode tok/s, prefill tok/s, TTFT, natural Top-K, speculative acceptance, and task quality: not measured.

## 2026-08-14 -- Packed raw expert-grid correctness gate

- Commit: `d7638af`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: two-expert nonzero synthetic BF16 fixture with GLM-shaped API dimensions reduced to a tiny test matrix; no full checkpoint.
- Mode: `raw_bf16_situ_mlp_grid_packed`, two experts, one candidate slab per expert, distinct input slab for each expert, pointer-array grid, synchronous transfer.
- Result: CPU BF16-rounded parity passed within the existing `2e-2` test tolerance. Activation H2D was 12 bytes and FP32 output D2H was 16 bytes; four resident-grid launches were recorded.
- Decode tok/s, prefill tok/s, TTFT, cache hit rate, natural Top-K, speculative acceptance, and task quality: not measured.
- Caveat: this validates per-expert pointer/input addressing only. It does not benchmark a ragged GLM route distribution or claim end-to-end throughput.

## 2026-08-14 -- Expert-major packed-plan correctness gate

- Commit: `f87bbdc`.
- Hardware: host C++ reference path; no CUDA and no checkpoint.
- Model/checkpoint: tiny synthetic hidden-state slab with two routed positions and three expert assignments.
- Mode: `build_expert_major_packed_plan`, stable first-use expert grouping, exact assignment metadata retained.
- Result: one expert received one `[hidden]` slab and another received two slabs in route order; the test verified hidden values, expert IDs, assignment count, and contributions metadata.
- Decode tok/s, prefill tok/s, TTFT, VRAM, H2D/NVMe traffic, cache hit rate, Top-K quality, speculative acceptance, and task quality: not measured.
- Caveat: this is scheduler preparation correctness, not a CUDA or end-to-end throughput result.

## 2026-08-14 -- Ragged expert-major packed-batch bucketing gate

- Commit: `46f2e8e`.
- Hardware: host C++ reference path; no CUDA execution and no checkpoint.
- Model/checkpoint: tiny synthetic hidden-state slab and route metadata only.
- Mode: `bucket_expert_major_packed_plan`, stable first-use assignment-count buckets, source group-index retention, no padding.
- Result: the C++ gate verified separate one- and two-assignment batches, repeated-shape concatenation, exact packed input order, and malformed payload rejection. WSL CTest passed 26/26 and the focused GLM Python suite passed 35/35.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D traffic, cache hit rate, natural Top-K, speculative acceptance, and task quality: not measured.
- Caveat: this is a scheduler-shape correctness boundary. No GLM router, CUDA dispatch, output scatter, full layer, or end-to-end throughput is connected yet.

## 2026-08-14 -- Expert-major contribution-scatter correctness gate

- Commit: `b777b1b`.
- Hardware: host C++ reference path; no CUDA execution and no checkpoint.
- Model/checkpoint: tiny synthetic hidden-state slab, route contributions, and group output slabs only.
- Mode: `scatter_expert_major_outputs`, explicit group-order output validation and contribution-weighted token-major accumulation.
- Result: the C++ gate reconstructed token 0 as `[6.4, 12.8]` and token 1 as `[3.0, 4.0]` from two routed expert groups, and rejected a short output slab. WSL CTest passed 26/26 and the focused GLM Python suite passed 35/35.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D traffic, cache hit rate, natural Top-K, speculative acceptance, and task quality: not measured.
- Caveat: this validates contribution semantics only. It is not a CUDA launch, real GLM layer, learned routing, or end-to-end throughput result.

## 2026-08-14 -- Latest sparse-packed real-shard rerun

- Commit: `1f43e1a` (runtime behavior from `b777b1b`).
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: the same two bounded GLM-5.2 probe artifacts, layer 10, selected experts 0-15, exact raw BF16 payloads; no full checkpoint.
- Mode: 10 warmups, 30 measured iterations, FP32 output, synchronous raw-BF16 resident pointer-array grid. `common` used a 2-token slab for every expert; `sparse-packed` used the existing deterministic alternating two-token assignment pattern. Neither is learned GLM routing.
- Common result: 927,744 ns warm median/block, 160,786,678 ns cold latency, 603,979,776 cold weight H2D bytes, 603,979,776 resident bytes, 0 warm weight H2D, and 0.00177719456 maximum CPU-relative difference. Host payload load was 4,241,858,678 ns.
- Sparse-packed result: 939,149 ns warm median/block, 175,614,990 ns cold latency, 603,979,776 cold weight H2D bytes, 603,979,776 resident bytes, 0 warm weight H2D, and 0.001662744442 maximum CPU-relative difference. Host payload load was 3,906,998,594 ns.
- Relative result: sparse-packed was approximately 1.2% slower in this rerun, reversing the earlier 7.2% lower sample. This confirms the mode is shape/addressing evidence only; no stable speedup or tok/s claim follows.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, natural Top-K, speculative acceptance, and task quality: not measured.

## 2026-08-14 -- Lazy real layer-10 MoE reference smoke

- Commit: `b94c8b8`.
- Hardware: Windows Python reference runtime; no CUDA execution and no full-model load.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts; complete layer-10 expert bundle with 277 complete groups overall.
- Mode: official sigmoid router, exact Top-8 selection, shared SwiGLU, lazy raw-BF16 expert loads, two random BF16 hidden tokens.
- Result: bundle validation/open took `30.662874 s`; the cold forward took `0.482775 s`; the cached repeat took `0.059090 s`; 15 unique layer-10 experts were selected; cold/cached output maximum absolute difference was `0.0`; output shape was `[2,6144]` BF16.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, speculative acceptance, and task quality: not measured.
- Caveat: this is a reference MoE-layer and cache-reuse gate. It is not a full-layer, full-model, or end-to-end throughput result.

## 2026-08-14 -- Public Linux correctness gate after evidence-boundary fix

- Commit: `a00beec`.
- Hardware: GitHub Actions Ubuntu runner; C++ CPU build, no CUDA and no checkpoint.
- Mode: CMake configure/build, CTest, Python/cross-language suite with explicit skips only for absent migrated B-0006 through B-0024 historical artifacts.
- Result: correctness workflow `31795971197` passed in `3m21s`; CodeQL workflow `31795971207` passed. The previous 50 `FileNotFoundError` failures were eliminated without skipping new GLM5X tests.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality: not measured.

## 2026-08-14 -- Exact GLM layer-10 MLA/DSA bundle smoke after reader reuse

- Commit: `a2d6b6d`.
- Hardware: WSL2 on the configured target PC; CPU PyTorch reference path, CUDA not used for this record.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts, complete layer 10, 277 complete expert groups in the bundle; no full checkpoint.
- Mode: exact q-residual projection, official-shaped causal DSA indexer, compressed MLA KV state, natural Top-8 sigmoid MoE, `cache_experts=true`, two random BF16 tokens.
- Bundle construction/root verification: `250.637263 s`.
- Cold two-token layer forward: `5.969859 s`; cached repeat: `0.057331 s`.
- Output: `[1,2,6144]` BF16; 16 unique routed experts loaded in the cold block; DSA selection shape `[1,2,2]` because the causal sequence contains two positions; cached output maximum absolute difference `0.0`.
- Reader-reuse comparison: the pre-reuse sample on the same five artifacts took `491.483777 s` to construct the layer, so one shared validated reader reduced open/verification time by approximately `49.0%`. This comparison is storage/open latency, not decode throughput.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, measured expert-cache hit rate, speculative acceptance, quality score, and end-to-end model throughput: not measured.
- Caveat: this is a correctness and storage-latency gate for one real layer boundary. It is not a full 78-layer run, a CUDA result, or a TPS claim.

## 2026-08-14 -- Ragged expert-major raw-BF16 CUDA dispatch on bounded real shards

- Commit: `d09eb3a`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts, layer 10, eight selected raw-BF16 experts; no full checkpoint.
- Mode: `expert-major`, deterministic two-token route pattern, 10 warmups and 30 measured iterations, FP32 output, synchronous transfers. The route is a scheduler exercise and is not learned GLM routing.
- Route shape: 8 expert groups, 10 total assignments, 2 candidate tokens, one packed CUDA call per assignment-count bucket.
- Expert-major result: `1,380,314 ns` warm median/block, `168,543,514 ns` cold latency, `4,687,726,797 ns` host payload load, `603,979,776` resident bytes, `603,979,776` cold weight H2D bytes, `0` warm weight H2D bytes, and `0.0014705552021` maximum CPU-relative difference.
- Paired common-input result: `1,651,193 ns` warm median/block, `165,055,925 ns` cold latency, `603,979,776` resident bytes, and `0.00146513758227` maximum CPU-relative difference.
- Paired sparse-packed result: `1,631,127 ns` warm median/block, `192,610,519 ns` cold latency, `603,979,776` resident bytes, and `0.00146589963697` maximum CPU-relative difference.
- Relative result: the expert-major bucket/scatter path was approximately `16.4%` lower warm block latency than common in this sample. This is a bounded FFN scheduling measurement, not a decode tok/s estimate.
- Not measured: end-to-end decode/prefill tok/s, TTFT, full-layer quality, natural Top-K quality, speculative acceptance, cache hit rate, NVMe GB/token, system RAM, and full-model VRAM pressure.

## 2026-08-14 -- Learned GLM router to raw-BF16 expert-major CUDA

- Commit: `e599dfb`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts, complete layer-10 expert union; no full checkpoint.
- Mode: official router tensors from the shard (`gate.weight` BF16 and `e_score_correction_bias` FP32), float32 sigmoid scores, natural Top-8, routed scale 2.5, raw-BF16 expert-major bucket/scatter, 20 warmups and 100 measured iterations, synchronous transfers.
- Two-token result: 15 unique expert groups, 16 assignments, `1,905,668 ns` warm median/block, `214,569,151 ns` cold latency, `9,619,390,326 ns` host load, `1,132,462,080` expert bytes resident, `3,146,752` router bytes read, `1,132,462,080` cold weight H2D bytes, `0` warm weight H2D bytes, and `0.000865828245878` maximum CPU-relative difference. VRAM admission budget was 2 GiB.
- Four-token result: 29 unique expert groups, 32 assignments, `3,757,986 ns` warm median/block, `327,788,285 ns` cold latency, `18,399,328,859 ns` host load, `2,189,426,688` expert bytes resident, `3,146,752` router bytes read, `2,189,426,688` cold weight H2D bytes, `0` warm weight H2D bytes, and `0.000666717009153` maximum CPU-relative difference. VRAM admission budget was 4 GiB.
- BF16-output cross-check: two tokens measured `1,937,250 ns` and `0.00194821879268` maximum CPU-relative difference, so BF16 output remains opt-in rather than a default quality path.
- Interpretation: this is the first real routing-aware GLM MoE/FFN measurement. The route and expert union are real, but MLA/DSA, trunk residuals, logits, and token generation are not included; no end-to-end tok/s claim follows.

## 2026-08-14 -- Learned GLM MoE sublayer with shared expert

- Commit: `8017bd2`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL on the configured target PC.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts, complete layer 10; no full checkpoint.
- Mode: `k3x_cuda_glm5x_real_expert_bench --input-mode learned-moe-layer`, official sigmoid natural Top-8, routed scale 2.5, raw-BF16 expert-major routed scatter plus the real shared-expert raw-BF16 grid, FP32 output, 20 warmups and 100 measured iterations, synchronous transfers.
- Two-token result: 15 routed expert groups, 16 assignments, one shared expert, `2,155,188 ns` warm median/block, `246,997,686 ns` cold latency, `10,123,913,076 ns` host load, `1,207,959,552` resident bytes, `1,207,959,552` cold weight H2D bytes, `0` warm weight H2D bytes, `3,146,752` router bytes, `75,497,472` shared payload bytes, and `0.000585675588809` maximum CPU-relative difference. Resident budget was 2 GiB.
- Four-token result: 29 routed expert groups, 32 assignments, one shared expert, `3,968,243 ns` warm median/block, `331,985,997 ns` cold latency, `18,897,178,153 ns` host load, `2,264,924,160` resident bytes, `2,264,924,160` cold weight H2D bytes, `0` warm weight H2D bytes, `3,146,752` router bytes, `75,497,472` shared payload bytes, and `0.000429985491792` maximum CPU-relative difference. Resident budget was 4 GiB.
- BF16-output cross-check: two tokens measured `2,374,827 ns` warm median/block with `0.00111866334919` maximum CPU-relative difference, so FP32 output remains the default.
- Decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, cache hit rate, speculative acceptance, quality score, and full-model VRAM pressure: not measured.
- Interpretation: this is the complete bounded learned MoE sublayer, including shared SwiGLU. It still excludes q-residual/MLA/DSA, trunk residuals, final logits, incremental full-layer state, and token generation; no end-to-end tok/s claim follows.

## 2026-08-14 -- GLM5XACT activation boundary verification

- Commits: `11fd058`, `30bf5d4`.
- Hardware: WSL2 Ubuntu-24.04 with CUDA 13.3 build configured for RTX 5080; no full checkpoint.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts remain the only real weights.
- Mode: portable `GLM5XACT` v1 BF16 activation writer/loader, fixed 40-byte header, atomic Python write, C++ shape/extent/CRC validation, `GLM5XDecoderLayerForward.moe_input` exposure, and optional benchmark `--input-bf16`/`--expected-bf16` boundary.
- Correctness result: WSL CUDA build passed; CTest `27/27` passed in `5.98 s`; public correctness workflow `31806277016` and CodeQL workflow `31806277022` passed, including the focused Python producer test. The Windows interpreter still lacks pytest, so no local Python rerun is claimed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, and quality: not measured. This is an artifact-boundary test, not model execution.

## 2026-08-15 -- Real layer-10 GLM SiLU and GLM5XACT parity boundary

- Commit: working tree after the GLM SiLU activation change; commit pending.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL2 Ubuntu-24.04. The target CPU/RAM/NVMe were not independently sampled by this benchmark.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` `.k3x` probe artifacts, complete layer 10 only; no full checkpoint.
- Mode: `learned-moe-layer`, official natural Top-8 sigmoid router with routed scale 2.5, 16 routed experts plus one shared expert, raw-BF16 resident expert-major/grid CUDA, explicit GLM SiLU, FP32 output, BF16-rounded input, 5 warmups, 20 measured iterations, 2 tokens.
- Route: 16 unique routed experts and 16 assignments, average Top-K `8.0`; the Python and C++ route IDs/contributions matched for both tokens. Speculative acceptance, adaptive-K changes, and task/session profiles were not active.
- Latency: host payload/router/shared load `10,690,416,182 ns`; cold execution `229,905,390 ns`; warm median `2,091,698 ns` per two-token MoE sublayer block. This is not decode tok/s, prefill tok/s, TTFT, or a complete decoder-layer measurement.
- Memory/traffic: resident expert bytes `1,283,457,024`; cold H2D `1,283,457,024 bytes` (`641,728,512 bytes/token`); warm H2D `0`; router payload `3,146,752 bytes`; shared payload `75,497,472 bytes`. Process peak VRAM, system RAM, NVMe GB/token, and cache-hit rate were not instrumented; zero warm H2D is the observable resident-table reuse signal.
- Correctness: GPU-versus-C++ CPU maximum absolute error `0.000018091101083` and relative error `0.000452667358331`. The expected GLM5XACT artifact is compared after the actual FP32 output is rounded to BF16: maximum absolute error `0.00006103515625`, relative error `0.00152439018711`. The separate unrounded CPU-versus-Python diagnostic is `0.000105291604996` absolute / `0.00262972200289` relative and reflects accumulation-order plus BF16 storage precision, not routing divergence.
- Tests: WSL host CTest `15/15`, WSL CUDA CTest `27/27`, and the full WSL Python suite `301 passed, 124 skipped` after building the CI-compatible `build/k3x_run` executable. No end-to-end quality score, token generation, or full-model throughput was measured.
- Interpretation: this closes the GLM MoE activation and bounded real-output handoff boundary. The next bottleneck is exact q-residual/MLA/DSA hidden-state export and full-layer parity; the repository must not claim a GLM tok/s number from this record.

## 2026-08-15 -- Rejected expert-major bucket-cache and shared-dispatch experiment

- Date: 2026-08-15.
- Commit: working-tree experiment compared with `f07d78c` baseline; reverted after measurement.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL2 Ubuntu-24.04.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` probe artifacts, layer 10, exact raw-BF16 expert roles; no full checkpoint.
- Mode: `learned-moe-layer`, natural Top-8 sigmoid routing, explicit GLM SiLU, FP32 output, resident experts. The experiment cached ragged expert-major buckets and fused the shared expert only for one-token calls. The baseline rebuilt buckets per call and dispatched routed/shared paths separately.
- Token-1 result: experiment five-run medians were `1,317,339`, `1,307,995`, `1,206,625`, `1,147,763`, and `1,374,589 ns`; median-of-runs `1,307,995 ns`. Baseline medians were `1,255,004`, `1,304,073`, `1,252,984`, `1,484,929`, and `1,265,441 ns`; median-of-runs `1,265,441 ns`. The experiment was approximately `3.4%` slower by median-of-runs.
- Token-2 result: bucket-cache-only experiment three-run medians were `2,238,866`, `2,191,291`, and `2,157,692 ns`; median `2,191,291 ns`. Baseline medians were `2,529,160`, `2,166,726`, and `2,120,466 ns`; median `2,166,726 ns`. The experiment was approximately `1.1%` slower by median-of-runs.
- Correctness: the existing real layer-10 CPU/GPU and BF16-boundary parity remained within the previously recorded tolerance. No route or output divergence was observed.
- Decision: rejected as a default optimization because the measured latency did not improve despite fewer token-1 grid calls. No decode/prefill tok/s, TTFT, full-layer quality, cache hit rate, NVMe traffic, or end-to-end throughput was measured.

## 2026-08-15 -- Opt-in device-side ragged expert accumulation

- Commit: `1514d11`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL2 Ubuntu-24.04. The target CPU/RAM/NVMe were not independently sampled by this benchmark.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` `.k3x` probe artifacts, complete layer 10 only; no full checkpoint.
- Mode: `learned-moe-layer`, official sigmoid natural Top-8 routing with scale 2.5, explicit GLM SiLU, FP32 output, BF16-rounded inputs, resident raw-BF16 expert union, 5 warmups and 100 measured iterations. Baseline uses host output copies plus CPU scatter; experimental mode uses `--device-accumulate 1` and one final device-to-host output copy.
- Route and residency: 15 routed expert groups, 16 assignments, one shared expert, 2 tokens, average Top-K `8.0`, resident expert bytes `1,207,959,552`, warm weight H2D `0`, router payload `3,146,752` bytes, shared payload `75,497,472` bytes.
- Repeated warm medians, baseline: `2,198,145`, `2,736,064`, `2,492,351 ns`; device accumulation: `1,991,721`, `1,981,629`, `2,446,610 ns`. Median-of-runs: `2,492,351 ns` versus `1,991,721 ns`, approximately `20.1%` lower for the experimental path. The spread is material, so this is not a guaranteed speedup.
- Correctness: both paths reported GPU/CPU maximum relative error `0.000571510172449` and absolute error `0.0000379583798349`; route IDs and contributions were unchanged. CUDA ragged primitive and varied-bucket parity tests pass.
- Traffic/quality: the experimental path reduces per-bucket output D2H and CPU scatter, but no full-layer quality, decode tok/s, prefill tok/s, TTFT, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, speculative acceptance, or final-token result was measured.
- Status: experimental and runtime-switchable only. The default remains the host-scatter path until exact full-layer GLM parity and quality results exist.

## 2026-08-15 -- Shared-expert device accumulation on exact GLM5XACT handoff

- Commit: `8090b90`.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3 in WSL2 Ubuntu-24.04. The target CPU/RAM/NVMe were not independently sampled by this benchmark.
- Model/checkpoint: five bounded `zai-org/GLM-5.2` `.k3x` probe artifacts, complete layer 10 only; exact `layer10-moe-input.gmlxact` and `layer10-moe-output.gmlxact` produced by the Python layer reference; no full checkpoint.
- Mode: learned natural Top-8, routed scale 2.5, 16 routed experts plus one shared expert, raw-BF16 resident grid, BF16-rounded input, FP32 output, 10 warmups and 100 measured iterations. The baseline uses separate routed/shared output handling; the fused mode uses `--device-accumulate 1 --fuse-shared 1` and adds the shared device output before one final D2H.
- Route/residency: 16 routed assignments across 16 unique experts, average Top-K `8.0`, resident expert bytes `1,283,457,024`, cold weight H2D `1,283,457,024` bytes, warm weight H2D `0`, router payload `3,146,752` bytes, shared payload `75,497,472` bytes. Host payload load was approximately `10.0–11.0 s` per fresh process.
- Baseline warm medians: `2,180,810`, `2,194,670`, and `2,371,374 ns`; median-of-runs `2,194,670 ns`.
- Device accumulation without shared fusion: `2,326,186`, `2,590,515`, and `2,098,680 ns`; median-of-runs `2,326,186 ns`, approximately `5.99%` slower than this longer baseline sweep. This does not reproduce the earlier standalone `20.1%` sample and shows material run-to-run variance.
- Fused shared accumulation: `1,984,222`, `1,986,460`, and `2,090,547 ns`; median-of-runs `1,986,460 ns`, approximately `9.49%` lower than baseline and `14.60%` lower than device accumulation without fusion in this sweep.
- Correctness: CUDA synthetic fused routed-plus-shared parity passed. On the exact GLM5XACT handoff, GPU/CPU maximum relative error was `0.00045266628149`, expected BF16-artifact relative error was `0.00152439018711`, and route IDs/contributions matched Python. The fused path reported one final output D2H per call.
- Tests: WSL host CTest `15/15`, CUDA CTest `27/27`, and Python `301 passed, 124 skipped`.
- Status: experimental and default-off. This is still one real MoE sublayer, not a complete decoder layer or end-to-end tok/s result. No TPS, TTFT, NVMe GB/token, full-layer quality, or final-token result was measured.

## 2026-08-15 -- Multi-layer CPU reference logits and greedy parity

- Date: 2026-08-15.
- Commit: `f5a3e3a`.
- Hardware: WSL2 Ubuntu-24.04 CPU reference; no CUDA execution and no full checkpoint.
- Model/checkpoint: synthetic GLM5X-compatible two-layer graph built from the existing tiny MLA/DSA/MoE fixture.
- Mode: prompt prefill followed by one-token incremental state reuse; final RMSNorm, LM head, and greedy generation enabled. No adaptive Top-K, speculative decoding, or proxy path.
- Correctness: one focused test matched each incremental prompt logit against the corresponding prefill output and matched greedy generation against an explicit loop. Full WSL Python suite passed `302 passed, 124 skipped` in `69.41 s`.
- Performance/traffic: decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, and quality benchmark were not measured. This is a correctness boundary, not a throughput result.
- Interpretation: final-logit/state ownership is now explicit for the synthetic graph. The next bottleneck is real all-layer tensor loading and exact MLA/DSA-to-CUDA hidden-state parity.

## 2026-08-15 -- Out-of-core reference layer-loader contract

- Date: 2026-08-15.
- Commit: `a33b25b`.
- Hardware: WSL2 Ubuntu-24.04 CPU reference; no full checkpoint and no new CUDA kernel path.
- Model/checkpoint: synthetic GLM5X-compatible two-layer graph built from the existing tiny MLA/DSA/MoE fixture.
- Mode: `GLM5XDecoderModelReference.from_layer_loader`, two layer IDs, prompt prefill, final logits, and recurrent state enabled.
- Correctness: the loader was called exactly in order `[0, 1]`; output contains both layer forwards. Full WSL Python passed `303 passed, 124 skipped` in `70.47 s`; WSL host CTest `15/15` and CUDA CTest `27/27` passed.
- Performance/traffic: no tok/s, VRAM, RAM, NVMe, H2D, or quality result was measured. The test proves the residency contract only.
- Interpretation: layer weights can now be supplied lazily without changing model-level state semantics. A real-shard provider and async transfer overlap are the next measurable boundaries.

## 2026-08-15 -- Cross-shard bundle reader reuse

- Date: 2026-08-15.
- Commit: `75dc00c`.
- Hardware/model: WSL2 Ubuntu-24.04 CPU reference and the bounded synthetic bundle fixture; no full checkpoint.
- Mode: `GLM5XDecoderLayerReference.bundle_layer_loader`, one bundle open, two requests for the same layer.
- Correctness/overhead contract: the monkeypatched open counter reported `1` bundle open for both layer requests; selected expert roles still loaded lazily through the verified bundle. Full Python passed `303 passed, 124 skipped`.
- Performance/traffic: no tok/s, VRAM, RAM, NVMe, H2D, or quality metric was measured. The result removes repeated reader initialization but is not a throughput claim.

## 2026-08-15 -- Lazy K3X payload/root admission on a real shard

- Date: 2026-08-15.
- Commit: `a726368`.
- Hardware: Solidigm NVMe on the Windows host, measured through WSL2 Ubuntu-24.04 Python; no GPU execution.
- Model/checkpoint: `zai-org/GLM-5.2`, `build-glm5x-hf-probe/first-shard.k3x`, 5,342,863,616 bytes, 35 tensors. No full checkpoint.
- Strict mode: `K3XReader.open(verify_payloads=True, verify_root=True)` took `49.816001 s`.
- Lazy mode: `verify_payloads=False, verify_root=False` opened and decoded directories in `0.003153 s`; reading the first selected tensor and performing its deferred CRC took `8.989272 s` for a `1,903,165,440`-byte payload.
- Correctness: a one-byte mutation in lazy mode raised `DATA_CRC_MISMATCH` on first tensor read. Default eager mode and all existing bundle identity/CRC tests remain active.
- Traffic/quality: no tok/s, TTFT, VRAM, system RAM, H2D, cache hit, or quality score was measured. The result is cold-start and selective-admission evidence only.
- Status: experimental and opt-in. `verify_root=False` deliberately weakens whole-artifact integrity timing and is not a QUALITY default.

## 2026-08-15 -- Reference trunk-layer cache contract

- Date: 2026-08-15.
- Commit: `761b881`; no CUDA or full checkpoint.
- Hardware/model: WSL2 Ubuntu-24.04 CPU reference, synthetic GLM5X-compatible two-layer graph.
- Mode: `GLM5XDecoderModelReference.from_layer_loader`, two forwards over the same two layers, `layer_cache_capacity=0` versus `2`.
- Correctness: both modes produced identical logits. Capacity 2 retained two validated layer objects and the loader call sequence over two forwards was `[0, 1]`; the no-cache regression remains `[0, 1, 0, 1]`.
- Performance/traffic: the synthetic model is too small for a meaningful latency claim. A single-threaded 8-forward sample measured `1.0073 ms/forward` with capacity 0 and `1.0310 ms/forward` with capacity 2, so no speedup is claimed. The measurable benefit is eliminating repeated layer construction and future real-shard admission calls; real 78-layer RAM/NVMe traffic remains unmeasured.
- Status: opt-in and runtime-selectable. Capacity 0 remains the default until real trunk footprint and quality parity are measured.

## 2026-08-15 -- Dense GLM MLP reference boundary

- Date: 2026-08-15.
- Commit: `fb3aa7d`.
- Hardware/model: WSL2 Ubuntu-24.04 CPU reference; synthetic GLM5X-compatible bundle; no full checkpoint.
- Mode: `GLM5XDenseMlpReference` plus `GLM5XDecoderLayerReference.from_bundle(..., mlp_type="dense")`, BF16 SwiGLU, explicit empty routing contract.
- Correctness: focused layer-reference tests passed `3/3`; the full WSL Python suite passed `306 passed, 124 skipped` in `73.57 s`. The dense bundle path produced zero expert loads and retained the decoder-layer shape/state contract.
- Performance/traffic: decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, and quality were not measured. This is a model-graph correctness boundary only.
- Status: implemented in the reference path; public correctness `31826654966` and CodeQL `31826655082` both passed. The hosted Linux job took about 7 minutes 10 seconds, including a roughly 5 minute 19 second Python step; this is CI wall time, not model throughput.

## 2026-08-15 -- Configuration-driven real GLM bundle factory

- Date: 2026-08-15.
- Commit: `1f123ca`.
- Hardware/model: WSL2 Ubuntu-24.04 CPU reference; five bounded `zai-org/GLM-5.2` `.k3x` probe artifacts; no full checkpoint.
- Mode: `GLM5XDecoderModelReference.from_bundle`, `verify_payloads=False`, `verify_root=False`, explicit small head-tensor overrides because the bounded bundle does not contain `model.norm.weight` or the complete final head. The decoder provider still reads real layer payloads.
- Synthetic correctness: three-layer bundle with dense/dense/sparse MLPs and a shared indexer passed model prefill/incremental parity; full WSL Python passed `307 passed, 124 skipped`; host CTest passed `15/15`.
- Real layer-0 gate: factory setup `0.066806 s`; layer-0 dense admission `4.278880 s`; one-token layer forward `0.036846 s`; output shape `[1,1,6144]`; DSA Top-K shape `[1,1,1]`; MoE routing shape `[1,1,0]` because layer 0 is dense.
- Performance/traffic: no full-model decode tok/s, prefill tok/s, TTFT, quality score, full VRAM, full RAM, NVMe GB/token, H2D GB/token, or final-token result was measured. These are real-layer admission/reference timings only.
- Interpretation: the next bottleneck is complete all-layer payload availability and exact final-head state, followed by CUDA hidden-state handoff and asynchronous layer overlap.
- Public verification: correctness `31828512721` and CodeQL `31828512789` passed for the implementation plus documentation HEAD `3a86ca3`.

## 2026-08-15 -- CUDA staging and local shard-stream gate

- Date: 2026-08-15.
- Commit: `6fb2da1`.
- Hardware/model: WSL2 Ubuntu-24.04 with CUDA 13.3 and RTX 5080; official `zai-org/GLM-5.2`; no quality benchmark.
- Correctness: CUDA-only layer parity and model-factory parity passed; the complete WSL Python suite passed `311 passed, 124 skipped` in `79.98 s`; host CTest passed `15/15`.
- Device timing: a four-token synthetic layer smoke measured `5.717097 ms` on CUDA versus `1.321640 ms` on CPU. This fixture is intentionally tiny and is not decode tok/s or a full-model estimate.
- Materialization: `model-00001-of-00282.k3x` finalized at `5,342,863,616` bytes and `model-00002-of-00282.k3x` at `5,351,993,600` bytes. Both artifacts passed strict reader verification and their source shards were removed after atomic deletion markers. Shard 3 was downloading at the time of recording.
- Throughput/quality: decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, adaptive Top-K, speculation acceptance, and quality were not measured. The active stream is a progress/traffic gate only.

## 2026-08-15 -- Public CI verification for the stream/device commit

- Commit: `6fb2da1`.
- Linux correctness run `31831711520`: success in `2m38s`; C++ configure/build/CTest and Python/cross-language tests passed.
- CodeQL run `31831711580`: success for both Python and C++ analysis.
- These are CI wall times and static/correctness gates, not model throughput measurements.

## 2026-08-15 -- Resume-path regression and public verification

- Commit: `db2cf37`.
- Focused bundle/stream/converter tests: `7 passed` in `1.53 s` under WSL.
- Linux correctness run `31833153961`: success in `2m45s`.
- CodeQL run `31833154040`: success; Python analysis `2m24s`, C++ analysis `3m40s`.
- These results cover restart semantics only. Decode tok/s, prefill tok/s, TTFT, memory traffic, and quality remain unmeasured.

## 2026-08-15 -- Initial three-worker shard overlap

- Commit: `6cd4e85`.
- Hardware: RTX 5080 PC, WSL2 Ubuntu-24.04, 1 GbE, C: NVMe workspace; three Python stream workers, no final bundle assembly.
- Range assignment: worker 0 `10..100`, worker 1 `101..191`, worker 2 `192..281` (half-open indices); first ten artifacts were already complete.
- Observed: worker 0 finalized shard 12 at `04:46:52`; worker 1 finalized 102 and 103 at `04:44:16` and `04:49:37`; worker 2 finalized 193 and 194 at `04:44:14` and `04:49:36`. Source downloads for the next shard overlapped conversion.
- Interpretation: the sample is an aggregate overlap signal, not a final sustained shards/hour benchmark. No model tok/s, quality, or runtime latency was measured.

## 2026-08-15 -- Real layer-10 four-token CUDA fusion comparison

- Date: 2026-08-15.
- Commit: `280f330` working tree measurement; no source-code change in this run.
- Hardware: RTX 5080, WSL2 Ubuntu-24.04, CUDA 13.3; official `zai-org/GLM-5.2` layer-10 real BF16 expert payloads.
- Mode: `learned-moe-layer`, four candidate tokens, 29 routed experts plus one shared expert, 4 GB resident budget, 10 warmup iterations and 100 measured iterations. Baseline used host accumulation/shared path; fused used device accumulation plus fused shared path.
- Baseline: warm median `4,317,561 ns/block`, host-load `32,058,777,747 ns`, cold latency `564,445,113 ns`, cold H2D `2,415,919,104 B`, resident expert bytes `2,415,919,104 B`, warm H2D `0 B`.
- Fused: warm median `3,741,291 ns/block`, host-load `31,820,005,590 ns`, cold latency `873,953,647 ns`, cold H2D `2,415,919,104 B`, resident expert bytes `2,415,919,104 B`, warm H2D `0 B`.
- Difference: the fused/device-accumulate warm median is approximately `13.3%` lower in this paired bounded sublayer sample. Cold latency was higher in the fused run, so this does not establish an end-to-end speedup or a default policy.
- Correctness: both modes reported GPU-vs-CPU maximum relative error `0.000643727718852` and CPU-expected maximum relative error `0.00484962016344`; route IDs/contributions were identical. No decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, or quality score was measured.
- Interpretation: this is a real-weight, real-routing CUDA kernel boundary only. The next bottleneck remains all-layer hidden-state/final-head parity and full-model scheduling; the result must not be converted into a model tok/s claim.

## 2026-08-15 -- Reference MXFP4 encoding quality gate

- Date: 2026-08-15.
- Commit: working-tree experiment after `2dafbcf`; Python implementation and tests are now recorded in the next code commit.
- Hardware/model: WSL2 Ubuntu-24.04 CPU reference, official `zai-org/GLM-5.2` layer-10 expert 4 from `glm5x-experts-partial.json`, real four-token `layer10-real4-moe-input.gmlxact`.
- Payload: three BF16 projections, each `25,165,824` bytes; total `75,497,472` bytes. Native MXFP4 output is `6,684,672` bytes per projection and `20,054,016` bytes total, exactly `26.5625%` of BF16 before any outlier metadata.
- `max_abs` mode: all three projections encoded in `0.442 s`; weight relative L2 was `11.735%`, `11.732%`, and `11.654%`; FFN output max absolute error was `0.0018968`, mean absolute error `0.00038270`, and relative L2 error `19.861969%`.
- `mse` mode: all three projections encoded in `7.439 s`; weight relative L2 was `11.127%`, `11.125%`, and `11.116%`; FFN output max absolute error was `0.0018991`, mean absolute error `0.00036631`, and relative L2 error `19.069034%`.
- Correctness: both encoded payloads decoded to valid native MXFP4 shapes and passed the focused round-trip suite; the full WSL Python suite passed `318 passed, 124 skipped` in `141.53 s`.
- Interpretation: the compression ratio is promising for bandwidth, but uncalibrated BF16-to-MXFP4 quality is not acceptable as a default. No converter/runtime integration, full-model quality score, or end-to-end tok/s is claimed.

## 2026-08-15 -- Reference expert-major MoE batching gate

- Date: 2026-08-15.
- Commit: `553a2a1`.
- Hardware/model: RTX 5080, WSL2 Ubuntu-24.04, CUDA 13.3, official GLM-5.2 layer-10 partial bundle, exact raw-BF16 expert roles, no full checkpoint.
- Correctness: `tests/python/test_glm5x_layer10_moe.py`, layer-reference, and model-reference focused tests passed `11/11`; the full WSL Python suite passed `319 passed, 124 skipped` in `132.68 s`. Router logits, Top-8 indices/weights, loaded-expert order, and output parity were checked between loop and expert-major modes.
- Four-token MoE sublayer: loop warm median `21.670 ms`; expert-major warm median `18.652 ms`; selected expert union `26`; output maximum absolute difference `0.03125` at BF16 boundary.
- One-token MoE sublayer: loop warm median `5.584 ms`; expert-major warm median `7.359 ms`; selected expert union `8`; output maximum absolute difference `0.015625` at BF16 boundary.
- Four-token full layer: loop warm median `18.676 ms`; expert-major warm median `20.082 ms`; this is a reference-layer timing, not decode tok/s.
- Memory: loop added about `0.20 MB` peak over its cached baseline; expert-major added about `1.97 GB` peak because it stacks selected expert weights for batched `bmm`.
- Interpretation: the grouped reference path is an opt-in experiment only. It is not enabled by default, not a full-model throughput result, and not a reason to claim a 15--25 tok/s target.

## 2026-08-15 -- Opt-in parallel exact expert reads

- Date: 2026-08-15.
- Commit: `db2e180`.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.3, five-shard official GLM-5.2 probe bundle, layer 10, exact raw-BF16 expert roles.
- Mode: one BF16 hidden-state token through `GLM5XLayer10MoEReference._from_open_bundle`, lazy bundle admission, `cache_experts=false`, loop routing, same input for both runs. The serial reference used `expert_load_workers=1`; the overlap experiment used `expert_load_workers=4` and a bounded thread pool for payload reads only.
- Cold MoE sublayer wall time: `5.403227270 s` with one worker versus `2.126850501 s` with four workers, approximately `60.6%` lower in this single cold sample. Eight natural selected experts were loaded in both runs.
- Peak CUDA allocated bytes: `719,413,248` for both modes. The same input produced output maximum absolute difference `0.0`; route and loaded-expert counts matched.
- Decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, H2D GB/token, cache hit rate, speculative acceptance, and quality benchmark were not measured. This is an exact one-layer I/O boundary result, not a full-model throughput result.
- Interpretation: retain the serial default and expose four workers only for the full-bundle gate. Re-measure on all 78 layers with real cold/warm traffic before promoting the policy; concurrent reads may contend with compute or filesystem cache.

## 2026-08-15 -- Opt-in exact host payload cache

- Date: 2026-08-15.
- Commit: `c2e5980`.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.3, five-shard official GLM-5.2 probe bundle, layer 10, exact raw-BF16 expert roles.
- Mode: one BF16 hidden-state token, `expert_load_workers=4`, `cache_experts=false`, a shared `1,000,000,000`-byte exact host payload cache, same layer and same input for two calls.
- Cold/warm MoE sublayer wall time: `3.070666336 s` on the first call and `0.094567934 s` on the second call. The cache held `603,979,776` bytes for 8 experts and reported `8` misses, `8` hits, and `0` evictions.
- Correctness: output maximum absolute difference between the cold and cached calls was `0.0`; selected expert count and route were unchanged.
- Decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, H2D GB/token, speculative acceptance, and quality benchmark were not measured. This is a bounded exact sublayer cache result, not a full-model throughput result.
- Interpretation: the cache can remove repeat NVMe reads across layer-object lifetimes, but its useful capacity and hit rate are unknown until the 78-layer full bundle runs. Capacity `0` remains the default.

## 2026-08-15 -- Opt-in decoded expert GPU cache

- Date: 2026-08-15.
- Commit: `5e4d86f`.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.3, five-shard official GLM-5.2 probe bundle, layer 10, exact raw-BF16 expert roles.
- Mode: one BF16 hidden-state token, `expert_load_workers=4`, host payload cache `1,000,000,000` bytes, decoded expert device cache `1,000,000,000` bytes, `cache_experts=false`, same input for two calls.
- Cold/warm MoE sublayer wall time: `3.124653389 s` on the first call and `0.003793419 s` on the second call. The device cache held `603,979,776` bytes for 8 experts and reported `8` misses, `8` hits, and `0` evictions; host cache reported 8 misses and 0 hits on the first/second pair because the decoded cache prevented the second host read.
- Correctness: output maximum absolute difference between cold and warm calls was `0.0`; selected expert count and route were unchanged. Peak CUDA allocated bytes were `719,427,584`.
- Decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, H2D GB/token, speculative acceptance, and quality benchmark were not measured. This is a bounded exact sublayer residency result, not a full-model throughput result.
- Interpretation: the device cache removes repeat H2D and decode work when the same experts recur, but 1 GiB only holds a small subset of the full model. The full gate will compare it against the cold path before any policy change.

## 2026-08-15 -- Reuse q-residual between DSA and MLA

- Date: 2026-08-15.
- Commit: `7c79976`.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.3, official GLM-5.2 five-shard probe bundle, layer 10 attention boundary, one BF16 token.
- Mode: identical hidden state and natural DSA Top-K inputs for both runs. The baseline computed q-residual for DSA and let MLA recompute it; the optimized path passed the already computed q-residual into MLA. Ten synchronized warm samples were compared by median.
- Baseline/reuse: `2.298938 ms` versus `2.224939 ms`, approximately `3.22%` lower for the optimized attention boundary.
- Correctness: output maximum absolute difference was `0.0`; route inputs and state construction were unchanged.
- Full-model decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, H2D GB/token, speculative acceptance, and quality benchmark were not measured. This is a bounded exact attention result, not a full-model throughput result.

## 2026-08-15 -- Experimental sparse Top-K MLA attention

- Date: 2026-08-15.
- Commit: `2a1eafc`.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.3, official GLM-5.2 five-shard probe bundle, layer 10 attention boundary.
- Mode: one BF16 incremental query after a manually constructed exact compressed MLA state of context length `16,384`; both paths used the same 128 selected key positions. Dense mode projected/masked the full context, while sparse mode gathered selected compressed KV positions before `kv_b_proj`.
- Timing: dense `12.154 ms`, sparse `2.040 ms`, approximately `83.2%` lower in this sample.
- Correctness/quality: output maximum absolute difference `0.000244140625`; relative L2 difference `0.0278%`. Synthetic prefill and incremental parity tests were exact within the test tolerance. The sparse switch is experimental and default-off.
- Full-model decode tok/s, prefill tok/s, TTFT, full-model VRAM/RAM, NVMe GB/token, speculative acceptance, and task quality were not measured. This is a long-context attention boundary result, not a full-model throughput result.

## 2026-08-15 -- Indexed exact expert-bundle metadata lookup

- Date: 2026-08-15.
- Commit: pending local implementation (focused verification before commit).
- Hardware/model: WSL2 Ubuntu-24.04 Python reference bundle fixture; no full checkpoint or CUDA execution.
- Change: `GLM5XExpertBundle.open()` now constructs one record dictionary per artifact, and `read_expert()` resolves each role by tensor ID without a linear scan through `reader.tensor_records`.
- Correctness: the expert-bundle, MLA, layer-reference, model-reference, and MoE focused suite passed `22/22`; payload equality and existing metadata/CRC checks remain covered.
- Performance: no end-to-end decode tok/s, prefill tok/s, TTFT, NVMe GB/token, H2D GB/token, or full-model timing was measured. The optimization is a metadata-path reduction only until the complete bundle gate runs.

## 2026-08-15 -- Zero-copy decode view for large role payloads

- Date: 2026-08-15.
- Commit: pending local implementation (focused verification before commit).
- Hardware/model: WSL2 Ubuntu-24.04 Python reference bundle fixtures; no full checkpoint or CUDA execution.
- Change: raw-BF16/FP32 tensor views now use `memoryview` above 4 KiB, while tiny fixtures use a writable fallback to avoid a PyTorch warning.
- Correctness: the expert-bundle, MLA, layer-reference, model-reference, and MoE focused suite passed `16/16` with no warnings; tensor shapes, route metadata, and exact output assertions remain unchanged.
- Performance: no end-to-end decode tok/s, prefill tok/s, TTFT, NVMe GB/token, H2D GB/token, or full-model timing was measured. The expected benefit is removal of one CPU-side payload copy before the existing device staging boundary.

## 2026-08-15 -- Public CI and Dependabot run verification

- Commit: `1ff787b`.
- Linux correctness run `31849485242`: success in approximately `2m36s`.
- CodeQL run `31849485259`: success in approximately `3m37s`.
- Dependabot pip update run `31849582869`: success; GitHub Actions update run `31849582874`: success. No update PR was opened; existing Dependabot PRs `#1`--`#4` remain closed.
- Repository security-alert enumeration is disabled and returned HTTP `403`, so this confirms workflow/update health only, not a CVE count. Local WSL `python -m pip check` reported `No broken requirements found`.

## 2026-08-15 -- Experimental row-scaled FP8 expert path

- Date: 2026-08-15.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, official GLM-5.2 five-shard layer-10 probe, exact raw-BF16 source roles.
- Mode: one token through the real layer-10 MoE, four parallel payload readers, exact router and shared expert, with host-side row-scaled E4M3 conversion before CUDA staging. The exact comparison used the same bundle and hidden input.
- Timing: exact `2.751867 s` cold and `4.731 ms` warm; FP8 `2.901232 s` cold and `5.713 ms` warm. The current experimental path is slower in this bounded sample because quantization and scaled GEMM overhead exceed the saved device bytes; no on-disk packed FP8 artifact was used.
- Correctness/quality: Top-K route IDs were identical; output relative L2 drift was `5.603%`, maximum absolute difference `0.265625`. Focused layer/model/reference tests passed `14/14`.
- Interpretation: retain the switch as default-off research. A persistent packed artifact, mixed-precision residuals, and a full-model quality/traffic gate are required before considering FP8 for BALANCED or QUALITY modes. No end-to-end tok/s was measured.

## 2026-08-15 -- Grouped exact expert-role reads

- Date: 2026-08-15.
- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, official GLM-5.2 five-shard layer-10 probe, exact raw-BF16 roles.
- Change: one expert's three role extents are read from one artifact open lifetime, with the existing lazy CRC checks applied to every record.
- Timing: one-token layer-10 cold sample with four readers measured `2.183734 s`; the same grouped path with one reader measured `4.978850 s`. An earlier pre-group four-reader sample was `2.751867 s`; the bounded comparison is approximately `20.6%` lower, but filesystem cache and worker overlap are not controlled across separate processes.
- Correctness: selected expert count and output route remained unchanged; focused bundle/reader/layer/model tests passed `20` cases with `4` capability skips. No full-model tok/s or quality score was measured.

## 2026-08-15 -- Rejected physical-offset ordering experiment

- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, official GLM-5.2 five-shard layer-10 probe, two-token BF16 activation, 16 selected experts, four exact payload readers, lazy bundle admission, no host/device expert cache.
- Comparison: grouped role reads in existing request order (`gate`, `up`, `down`) versus an unshipped variant sorted by each record's physical `data_offset`.
- Timing: four paired samples produced `4.565814 s` median for the existing order and `4.745335 s` for sorted order; sorted order was `3.93%` slower. This is a bounded sublayer measurement and does not represent full-model tok/s.
- Correctness: both variants selected 16 experts and returned `(1, 2, 6144)` outputs. The production sort was reverted; focused reader/bundle/CPP tests passed `24` with `4` capability skips.
- Decision: reject the optimization. Filesystem cache, thread scheduling, and Python decode variance remain uncontrolled; do not infer a benefit from the physical layout alone.

## 2026-08-15 -- Rejected artifact-wide expert batch reads

- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, official GLM-5.2 five-shard layer-10 probe, two-token BF16 activation, 16 selected experts, four workers, lazy bundle admission, no host/device expert cache.
- Comparison: existing concurrent per-expert `read_expert()` tasks versus an unshipped artifact-grouped `read_experts()` variant with one sequential task per artifact.
- Timing: existing per-expert tasks measured `4.175182 s` median; artifact-grouped tasks measured `4.976070 s` median, `16.09%` slower. This is a bounded layer-10 measurement, not full-model tok/s.
- Correctness: both variants selected 16 experts and returned the same `(1, 2, 6144)` output shape. The batch API and layer integration were reverted; the focused MoE/layer/model tests remain green at `13 passed`.
- Decision: reject artifact-wide batching for the current WSL/NTFS storage path. Keep parallel expert tasks until a full-model I/O trace justifies a different queue granularity.

## 2026-08-15 -- Expert-read worker sweep for full-gate tuning

- Hardware/model: RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, official GLM-5.2 five-shard layer-10 probe, two-token BF16 activation, 16 selected experts, lazy bundle admission, no host/device expert cache.
- Timing medians: `1=7.635803 s`, `2=5.390695 s`, `4=4.428789 s`, `8=3.704820 s`, `16=3.159413 s` for one layer-10 MoE forward. Sixteen readers were approximately `28.7%` faster than four in this bounded sample.
- Correctness: all points selected the same 16 experts and returned the same output shape. No end-to-end tok/s or quality benchmark was measured.
- Operational choice: the local full-gate monitor now defaults to `EXPERT_LOAD_WORKERS=16`; the environment variable allows reproduction of another setting, and the serial correctness default is unchanged.

## 2026-08-15 -- C++ deadline worker-pool correctness gate

- Hardware/model: WSL2 Ubuntu-24.04 C++ CPU build with synthetic K3X payloads; no full GLM bundle and no CUDA throughput claim.
- Change: `DeadlineExpertLoader(max_pending, worker_count)` now supports a bounded worker pool. `RuntimeSession` defaults to eight deadline workers, while `HostExpertStore` performs payload I/O outside its mutex and deduplicates concurrent loads for the same `(layer, expert)` key.
- Verification: C++ build succeeded; CTest passed `15/15`; Python regression passed `325` with `124` capability skips; `py_compile tools/benchmark_synthetic.py` passed. The scheduler overlap test observed at least two concurrent loads, and the store test confirmed different keys overlap while the existing same-key single-loader test remains green.
- Synthetic timing: the tiny CPU fixture did not show a stable decode improvement across `1/2/4/8` workers, so no tok/s or speedup is reported. The real full-model C++ gate remains pending completion of the 282-shard bundle.

## 2026-08-15 -- Logical storage-read telemetry

- Change: `K3XReader` now counts payload read calls and data-plus-auxiliary bytes; `GLM5XExpertBundle` aggregates those counters and `benchmark_glm5x_reference.py` records separate prefill/decode `storage_read_*` fields.
- Verification: the focused K3X format, expert-bundle, and benchmark-schema suite passed `29` with `6` capability skips; the complete Python suite passed `326` with `124` skips.
- Measurement boundary: these are logical artifact-file reads. They are not physical NVMe GB/token because the OS page cache may serve the read. Full-model values remain pending bundle completion.

## 2026-08-15 -- Exact FP32 LM-head reuse microbenchmark

- Shape: GLM-5.2 `lm_head.weight` shape `(154880, 6144)` on RTX 5080, BF16 source.
- Fresh conversion: `0.061629 s` median over three synchronized conversions; FP32 resident bytes `3,806,330,880` and temporary peak allocation for BF16 plus FP32 `5,710,544,896` bytes in the isolated probe. The steady-state model head replaces the BF16 source after preparation.
- Reuse: prepared transpose-view access `3.13 us` median over ten synchronized accesses. This is a bounded component measurement, not end-to-end tok/s.
- Correctness: model-reference focused suite `9 passed`; full-model logits and quality remain pending the 282-shard bundle.

## 2026-08-15 -- Grouped decoder-layer trunk tensor reads

- Commit: `5fc8d07`.
- Hardware/model: WSL2 Ubuntu-24.04 Python reference fixture; no full checkpoint or end-to-end CUDA execution.
- Mode: `GLM5XDecoderLayerReference.from_bundle()` with one synthetic layer and all attention/indexer/norm/router/shared-MoE tensors in one `.k3x` artifact. The regression counted reader calls during layer construction.
- Result: the pre-change RED test observed `19` individual `read_tensor_extents()` calls. The grouped path observed `0` individual calls and at least one `read_tensor_extents_many()` call, while the layer output and expert route assertions remained green.
- Verification: focused bundle/layer/MoE/model suite `18 passed`; complete Python suite `327 passed, 124 skipped`; `py_compile` passed for the changed modules.
- Boundary: no decode tok/s, prefill tok/s, TTFT, physical NVMe GB/token, H2D GB/token, VRAM, or task-quality result was measured. This is an exact metadata/open-path optimization pending the full 282-shard gate.

## 2026-08-15 -- Full 282-shard exact BF16 CUDA cold gate

- Date: 2026-08-15.
- Commit: local measurement from public base `6e8c289`; the later monitor repair was documentation/runner-only and does not change this measured model path.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0, C: NVMe workspace; target CPU/RAM were not independently sampled.
- Model/checkpoint: official `zai-org/GLM-5.2`; `282` `.k3x` artifacts, `59,585` tensors, `19,456` complete experts; `78` layers; natural Top-16; exact BF16 payloads; lazy bundle admission; `EXPERT_LOAD_WORKERS=16`; no host/device expert cache.
- Context/mode: prompt token `[0]`, one prefill token and one greedy decode token, `execution_mode=loop`, `layer_cache_capacity=0`, `sparse_topk_attention=false`.
- Decode: `302.68789011100307 s`, `0.0033037330949489767 tok/s`.
- Prefill: `306.93307032898883 s`, `0.003258039281750062 tok/s`.
- TTFT: `609.6209604399919 s`.
- Logical storage reads: prefill `79,763,152,896` bytes over `813` calls; decode `79,763,152,896` bytes over `815` calls; `79,763,152,896` logical bytes/token. These are artifact-file requests, not physical NVMe counters.
- VRAM: peak allocated `8,083,474,944` bytes; peak reserved `9,267,314,688` bytes. System RAM, physical NVMe GB/token, H2D GB/token, quality score, and speculative acceptance were not measured by this CLI.
- Result: correctness baseline completed and generated token `[565]`. The `0.0033 tok/s` number is measured exact reference performance, not an optimized target or a promise. The dominant observed cost is repeated storage reload.

## 2026-08-15 -- Full 282-shard exact BF16 CUDA cached gate

- Date: 2026-08-15.
- Commit: local measurement from public base `6e8c289`.
- Hardware/model: same RTX 5080/WSL2 and official GLM-5.2 full bundle as the cold gate; exact BF16 natural Top-16; `78` layers; `EXPERT_LOAD_WORKERS=16`.
- Cache/mode: prompt token `[0]`, one prefill token plus two decode tokens, `execution_mode=loop`, host expert cache `8,589,934,592` bytes, device expert cache `4,294,967,296` bytes, `layer_cache_capacity=0`, lazy bundle admission.
- Decode: `611.385999924998 s` total, `0.0032712558027912816 tok/s`; step tok/s were `0.003265396166870458` and `0.0032771365063288043`.
- Prefill: `303.5853812530113 s`, `0.003293966250524393 tok/s`; TTFT `609.8269363040163 s`.
- Logical storage reads: prefill `79,763,152,896` bytes; decode `159,526,305,792` bytes (`79,763,152,896` per token); decode read calls `1,625`.
- Cache telemetry: host hit rate `0.0`, `1,800` misses, `1,687` evictions, `8,531,214,336` resident bytes; device hit rate `0.0`, `1,800` misses, `1,744` evictions, `4,227,858,432` resident bytes. The configured capacities do not hold the full expert working set and do not retain non-expert trunk tensors.
- VRAM: peak allocated `12,312,381,952` bytes; peak reserved `13,042,188,288` bytes. Generated tokens were `[565, 8009]`; no quality benchmark or physical device traffic measurement was run.
- Result: enabling the bounded expert caches did not improve this workload. The next benchmark must change residency policy, not merely raise the same small cache, and must re-run exact output parity before any fast mode is enabled.

## 2026-08-15 -- Public CI and dependency-alert verification

- Public head: `5dfe036`.
- Correctness run `31863769799`: completed successfully in about `3m02s`.
- CodeQL run `31863769798`: completed successfully in about `3m46s`.
- The recurring red `correctness / Linux (push)` notification is historical run `31795400168` on stale commit `b94c8b8`; its Python step lacked 50 migrated historical `results/b0006..b0024` files. It is not an active failure on current `main`.
- Dependabot PRs `#1`--`#4` are closed; no open update PR is present. Dependabot security alerts and vulnerability alerts are disabled at the repository endpoint (`403`/`404`), so no CVE count can be verified from the alarm banner.

## 2026-08-15 -- Experimental full-model INT4 expert rejection gate

- Date: 2026-08-15.
- Commit: working tree based on `2e122a0`; this was an experimental run before the GPU-side qparam refinement was recorded, so it is not a current default-path result.
- Hardware: NVIDIA GeForce RTX 5080 16 GB, WSL2 Ubuntu-24.04, CUDA 13.0; system RAM and physical NVMe counters were not independently sampled.
- Model/checkpoint: official GLM-5.2 full bundle, `282` artifacts, `78` layers, natural Top-8 from the current descriptor, exact raw-BF16 source payloads packed to CUDA TinyGEMM INT4 at load time.
- Mode: prompt `[0]`, one prefill token and one greedy decode token, `layer_cache_capacity=78`, `expert_load_workers=16`, `expert_precision=int4`, `trunk_precision=int4`, no host/device expert cache, lazy bundle admission.
- Decode: `353.3312996799941 s`, `0.002830204968837129 tok/s`.
- Prefill: `657.3077802089974 s`, `0.0015213573155669026 tok/s`.
- TTFT: `1010.6390798889915 s`.
- Logical storage reads: prefill `79,763,152,896` bytes; decode `45,298,483,200` bytes/token over `280` grouped calls. The INT4 representation did not reduce source artifact bytes because no packed sidecar exists yet.
- VRAM: peak allocated `17,341,184,512` bytes and peak reserved `17,624,465,408` bytes, above the nominal 16 GiB RTX 5080 budget. Quality and physical NVMe/H2D counters were not measured by this run.
- Result: rejected as a default fast mode. Cold per-token packing and the unchanged expert-read bound dominate; the generated token was `[154820]`.

## 2026-08-15 -- Layer-10 packed expert cache probe

- Date: 2026-08-15.
- Commit: working tree based on `2e122a0` with CUDA-side INT4 qparam/packing refinement.
- Hardware/model: same RTX 5080/WSL2; official GLM-5.2 layer-10 bundle payloads and `layer10-real4-moe-input.gmlxact` with four tokens.
- Mode: exact natural Top-8 router, `expert_precision=int4`, loop execution, `expert_load_workers=16`, 2 GiB decoded expert device cache. This is a MoE-sublayer probe, not a decoder-token benchmark.
- First call: `13.281714103999548 s` (`0.301165946554705` input tokens/s), including payload reads and GPU packing.
- Repeated identical call: `0.00977079599397257 s` (`409.38322757608785` input tokens/s), with `31` cache hits, `31` misses, `31` entries, `621,674,496` resident bytes, and zero evictions.
- Quality: route/output parity was not a full-layer quality score; the probe only confirmed finite packed execution and repeated-route cache reuse.
- Result: confirms that resident packed reuse is valuable, but it cannot be extrapolated to full-model tok/s. The missing requirement is a storage-side packed expert artifact or a route-stable residency policy that materially lowers full-model bytes/token.

## 2026-08-15 -- Verification after INT4 expert changes

- WSL Python focused suite: `33 passed, 6 skipped` across INT4, bundle, layer, model, MoE, and benchmark-schema tests.
- `py_compile`: passed for the changed INT4/reference/benchmark modules.
- No 10--20 tok/s full-model result exists. The measured full-model numbers above remain the only throughput evidence and are explicitly not targets.

## 2026-08-15 -- Fingerprinted packed INT4 sidecar probe

- Date: 2026-08-15.
- Commit: working tree based on `1598ae6` before the sidecar commit.
- Hardware/model: RTX 5080 16 GB under WSL2; official GLM-5.2 layer-10 bundle; four-token `layer10-real4-moe-input.gmlxact` route.
- Mode: `expert_precision=int4`, exact natural Top-8, loop execution, optional `GLM5XPackedExpertCache`, 31 selected expert records, fresh layer instance for the hit measurement.
- Cold sidecar population: `18.112762928998563 s` and `31` sidecar writes.
- Fresh-process sidecar reuse: `1.152440828998806 s` (`3.4708940358135685` input tokens/s), `31` sidecar hits, `31` misses, `31` writes total across both phases, `0` bundle-read calls, `0` bundle-read bytes, and route equality `true`.
- Result: the sidecar removes repeated BF16 source-bundle reads for this bounded sublayer. It is not a full-model decode measurement, does not prove 10--20 tok/s, and remains opt-in pending a 78-layer gate.

## 2026-08-15 -- Verification after packed sidecar integration

- WSL Python suite: `332 passed, 124 skipped` in `76.14 s`.
- Focused sidecar/bundle/layer/model/MoE/schema suite: `32 passed, 6 skipped`.
- Changed-module `py_compile` and `git diff --check`: passed.
- No full-model rerun was started because the prior exact/INT4 gates already take several minutes and the current sidecar evidence is intentionally bounded.
