# GLM5X Project State

Latest verified complete Python regression: `362 passed, 124 skipped` in `106.39 s` against the WSL2 CPU build. CUDA CTest is green at `27/27`, and CPU CTest is green at `15/15`. The prior milestone text below retains historical counts for context.

## Current milestone

GLM-5.2 shape/manifest boundary, exact cross-shard raw-BF16 loading, the exact q-residual/MLA/DSA/MoE reference, configuration-driven all-layer loading, resumable local full-checkpoint streaming, lazy final bundle indexing, disjoint shard-range workers, exact cache/prefetch experiments, CUDA expert-major sublayer boundaries, logical storage telemetry, grouped decoder-layer reads, exact FP32 LM-head reuse, reduced-routing/proxy controls, INT4/FP8 comparison sidecars, experimental MXFP4 sidecars, and the explicit full-bundle benchmark gate are implemented. The latest local regression is `362 passed, 124 skipped`; WSL CUDA CTest is green at `27/27`, and CPU CTest is green at `15/15`. All 282 real shards now have verified `.k3x` artifacts and source-deletion markers, and lazy assembly reports 19,456 complete experts. The first exact full-model CUDA gate is measured: cold prefill `0.003258 tok/s`, cold decode `0.003304 tok/s`, TTFT `609.621 s`, and logical storage reads `79,763,152,896` bytes per token. The cached two-token gate remains exact but records `0.003271` tok/s with zero expert-cache hits because the current capacities do not retain the working set. These are baseline measurements, not a target claim. FP8 full-model population was stopped with no result. Full CUDA final-logit quality, MTP, paged KV, asynchronous residency, calibrated/native FP4, and optimized end-to-end execution remain open.

## Current live snapshot -- 2026-08-15

- FP4 direction: native NVFP4 scaled-GEMM and `.pn4`/`.pgu` sidecars are implemented and pushed in `1db2e0a`. The latest selected NVFP4/layer/cache group is `17 passed`; the first full 78-layer gate completed but is rejected as a default because it changed the final token and measured `0.005469` decode tok/s versus `0.009696` for the paired resident-trunk BF16 control.

- Local source/output: `build-glm5x-full-source` -> `build-glm5x-full-k3x`; the three disjoint workers finished and no conversion restart is required.
- Materialization: `282/282` `.k3x` artifacts and `282/282` atomic source-deleted markers are present; no active conversion partials remain. The final bundle index contains `59,585` tensors and `19,456` complete experts.
- Storage: the source manifest totals `1,506,659,919,872` bytes. The raw-BF16 artifact set is therefore approximately `1.507 TB` decimal before any future derived/quantized copy. Free-space headroom must be checked before any second representation is created.
- Full-gate result: exact cold one-token run took `306.933 s` prefill plus `302.688 s` decode, with `79.763 GB` of logical artifact reads per token and `8.083 GB` peak allocated VRAM. The cached two-token run took `303.585 s` prefill and `611.386 s` decode, with zero expert-cache hits because the configured cache capacity evicted the working set.
- Current bottleneck: the exact reference is still limited by storage/reload, while the first NVFP4 full gate exposes additional sidecar I/O/native FP4 overhead and final-token divergence. Stable per-layer admission now reduces repeated H2D in a bounded real-sidecar trace, but the full runtime still lacks pooled asynchronous layer-window H2D and a natural-router full-model residency gate.
- New host packed tier: `--expert-packed-host-cache-bytes` is implemented as an opt-in bounded RAM LRU for validated `.pi4/.pf8/.pm4/.pn4/.pgu` payloads. A real 16-sidecar probe improved `1.715 s` first admission to `0.282 s` on reuse with `629,145,728` resident bytes. A 40 GiB host tier combined with a 40 GiB trunk tier reached approximately `72 GiB` WSL RSS without a completed full gate, so large capacities are unsafe without a joint memory budget.
- Reduced-routing/proxy gate: the real layer-10 four-token shared Top-4 proxy measured `5.043440291978186 s` versus natural Top-8 `12.43729756900575 s`, but relative L2 drift was `0.8120684623718262`; the proxy is default-off and does not count toward the TPS target.
- Verification: the latest complete WSL Python suite passed `354 passed, 124 skipped` in `79.53 s`; the focused packed-cache/benchmark regression passed `8` selected tests. These are current local results; no full-model speed claim changed.
- GitHub: public head `bb85223` contains the native NVFP4 path, full-gate measurements, and documentation. The latest local verification is `341 passed, 124 skipped`; the recurring red `correctness / Linux (push)` notification is stale run `31795400168` on `b94c8b8`, where 50 historical evidence files were absent; it is not an active failure on `main`.
- Public PR `#5` (`agent/nvfp4-grouped-cache`) contains the grouped NVFP4, protected-cache, host packed-tier, benchmark-repair, and documentation updates. The current PR head has green Linux correctness and C++/Python CodeQL checks; the PR remains draft and unmerged.
- Dependency status: Dependabot update PRs `#1`--`#4` are closed and no open Dependabot PR exists. The repository security-alert endpoints are disabled (`403`/`404`), so the visible alarm cannot be converted into a verified CVE count.

## Completed

- Created the independent repository at `C:\Users\jolib\Documents\project-glm5x`.
- Migrated storage, converter, reference, runtime, test, and benchmark source directories without K3 official weight artifacts.
- Deleted the six verified official K3 artifact directories from the old K3X worktree. Synthetic fixtures remain.
- Added GLM-5.x descriptor validation and the `glm5x-convert` CLI wrapper.
- Added a tiny synthetic GLM5X reference fixture covering recurrent state, Top-K routing, and greedy generation parity.
- Added the GLM5X architecture/design and bootstrap plan documents.
- Added `TurboQuantConfig`, `QuantizedVector`, and `TurboQuantKVCache` with Hadamard rotation, 2/2.5/3/3.5/4/6/8/16-bit schedules, asymmetric K/V settings, incremental attention, and logical 1M-token capacity estimation.
- Added six focused TurboQuant correctness/capacity tests.
- Added GLM-5.2 descriptor fields for MoE intermediate width, DSA index shape, MTP sharing, and maximum position length.
- Added `GLM5XTensorManifest` validation for safetensors weight maps, shard names, tensor count, source byte totals, and official indexer role resolution across `full/shared` layers.
- Added header-only safetensors inspection and manifest parity validation without materializing tensor payloads.
- Added `glm5x-convert convert-shard`, bounded raw BF16 extent writing, sidecar tensor names, and BF16 reader metadata support; the first real shard round-trips through Python and C++ readers.
- Added source/config-fingerprinted GLM shard resume ledgers with canonical extent/length/source-CRC/partial-CRC validation and crash-safe finalization recovery.
- Added complete same-shard raw-BF16 expert-role directory records and `glm5x-convert convert-shards`, which converts manifest shards as independently restartable artifacts and verifies completed outputs on retry.
- Updated the portable C++ reader to accept only validated raw-BF16 `EXPT` links in addition to native MXFP4 links. The storage-slice loader remains MXFP4-only, so this is a staging-format gate rather than BF16 execution.
- Added `glm5x-convert assemble-experts`, a copy-free bundle index that joins complete expert role triples across finalized shard artifacts and records exact artifact-relative extents and CRCs.
- Added `GLM5XExpertBundle`, which rechecks artifact identities and role extents before returning exact BF16 bytes. Layer 10 expert 0 from the second real shard matches all three source roles byte-for-byte.
- Added `k3x::load_glm5x_bf16_expert` and a C++ real-artifact gate. It finds GLM role IDs across multiple readers, validates released dimensions and CRC32C, and returns three host payload vectors.
- Added `k3x_cuda_glm5x_real_expert_bench`, which feeds one exact real expert into the resident CUDA dense SiTU path and compares output against the CPU reference.
- Added a dense BF16 `resident_grid` backend path for real GLM experts, with candidate-token batching, resident-table admission, one activation upload, and CPU-parity regression coverage.
- Extended the real-expert benchmark with `--tokens`; 8 real experts over 4 candidate tokens now run through the BF16 grid while FP32 keeps the scalar numerical reference path.
- Added direct `RawBf16MlpView` admission so selected `.k3x` BF16 expert bytes do not pass through FP32 staging; the real probe decodes only the last expert for CPU comparison.
- Added cublasLt pointer-array batching for multi-expert raw BF16 grids, reducing three projection phases to three batched GEMM calls plus one SiTU launch; the single-expert path remains scalar.
- Added an opt-in BF16-output raw grid path that keeps projection intermediates and final device output in BF16, with a separate BF16 SiTU kernel; FP32 output remains the default reference path.
- Added an optional raw-grid cublasLt workspace budget and CLI `--workspace-bytes`; zero remains the default and the setting is shape-sensitive.
- Added `raw_bf16_situ_mlp_grid_packed` so expert-major callers can provide per-expert candidate slabs without broadcasting one input block to every expert; common-input behavior remains unchanged.
- Added model-neutral `ExpertMajorPackedPlan` preparation from token hidden states and route assignments, retaining explicit token/router-slot/contribution metadata for future GLM scheduler integration.
- Added stable ragged `ExpertMajorPackedBatch` bucketing by assignment count, retaining source group indices and rejecting malformed slab lengths before CUDA dispatch.
- Added exact `scatter_expert_major_outputs` contribution accumulation from group-order expert slabs back to token-major outputs with shape validation.
- Added `--input-mode sparse-packed` to the real-shard probe so the packed raw grid can be measured on real BF16 payloads with an explicitly deterministic two-token assignment pattern.
- Added and measured `k3x_cuda_glm5x_moe_bench` on the real RTX 5080 at GLM-5.2 expert dimensions.
- Added an expert-major candidate-token benchmark mode for 1/2/4/8 tokens.
- Added resident exact MXFP4 reuse to the CUDA expert-major batch backend and allowed resident weights in the CLI validation contract.
- Added opt-in resident BF16 dequantized expert-grid execution through cublasLt, with native MXFP4 fallback when resident capacity is insufficient.
- Added `GLM5XDSAConfig`, `GLM5XDSAIndexer`, and `GLM5XDSAState`, connecting descriptor index metadata and explicit query/key projections to compressed KV blocks, exact top-k refresh, and a separately marked stale fast refresh cadence.
- Added `GLM5XOfficialDSAIndexer` with official-shaped `wq_b/wk/k_norm/weights_proj` tensors, interleaved/non-interleaved indexer RoPE, ReLU score aggregation, causal masking, and Top-K reference parity. Its safetensors loader reads only the five indexer tensors needed for a selected layer.
- Added `GLM5XLayer10MoEReference` with official sigmoid routing, exact Top-8 normalization/routed scale, shared SwiGLU, and lazy exact raw-BF16 expert loading from the copy-free bundle.
- Added an opt-in reference expert-major MoE execution mode, propagated through decoder-layer and model factories. The loop reference remains the default; parity and real RTX 5080 trade-offs are recorded in `BENCHMARKS.md`.
- Added `tools/benchmark_glm5x_reference.py` for explicit-token full-bundle prefill/TTFT/decode measurement with strict/lazy admission, cache switches, execution-mode selection, and CUDA peak-memory telemetry. It has no full-checkpoint result yet.
- Added opt-in parallel exact expert payload reads through `--expert-load-workers`; the serial default remains unchanged, and the focused MoE/model-reference regression passes.
- Added a bounded C++ deadline worker pool through `--l2-expert-workers`, with eight workers as the `RuntimeSession` deadline default and per-key in-flight host-cache deduplication. Serial direct-scheduler behavior remains available.
- Added an opt-in bounded exact host payload cache keyed by layer/expert, with hit/miss/eviction telemetry in the full-bundle benchmark. Capacity `0` remains the default until full-model traffic is measured.
- Added an opt-in bounded decoded expert tensor cache on the target device, shared across layer objects and token forwards. Capacity `0` remains the default; the full monitor's cached comparison uses 4 GiB.
- Added `GLM5XMLAReference`, incremental compressed MLA state, official causal DSA state, and `GLM5XDecoderLayerReference` with full-vs-incremental parity and bundle-backed attention/indexer/norm/MoE construction.
- Added `GLM5XDecoderModelReference` with per-layer MLA/DSA state, final RMSNorm, LM-head logits, prompt prefill, one-token continuation, and greedy generation parity over a synthetic two-layer graph.
- Reused one root-verified `GLM5XExpertBundle` reader across the layer loader and lazy MoE closure; the real five-shard construction smoke fell from 491.483777 s to 250.637263 s.
- Connected `ExpertMajorPackedPlan` bucketing and contribution scatter to `CudaBackend::raw_bf16_situ_mlp_expert_major`; the real bounded probe measured 1,380,314 ns warm median/block for a deterministic 8-group/10-assignment/2-token route.
- Added `learned-expert-major` to load the official GLM router and correction bias, select the actual Top-8 expert union, and expose an explicit resident-byte budget. The bounded two-token route selected 15 experts and measured 1,905,668 ns warm median/block with 0.0866% maximum CPU-relative difference.
- Added `learned-moe-layer` to execute the same natural routed union plus the actual layer shared expert through raw-BF16 CUDA, with combined CPU parity and separate shared-payload/residency telemetry.
- Added `GLM5XACT` v1 Python/C++ BF16 activation artifacts with atomic writing, fixed headers, shape/extent/CRC validation, and optional benchmark input/expected-output comparison.
- Added a narrow pytest boundary for historical K3X evidence files absent from the GLM5X repository; new GLM5X tests remain active. GitHub Actions correctness and CodeQL pass on `f07d78c`.
- Added an explicit GLM SiLU gated-MoE activation path in CPU and CUDA while retaining the inherited SiTU path for legacy callers. The bounded layer-10 learned-MoE probe now has route parity and a BF16-boundary expected-output comparison.
- Added `GLM5XDecoderModelReference.from_layer_loader` so full-model reference state can request one validated layer at a time without retaining all layer objects in RAM; eager construction remains available for parity.
- Added `GLM5XDecoderLayerReference.bundle_layer_loader`, which reuses one root-verified bundle reader and tensor-reference map across layer requests while keeping selected expert payloads lazy.
- Added opt-in `K3XReader`/`GLM5XExpertBundle` lazy admission. Directory metadata is checked immediately, selected tensor CRCs are verified on first read, and strict eager payload/root verification remains the default correctness mode.
- Added an opt-in bounded LRU for validated reference trunk layers. Capacity zero preserves strict layer-at-a-time loading; positive capacity removes repeated layer-provider calls without changing logits. Expert payload residency remains a separate policy.
- Added `GLM5XDenseMlpReference` and explicit `mlp_type="dense"` bundle loading for GLM-5.2's first three dense layers. The path uses official SwiGLU, preserves the decoder output schema, and reports empty routing with zero expert loads.
- Added `GLM5XDecoderModelReference.from_bundle`, which resolves official dense/sparse MLP types and shared-indexer sources while keeping decoder layers provider-owned. Bounded probes can override only missing head tensors; real decoder-layer payloads remain exact and lazy.
- Added a reference-only native MXFP4 encoder with deterministic E8M0 `max_abs` and `mse` scale modes, chunked packing, and BF16/FP32 shape/finite-value validation. It is not integrated into the converter because the first real layer-10 quality probe is still too lossy for a default path.

## In progress

- Load all real GLM layers into the model-level reference state contract and add MTP state once the required tensor roles are available. The configuration-driven provider now exists; complete payload coverage is the remaining gate.
- Extend the layer-10 exact q-residual/MLA/DSA boundary to the remaining layers, final logits, and MTP state.
- Rename user-facing runtime and benchmark commands where that does not break the inherited storage ABI.
- Connect the exact layer-10 reference's MLA/DSA hidden state to the learned router/expert-major CUDA grid while retaining strict natural Top-8 verification.
- Validate nonzero full-layer routing/MLA/DSA parity and measure VRAM-bank pressure before considering BF16 grid mode a default.
- Replace host float decode plus conversion with direct raw-BF16/tensor-core storage views where quality permits.
- Measure whether BF16-output resident grids remain inside the chosen quality budget on nonzero full-layer GLM data.
- Measure the new CUDA bucket loop with exact nonzero GLM router assignments, then add pinned asynchronous staging after full-layer parity is established.
- Calibrate expert quantization with outlier residuals or mixed precision; direct BF16-to-MXFP4 remains experimental after the measured real-layer quality gate.
- Real five-shard layer-10 reference smoke at `a2d6b6d`: bundle open/root verification 250.637263 s, cold two-token forward 5.969859 s, cached repeat 0.057331 s, 16 unique selected experts, cached output max difference 0.0. These are layer/storage reference timings, not tok/s.
- Latest bounded real layer-10 learned-MoE result at `f07d78c`: two tokens, 16 routed experts plus one shared expert, 2,091,698 ns warm median per MoE sublayer block, 1,283,457,024 resident expert bytes, zero warm H2D, GPU-versus-CPU relative error `0.000452667358331`, and BF16-rounded expected-artifact relative error `0.00152439018711`. Route IDs and contributions match between Python and C++.
- Latest device-accumulate rerun at `1514d11`: the same two-token learned-MoE boundary measured three baseline medians `2,198,145`, `2,736,064`, `2,492,351 ns` and three device-accumulate medians `1,991,721`, `1,981,629`, `2,446,610 ns`. Median-of-runs improved from `2,492,351 ns` to `1,991,721 ns` (about 20.1%) with unchanged GPU/CPU relative error `0.000571510172449`; this remains an opt-in bounded sublayer result, not tok/s.

## Known blockers

- Full GLM-5.2 local materialization and bundle assembly are complete. The exact full-model gate is now measured, but its `0.0033 tok/s` baseline exposes the current reload-everything bottleneck; this is not a production-speed result.
- The reference MXFP4 encoder is not a production weight path yet. A real layer-10 expert's three projections compressed to 26.56% of BF16 storage, while FFN relative L2 error remained 19.86% (`max_abs`) or 19.07% (`mse`).
- No Cloud Run conversion or paid resource has been authorized or attempted. The active stream is local and uses resumable HTTP-range downloads with one-shard-at-a-time conversion.
- Current Dependabot state: PRs 1-4 were closed after their setup-python, checkout, numpy, and setuptools bumps were integrated and verified on `a3fb8a8`; the repository Dependabot security-alert endpoint is disabled, so no CVE alert was verified.
- Historical Dependabot note: the original PR branches first failed with 50 `FileNotFoundError` cases from absent migrated K3X evidence; their rebased replacement checks were green before the exact updates were integrated on main and the PRs were closed. Repository Dependabot and vulnerability-alert APIs are disabled, so no CVE alert is confirmed.
- The migrated C++ runtime still has K3-oriented names and graph assumptions in several files.
- CUDA build and bounded RTX 5080 kernel measurements are validated in WSL; native Linux end-to-end throughput has not been measured. The current WSL exact full-model baseline is storage-bound and must not be presented as optimized performance.
- The TurboQuant implementation is CPU/reference only; it does not yet contain a packed CUDA kernel or full PolarQuant/QJL production path.
- 600k/1M capacity is a formula estimate only until a real GLM-5.2 DSA state is allocated and restored.
- The Python expert-major experiment temporarily stacks selected real expert weights and added about 1.97 GB peak VRAM in the four-token probe; it is not safe as a default 16 GB runtime policy until a resident-weight-aware C++ path replaces the temporary stack.

## Hardware assumptions

- CPU: AMD Ryzen 7 9800X3D.
- GPU: NVIDIA RTX 5080 16 GB.
- RAM: 96 GB DDR5-4200.
- Storage: Solidigm P44 Pro 2 TB NVMe.
- Preferred execution environment: native Linux.

## Latest verified state

- Focused GLM descriptor, manifest, CLI, toy/reference graph, TurboQuant, DSA, official indexer, shard-converter, bundle loader, exact MLA/DSA layer/model reference, dense first-three-layer path, configuration-driven bundle factory, lazy layer/bundle admission, bounded trunk-layer caching, device staging, source-deletion resume, expert-major reference mode, reduced-routing/proxy controls, and multi-shard tests: the complete WSL Python suite is `333 passed, 124 skipped` (`77.56 s`). Host CTest remains green. CUDA layer/model parity tests pass in WSL; the Windows Python interpreter still lacks pytest.
- CUDA CMake build: successful in WSL with CUDA 13.3 and RTX 5080 compute capability 12.0.
- CTest: 27/27 tests passed in WSL, including `glm5x_activation`.
- CPU WSL CTest: 15/15 tests passed after the expert-major CUDA API change.
- Focused CUDA regression now includes a nonzero two-expert/two-token BF16 resident-grid parity case; the 26-test CTest count is unchanged because it extends `cuda_dense`.
- Real-shard C++ metadata gate: five downloaded GLM probe artifacts passed the reader gates; the assembled bundle contains 888 tensors, 277 complete expert groups, and one incomplete group.
- Cross-shard bundle gate: five probe artifacts indexed without copying payload bytes; layer 10 is complete and the lazy reference selected 15 unique experts for two random tokens.
- Real BF16 payload gate: layer 10 expert 0 returned three 25,165,824-byte roles that matched source safetensors bytes exactly; a tampered offset is rejected.
- C++ host loader gate: 75,497,472 bytes loaded and CRC-checked in 465,087,758 ns under WSL; no CUDA execution or token generation.
- Real CUDA expert gate: FP32 resident rerun warm median 271,493 ns with 150,994,944 resident bytes and CPU max absolute error `8.38190317154e-09`; cached BF16-rounded rerun warm median 236,593 ns with 75,497,472 resident bytes and 0.1828% max relative error. These are one-expert FFN records only.
- Multi-expert pressure gate: 8 real layer-10 experts measured 1,854,140 ns sequential warm median with BF16/604 MB resident and zero warm H2D; FP32 exceeded the 1 GiB budget and measured 13,153,048 ns with 3.02 GB warm H2D.
- Real expert-major gate: 8 real layer-10 experts over 4 candidate tokens measured 1,758,739 ns BF16 grid warm median (approximately 439,685 ns per candidate), 603,979,776 resident bytes, zero warm H2D, and 0.1351% maximum relative CPU error.
- Direct raw-BF16 gate: the same 8-expert/4-token probe measured 1,648,927 ns warm median (approximately 412,232 ns per candidate), 135,877,327 ns cold, 603,979,776 resident bytes, zero warm H2D, and unchanged 0.1351% maximum relative CPU error.
- Pointer-array gate: the same 8-expert/4-token probe measured 1,065,026 ns warm median (approximately 266,257 ns per candidate), 153,395,924 ns cold, 603,979,776 resident bytes, zero warm H2D, four grid launches/call, and 0.1359% maximum relative CPU error.
- BF16-output gate: paired 8-expert/4-token real-shard runs measured 1,034,950 ns warm median versus 1,091,122 ns with FP32 output; maximum CPU-relative error was 0.3167% versus 0.1359%. This is an opt-in bounded FFN experiment, not a model tok/s result.
- cublasLt workspace gate: with the same shape, 64 MiB measured 967,790 ns FP32-output versus 994,529 ns at zero workspace; the same budget measured 1,080,469 ns for BF16 output versus 1,034,950 ns at zero. Keep the knob runtime-selectable and default-off.
- Packed-grid gate: `cuda_dense` now verifies two experts receiving different one-token slabs, CPU parity, and packed activation/output byte accounting. This is a correctness boundary only; no ragged real-router throughput has been measured.
- Packed-plan gate: `test_expert_major` verifies stable per-expert slab order for one-token and two-token assignment groups. No GLM router or end-to-end throughput is connected yet.
- Ragged packed-batch gate: `test_expert_major` verifies stable one-/two-assignment buckets, repeated-shape slab concatenation, source group indices, and malformed payload rejection. The CUDA bucket loop consumes this contract for both deterministic and learned route plans.
- Contribution-scatter gate: `test_expert_major` verifies weighted group-output accumulation in token order and short-output rejection. The CUDA path uses the same helper after raw-BF16 bucket execution; full-layer GLM parity remains open.
- Learned-router real-shard gate: official layer-10 router tensors selected 15 experts/16 assignments for two tokens and 29 experts/32 assignments for four tokens. FP32-output warm medians were 1,905,668 ns/block and 3,757,986 ns/block, respectively; no full-layer or end-to-end tok/s was measured.
- Learned-MoE-sublayer real-shard gate: two tokens selected 15 routed experts plus one shared expert and measured 2,155,188 ns/block with 1,207,959,552 resident bytes and 0.0586% maximum CPU-relative difference. Four tokens selected 29 routed experts plus one shared expert and measured 3,968,243 ns/block with 2,264,924,160 resident bytes and 0.0430% relative difference. BF16-output was slower at 2,374,827 ns and had 0.1119% relative difference, so FP32 remains default. This is not a full-layer or end-to-end tok/s result.
- Expert-major real-shard gate: five probe artifacts, deterministic 8-group/10-assignment/2-token route, 1,380,314 ns warm median/block, 168,543,514 ns cold latency, 603,979,776 resident bytes, zero warm weight H2D, and 0.1471% maximum CPU-relative difference. Common was 1,651,193 ns and sparse-packed was 1,631,127 ns in paired samples. This is not learned routing or end-to-end tok/s.
- Sparse-packed probe: deterministic 8-expert/2-token pattern measured 965,550 ns/block versus 1,040,559 ns for the common-input rerun; this is not learned routing or end-to-end throughput.
- Latest rerun at `1f43e1a`: common 927,744 ns/block versus sparse-packed 939,149 ns/block, so sparse-packed was about 1.2% slower in this sample. The direction changed from the earlier sample; no stable packed speedup is assumed.
- Historical K3X evidence checks are explicitly skipped when their absent `results/` artifacts are not shipped; the Linux workflow still builds C++, runs CTest, and runs all new GLM5X tests. The Windows local environment still lacks the Linux-built executable for one cross-language test.
- Public Linux correctness workflow `31812923197` and CodeQL workflow `31812923191` both passed for implementation commit `f07d78c`. The Linux job completed in about 3 minutes 51 seconds and the CodeQL jobs completed in about 3 minutes 20 seconds. The workflows still emit non-failing Node 20 and CodeQL v3 deprecation annotations.
- Public Linux correctness workflow `31822433552` and CodeQL workflow `31822433677` both passed for verified HEAD `4f9c3c2`. Linux completed in `3m13s`; CodeQL completed in `4m28s` for C++ and `2m15s` for Python. No failure or timeout occurred on this head.
- Public Linux correctness workflow `31824430721` and CodeQL workflow `31824430714` both passed for implementation/docs HEAD `0040791`. Linux completed in `2m43s`; CodeQL completed in `3m39s` for C++ and `2m29s` for Python. The CodeQL overlay-base message was a non-failing fallback annotation, not a failed check.
- Public Linux correctness workflow `31824846842` and CodeQL workflow `31824846833` both passed for docs-only HEAD `31fe66f`. Linux completed successfully; CodeQL completed in `2m59s` for C++ and `2m19s` for Python. The overlay-base annotation remained non-failing.
- Public Linux correctness workflow `31825623428` and CodeQL workflow `31825623418` both passed for `761b881`. Linux completed in `2m53s`; CodeQL completed successfully with no failing job. The cache/eviction test was included in the Python step.
- No end-to-end GLM decode tok/s or quality result exists yet.
- The 2026-08-15 prepared-bucket cache and one-token shared-dispatch experiment was reverted after paired RTX 5080 medians were approximately 3.4% slower for token-1 and 1.1% slower for token-2 than the `f07d78c` baseline. The next performance experiment is device-side expert-output accumulation; no optimization is accepted from theory alone.
- The device-side ragged expert accumulation experiment now passes direct, varied-bucket, host, and CUDA parity. It remains runtime-switchable and default-off because the three-run latency spread is material and the full layer, final logits, and quality path are not yet connected.
- Bounded GLM-5.2-shaped CUDA result: 8 experts/1 token median 2,662,772 ns; 8 experts/4 tokens 1,344,816 ns per candidate token; maximum absolute error 0.
- Resident expert-major batch result: 8 groups x 4 candidates, 1,641,591 ns/candidate token, cold weight H2D 160,432,128 bytes and warm weight H2D 0 bytes.
- Resident BF16 grid result: 8 experts x 4 candidates, 2,582,527 ns/block versus native 5,394,131 ns/block; BF16 resident weight bytes 603,979,776 versus native 160,432,128; maximum absolute error 0 on the zero-weight fixture.
- Latest rerun: native zero-pattern grid 5,510,632 ns/block; BF16 zero-pattern grid 4,386,083 ns/block; nonzero BF16 grid 4,044,675 ns/block with 0.9505% maximum relative difference versus the native GPU reference. These remain bounded synthetic measurements only.
- DSA reference capacity estimate: 201,637,504 bytes at 600,000 tokens and 336,062,512 bytes at 1,000,000 tokens for BF16 index keys plus K6/V4 KV with index width 4096; this is formula-only.
- Official manifest metadata probe: 59,585 tensors across 282 shards; shared indexer layer mapping is resolved without opening payloads.
- First official shard header probe: 35/35 names matched `model-00001-of-00282.safetensors`; representative indexer tensors are BF16 with `wk=(128,6144)`, `wq_b=(4096,2048)`, and `weights_proj=(32,6144)`.
- First bounded artifact: 35 BF16 tensors, 78 layer records, Python reader checks green, and WSL C++ `test_reader` exit 0; no full model loaded.
- Real indexer payload gate: loaded only five layer-0 indexer tensors from the 5.3 GB first shard (`wq_b=(4096,2048)`, `wk=(128,6144)`, `weights_proj=(32,6144)`, and two 128-element LayerNorm vectors) and ran causal Top-K on zero activations. This is payload/shape evidence, not model quality or throughput.
- Last known-good implementation HEAD: `f07d78c` (`feat: align GLM MoE activation with SiLU`). WSL host CTest `15/15`, WSL CUDA CTest `27/27`, and the complete WSL Python suite `301 passed, 124 skipped in 71.25 s` are green. Public Linux correctness `31812923197` and CodeQL `31812923191` also passed.
- Last known-good implementation/docs HEAD: `761b881` (`test: cover trunk cache eviction`). WSL host CTest `15/15`, WSL CUDA CTest `27/27`, and the complete WSL Python suite `305 passed, 124 skipped` are green. Public correctness `31825623428` and CodeQL `31825623418` also passed.
- Last known-good implementation/docs HEAD: `fb3aa7d` (`feat: support dense GLM MLP layers`). WSL focused tests passed `3/3`, the complete Python suite passed `306/124`, and public correctness `31826654966` plus CodeQL `31826655082` passed. The Linux job took about 7 minutes 10 seconds because the hosted Python step ran for about 5 minutes 19 seconds; this is a successful run, not a timeout failure.
- Latest verified implementation/docs HEAD: `3a86ca3` (`docs: record bundle factory gate`, including implementation `1f123ca`). WSL model-reference tests passed `4/4`, full Python passed `307/124`, host CTest passed `15/15`, real layer-0 admission measured `4.278880 s` plus `0.036846 s` one-token CPU forward, and public correctness `31828512721` plus CodeQL `31828512789` passed. No full-model tok/s is claimed.
- Next bottleneck: finish the local 282-shard stream, assemble the complete bundle, load all real layers into the model reference, export the exact q-residual/MLA/DSA hidden state into `GLM5XACT`, verify nonzero full-layer parity, then add pinned/asynchronous raw H2D, direct tensor-core selection, MTP, and VRAM-pressure-aware residency. No end-to-end tok/s is claimed until those gates run.

## 2026-08-15 -- Device staging and resumable local materialization

- Added explicit `device="cuda"` staging to the Python reference bundle/model path. CUDA layer and model factory parity tests passed against CPU reference outputs; this is not a complete CUDA decoder.
- Added `convert-shards --delete-source` with an atomic source-deleted marker. A finalized artifact is strict-reader verified before its source shard is removed, and a retry can recover from the marker without redownloading the shard.
- Added `tools/stream_glm5x_checkpoint.py` with public repository metadata discovery, HTTP Range `.part` resume, one-shard conversion, source deletion, and final bundle assembly.
- The local stream has finalized `model-00001-of-00282.k3x` (`5,342,863,616` bytes) and `model-00002-of-00282.k3x` (`5,351,993,600` bytes); the third source shard is downloading. No quality, final-token, or tok/s result exists yet.
- WSL verification after the changes: Python `311 passed, 124 skipped` in `79.98 s`, host CTest `15/15`, and CUDA-only layer/model parity tests passed. Public correctness and CodeQL are green for `6fb2da1`.

## 2026-08-15 -- Public verification of streaming/device boundary

- Commit `6fb2da1` was pushed to public `main`.
- Linux correctness `31831711520` passed in `2m38s`; its C++ build/CTest and Python/cross-language step are green.
- CodeQL `31831711580` passed for both Python and C++ analysis.
- The local stream reached four finalized shards of 282 while CI ran. The stream remains active; no full-model logits or tok/s result exists.

## 2026-08-15 -- Avoid duplicate final payload scanning

- Added a lazy verification switch to bundle assembly. The local stream will use it only after strict per-shard conversion verification and source deletion markers; public CLI assembly remains strict by default.
- The running stream has reached seven finalized shards of 282. The next restart will load the updated stream code; no full bundle or TPS result exists yet.

## 2026-08-15 -- Resume without redownloading completed shards

- `db2cf37` checks for a finalized `.k3x` plus source-deleted marker before downloading. Existing `.part` files for incomplete shards are still resumed with HTTP Range.
- Focused bundle/stream/converter tests passed `7/7`. Public correctness `31833153961` and CodeQL `31833154040` passed for the implementation.
- The running stream was safely restarted from 8/282 completed shards; shard 9's `.part` is retained. No full-model logits or tok/s result exists.

## 2026-08-15 -- Parallel local shard workers

- Commit `6cd4e85` adds `--shard-start`, `--shard-end`, and `--no-assemble` to the stream driver. Three workers now own ranges `10..100`, `101..191`, and `192..281` while the first ten shards remain complete.
- At the first measurement point, 16 artifacts existed. Workers 1 and 2 completed new shards 102/193 at `04:44:14/16` and 103/194 at `04:49:36/37`; worker 0 completed shard 12 at `04:46:52`. The aggregate sample shows download/conversion overlap; sustained throughput is still being measured.
- Public correctness and CodeQL for `6cd4e85` are green. The full bundle, final logits, and tok/s remain open.

## 2026-08-15 -- Exact-read and CI status refresh

- The exact selected-expert loader now groups the three co-located role extents under one artifact open. The focused bundle/reader/layer/model selection passed `20` cases with `4` capability skips; a real layer-10 one-token cold sample measured `2.183734 s` with four readers versus an earlier `2.751867 s` sample. This is a bounded I/O comparison, not an end-to-end tok/s result.
- The opt-in host-quantized FP8 experiment remains default-off. On the real layer-10 probe it was `2.901232 s` cold versus exact `2.751867 s`, with `5.603%` output relative-L2 drift.
- The latest pushed head is `d539551`; Linux correctness `31851436292` and CodeQL `31851436264` both completed successfully. The old red notification run `31795400168` failed because historical `results/b0006..b0024` evidence files were absent on commit `b94c8b8`; it is not a failure of the current `main` head.
- Dependabot PRs 1--4 are closed and the scheduled Dependabot update workflows passed. The repository's security-alert API is disabled (`403`), so no vulnerability count is asserted from the alarm banner.

## 2026-08-15 -- CI, dependency-alert, and full-stream status refresh

- The red `correctness / Linux (push)` notification traced to historical run `31795400168` on commit `b94c8b8`. C++ configure/build/CTest passed; the Python step failed because that old checkout lacked committed historical `results/b0006..b0024` evidence files. Current `main` had successful correctness and CodeQL runs through `b45ba4a`; the new telemetry head `eed4a09` is queued for the same gates.
- Dependabot update PRs `#1`--`#4` are closed. The repository-level Dependabot security-alert and vulnerability-alert APIs are disabled, so the alarm banner cannot be converted into a verified CVE count without changing repository settings.
- Logical storage-read telemetry was added in `eed4a09`. Local verification: CTest `15/15`, focused Python `29 passed, 6 skipped`, full Python `326 passed, 124 skipped`, and targeted Python compilation passed. `storage_read_*` is explicitly not physical NVMe telemetry.
- The three local conversion workers remain alive. Latest probe at approximately `10:16 KST`: `212` artifacts, `211/282` source-deletion markers, `71` shards remaining, no partial artifacts, and approximately `537 GiB` free on `C:`. The observed rate remains about `30--35` shards/hour; conversion is estimated around `12:20--12:45 KST`, followed by assembly and the first CUDA cold/cached reference gate around `12:55--14:15 KST`. This is a scheduling estimate, not a throughput result.
- The latest public gates for `0cdb69d` are green: correctness run `31855870222` completed in `2m32s`; CodeQL run `31855870243` completed with Python analysis in `2m16s` and C++ analysis in `4m04s`. The C++ CodeQL overlay-base message is a non-failing fallback annotation.

## 2026-08-15 -- Exact logits allocation and traffic-boundary update

- `70d7d39` lazily prepares and reuses the exact FP32 LM head. The model-reference regression passed `9/9`; the full Python suite passed `327` with `124` capability skips; host CTest passed `15/15`.
- An RTX 5080 isolated GLM-shaped probe measured `61.629 ms` median for a fresh `(154880, 6144)` BF16-to-FP32 conversion and `3.13 us` median for reuse. The retained FP32 matrix is `3,806,330,880` bytes and is not an end-to-end result.
- `2a9778d` added `PERFORMANCE_MODEL.md`, which records the dimension-derived `31.88 GiB` non-routed BF16 trunk and `74.07 GiB` one-token Top-8 fetch bound. These figures are constraints only; physical NVMe/H2D and TPS remain unmeasured.
- Public correctness `31856761733` and CodeQL `31856761734` passed for `2a9778d`; the C++ CodeQL overlay-base message remains a non-failing fallback annotation.
- The three conversion workers remain alive. At approximately `10:36 KST`, `224` artifacts and `223/282` source-deletion markers were present, with no partial artifacts and approximately `477 GB` free on `C:`. The complete bundle and first real-model CUDA gate remain the next evidence boundary.

## 2026-08-15 -- LM-head steady-state memory update

- `f4fbcc1` now promotes the prepared FP32 LM head to the active model head and releases the BF16 source after first use. Focused model tests passed `5/5`; the complete Python suite passed `327/124`; host CTest remains `15/15`.
- The isolated RTX 5080 conversion timing is unchanged at `61.629 ms` median for first-use conversion and `3.13 us` median for reuse. The first-use peak includes BF16 plus FP32; steady-state retains the FP32 matrix only.
- Public Linux correctness `31857433809` and CodeQL `31857433741` passed for `f4fbcc1`. The C++ CodeQL overlay-base annotation is non-failing.
- At approximately `10:51 KST`, the three workers had `232/282` source-deletion markers, `232` finalized artifacts, no partial artifacts, and approximately `443 GB` free on `C:`. Full bundle assembly and the real CUDA gate remain pending.

## 2026-08-15 -- Full 282-shard exact CUDA baseline and monitor repair

- All `282/282` source-deletion markers and `.k3x` artifacts completed. Lazy bundle assembly succeeded with `282` artifacts, `59,585` tensors, and `19,456` complete experts; no payload copy was performed during assembly.
- The first exact full-model cold gate used the official GLM-5.2 bundle, `78` layers, natural Top-16 routing, BF16 payloads, `EXPERT_LOAD_WORKERS=16`, and zero host/device expert cache. It measured prefill `306.93307032898883 s` (`0.003258039281750062 tok/s`), decode `302.68789011100307 s` (`0.0033037330949489767 tok/s`), TTFT `609.6209604399919 s`, logical read bytes `79,763,152,896` per token, and peak allocated VRAM `8,083,474,944` bytes.
- The cached two-token gate used 8 GiB host and 4 GiB device expert caches. It measured prefill `303.5853812530113 s`, decode `611.385999924998 s` (`0.0032712558027912816 tok/s`), `0.0` expert-cache hit rate, `1,800` misses, `1,687/1,744` evictions, and `159,526,305,792` decode read bytes. The cache capacity is insufficient for this working set and does not retain the trunk, so no speedup is claimed.
- Root cause of the repeated local gate error: `tools/monitor_glm5x_full_gate.sh` used Bash-only syntax without a shebang and was invoked through `sh`, producing `line 27: syntax error near unexpected token 'then'`. The script is now POSIX-`sh` compatible, declares its interpreter, and selects `/home/jolib/.venvs/k3x-m1/bin/python` (override with `K3X_PYTHON`) before falling back to `python3`/`python`; `sh -n` and `bash -n` pass.
- Public verification after the script repair is green at `5dfe036`: correctness `31863769799` completed in about `3m02s`, and CodeQL `31863769798` completed in about `3m46s`; no cloud or paid resource was used.
- Next bottleneck: implement an exact resident non-expert trunk policy and layer-aware pinned/asynchronous staging so the measured `79.8 GB/token` logical reload is amortized. Re-run exact parity and the full gate before enabling any mixed precision, proxy, adaptive Top-K, or speculative mode.

## 2026-08-15 -- INT4 expert residency gate

- Implemented: CUDA TinyGEMM INT4 weight wrapper, GPU-side group quantization/packing, INT4 expert/shared projection support, device-cache byte accounting, expert-major safety fallback, CLI precision flag, and bundle grouped expert reads.
- Verification: WSL focused INT4/bundle/layer/model/MoE/schema tests `33 passed, 6 skipped`; targeted `py_compile` passed.
- Measured rejection: full-model cold `trunk=int4 + expert=int4` with no packed cache measured `0.002830204968837129` decode tok/s, `353.3312996799941 s` decode, `45,298,483,200` logical expert bytes/token, and `17,341,184,512` peak allocated VRAM bytes. This is not a usable 16 GiB default.
- Measured opportunity: real layer-10 four-token INT4 MoE with a 2 GiB packed device cache measured `13.281714103999548 s` on first call and `0.00977079599397257 s` on an identical cached call (`31` hits, `31` misses, `621,674,496` resident bytes). This is a sublayer-only result.
- In progress: design a storage-side packed expert artifact and an exact route-stable residency policy. The current source bundle still forces `45.3 GB/token` logical expert reads, so 10--20 tok/s is not yet physically plausible on the target NVMe/PCIe path.
- Known blocker: the long 2-token 8 GiB packed-cache full-model run was stopped before completion because initial full-bundle materialization exceeded the requested turnaround; no partial result is treated as a benchmark.
- Last known-good tests: the focused suite above; no new public push was made for the uncommitted INT4 work.

## 2026-08-15 -- Fingerprinted packed expert sidecar

- Public commit: `7c3f539` (`perf: add fingerprinted packed expert sidecar`), pushed to `origin/main`.
- Implemented: optional CUDA-only `.pi4` sidecars for packed INT4 gate/up/down projections, source-layout fingerprints, per-role CRC32C validation, atomic writes, model/CLI wiring, and benchmark telemetry.
- Measured bounded result: layer-10 sidecar population took `18.112762928998563 s`; a fresh layer instance reused 31 sidecars in `1.152440828998806 s` with `0` bundle-read calls and `0` bundle-read bytes. Route equality remained `true`.
- Verification: full WSL Python suite `332 passed, 124 skipped` in `76.14 s`; focused sidecar integration `32 passed, 6 skipped`; changed-module `py_compile` passed.
- Current limitation: this reduces repeat source-bundle I/O for selected INT4 experts but does not lower the measured full-model cold bound until the selected experts are actually reused. It is not enabled by default and no 10--20 tok/s claim is made.
- Next bottleneck: implement and measure exact multi-layer route-stable residency/trunk staging, then run one deliberate 78-layer full-model gate with quality and physical I/O telemetry.

## 2026-08-15 -- Reduced-routing and shared proxy quality gate

- Implemented: explicit `routing_top_k`, `proxy_mode`, and `proxy_top_k` controls in the reference model/layer factories and benchmark CLI. `proxy_mode="none"` is exact; `proxy_mode="shared"` evaluates only the requested routed subset and uses a documented shared-expert approximation for dropped mass.
- Verification: focused layer/model/schema coverage passed `24 passed, 6 skipped`; the complete WSL Python suite passed `333 passed, 124 skipped` in `77.56 s`; changed Python modules compiled successfully.
- Measured rejection: real layer-10 four-token natural Top-8 was `12.43729756900575 s` with 31 unique experts. Shared Top-4 was `5.043440291978186 s` with 16 unique experts, but relative L2 drift was `0.8120684623718262` and maximum absolute error was `0.01171150803565979`.
- Status: proxy and reduced routing remain experimental/default-off. The active performance blocker remains exact storage-side expert/trunk residency; no full-model TPS or quality claim changed.

## 2026-08-15 -- Fingerprinted FP8 expert sidecar

- Implemented: the fingerprint-bound sidecar now supports `.pi4` INT4 and `.pf8` row-scaled E4M3 FP8 role payloads with atomic writes, source-digest binding, and CRC checks. The layer/model reference and real-layer benchmark can select FP8 explicitly.
- Verification: packed-cache and MoE regressions passed `10/10`; changed modules compile successfully. The complete suite passed `334 passed, 124 skipped` in `76.00 s` after this extension.
- Measured bounded result: first population of 31 FP8 sidecars took `21.40642180899158 s`; a fresh process reused them in `4.820426017016871 s` versus BF16 `11.759381022013258 s`, with identical routes and `0.05696592479944229` relative L2 drift.
- Status: FP8 sidecar reuse is experimental/default-off. It is a warm layer result, not full-model tok/s or coding-quality evidence.

## 2026-08-15 -- FP4 pivot and MXFP4 sidecar gate

- Stopped the abandoned full-model FP8 sidecar population. Its partial `.pf8` artifacts are preserved for inspection, but no full-model FP8 result is recorded.
- Added experimental `.pm4` fingerprinted sidecars and `expert_precision=mxfp4`. The current reference path packs E2M1/E8M0 MXFP4, then decodes to BF16 for execution; BF16 remains the exact/default path.
- Real layer-10 one-token gate: eight routed experts occupied `160,440,156` sidecar bytes (`26.56%` of corresponding BF16 role bytes), route IDs matched, and MXFP4-vs-BF16 relative L2 error was `0.16359105706214905` with max absolute error `0.001750946044921875`. Fresh sidecar decode took `17.867729659978068 s` versus `2.79652249100036 s` BF16 because the native FP4 kernel is not connected yet.
- Status: FP4 storage plumbing is implemented and tested, but uncalibrated MXFP4 is not promoted. Next is calibrated residual metadata plus native RTX 5080 FP4 execution, followed by one fresh full-model quality/traffic gate.

## 2026-08-15 -- Luna parallel NVFP4/cache follow-up

- Implemented: `reference/glm5x_ref/nvfp4_batched.py`, the `grouped_nvfp4` layer/model switch, `--nvfp4-grouped`, and focused parity tests. The grouped API is an experimental CUDA kernel boundary and is not silently enabled for the full model.
- Implemented: explicit protected-key tracking for `GLM5XExpertTensorCache(policy="layer_balanced")`. A RED regression reproduced protected-entry eviction; the focused suite is green after the fix.
- Measured: grouped NVFP4 projection samples were variable (`0.783x` to `2.138x` versus sequential depending on expert count). The real sidecar admission/H2D probe measured roughly `89.505--102.677 ms` GPU-event time per expert, while gate/up projection itself was about `0.131 ms`; transfer/residency is the current dominant boundary.
- Verification complete: full WSL Python regression passed `352 passed, 124 skipped` in `75.19 s` after the focused `38 passed` result.
- Known blocker: the current full NVFP4 gate remains `0.0144835562212668` decode tok/s with final-token divergence from exact BF16. Ten tok/s is not achieved; the next concrete task is pinned asynchronous staging plus route-stable layer-window residency and separate sidecar/H2D telemetry.

## 2026-08-15 -- Verified packed-sidecar host tier

- Implemented `GLM5XPackedExpertCache(host_cache_capacity_bytes=...)` and the CLI flag `--expert-packed-host-cache-bytes`. The tier retains only validated metadata/payload pairs in a bounded RAM LRU; capacity `0` is the exact previous behavior, and decoded CUDA residency remains a separate cache.
- TDD evidence: the host-reuse regression failed on the missing constructor option, then passed after implementation. The full WSL Python suite passed `354 passed, 124 skipped` in `79.53 s`; changed modules compile successfully.
- Real sidecar probe: 16 `.pgu` entries, 16 readers, RTX 5080. Host cache disabled measured `2.022659/1.954926 s` for two passes; 2 GiB host cache measured `1.714678/0.281999 s`, with 16 host hits and `629,145,728` resident payload bytes. This is not full-model tok/s.
- A 40 GiB host sidecar cache plus a 40 GiB trunk cache was safely interrupted at approximately `72 GiB` WSL RSS before producing a full-gate result. Do not maximize both capacities independently on the 96 GB host.
- Fixed the synthetic benchmark's existing `l2_expert_workers` forwarding omission. The requested CUDA prefetch smoke completed at `55.250838` synthetic decode tok/s versus `49.563953` synchronous; this is synthetic-only evidence.
- Next bottleneck: integrate host-side sidecar reuse with route-stable layer-window admission and pinned/nonblocking H2D, then run a bounded full-layer quality gate before attempting another full 78-layer run.

## 2026-08-15 -- Linux CUDA-less INT4 guard repair

- GitHub correctness run `31884496150` exposed a CPU-only CI mismatch: the INT4 helper returned `RuntimeError(GLM5X_INT4_CUDA_UNAVAILABLE)` before the public CPU-target `ValueError(GLM5X_INT4_CUDA_REQUIRED)` contract. The explicit target/availability guard is fixed in `0d5621d`.
- Local focused INT4/packed-cache/benchmark tests passed `9`, and the complete WSL Python suite passed `354 passed, 124 skipped` in `78.52 s`. Public correctness and C++/Python CodeQL all pass on the current PR head.

## 2026-08-15 -- Protected residency and pinned staging milestone

- Current implementation commit: `49c386b` (`perf: add protected residency and pinned staging`). The C++ runtime now has a byte-bounded resident-weight LRU with explicit per-layer protected access sets. The Python packed-sidecar cache has an opt-in page-locked staging pool and non-blocking CUDA transfer mode.
- Defaults and correctness boundary are unchanged. Natural routing, exact BF16, synchronous sidecar loading, and zero-capacity caches remain the reference path. Pinned staging requires a packed expert precision, a positive capacity, and one expert reader; BF16 misuse is rejected.
- Verification: WSL CUDA build succeeded; CTest `27/27` passed; focused Python `26 passed, 6 skipped`; full WSL Python `356 passed, 124 skipped` in `77.29 s`; changed-module `py_compile` and `git diff --check` passed.
- Bounded measurements: the synthetic C++ residency fixture kept exact token IDs `[43, 32]` under 4 KiB and 1 MiB budgets, but the larger budget did not improve median latency. Real layer-10 pinned sidecar staging was slower on first use (`5.606441 s`) and slightly faster on the repeated bounded call (`3.370938 s` versus `3.469007 s` synchronous). No full-model rerun or quality promotion was made.
- Latest full-model truth remains exact BF16 `0.010559 tok/s` in the best measured resident-trunk/host configuration and NVFP4 `0.014484 tok/s` in the latest quality-rejected gate. The 10--20 tok/s target is not achieved, and no projected number is recorded as measured.

## In progress

- Replace the current per-layer residency boundary with route-stable multi-layer lookahead and a pooled asynchronous sidecar/H2D scheduler. Add separate sidecar bytes, H2D bytes/time, physical NVMe sampling, eviction counts, and deadline misses before another 78-layer gate.
- Connect the exact GLM MLA/DSA hidden-state path to the C++ expert-major backend and final logits, then run a quality gate before enabling any FP4 or adaptive Top-K mode.

## Known blockers

- Pinned staging currently improves the CUDA event component in an isolated transport probe but can worsen wall time when staging buffers are allocated per sample. It is a boundary for future reuse, not a proven full-model optimization.
- The C++ resident table still owns one active access context per backend/table. Concurrent forwards sharing one `RuntimeSession` are not a supported contract until the context is made per-forward or the session is serialized.
- The full-model bottleneck remains expert sidecar admission/H2D and trunk residency. Existing logical expert traffic is tens of GB per token, so isolated sub-millisecond projection results cannot reach 10 tok/s without a residency/traffic change.

## Last known-good test state

- Commit `49c386b`; WSL CUDA CMake build and CTest `27/27` green; full WSL Python `356 passed, 124 skipped`; no cloud or paid resources used.

## 2026-08-15 -- Packed-sidecar telemetry milestone

- Current milestone: exact packed-sidecar traffic accounting is implemented and measured. The work is committed locally as `25fc7c7` on `codex/sidecar-telemetry`, based on plan commit `5fa1fe8` and public `origin/main` `2b9c214`.
- Completed: opt-in sidecar file/decode/H2D counters, phase-separated reference benchmark fields, validation for packed path/precision, CUDA-event timing, host-cache/file-read separation, focused CUDA regressions, and a five-iteration real expert-48 paired measurement.
- Current hardware assumption: Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB system RAM, WSL2 Ubuntu-24.04/CUDA 13.0 for development; Linux native remains the deployment target.
- Latest measured bottleneck: one real mixed `.pgu` expert still requires `39,321,608` H2D bytes when it is not device-resident. Pooled pinned staging reduced the warm transfer event median from `3.061 ms` to `1.033 ms` and wall median from `17.896 ms` to `11.425 ms`, but this is an expert transport boundary and not token throughput.
- Full-model truth is unchanged: best exact measured decode remains `0.010559 tok/s`; latest mixed NVFP4 gate remains `0.014484 tok/s` and is quality-rejected. No 10 tok/s result exists.
- Known blocker: exact multi-layer residency/lookahead is not implemented, physical NVMe traffic is still unmeasured, and `gh auth status` reports an invalid token for account `rsb1813`, so the verified local commits are not yet published from this session.
- Next concrete task: design and implement a bounded N+1 exact-prefetch predictor over existing runtime-profile transitions, without changing natural routing. Measure recall, overfetch bytes, residency hit rate, H2D bytes, and final output parity before any full 78-layer rerun.
- Last known-good tests: commit `25fc7c7`; focused benchmark/cache suite `25 passed, 6 skipped`; full WSL Python `360 passed, 124 skipped` in `78.30 s`; changed-module `py_compile` and `git diff --check` passed. C++ was unchanged; prior CTest `27/27` remains the latest applicable C++ result.

## 2026-08-15 -- Exact N+1 transition-prefetch milestone

- Current milestone: deterministic exact N+1 transition prediction, deadline-ticket reuse, CLI gating, full telemetry, and standard JSON/CSV benchmark integration are implemented in code commit `faa12f4` on `codex/transition-prefetch`.
- Completed: prior/live normalized transition ranking, deterministic tie ordering, candidate bound `0..16`, exact selected-ticket reuse, exact fallback for every miss, recall/overfetch/readiness/byte accounting, route/token parity tests, and durable candidate `0/1/2` synthetic artifacts.
- Measured result: candidate `0/1/2` produced identical tokens and routes while measuring `150.141/129.338/113.328` synthetic CPU decode tok/s. Candidate 2 reduced exposed expert-load wait from `50.024 ms` to `33.962 ms`, but matched only `17/36` submissions and was about `24.5%` slower than candidate 0.
- Decision: the feature is implemented but remains default-off. It is not evidence for GLM full-model speed, does not perform sidecar-to-VRAM lookahead, and does not change any quality mode.
- Current hardware assumption: Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB system RAM, WSL2 Ubuntu-24.04/CUDA 13.0 for development; Linux native remains the deployment target.
- Latest measured bottleneck: real mixed `.pgu` admission still moves `39,321,608` bytes per nonresident expert. Host-only prediction can reduce wait but cannot solve synchronous H2D or the full-model cache-thrash pattern.
- Full-model truth: best exact measured decode remains `0.010559 tok/s`; the latest mixed NVFP4 gate remains `0.014484 tok/s` and quality-rejected. The requested 10 tok/s threshold is not achieved.
- Known failures/blockers: the C++ runtime still lacks the official 282-shard full-model execution boundary, pooled predicted sidecar H2D is not integrated, physical NVMe traffic is unmeasured, and the predictor's first synthetic recall/overfetch trade-off is poor. PR #6 for sidecar telemetry is open, draft, mergeable, and all checks are green; this stacked transition branch is not yet published.
- Next concrete task: implement exact packed-expert layer-window residency with pooled pinned asynchronous H2D and phase-separated sidecar/H2D telemetry on the real layer-10 path, then use that evidence to choose a full-model residency policy. Do not rerun the 78-layer gate until this boundary reduces transferred bytes or wall time while preserving logits/routes.
- Last known-good state: code commit `faa12f4`; CUDA CTest `27/27`, CPU CTest `15/15`, full CPU-build Python `362 passed, 124 skipped` in `106.39 s`, focused integration/schema tests green, `py_compile`, `git diff --check`, and benchmark artifact SHA-256 verification all passed.

## 2026-08-16 -- Stable per-layer hot-bank milestone

- Current milestone: an opt-in exact `stable_hot_bank` device-cache policy, CLI/benchmark telemetry, and a reproducible real-sidecar residency benchmark are implemented on `codex/stable-hot-bank`, stacked above `0437bc0`.
- Completed: per-layer access-frequency tracking, strictly-hotter promotion, transient bypass, byte-bound enforcement, bypass/promotion telemetry, cache/model/CLI regressions, digest-safe trace selection, and B-0002 RTX 5080 raw/summary artifacts.
- Measured result: on a structured 16-layer×8-expert repeated trace, LRU had zero warm hits and `5.033 GB` H2D per pass. The stable bank retained one expert per layer, produced `16/128` hits, moved `4.404 GB`, and reduced three-pass median wall time from `3.656` to `3.289 s` (`10.04%`).
- Decision: keep the policy experimental and default-off. The trace uses exact digest-matched payloads but is not natural-router full-model decode and did not compute logits or tokens.
- Current hardware assumption: Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB system RAM, WSL2 Ubuntu-24.04/CUDA 13.0 for development; Linux native remains the deployment target.
- Latest measured bottleneck: nonresident mixed `.pgu` experts still move about `39.3 MB` each, and synchronous per-expert decode/H2D remains dominant. Stable retention cuts a bounded fraction, but pooled asynchronous H2D and official full-model runtime integration are still missing.
- Full-model truth: best exact measured decode remains `0.010559 tok/s`; the latest mixed NVFP4 gate remains `0.014484 tok/s` and quality-rejected. The requested 10 tok/s threshold is not achieved.
- Known blockers: the official 282-shard model is still not connected to the C++ full-model execution path; hot-bank full-model hit rate and final-logit parity are unmeasured; physical NVMe traffic remains unmeasured; and pinned H2D currently synchronizes for telemetry/completion.
- Next concrete task: combine the exact stable bank with pooled pinned N+1 sidecar staging and asynchronous completion, then run a natural-router bounded/full-model quality and traffic gate. Do not infer full-model TPS from B-0002.
- Last known-good test state: B-0002 raw/summary SHA-256 parity passed; focused cache/model/CLI regression `35 passed, 6 skipped`; full CPU-build Python regression `365 passed, 124 skipped` in `107.34 s`; changed-module `py_compile` and `git diff --check` passed. Final commit hashes are pending this milestone close.
