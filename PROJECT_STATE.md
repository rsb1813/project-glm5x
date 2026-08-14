# GLM5X Project State

## Current milestone

GLM-5.2 shape/manifest boundary, exact cross-shard raw-BF16 loading, an exact q-residual/MLA/DSA/MoE layer-10 reference, a learned-router-aware raw-BF16 CUDA MoE sublayer boundary, and the portable `GLM5XACT` activation handoff are implemented over five bounded real shards. The implementation evidence is in `30bf5d4`; public Linux correctness `31806277016` and CodeQL `31806277022` are green, and the documentation head is `8c7351f`.

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
- Added `GLM5XMLAReference`, incremental compressed MLA state, official causal DSA state, and `GLM5XDecoderLayerReference` with full-vs-incremental parity and bundle-backed attention/indexer/norm/MoE construction.
- Reused one root-verified `GLM5XExpertBundle` reader across the layer loader and lazy MoE closure; the real five-shard construction smoke fell from 491.483777 s to 250.637263 s.
- Connected `ExpertMajorPackedPlan` bucketing and contribution scatter to `CudaBackend::raw_bf16_situ_mlp_expert_major`; the real bounded probe measured 1,380,314 ns warm median/block for a deterministic 8-group/10-assignment/2-token route.
- Added `learned-expert-major` to load the official GLM router and correction bias, select the actual Top-8 expert union, and expose an explicit resident-byte budget. The bounded two-token route selected 15 experts and measured 1,905,668 ns warm median/block with 0.0866% maximum CPU-relative difference.
- Added `learned-moe-layer` to execute the same natural routed union plus the actual layer shared expert through raw-BF16 CUDA, with combined CPU parity and separate shared-payload/residency telemetry.
- Added `GLM5XACT` v1 Python/C++ BF16 activation artifacts with atomic writing, fixed headers, shape/extent/CRC validation, and optional benchmark input/expected-output comparison.
- Added a narrow pytest boundary for historical K3X evidence files absent from the GLM5X repository; new GLM5X tests remain active. GitHub Actions correctness and CodeQL pass on `a00beec`.

## In progress

- Finish the tiny GLM-5.2-compatible reference graph and greedy parity tests around the remaining final-logits/MTP boundary.
- Extend the layer-10 exact q-residual/MLA/DSA boundary to the remaining layers, final logits, and MTP state.
- Rename user-facing runtime and benchmark commands where that does not break the inherited storage ABI.
- Connect the exact layer-10 reference's MLA/DSA hidden state to the learned router/expert-major CUDA grid while retaining strict natural Top-8 verification.
- Validate nonzero full-layer routing/MLA/DSA parity and measure VRAM-bank pressure before considering BF16 grid mode a default.
- Replace host float decode plus conversion with direct raw-BF16/tensor-core storage views where quality permits.
- Measure whether BF16-output resident grids remain inside the chosen quality budget on nonzero full-layer GLM data.
- Measure the new CUDA bucket loop with exact nonzero GLM router assignments, then add pinned asynchronous staging after full-layer parity is established.
- Real five-shard layer-10 reference smoke at `a2d6b6d`: bundle open/root verification 250.637263 s, cold two-token forward 5.969859 s, cached repeat 0.057331 s, 16 unique selected experts, cached output max difference 0.0. These are layer/storage reference timings, not tok/s.

## Known blockers

- No full GLM-5.2 checkpoint is present; five bounded probe shards are available, so full checkpoint correctness and local TPS are not measured.
- No full checkpoint download or Cloud Run conversion has been authorized or attempted; only five bounded probe shards are present.
- Dependabot PRs #1–#4 are open for action/setup-python, action/checkout, numpy, and setuptools updates. Their first Linux checks failed with 50 `FileNotFoundError` cases because the bot branches were stale; all four branches were rebased onto `main`, and the replacement Linux/CodeQL checks are green (`#1` runs `31805879968`/`31805879919`, `#2` `31805884019`/`31805884008`, `#3` `31805888033`/`31805888023`, `#4` `31805890586`/`31805890614`). Repository Dependabot and vulnerability-alert APIs are disabled, so no CVE alert is confirmed.
- The migrated C++ runtime still has K3-oriented names and graph assumptions in several files.
- CUDA build and bounded RTX 5080 kernel measurements are validated in WSL; native Linux end-to-end throughput has not been measured.
- The TurboQuant implementation is CPU/reference only; it does not yet contain a packed CUDA kernel or full PolarQuant/QJL production path.
- 600k/1M capacity is a formula estimate only until a real GLM-5.2 DSA state is allocated and restored.

## Hardware assumptions

- CPU: AMD Ryzen 7 9800X3D.
- GPU: NVIDIA RTX 5080 16 GB.
- RAM: 96 GB DDR5-4200.
- Storage: Solidigm P44 Pro 2 TB NVMe.
- Preferred execution environment: native Linux.

## Latest verified state

- Focused GLM descriptor, manifest, CLI, toy reference, TurboQuant, DSA, official indexer, shard-converter, bundle loader, exact MLA/DSA layer reference, and multi-shard tests: 42 passed in the last WSL Python environment. The Windows Python interpreter currently lacks pytest, so no new Python rerun was claimed. The CUDA Python test remains skipped on Windows because the WSL ELF binary is not a Windows executable.
- CUDA CMake build: successful in WSL with CUDA 13.3 and RTX 5080 compute capability 12.0.
- CTest: 27/27 tests passed in WSL, including `glm5x_activation`.
- CPU WSL CTest: 14/14 tests passed after the expert-major CUDA API change.
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
- Public Linux correctness workflow `31806277016` and CodeQL workflow `31806277022` both passed for implementation commit `30bf5d4`; documentation head `8c7351f` also passed correctness `31806748306` and CodeQL `31806748296`. The workflows still emit non-failing Node 20 and CodeQL v3 deprecation annotations.
- No end-to-end GLM decode tok/s or quality result exists yet.
- Bounded GLM-5.2-shaped CUDA result: 8 experts/1 token median 2,662,772 ns; 8 experts/4 tokens 1,344,816 ns per candidate token; maximum absolute error 0.
- Resident expert-major batch result: 8 groups x 4 candidates, 1,641,591 ns/candidate token, cold weight H2D 160,432,128 bytes and warm weight H2D 0 bytes.
- Resident BF16 grid result: 8 experts x 4 candidates, 2,582,527 ns/block versus native 5,394,131 ns/block; BF16 resident weight bytes 603,979,776 versus native 160,432,128; maximum absolute error 0 on the zero-weight fixture.
- Latest rerun: native zero-pattern grid 5,510,632 ns/block; BF16 zero-pattern grid 4,386,083 ns/block; nonzero BF16 grid 4,044,675 ns/block with 0.9505% maximum relative difference versus the native GPU reference. These remain bounded synthetic measurements only.
- DSA reference capacity estimate: 201,637,504 bytes at 600,000 tokens and 336,062,512 bytes at 1,000,000 tokens for BF16 index keys plus K6/V4 KV with index width 4096; this is formula-only.
- Official manifest metadata probe: 59,585 tensors across 282 shards; shared indexer layer mapping is resolved without opening payloads.
- First official shard header probe: 35/35 names matched `model-00001-of-00282.safetensors`; representative indexer tensors are BF16 with `wk=(128,6144)`, `wq_b=(4096,2048)`, and `weights_proj=(32,6144)`.
- First bounded artifact: 35 BF16 tensors, 78 layer records, Python reader checks green, and WSL C++ `test_reader` exit 0; no full model loaded.
- Real indexer payload gate: loaded only five layer-0 indexer tensors from the 5.3 GB first shard (`wq_b=(4096,2048)`, `wk=(128,6144)`, `weights_proj=(32,6144)`, and two 128-element LayerNorm vectors) and ran causal Top-K on zero activations. This is payload/shape evidence, not model quality or throughput.
- Last known-good implementation HEAD: `30bf5d4` (`feat: expose GLM MoE activation inputs`); latest documentation HEAD: `8c7351f` (`docs: record activation boundary and CI recovery`). WSL CUDA CTest 27/27 is green; the public Python/cross-language suite passed, while the Windows interpreter lacks pytest for a local rerun. Public Linux correctness `31806277016`, CodeQL `31806277022`, documentation correctness `31806748306`, and documentation CodeQL `31806748296` all passed.
- Next bottleneck: export the exact layer-10 q-residual/MLA/DSA hidden state into `GLM5XACT`, run the expected-output parity path on a real bounded artifact, then add pinned/asynchronous raw H2D, direct tensor-core algorithm selection, all-layer exact state, nonzero full-layer parity, and VRAM-pressure-aware residency; full weights remain intentionally absent.
