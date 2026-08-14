# GLM5X Project State

## Current milestone

GLM-5.2 shape/manifest boundary, exact cross-shard raw-BF16 loading, and a bounded real-shard RTX 5080 BF16 expert-major candidate grid. Full model routing, attention, quality, and end-to-end throughput remain open.

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
- Added and measured `k3x_cuda_glm5x_moe_bench` on the real RTX 5080 at GLM-5.2 expert dimensions.
- Added an expert-major candidate-token benchmark mode for 1/2/4/8 tokens.
- Added resident exact MXFP4 reuse to the CUDA expert-major batch backend and allowed resident weights in the CLI validation contract.
- Added opt-in resident BF16 dequantized expert-grid execution through cublasLt, with native MXFP4 fallback when resident capacity is insufficient.
- Added `GLM5XDSAConfig`, `GLM5XDSAIndexer`, and `GLM5XDSAState`, connecting descriptor index metadata and explicit query/key projections to compressed KV blocks, exact top-k refresh, and a separately marked stale fast refresh cadence.
- Added `GLM5XOfficialDSAIndexer` with official-shaped `wq_b/wk/k_norm/weights_proj` tensors, interleaved/non-interleaved indexer RoPE, ReLU score aggregation, causal masking, and Top-K reference parity. Its safetensors loader reads only the five indexer tensors needed for a selected layer.

## In progress

- Build the tiny GLM-5.2-compatible reference graph and greedy parity tests.
- Connect the official indexer to the production q-residual path and exact MLA/DSA state.
- Rename user-facing runtime and benchmark commands where that does not break the inherited storage ABI.
- Connect the real expert-major grid to exact GLM DSA/MTP state and retain strict natural Top-8 verification.
- Validate nonzero full-layer routing/MLA/DSA parity and measure VRAM-bank pressure before considering BF16 grid mode a default.
- Replace host float decode plus conversion with direct raw-BF16/tensor-core storage views where quality permits.
- Measure whether BF16-output resident grids remain inside the chosen quality budget on nonzero full-layer GLM data.

## Known blockers

- No full GLM-5.2 checkpoint is present; only two bounded probe shards are available, so full checkpoint correctness and local TPS are not measured.
- Only two bounded GLM-5.2 probe shards are present; no full checkpoint download or Cloud Run conversion has been authorized or attempted.
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

- Focused GLM descriptor, manifest, CLI, toy reference, TurboQuant, DSA, official indexer, shard-converter, bundle loader, and multi-shard tests: 35 passed. The CUDA Python test remains skipped on Windows because the WSL ELF binary is not a Windows executable.
- CUDA CMake build: successful in WSL with CUDA 13.3 and RTX 5080 compute capability 12.0.
- CTest: 26/26 tests passed in WSL.
- Focused CUDA regression now includes a nonzero two-expert/two-token BF16 resident-grid parity case; the 26-test CTest count is unchanged because it extends `cuda_dense`.
- Real-shard C++ metadata gate: both downloaded GLM probe artifacts passed `test_reader ... metadata`; the second artifact contains 212 tensors and 70 complete raw-BF16 expert records.
- Cross-shard bundle gate: 2 probe artifacts and 247 tensors indexed in approximately 11.9 seconds, producing 70 complete experts and 0 incomplete groups without copying payload bytes.
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
- Full inherited Python suite was not green because historical `results/` artifacts and a Windows `build/` executable path were intentionally not migrated; the focused GLM suite remains green.
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
- Last known-good code HEAD: `95f596d` (`perf: add optional bf16 resident grid outputs`). Focused Python and WSL CTest gates are green.
- Next bottleneck: pinned/asynchronous raw H2D, direct tensor-core algorithm selection, q-residual production projection, exact MLA/DSA state, natural Top-8 routing, nonzero full-layer parity, and VRAM-pressure-aware multi-expert residency; full weights remain intentionally absent.
