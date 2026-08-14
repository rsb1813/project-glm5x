# GLM5X Work Context

## 2026-08-14 bootstrap

- The project pivots from Kimi K3/K3X to GLM-5.2 first, with a GLM-5.3 descriptor and checkpoint swap planned later.
- K3X storage, tiered cache, deadline prefetch, expert-major scheduling, speculative interfaces, and benchmark schemas are retained as model-neutral foundations.
- KDA, Attention Residual, 896-way Top-16 routing, and Kimi-native MXFP4 graph assumptions are not part of the GLM5X default graph.
- No GLM-5.2 or GLM-5.3 weights are bundled in this bootstrap repository, and no throughput claim is made.
- Official K3 artifacts were deleted only from the approved old-worktree paths. Synthetic K3 fixtures remain for compatibility tests.
- The focused descriptor/CLI/toy reference suite and WSL CTest were green; the inherited full Python suite remains outside bootstrap because of historical results and Windows build-path assumptions.

## 2026-08-14 TurboQuant milestone

- The first optimization is deliberately KV-only. TurboQuant does not reduce GLM-5.2 expert weight bytes, so expert streaming remains the short-context throughput bottleneck.
- The reference implementation uses Hadamard rotation with deterministic seed metadata, row scales, integer bit widths, and half-bit schedules such as `(3, 4)` for a logical 3.5-bit path.
- Balanced candidate is asymmetric K6/V4. 3.5-bit and 2.5-bit modes remain experimental until GLM-5.2 long-context quality is measured.
- `TurboQuantKVCache` stores quantized blocks, restores them for attention, and reports logical storage. It is not yet a packed CUDA format.
- The 1M-token result is formula-only capacity arithmetic. No GLM weights, DSA state, or full-context allocation has been run.
- Focused GLM descriptor, CLI, toy, and TurboQuant tests pass 13/13 in `C:\g5xv` with Python 3.12, PyTorch 2.13.0, and pytest 9.1.1.

## 2026-08-14 GLM shape and CUDA baseline

- The local GLM-5.2 config/index probe fixes the runtime shape at hidden size 6144, 78 layers, 256 routed experts, Top-8, MoE intermediate size 2048, DSA index Top-K 2048/frequency 4, index heads 32 x 128, and maximum position 1,048,576. The source index reports 1,506,659,919,872 bytes; no weight shard was downloaded.
- `GLM5XTensorManifest` validates the descriptor plus safetensors `weight_map` and source byte total before conversion. It rejects missing maps, non-positive totals, path-bearing shard names, and invalid tensor names.
- Real RTX 5080 WSL CUDA build passed CTest 26/26. The new synthetic benchmark uses zero MXFP4 weights so the output reference is exact and the result is explicitly a layer/kernel record, not model tok/s.
- Final benchmark samples at commit `31678d1`: 8 experts/1 token median 2,662,772 ns, 8 experts/4 tokens median 5,379,264 ns (1,344,816 ns/token), 8 experts/8 tokens median 8,791,638 ns (1,098,955 ns/token), and 16 experts/1 token median 5,458,462 ns. All maximum absolute errors were 0.
- A shared-input row-tiled kernel and a lookup/bit-scale MXFP4 decoder were tested and rejected after repeated slowdowns. They were not kept in the accepted source.
- Expert-major candidate batching is a measured optimization candidate for MTP/DSpark integration. It must not be treated as speculative acceptance or full-model throughput until the exact target verifier is connected.

## 2026-08-14 Resident expert-major batch reuse

- `mxfp4_situ_mlp_batch` now accepts resident CUDA weights, acquires all three exact MXFP4 projections through `ResidentWeightTable`, and falls back to the existing transient scratch path only on capacity bypass.
- The CUDA FFN test is a red-green regression: before the patch, the second resident batch re-uploaded weights; after the patch, CPU parity is exact, cache hits increase by three, and warm weight H2D remains unchanged.
- The CLI expert-major contract no longer rejects resident weights. This is a storage/transfer optimization only; natural routing, exact candidate verification, and quality semantics are unchanged.
- GLM-shaped `expert-batch` is not faster than the all-expert grid for the synthetic 8-expert case, but it proves the important invariant that cold weight upload is one-time and warm upload is zero. The next performance work should target variable-union scheduling and tensor-core/dequantized GEMM, not proxy weights.

## 2026-08-14 BF16 resident grid

- Added `CudaMxfp4Execution::dequantized_bf16` as an opt-in path. It caches host BF16 dequantizations by tensor identity, admits them through the same resident table, and uses cublasLt batched projections across all selected experts with one activation upload and one output download.
- Added red-green CUDA parity coverage for BF16 expert-batch and BF16 expert-grid execution. Native MXFP4 remains the default; capacity bypass returns to the exact native group path.
- On the RTX 5080 shaped fixture, 8 experts/4 tokens measured 5,394,131 ns native grid versus 2,582,527 ns BF16 grid. BF16 resident bytes were 603,979,776 versus 160,432,128 native. Both report maximum absolute error 0 only because the fixture uses deterministic zero weights.
- A deterministic nonzero packed pattern was added to the benchmark. BF16 grid versus a separate native GPU reference measured maximum relative difference 0.00950465 (0.95%) at 2,563,496 ns warm median. This is numeric parity evidence only, not calibrated GLM quality.
- The CUDA parity gate initially exposed the expected BF16 accumulation rounding rather than an exact-float mismatch. The test now uses explicit bounded tolerances (2e-3 for the tiny batch, 4e-3 for the zero-grid fixture, and 0.5 for the larger nonzero fixture) and the full 26-test CTest suite is green.
- BF16 grid capacity preflight now excludes already-resident BF16 keys and rejects the whole BF16 admission before native fallback if the remaining VRAM budget is insufficient. This prevents mixed BF16/native metadata collisions on repeated calls. The released-dimension 16-expert check now returns exact native output instead of `INVALID_EXTENT` under the 1 GiB budget.

## 2026-08-14 Latest benchmark rerun

- Re-ran the bounded GLM-5.2-shaped 8-expert/4-token grid on the RTX 5080 after the capacity guard was committed. Native zero-pattern median was 5,510,632 ns/block and resident BF16 median was 4,386,083 ns/block.
- Re-ran the nonzero BF16 comparison at 4,044,675 ns/block. The maximum relative difference against the native GPU reference was 0.0095046479255 (0.9505%).
- Recorded the new values as an additional benchmark entry rather than overwriting prior samples. The benchmark remains a layer/kernel measurement; no model tok/s or quality conclusion is permitted.

## 2026-08-14 DSA/indexer reference state

- Added `GLM5XDSAConfig`, `GLM5XDSAIndexer`, and `GLM5XDSAState` instead of pretending the standalone TurboQuant cache was already a GLM DSA graph. Descriptor metadata now determines index width/top-k policy, and explicit query/key matrices project hidden states into index keys.
- The correctness path (`reference_mode=True`) refreshes index top-k on every query. The experimental fast path reuses the previous selection until `index_topk_freq` new tokens arrive; the distinction is explicit in the API and tests.
- The state stores projected synthetic index keys plus the existing TurboQuant KV cache. Official role mapping is now present, while tensor shapes/values, MLA latent transforms, and CUDA storage are still absent, so 600k/1M numbers remain arithmetic estimates.
- Focused GLM Python coverage was 21 passing tests before the role-resolution regression. The next implementation boundary is official tensor shape/value parity and DSA/MLA reference parity, not a speculative end-to-end TPS claim.

## 2026-08-14 Official manifest role probe

- The local HF metadata cache identifies `zai-org/GLM-5.2`; `hf download --dry-run` reports 295 files totaling about 1.5 TB. No shard payload was downloaded. XET support is installed and enabled.
- The official index contains 59,585 tensors across 282 shards and a 78-entry `indexer_types` list. `GLM5XTensorManifest` now resolves shared indexer layers to the nearest preceding full layer and validates the observed component names.
- Example metadata-only resolutions: layer 3 uses layer 2's indexer tensors, layer 7 uses layer 6's, and layer 77 uses layer 74's. The first bounded shard now supplies the initial shape/dtype parity gate.
- Focused GLM Python coverage becomes 23 passing tests after the role-resolution and header-parity regressions.

## 2026-08-14 First real shard header gate

- Downloaded only `model-00001-of-00282.safetensors` (5,342,821,416 bytes) using XET high-performance mode; no other model shard was downloaded.
- Header-only inspection validated all 35 index-listed names. The observed indexer shapes are BF16 `wk=(128,6144)`, `wq_b=(4096,2048)`, `weights_proj=(32,6144)`, and `k_norm=(128,)` for layers 0 and 1.
- Added `inspect_safetensors_shard` and `GLM5XTensorManifest.validate_safetensors_shard`; both compare headers without materializing tensor payloads. The next boundary is bounded streaming conversion, not a full checkpoint load.

## 2026-08-14 First bounded GLM shard artifact

- Added `glm5x-convert convert-shard` and a raw BF16 bounded writer that reuses K3X aligned extents, CRC32C, directory/root SHA-256, and a name-preserving sidecar.
- Converted only `model-00001-of-00282.safetensors` with an 8 MiB chunk limit. The artifact contains 35 tensors and 78 layer records; Python checksum/root verification and WSL C++ `test_reader` both passed.
- `DType.BF16` is now accepted by the portable reader, but no CUDA BF16 weight consumption or expert-directory semantics are claimed yet. The next task is resumable multi-shard conversion.

## 2026-08-14 Resumable multi-shard conversion

- `convert_glm5x_shard` now writes a source/config-fingerprinted `.resume.json` ledger beside the `.partial` artifact. Each completed tensor extent is recorded only after fsync, source CRC verification, and partial-file readback; resume accepts only the canonical prefix with expected IDs, aligned offsets, lengths, and recomputed source CRCs.
- Finalization is crash-safe across the output rename, sidecar rename, and ledger cleanup. If a worker dies after the `.k3x` rename, the next invocation validates the final reader metadata and repairs the sidecar/ledger boundary.
- `convert_glm5x_shards` treats every manifest shard as an independent unit and skips only finalized artifacts whose sidecar/source SHA-256 and K3X reader metadata match. This is the local equivalent of a restartable Cloud Run worker unit; object-store uploads and multi-worker scheduling remain future work.
- Complete same-shard GLM raw-BF16 expert role triples receive `EXPT` records linked to their tensor IDs. Incomplete triples stay in the sidecar to avoid fabricating cross-shard links.
- Focused GLM coverage is now 28 passing tests. No end-to-end GLM decode or conversion-throughput benchmark was added; the next bottleneck is exact learned DSA/indexer projection and cross-shard expert assembly.

## 2026-08-14 Official DSA indexer reference

- Added a separate `GLM5XOfficialDSAIndexer` rather than flattening GLM's `[heads, head_dim]` query and `[head_dim]` key into the earlier equal-width DSA experiment. The reference follows the official order: `wq_b`, `wk` + LayerNorm, indexer RoPE, per-head dot products, ReLU, `weights_proj` aggregation, causal mask, and Top-K.
- The loader uses `safe_open().get_tensor()` for only `wq_b`, `wk`, `k_norm.weight`, `k_norm.bias`, and `weights_proj`. A manual run on the real first shard loaded shapes `(4096,2048)`, `(128,6144)`, `(128,)`, `(128,)`, and `(32,6144)` without materializing the other 30 tensors.
- Automated parity is synthetic and independent of the production method; the real run used zero activations only. No quality, full-layer, or tok/s claim is made. Focused GLM coverage is now 31 passing tests.
- The official Transformers source is the reference boundary for this formula: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py`. q-residual production projection, MLA latent path, cache updates, and nonzero real-shard parity remain the next implementation boundary.

## 2026-08-14 Raw-BF16 expert directory reader gate

- The first real second shard contained complete same-shard raw-BF16 expert triples, but the C++ directory validator still required `quantization=MXFP4` for every `EXPT` link. Python accepted the artifact while the WSL C++ reader returned `INVALID_DIRECTORY`.
- The reader now accepts only two expert payload classes: native `dtype=UINT8, quantization=MXFP4`, or raw `dtype=BF16, quantization=NONE` with no auxiliary extent or checksum. Other expert metadata remains rejected.
- A metadata-only mode was added to `test_reader` so multi-gigabyte real artifacts can exercise directory validation without rescanning every payload checksum. Both downloaded GLM probe artifacts now pass this gate; the full 26-test CTest suite and 31 focused GLM Python tests remain green.
- This does not enable BF16 CUDA execution. `load_storage_expert` deliberately remains native-MXFP4-only until a separate GLM payload path is implemented and benchmarked.

## 2026-08-14 Cross-shard expert bundle index

- Added `glm5x-convert assemble-experts`, which opens finalized shard artifacts, validates their sidecar source digest and tensor IDs, and emits a copy-free `glm5x-expert-bundle-v1` JSON index.
- Each complete `(layer, expert)` record contains the three role names plus artifact-relative path, tensor ID, dtype, quantization, aligned data offset, byte length, logical length, and data CRC. Duplicate roles are rejected and incomplete role groups are listed separately.
- The two downloaded GLM probe artifacts were indexed in about 12 seconds on the Windows host. The result covers 2 artifacts, 247 tensors, and 70 complete experts with no incomplete groups. This is storage/indexer evidence, not a model execution or throughput result.
- The bundle does not copy or quantize payload bytes. The next runtime boundary is a bounded exact BF16 expert loader that consumes these references, followed by cross-shard nonzero numerical parity.

## 2026-08-14 Exact bundle BF16 payload parity

- Added `GLM5XExpertBundle.open/read_expert`. It reopens every referenced `.k3x`, checks file UUID/root/source digests and tensor counts, then checks role tensor IDs, dtype, quantization, shape, aligned offset, length, logical length, and CRC before returning bytes.
- The loader intentionally accepts only raw BF16/no-auxiliary payloads for this GLM staging path. Native MXFP4 remains a separate exact storage-slice path.
- A real nonzero gate on layer 10 expert 0 matched all three 25,165,824-byte role tensors byte-for-byte against `model-00002-of-00282.safetensors`; the printed role SHA-256 values were `d2e72bbf...`, `39ebf198...`, and `601b9d1f...`.
- Focused GLM coverage is now 35 passing tests. This is exact payload/reference evidence, not CUDA execution or end-to-end model throughput.

## 2026-08-14 C++ cross-shard BF16 loader

- Added `k3x::load_glm5x_bf16_expert`, which accepts multiple `Reader` instances, locates canonical GLM role IDs across them, rejects duplicate/missing roles, validates `BF16/NONE`, released `6144 x 2048` shapes, layer/expert IDs, lengths, and CRC32C, then returns three exact host payloads.
- Added `test_glm5x_bf16_bundle` as a real-artifact gate. With the two downloaded probe artifacts, layer 10 expert 0 loaded 75,497,472 bytes in 465,087,758 ns under WSL. The result is host storage latency only; no CUDA weights or model layer were executed.
- The next performance boundary is moving these three host vectors into the resident BF16 CUDA grid without an intermediate copy, then comparing that layer output against a CPU BF16 reference.

## 2026-08-14 First real GLM expert CUDA bridge

- Added `k3x_cuda_glm5x_real_expert_bench`. It loads layer 10 expert 0 across the two real probe artifacts, decodes BF16 bytes to host floats, and executes the existing resident CUDA dense SiTU FFN against a deterministic nonzero input.
- FP32 resident run with 5 warm samples measured `latency_nanoseconds_median=457802` in the first smoke; the 20-sample rerun measured `275473` ns. Cold weight H2D was 150,994,944 bytes, warm H2D was 0, resident bytes were 150,994,944, and GPU-vs-CPU maximum absolute error was `8.38190317154e-09`.
- Pre-cache `bf16-rounded` resident run reduced weight H2D/residency to 75,497,472 bytes but measured 28,154,650 ns warm median and 0.1828% maximum relative CPU difference. This historical sample exposed the repeated host conversion bottleneck.
- Neither run is a full GLM layer or token-generation benchmark. Router, DSA/MLA, dense trunk, residuals, and other experts remain outside the measurement.

## 2026-08-14 BF16 host-conversion cache rerun

- The first BF16-rounded real-expert run spent most of its 28.15 ms warm median reconverting 75 MiB of FP32 views to BF16 on every call. Added a tensor-identity/shape/pointer keyed host BF16 cache in the CUDA backend; input conversion remains per call because activations can change.
- Rerun with 5 warmups and 20 samples measured BF16-rounded warm median 236,593 ns, cold latency 197,436,559 ns, cold H2D 75,497,472 bytes, warm H2D 0, and resident bytes 75,497,472. CPU relative difference remained 0.00182774465 (0.1828%).
- FP32 rerun measured 271,493 ns warm median, 150,994,944 resident bytes, and `8.38190317154e-09` maximum absolute CPU difference. The cached BF16 path is now the faster bounded candidate, but remains experimental until full-layer/model quality is measured.

## 2026-08-14 Real multi-expert resident pressure

- Extended `k3x_cuda_glm5x_real_expert_bench` with `--experts N`; for `N>1` it selects the first available expert IDs on the requested layer and executes them sequentially through the existing resident dense path.
- Eight layer-10 experts in BF16-rounded mode loaded 603,979,776 payload bytes, took 4,859,331,588 ns for the cold host reads, and measured 1,854,140 ns warm median over 20 iterations with warm H2D 0 and resident bytes 603,979,776.
- Eight FP32 experts required 1,207,959,552 resident bytes. Under the 1 GiB budget, residency bypass caused 3,019,898,880 warm H2D bytes and 13,153,048 ns warm median. This confirms BF16 residency is required for multi-expert pressure on the target card.
- The current path is sequential, so this is a lower bound for an expert-major batch implementation rather than a final MoE layer result.

## 2026-08-14 Real BF16 expert-major candidate grid

- Added `dense_situ_mlp_grid` to the CUDA backend for the validated raw-BF16 GLM path. It batches all selected experts over a candidate-token block, admits each projection through the resident table, uploads the activation block once, and returns one flattened output per expert. FP32 remains on the scalar reference path.
- Added `--tokens` to `k3x_cuda_glm5x_real_expert_bench` with a 1..65535 guard. The CPU reference now evaluates every candidate token, so the reported error covers the full last-expert block rather than one token only.
- The two-token CUDA regression uses nonzero matrices and CPU BF16-rounded references. The full WSL CTest suite remains 26/26 and the focused GLM Python suite remains 35/35.
- Rerun on the two real probe artifacts: 8 experts x 4 tokens measured 1,758,739 ns warm block median, 603,979,776 resident bytes, zero warm H2D, and 0.1351% maximum relative CPU difference. This is a bounded FFN block result, not model tok/s.
- The next bottleneck is now direct raw-BF16/tensor-core storage plus exact natural Top-8 routing and nonzero full-layer parity. The dense grid is opt-in until those quality and capacity gates exist.

## 2026-08-14 Direct raw-BF16 resident admission

- Added public `RawBf16WeightView`/`RawBf16MlpView` contracts and a CUDA `raw_bf16_situ_mlp_grid` entry point. The implementation validates byte lengths, tensor IDs, shapes, CRC-checked caller payloads, and resident capacity before admitting role bytes directly.
- The real benchmark now keeps raw `.k3x` role bytes for every selected expert and decodes only the last expert for the CPU comparison. FP32 mode remains unchanged and still exercises the high-precision scalar reference.
- Direct raw BF16 on the two probe artifacts measured 8 experts x 4 tokens at 1,648,927 ns warm block median, 135,877,327 ns cold, 603,979,776 resident bytes, zero warm H2D, and 0.1351% maximum relative CPU difference. The prior dense wrapper measured 1,758,739 ns warm and 759,804,032 ns cold under the same command shape.
- `test_cuda_dense` now exercises both the dense-wrapper and raw-byte APIs with a nonzero two-expert/two-token fixture. Full WSL CTest remains 26/26 and focused GLM Python remains 35/35.
- The next bottleneck is pinned/asynchronous raw H2D plus exact natural Top-8/MLA/DSA layer integration; raw grid results are still FFN-block evidence, not model tok/s.

## 2026-08-14 Pointer-array expert GEMM batching

- Added cached cublasLt pointer-array layouts for the raw BF16 grid. Each multi-expert call now submits gate, up, and down as one batched projection plus one SiTU launch; the pointer arrays live in a reusable device scratch buffer.
- The 2-expert nonzero CUDA regression now checks the pointer descriptor transfer counter (`144` bytes for its three pointer sets) and four kernel launches. Full WSL CTest remains 26/26 and focused GLM Python remains 35/35.
- Isolated real-shard rerun: 8 experts x 4 tokens measured 1,065,026 ns warm median, 153,395,924 ns cold, 603,979,776 resident bytes, zero warm H2D, and 0.1359% maximum relative CPU error. The latest direct raw per-expert run was 1,648,927 ns warm, so the hot block improved by about 35.4%.
- Single-expert measurements remained on the scalar plan because pointer-array setup was slower. The next bottleneck is pinned/asynchronous staging and full GLM layer integration, not another unverified TPS extrapolation.

## 2026-08-14 BF16 resident-grid output experiment

- Added `CudaBf16OutputMode` with `fp32` as the default and `bf16` as an explicit raw-grid experiment. Gate/up/down cublasLt outputs use BF16 layouts in the experimental mode; a separate BF16 SiTU kernel keeps the intermediate activation in BF16; the public result is converted back to float only after the final D2H copy.
- `test_cuda_dense` now checks the BF16-output path against the rounded CPU reference and checks the halved physical D2H byte count on the two-expert fixture. The focused CUDA regression passed after the change.
- Paired real-shard probe on RTX 5080: FP32 output 1,091,122 ns warm median and 0.135860% maximum relative CPU difference; BF16 output 1,034,950 ns and 0.316691% difference. The speed improvement is modest and the error increases, so the default remains FP32.
- This is a bounded FFN-block experiment. It does not justify a full-model tok/s estimate or automatic quality-mode promotion.

## 2026-08-14 cublasLt workspace tuning

- Added `cuda_cublas_workspace_bytes` and `--workspace-bytes` for the raw pointer-array grid only. The backend reserves one reusable device scratch buffer and passes it to the three projection calls; the default remains zero.
- On the real RTX 5080 probe, zero/8 MiB/16 MiB/64 MiB FP32-output medians were 994,529/986,393/1,073,612/967,790 ns for the same 8-expert/4-token command. The 64 MiB BF16-output run was 1,080,469 ns versus 1,034,950 ns without workspace.
- The workspace preference changes cublasLt heuristic selection and is therefore shape- and output-mode-sensitive. It is exposed for explicit tuning, not enabled globally.

## 2026-08-14 Packed raw expert-grid inputs

- Added `raw_bf16_situ_mlp_grid_packed` without changing the existing common-input API. The caller supplies a flat `[expert][candidate][hidden]` slab; the backend uploads the slab once and points each pointer-array B operand at its expert segment. Scalar fallback also offsets each expert input correctly.
- `test_cuda_dense` uses two nonzero experts with distinct one-token slabs and compares each result to the rounded CPU reference. It also checks 12 bytes of packed activation H2D and 16 bytes of FP32 output D2H on the tiny fixture.
- This is the kernel/scheduling contract needed before exact GLM route assignment can be connected. It is not yet a ragged scheduler or a throughput result.

## 2026-08-14 Expert-major packed-plan preparation

- Added `build_expert_major_packed_plan` next to the existing stable first-use grouping. It validates the `[token][hidden]` input slab, copies each assignment's hidden state into its expert's assignment order, and retains token index, router slot, and contribution.
- The implementation is model-neutral and CPU-only. It deliberately does not infer routes, load GLM tensors, or hide route scatter in CUDA. The next integration step is to bucket these groups by assignment count and feed their slabs to the packed raw-BF16 grid.

## 2026-08-14 Sparse-packed real-shard probe

- Added `--input-mode common|sparse-packed` to `k3x_cuda_glm5x_real_expert_bench`. The sparse mode is deliberately constrained to BF16-rounded, two logical tokens, and alternates token 0/1 across the selected experts before calling the packed raw grid.
- On the two downloaded probe artifacts, common 8-expert/2-token input measured 1,040,559 ns warm median; sparse-packed measured 965,550 ns. The lower latency is a bounded input-addressing result, not a claim about GLM's learned router or full decode.
- BF16-output sparse-packed measured 995,611 ns with 0.3967% maximum CPU-relative difference, so FP32 output remains the safer default for quality-sensitive modes.

## 2026-08-14 Ragged packed-batch dispatch boundary

- The raw BF16 CUDA grid accepts a rectangular `[expert][candidate][hidden]` slab, while real router assignments are ragged. The next bounded step is a stable CPU-only bucketing helper that groups packed expert records by assignment count and retains original group indices.
- The helper will not infer routes, load weights, or scatter outputs. It only makes the existing packed-grid contract callable without padding or silently changing token order; validation remains explicit for hidden width, group payload length, and assignment totals.
- `bucket_expert_major_packed_plan` now groups by assignment count in first-use order, concatenates each group's already-packed hidden slab, and retains source group indices. The C++ test covers separate buckets, repeated-shape grouping, and malformed payload rejection; WSL CTest remains 26/26 and the focused GLM Python suite remains 35/35.

## 2026-08-14 Expert-major output scatter boundary

- The packed CUDA grid returns one output slab per expert group, but exact MoE semantics require contribution-weighted accumulation back into token order. The next bounded helper will consume group-order output slabs and perform explicit validation and scatter on the CPU reference side.
- `scatter_expert_major_outputs` now validates group-output shape and assignment totals, then accumulates each group slab by its retained token index and router contribution. The route scatter remains outside CUDA so the same helper can verify future packed dispatch against the CPU reference.

## 2026-08-14 Latest bounded sparse-packed rerun

- On the latest pushed HEAD, the same two-shard layer-10 probe measured common 927,744 ns/block and sparse-packed 939,149 ns/block, with zero warm weight H2D and 603,979,776 resident bytes in both runs.
- The earlier sample showed sparse-packed lower latency, but this rerun reversed the direction. Treat packed input as a correctness/scheduling contract until a larger repeated sweep explains the variance; do not enable it as a universal speed optimization.

## 2026-08-14 CI evidence boundary and lazy real MoE reference

- The public `correctness` workflow failed on `b94c8b8` because migrated K3X tests referenced historical `results/b0006` through `b0024` artifacts that are intentionally not in the GLM5X repository. The actual C++ build/tests passed; CodeQL also passed.
- Commit `a00beec` marks only those historical evidence/manifest tests as skipped when their recorded artifacts are absent and skips old-baseline-dependent ablations with an explicit reason. New GLM5X tests remain active. GitHub Actions correctness and CodeQL both passed on the pushed commit.
- Added `GLM5XLayer10MoEReference` with official sigmoid router, exact Top-8 normalization/routed scale, shared SwiGLU, and lazy exact raw-BF16 bundle reads. A five-shard real bundle smoke selected 15 unique layer-10 experts and returned identical cold/cached outputs; this is reference correctness evidence only.

## 2026-08-14 Exact MLA/DSA layer boundary and bundle-open reduction

- Added q-residual MLA projection, compressed incremental MLA state, official causal DSA index-key state, and the decoder-layer order connecting them to the lazy natural Top-8 MoE reference. Synthetic full-vs-incremental parity and bundle-loader coverage are green.
- The first real five-shard layer-10 smoke exposed duplicate `GLM5XExpertBundle.open()` calls. A test reproduced two opens, then the loader was changed to share one root-verified bundle object with the attention/indexer/norm readers and lazy MoE closure.
- Focused GLM Python tests passed 42/42 and WSL CTest passed 14/14 after the change. The real smoke measured `250.637263 s` bundle construction, `5.969859 s` cold forward, `0.057331 s` cached forward, and `0.0` maximum cached-output difference. Full-model throughput remains unmeasured.

## 2026-08-14 Ragged expert-major CUDA dispatch

- Added `CudaBackend::raw_bf16_situ_mlp_expert_major` after a compile-failing CUDA test first established the missing API. The backend now consumes the model-neutral packed plan, groups assignments by count, reuses the existing raw-BF16 packed grid per bucket, and scatters weighted outputs through the audited helper.
- WSL CUDA CTest passed 26/26 and CPU CTest passed 14/14. The Windows Python interpreter has no pytest module, so the focused 42/42 Python result remains the last WSL Python evidence rather than a new local rerun.
- On five bounded real GLM-5.2 shard artifacts, deterministic 8-group/10-assignment/2-token expert-major measured `1,380,314 ns` warm median per block versus `1,651,193 ns` common and `1,631,127 ns` sparse-packed. Maximum CPU-relative error was `0.1471%`, resident bytes `603,979,776`, and warm weight H2D `0`.
- This is an executable ragged FFN scheduling boundary, not learned GLM routing, a full layer, or end-to-end tok/s. The next bottleneck is connecting exact router/MLA/DSA outputs and measuring pinned asynchronous staging.

## 2026-08-14 Learned GLM router expert-major probe

- Extended the real-expert benchmark with `learned-expert-major`. It validates and reads the actual layer-10 router BF16 matrix plus FP32 correction bias, uses the already-tested natural Top-8 policy, computes routed contributions with scale 2.5, and loads only the selected expert union.
- A 2 GiB budget was required for two tokens because the real route selected 15 experts and 1.132 GB of BF16 expert roles. The 20-warmup/100-iteration FP32-output run measured 1.905668 ms/block, 0.0866% maximum CPU-relative difference, and zero warm weight H2D. A 4-token run selected 29 experts and measured 3.757986 ms/block with 0.0667% relative difference under a 4 GiB budget.
- The earlier deterministic route is now clearly separated from this learned route. Neither includes MLA/DSA, trunk residuals, logits, or generation, so neither is a model tok/s result.

## 2026-08-14 -- Learned GLM MoE sublayer CUDA boundary

- Commit `8017bd2` adds `learned-moe-layer`. It loads the real layer-10 shared expert (`gate_proj`, `up_proj`, `down_proj`) in addition to the selected routed union, executes routed expert-major scatter plus shared raw-BF16 grid, and compares the sum with the CPU dense reference.
- With two tokens and a 2 GiB resident budget, 15 routed experts plus one shared expert used `1,207,959,552` resident bytes and measured `2,155,188 ns` warm median/block. Maximum CPU-relative difference was `0.000585675588809` and warm weight H2D was `0`.
- With four tokens and a 4 GiB budget, 29 routed experts plus one shared expert used `2,264,924,160` resident bytes and measured `3,968,243 ns` warm median/block. Maximum CPU-relative difference was `0.000429985491792` and warm weight H2D was `0`.
- The opt-in BF16-output cross-check measured `2,374,827 ns` for two tokens and `0.00111866334919` maximum CPU-relative difference, so FP32 output remains the default for this boundary.
- This is the complete bounded MoE sublayer only. MLA/DSA, trunk residuals, final logits, incremental full-layer state, and end-to-end tok/s remain unmeasured.

- Public verification for the implementation/docs pair passed after the change: correctness workflow `31804542819` and CodeQL workflow `31804542791` are green.

## 2026-08-14 — public CI and Dependabot diagnosis

- Pushed learned-router evidence at `222d113`; Linux correctness `31802692875` passed in 2m43s and CodeQL `31802692977` passed in 4m10s.
- Dependabot PRs #1–#4 independently fail only in the Linux job after about 2m30s with 50 missing historical `results/b0006`–`b0024` files. Their static-analysis jobs pass. No Dependabot PR branch was changed.
- The visible Node 20 and CodeQL v3 messages are deprecation annotations, not failed steps. Dependabot and vulnerability-alert APIs return disabled/not-authorized responses, so no CVE alert was verified.
- Rebased Dependabot PRs #1–#4 onto current `main`; their replacement Linux and CodeQL checks are green. The original 50 missing-artifact failures were stale-branch evidence failures, not dependency test regressions.

## 2026-08-14 -- Portable activation handoff

- Added `GLM5XACT` v1 with a 40-byte fixed header, BF16 dtype tag, token/hidden dimensions, payload length, and CRC32C. The Python writer uses a temporary file plus `fsync`/`replace`; the C++ loader validates the complete extent before returning bytes.
- Extended `k3x_cuda_glm5x_real_expert_bench` with `--input-bf16` and `--expected-bf16`. The latter reports absolute/relative output error against an independently written BF16 artifact and is opt-in.
- Verification: commits `11fd058` and `30bf5d4`; WSL CUDA build successful; CTest 27/27 passed in 5.98 seconds; public correctness `31806277016` and CodeQL `31806277022` passed, including the Python writer test. Python syntax compilation also passed locally, while the Windows Python has no pytest.
- Boundary remains intentionally narrow: exact q-residual/MLA/DSA export and full-layer generation are still pending, and no tok/s number is inferred from the artifact test.

## 2026-08-15 -- Linux CI diagnosis and official GLM activation parity

- The recurring `correctness / linux` red check was reproduced from the stale historical boundary: migrated K3X tests attempted to open absent `results/b0006` through `b0024` files. The current `main` head already contains the narrow collection skip, and the latest public correctness run `31807144477` and CodeQL run `31807144571` are green. A local full run initially produced the same class of misleading failures only because `build/k3x_run` had not been built; after configuring the CI-compatible `build` directory, host CTest passed `15/15` and Python passed `300` with `124` explicit skips.
- Dependabot PRs #1 through #4 are open replacement-update branches for setup-python, checkout, numpy, and setuptools. Their replacement Linux/CodeQL checks are green. Repository Dependabot security alerts are disabled (`403` from the alerts endpoint and `security_and_analysis.dependabot_security_updates=disabled`), so no vulnerability alert was verified. No dependency PR was merged or security setting changed.
- The official GLM implementation uses SiLU gated MLP, not the inherited K3X SiTU function. Added `MlpActivation::silu` to CPU/CUDA dense and raw-BF16 grid paths while preserving SiTU as the legacy default. The real layer-10 learned-MoE probe now selects 16 routed experts plus one shared expert and reports GPU/CPU relative error `0.000452667358331`; expected BF16-boundary comparison reports `0.00152439018711` relative error and `0.00006103515625` absolute error.
- Python and C++ route IDs/contributions were compared directly on the same two-token activation. The route sets and weights matched; the former approximately `0.26%` CPU-versus-Python diagnostic was traced to FP32 accumulation order plus comparing FP32 output with a BF16 artifact, not a router or expert-selection bug. The expected-output comparator now rounds the actual output to BF16 before comparison.
- This is still a bounded MoE sublayer boundary. Exact q-residual/MLA/DSA state export, all-layer residuals, final logits, incremental generation, quality evaluation, and end-to-end tok/s remain the next work. Full weights and paid Cloud Run resources remain absent.

## 2026-08-15 -- Final f07d78c CI and Dependabot verification

- Public correctness run `31812923197` for `f07d78c` completed successfully. Linux checkout, Python installation, C++ build, CTest, and the complete Python/cross-language suite all passed; the job took about 3 minutes 51 seconds.
- Public CodeQL run `31812923191` for `f07d78c` completed successfully for both Python and C++. The non-failing Node 20 and CodeQL v3 deprecation annotations remain informational.
- The local WSL regression reproduced the public path with host CTest `15/15`, CUDA CTest `27/27`, and `301 passed, 124 skipped` in `71.25 s` using `/tmp/glm5x-venv/bin/python`.
- Dependabot PRs #1 through #4 remain open update proposals only. Their rebased replacement checks are green: #1 `31805879968`/`31805879919`, #2 `31805884019`/`31805884008`, #3 `31805888033`/`31805888023`, and #4 `31805890586`/`31805890614`. The repository is public, Dependabot security updates are disabled, and the alerts endpoint returns `403` disabled; no vulnerability alert was verified and no PR was merged.

## 2026-08-15 -- Expert-major bucket reuse and shared-dispatch experiment rejected

- The learned CUDA MoE benchmark was tested with a cached immutable bucket list and, for one-token decode, a fused routed-plus-shared expert-major dispatch. Both changes preserved route/output parity but did not improve paired RTX 5080 medians.
- Token-1 five-run medians were `1,307,995 ns` with both changes versus `1,265,441 ns` at the baseline. Token-2 three-run medians were `2,191,291 ns` with bucket caching versus `2,166,726 ns` at the baseline. The token-1 fused path cut grid calls but remained slower overall.
- The experiment was reverted from the default code path and recorded as D-0042. The next performance hypothesis is device-side accumulation to remove per-expert host output copies and CPU scatter; it must first pass exact parity and a paired benchmark.

## 2026-08-15 -- Device-side ragged expert accumulation

- Added a CUDA ragged mix primitive that consumes per-assignment output offsets, token indices, and router contributions. The accumulator is zeroed once per expert-major forward and each bucket kernel adds into the token-major device buffer, so later assignment-count buckets cannot erase earlier token results.
- The backend exposes the path only when FP32 output and `cuda_expert_major_device_accumulate` are selected. It retains the existing host scatter path as the default and falls back automatically for BF16 output.
- Added `raw_bf16_situ_mlp_expert_major_with_shared` and `--fuse-shared 1`. The experimental path adds the shared expert's device output into the routed accumulator before one final D2H, and is restricted to learned-MoE, device accumulation, and FP32 output.
- CUDA synthetic routed-plus-shared parity passed. On the exact two-token layer-10 GLM5XACT handoff, 100-iteration median-of-runs were `2,194,670 ns` baseline, `2,326,186 ns` device accumulation without fusion, and `1,986,460 ns` fused shared accumulation. Fused was approximately `9.49%` lower than baseline in this sweep, while the standalone device path was approximately `5.99%` slower; the spread means neither path is promoted by default.
- WSL host CTest `15/15`, CUDA CTest `27/27`, and Python `301 passed, 124 skipped` passed after the change. The next bottleneck remains full-layer MLA/DSA-to-CUDA state and final-logit/incremental generation, not this bounded MoE handoff.
- A CUDA regression now covers both one-bucket and varied one-/two-assignment bucket plans. WSL host CTest 15/15, CUDA CTest 27/27, and Python 301 passed/124 skipped are green.
- Three paired RTX 5080 layer-10 learned-MoE runs measured median-of-runs `2.492351 ms` baseline versus `1.991721 ms` device accumulation, about `20.1%` lower block latency. Relative GPU/CPU error stayed `0.000571510172449`; this is still only the bounded MoE sublayer, not end-to-end token generation.

## 2026-08-15 -- CI and Dependabot maintenance

- The recurring historical `correctness / linux` failure was not a two-minute runtime timeout. On stale commit `b94c8b8`, C++ build/CTest passed and the Python step failed after about one minute with 50 `FileNotFoundError` cases for K3X `results/b0006` through `b0024` artifacts that GLM5X does not ship. Current `main` already has the narrow explicit-skip boundary and its latest public checks were green.
- Dependabot PRs 1-4 were green replacement proposals for setup-python 7, checkout 7, numpy 2.5.2, and setuptools 84; they were closed after those exact changes were integrated on `a3fb8a8`. The repository vulnerability-alert endpoint returns disabled/403, so no CVE alert was confirmed. CodeQL v4/Node24-compatible action majors are also on main; no paid resource or security setting was changed.

## 2026-08-15 -- Multi-layer CPU logits reference

- Added `GLM5XDecoderModelReference` with per-layer MLA/DSA state tuples, final RMSNorm, LM-head logits, prompt prefill, one-token incremental forward, and greedy generation.
- The focused parity test compares every incremental prompt logit against the multi-token prefill and compares generated tokens with an explicit greedy loop. WSL full Python passed `302 passed, 124 skipped`; no real full-checkpoint or tok/s claim is made.
- The next implementation boundary is loading all real GLM layers into this state contract and exporting the exact hidden handoff to CUDA without changing natural routing.

## 2026-08-15 -- Out-of-core reference layer loader

- Added `GLM5XDecoderModelReference.from_layer_loader`. The model keeps embedding/final-logit tensors plus recurrent per-layer MLA/DSA state, while requesting a decoder layer by ID only during a forward.
- The focused loader test verifies a two-layer forward calls IDs `[0, 1]` and returns both layer records. WSL full Python passed `303 passed, 124 skipped`; WSL host/CUDA CTest remained `15/15` and `27/27`.
- This is a residency contract, not a performance claim. The next step is a real-shard layer provider that reads only the selected layer and its routed expert union, then overlaps its transfer with the preceding layer.

## 2026-08-15 -- Reuse cross-shard bundle readers

- Added `GLM5XDecoderLayerReference.bundle_layer_loader`. It opens the bundle and builds the tensor-reference map once, then creates requested layer objects against the same verified readers; expert payloads remain lazy.
- The bounded bundle test requested the same layer twice and observed one bundle open, while full Python stayed green at `303 passed, 124 skipped`.
- This removes repeated root verification from the reference layer provider but does not claim a tok/s gain; real layer admission and CUDA overlap remain the next performance boundary.

## 2026-08-15 -- Lazy K3X payload admission measurement

- Added `K3XReader.open(..., verify_payloads=False, verify_root=False)` and matching `GLM5XExpertBundle.open`/layer-loader flags. Eager verification remains the default; lazy reads verify selected tensor and auxiliary CRCs on first access under a lock.
- On the real `build-glm5x-hf-probe/first-shard.k3x` (5.34 GB, 35 tensors), lazy directory open took `0.003153 s`; first selected tensor read plus CRC took `8.989272 s`; strict eager open took `49.816001 s`. This is cold-start/traffic evidence, not tok/s.
- The measured startup reduction is useful for layer-at-a-time runtime admission, but `verify_root=False` is an explicitly weaker integrity mode and remains experimental until full runtime telemetry and recovery policy are added.

## 2026-08-15 -- Latest public verification

- After `0040791`, GitHub Actions correctness run `31824430721` passed in `2m43s`; C++ build/CTest and the full Python/cross-language step passed. CodeQL run `31824430714` also passed, with C++ analysis completing in `3m39s` and Python analysis in `2m29s`.
- The CodeQL overlay-base warning was a non-failing fallback annotation. No current Linux failure or two-minute timeout is present on `main`.
- The repository has zero open Dependabot PRs. Dependabot security alerts are disabled by repository configuration, so the API's `403` response confirms unavailable alerting rather than a confirmed vulnerability count.

- The documentation-only follow-up `31fe66f` also passed Linux correctness as `31824846842`; CodeQL `31824846833` passed with C++ in `2m59s` and Python in `2m19s`. The overlay-base annotation remains non-failing.

## 2026-08-15 -- Bounded reference trunk-layer cache

- Added `layer_cache_capacity` to `GLM5XDecoderModelReference.from_layer_loader`. Capacity zero preserves the prior out-of-core behavior; a positive value retains validated layer objects in an LRU and leaves expert payload caching to the layer/provider policy.
- The focused cache test passed with identical logits and loader calls `[0, 1]` across two forwards at capacity 2. The full WSL Python suite passed `305 passed, 124 skipped` in `73.05 s`.
- A small single-threaded synthetic sample measured `1.0073 ms` per forward with capacity 0 and `1.0310 ms` with capacity 2. This is too small to claim a speedup; the value is the measured elimination of repeated layer construction/admission calls, which must be rerun on real all-layer data.

## 2026-08-15 -- Dense GLM MLP reference boundary

- GLM-5.2 declares the first three decoder layers as dense MLPs. Added `GLM5XDenseMlpReference` with official `silu(gate_proj(x)) * up_proj(x)` then `down_proj` ordering and the shared `GLM5XMoEForward` schema.
- `GLM5XDecoderLayerReference.from_bundle` and `bundle_layer_loader` now accept `mlp_type="dense"` plus an optional `indexer_source_layer`; sparse MoE remains the default. Dense forwards report empty routing and zero expert loads rather than inventing a router result.
- Focused layer-reference tests passed `3/3`; the WSL Python suite passed `306 passed, 124 skipped` in `73.57 s`. No real all-layer weights, final logits, MTP, CUDA logits, or tok/s were measured.

## 2026-08-15 -- Configuration-driven real bundle model factory

- Added `GLM5XDecoderModelReference.from_bundle`, which retains one cross-shard bundle reader and creates layers on demand from the official configuration. It resolves explicit `mlp_layer_types` or `first_k_dense_replace`, and maps each `shared` indexer layer to its nearest preceding `full` source.
- The factory reads embedding/final norm/LM-head tensors when present and accepts explicit head overrides for bounded partial probes. It never treats those overrides as a full-checkpoint result.
- A synthetic three-layer bundle exercised dense/dense/sparse selection, shared indexer reuse, sparse expert loading, and full-vs-incremental logits. Full WSL Python passed `307 passed, 124 skipped`; host CTest passed `15/15`.
- On the five bounded real artifacts, lazy factory setup took `0.066806 s`, layer-0 admission took `4.278880 s`, and a one-token real layer-0 CPU forward took `0.036846 s` with `[1,1,6144]` output. This is not full-model throughput.

## 2026-08-15 -- Bundle factory public verification

- The factory implementation plus documentation was pushed as `3a86ca3`. GitHub correctness `31828512721` and CodeQL `31828512789` both passed.
- The implementation is now the verified model-level starting point for complete shard coverage. Full checkpoint materialization, CUDA final logits, MTP, and end-to-end tok/s remain open.

## 2026-08-15 -- Dense-layer public verification

- Pushed implementation commit `fb3aa7d`. GitHub correctness `31826654966` passed C++ configure/build/CTest and the full Python/cross-language suite; CodeQL `31826655082` also passed.
- The hosted Linux job took about 7 minutes 10 seconds, with the Python step taking about 5 minutes 19 seconds. This is a hosted CI wall-time observation and does not change the local 73.57-second WSL test measurement or imply a runtime timeout.

## 2026-08-15 -- CUDA staging and local full-checkpoint stream

- The Python reference bundle/model factories now accept `device="cuda"`. Embedding, norms, projections, dense tensors, router tensors, and selected expert payloads are staged on the target device; CPU remains the default. CUDA layer/model parity passed in WSL.
- `convert-shards --delete-source` writes a source-deleted marker only after strict artifact verification and before unlinking the source shard. The marker lets a retry trust the completed artifact without redownloading a shard.
- `tools/stream_glm5x_checkpoint.py` uses public HF metadata and resumable HTTP Range downloads. The first two official GLM-5.2 shards converted successfully to `.k3x` and their source files were deleted; the third shard is currently downloading. No full model has been assembled and no TPS claim is permitted yet.
- Current local gates after the change are Python `311 passed, 124 skipped` in `79.98 s`, host CTest `15/15`, and CUDA-only layer/model parity. The next bottleneck is completing the 282-shard bundle and then proving full-layer hidden-state/final-logit parity before optimizing CUDA scheduling.

## 2026-08-15 -- Public verification of `6fb2da1`

- Pushed `6fb2da1` to `main`. Linux correctness run `31831711520` passed in `2m38s`; CodeQL run `31831711580` passed for both Python and C++.
- The local materializer reached 4/282 finalized `.k3x` shards while the hosted checks ran. The historical Linux failure notification remains tied to stale commit `b94c8b8`, not this push.

## 2026-08-15 -- Remove duplicate final bundle scan

- The stream previously used strict `K3XReader.open` for every artifact again during final bundle assembly. Since each artifact is strict-verified before its deletion marker, the stream now indexes the final bundle with lazy payload/root admission while the public CLI remains strict.
- Focused bundle and stream tests passed `4/4`. The active process was started from the prior code, so it must be resumed/restarted before the new final-assembly path is used.

## 2026-08-15 -- Resume guard and final public verification

- `db2cf37` adds a completed-artifact/marker check before download. A restarted process now retains the incomplete shard `.part` and avoids redownloading finalized source-deleted shards.
- Focused tests passed `7/7`; Linux correctness `31833153961` passed in `2m45s`; CodeQL `31833154040` passed for Python and C++.
- A redundant shard-1 partial created by the pre-fix process remains as an isolated `.part` file; it is not used by the marker-aware path. Shard 9's large `.part` is the valid resumable input.

## 2026-08-15 -- Three-worker local conversion

- Added disjoint range arguments and launched three `--no-assemble` workers. The first overlap sample finalized 12, 102, 103, 193, and 194 while each worker continued to prefetch its next source.
- The stream is now a coordinator/worker pipeline: workers never write the shared bundle, and final lazy assembly waits for all 282 source-deleted markers. Sustained throughput and failure recovery still need a longer sample.

## 2026-08-15 -- CI, Dependabot, and live conversion verification

- The recurring Linux failure was reproduced from historical run `31795400168` on stale commit `b94c8b8`: the Python step ended after `60.08s` with `50 failed, 294 passed, 74 skipped`, primarily because migrated historical `results/` files were absent. The follow-up guard commit `a00beec` made absent historical evidence an explicit skip; current `main` no longer exhibits that failure.
- Latest pushed head `5c5a2eb` is green in Linux correctness run `31835150475` and CodeQL run `31835150428`. The Linux job completed in `2m55s`; this is CI wall time, not model execution time.
- Dependabot PRs 1-4 are closed and their dependency/action changes are already present on `main`. The repository has Dependabot security alerts disabled, so the REST endpoint's `403` means alerting is unavailable, not that a number of vulnerabilities was confirmed.
- At 05:03 KST, 23/282 `.k3x` artifacts and 23 source-deleted markers were present. Three workers had produced 13 new artifacts in 22m37s, an initial aggregate sample of `34.5 shards/hour`; 259 artifacts remained. The estimate is conditional and must be refreshed after a longer sample.

## 2026-08-15 -- Current CI, dependency, conversion, and CUDA verification

- The docs and expert-major follow-up head `142ec22` is now public. GitHub check-runs `31840980835` (Linux correctness) and `31840980920` (CodeQL Python/C++) are both successful. The recurring `Linux (push)` failure notification is from historical run `31795400168` on stale commit `b94c8b8`, where absent migrated `results/b0006..b0024` files caused `50 failed, 294 passed, 74 skipped`; it is not an active failure on `main`.
- Dependabot update PRs 1–4 are closed and their action/package bumps are present on `main`. Repository Dependabot security alerts are disabled; the API returns `403`, so no vulnerability count is confirmed. The local WSL environment passes `python -m pip check`; the stale Windows `.venv` has missing optional torch dependencies and was not changed.
- At 05:17:32 KST, three disjoint workers had finalized 32/282 artifacts and source-deleted markers. 22 artifacts were produced after the 04:39:53 launch in 37m39s, approximately 35.1 shards/hour; 250 remain. This gives a conditional `~7.1 h` conversion estimate and a conservative `7.5–9 h` planning window, not a completion promise. C: had approximately 1.550 TB decimal free.
- A four-token real layer-10 learned-MoE CUDA comparison at 4 GB resident budget measured `4,317,561 ns` baseline versus `3,741,291 ns` fused/device-accumulate warm median, about 13.3% lower in the bounded warm sample. Both paths retained GPU-vs-CPU maximum relative error `0.000643727718852`; the fused cold latency was higher. This is not full-model tok/s.

## 2026-08-15 -- MXFP4 encoder quality gate

- Added a chunked reference encoder for native E2M1/E8M0 payloads with explicit `max_abs` and `mse` scale modes. Focused tests passed `11/11`; the full WSL Python suite passed `318 passed, 124 skipped` in `141.53 s`.
- On real layer-10 expert 4, three BF16 projections shrank from `75,497,472` to `20,054,016` bytes (`26.5625%`). Max-abs scales produced `19.861969%` FFN relative L2 error; MSE scales produced `19.069034%` while taking `7.439 s` instead of `0.442 s` for the three projections.
- The encoder is deliberately not connected to conversion or the default runtime. The next quantization task is calibrated outlier/mixed-precision storage, with exact raw-BF16 reference retained for quality comparisons.

## 2026-08-15 -- Reference expert-major batching gate

- Added an explicit `execution_mode` switch to the GLM5X reference MoE, decoder-layer factory, and model factory. The default loop path is unchanged; the new expert-major path batches selected assignments with `torch.bmm` and retains the exact route/scatter contract.
- Focused parity passed `11/11`; the full WSL Python suite passed `319 passed, 124 skipped` in `132.68 s`.
- Real RTX 5080 evidence was mixed: four-token direct MoE improved from `21.670 ms` to `18.652 ms`, but one-token direct MoE worsened from `5.584 ms` to `7.359 ms`, and the full four-token layer worsened from `18.676 ms` to `20.082 ms`.
- The grouped path temporarily allocates about `1.97 GB` for stacked weights on the four-token probe. Keep it opt-in and pursue the resident-weight-aware C++ expert-major path for production.

## 2026-08-15 -- Full-bundle reference benchmark gate

- Added `tools/benchmark_glm5x_reference.py`. It takes `--bundle`, `--config`, and explicit comma-separated `--prompt` IDs, then measures prefill, TTFT, incremental decode, generated IDs, cache/execution settings, and CUDA peak memory.
- Strict bundle admission is the default; `--lazy-bundle` is explicit. No tokenizer or estimated TPS is hidden inside the tool. The first real full-model JSON remains pending the 282-shard assembly.

## 2026-08-15 -- Current CI and materialization refresh

- Commit `0e6972d` passed Linux correctness run `31842116346` and CodeQL run `31842116338`; the red Linux notification refers to the historical stale run documented above.
- Dependabot update PRs 1--4 are closed and their dependency/action changes are present on `main`. Dependabot security alerts are disabled for this public repository, so the API returned `403` and no CVE count was verified.
- At 06:24 KST, 72/282 shard artifacts and source-deleted markers were finalized. The three disjoint workers produced 62 artifacts after the 04:39:53 launch, an observed aggregate of approximately 35.8 shards/hour. 210 artifacts remain, giving approximately 5.9 hours at the observed rate and a conservative 6--7 hour conversion/indexing window. Full-model logits and end-to-end tok/s remain unmeasured.
- Added `tools/monitor_glm5x_full_gate.sh` and started it as a hidden local WSL process. It polls the 282-marker gate, then runs lazy bundle assembly and separate cold (`build-glm5x-full-reference-cuda-cold.json`) and 8 GiB cached (`build-glm5x-full-reference-cuda-cached.json`) CUDA reference benchmarks; no cloud or paid resource is involved.
- Added opt-in `expert_load_workers` to the Python reference bundle path. Selected expert payload reads can overlap through a bounded thread pool while tensor decoding remains serial; focused MoE tests passed `4/4` and layer/model reference tests passed `8/8`. No real full-bundle timing is claimed before the active materialization completes.
- Added `GLM5XExpertPayloadCache`, an exact bounded host-byte cache shared by the bundle reader. In a five-shard real layer-10 cold probe, the same eight experts took `3.070666336 s` on the first call and `0.094567934 s` on the second with a 1,000,000,000-byte cache; output max difference was `0.0`, with 8 hits, 8 misses, and 0 evictions. This is a bounded sublayer result, not full-model TPS.
