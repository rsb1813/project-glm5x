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
