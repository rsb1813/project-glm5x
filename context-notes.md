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
