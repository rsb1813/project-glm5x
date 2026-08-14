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
- Final benchmark samples at commit `31678d1`: 8 experts/1 token median 2,675,694 ns, 8 experts/4 tokens median 5,437,941 ns (1,359,485 ns/token), 8 experts/8 tokens median 8,945,204 ns (1,118,151 ns/token), and 16 experts/1 token median 5,318,664 ns. All maximum absolute errors were 0.
- A shared-input row-tiled kernel and a lookup/bit-scale MXFP4 decoder were tested and rejected after repeated slowdowns. They were not kept in the accepted source.
- Expert-major candidate batching is a measured optimization candidate for MTP/DSpark integration. It must not be treated as speculative acceptance or full-model throughput until the exact target verifier is connected.
