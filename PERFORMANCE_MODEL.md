# GLM5X Performance Model

This document separates dimension-derived traffic bounds from measured runtime results. It must not be read as a TPS claim.

## Current GLM-5.2 shape

The official configuration used by the local stream is 78 decoder layers, hidden size 6144, 64 attention heads, q-lora rank 2048, qk-nope 192, qk-rope 64, value head 256, KV rank 512, 256 routed experts, Top-8 routing, and BF16 source tensors. The first three layers are dense; the remaining 75 layers are sparse with one shared expert.

## Derived BF16 traffic bound

Using two bytes per BF16 element and the official dimensions, the storage footprint of the non-routed trunk is approximately:

| Component | Bytes per layer | Count | Total |
| --- | ---: | ---: | ---: |
| Sparse attention, norms, router, indexer, and shared expert | 424,305,664 | 75 | 31,822,924,800 |
| Dense first-layer MLP variant | 801,792,512 | 3 | 2,405,377,536 |
| **All non-routed trunk** |  | 78 | **34,228,302,336 (31.88 GiB)** |

One routed expert has three BF16 projections totaling 75,497,472 bytes. A one-token Top-8 union therefore adds 603,979,776 bytes per sparse layer. If every layer is reloaded for every token, the dimension-derived one-token fetch is approximately 79,526,785,536 bytes (74.07 GiB), before file headers, alignment, runtime buffers, or duplicate reads.

This bound explains why a naive BF16 out-of-core loop cannot meet the 10 tok/s objective over one PCIe link. It is a design constraint, not a measured device bandwidth or latency result.

## Resident-memory implications

- The BF16 embedding and LM head each occupy about 1.90 GiB at the configured vocabulary and hidden size.
- The exact FP32 logits path now lazily promotes a 3,806,330,880-byte LM-head matrix after its first use and releases the BF16 source from the active model object. This avoids repeated full-vocabulary conversion; the first-use peak includes both matrices and the steady-state FP32 residency must be included in the 16 GiB VRAM budget.
- Keeping the complete BF16 trunk resident is not possible on the target GPU. The path toward 10 tok/s requires a measured mixed-precision trunk representation, fused dequantization/GEMM, or a different resident partition; expert streaming alone cannot overcome the trunk traffic bound.

## Measurement gates

The following remain measured-runtime requirements and are not inferred from this document.

1. Complete the 282-shard K3X bundle and verify exact final logits and greedy token parity.
2. Record prefill/decode tok/s, TTFT, peak VRAM, host RSS, logical storage reads, physical NVMe reads, H2D bytes, layer timings, and cache residency on the RTX 5080.
3. Compare exact BF16, mixed-precision trunk, and expert-major residency modes with the same prompt and quality checks.
4. Accept an optimization only when its measured traffic/latency improvement is larger than its quality and memory cost.

Until those gates run, no end-to-end TPS number is reported.
