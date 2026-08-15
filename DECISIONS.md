# GLM5X Engineering Decisions

## D-0001 — Pivot from Kimi K3 to GLM-5.x

- Decision: build GLM5X as a new repository and use GLM-5.2 as the first executable target.
- Alternatives: continue K3X, or fork K3X in place.
- Evidence: GLM-5.x has a smaller active expert set and a built-in MTP path that matches the existing expert-cache and speculative roadmap better.
- Accepted because: a clean repository prevents Kimi-specific graph and weight assumptions from leaking into the GLM runtime.
- Revisit: when GLM-5.3 configuration and weights are public.

## D-0002 — Preserve the K3X storage core

- Decision: retain the aligned extent, checksum, resume, cache, prefetch, and benchmark interfaces as an internal compatibility core.
- Alternatives: rewrite storage immediately, or use a general safetensors/GGUF runtime.
- Evidence: these boundaries are independent of Kimi tensor names and already have focused tests.
- Accepted because: it shortens the path to a bounded GLM artifact while keeping storage streaming and correctness gates.
- Revisit: after the first GLM-5.2 end-to-end trace.

## D-0003 — Delete official K3 artifacts, preserve synthetic fixtures

- Decision: delete only verified official K3 source/derived artifact directories and retain synthetic fixtures used by the old repository's tests.
- Alternatives: delete the entire old K3X repository or delete every `.safetensors` file.
- Evidence: the pre-delete inventory identified official artifacts separately from `synthetic-k3-source-v1` fixtures.
- Accepted because: it removes Kimi weight storage without destroying reusable correctness fixtures or historical code.
- Revisit: never unless the user explicitly requests full old-repository archival deletion.

## D-0004 — Start TurboQuant as a reference KV path, not a weight format

- Decision: implement TurboQuant-inspired rotation and vector quantization first for KV cache only, with exact/lossless reference mode and asymmetric K6/V4 as the initial balanced candidate.
- Alternatives: apply TurboQuant-like quantization directly to expert weights, use uniform FP8 KV only, or implement a CUDA kernel before a reference contract exists.
- Evidence: the TurboQuant paper targets online vector/KV quantization, while GLM-5.2 short-context performance is still expected to be dominated by expert weight movement. The paper's quality observations were not measured on GLM-5.2.
- Accepted because: separating KV capacity from weight streaming prevents an unsupported 3-bit weight claim and gives the 600k/1M path a testable, reversible baseline.
- Revisit: after GLM-5.2 DSA state parity, long-context quality tests, and a packed RTX 5080 kernel benchmark.

## D-0005 -- Fix the GLM-5.2 shape at a manifest boundary before weights

- Decision: load GLM-5.2 dimensions from a validated descriptor plus safetensors index manifest, and keep the first CUDA evidence as a bounded synthetic expert benchmark.
- Alternatives: hard-code tensor names in CUDA, download the 1.5 TB source checkpoint first, or reuse Kimi K3 graph dimensions.
- Evidence: the local GLM-5.2 config/index reports hidden size 6144, 78 layers, 256 routed experts, Top-8, expert intermediate size 2048, DSA index Top-K 2048, and a 1,506,659,919,872-byte source index. The 8-expert shaped CUDA block measured 2,675,694 ns warm median for one token and 1,359,485 ns/token in a four-candidate block with maximum absolute error 0.
- Accepted because: shape and storage contracts can be verified without a full checkpoint, and expert-major candidate batching already shows a measured amortization path.
- Revisit: after a bounded real shard is materialized and exact GLM DSA/MoE layer parity exists.

## D-0006 -- Do not keep unverified MXFP4 micro-optimizations

- Decision: retain the existing scalar MXFP4 grid kernel as the accepted reference CUDA path for this milestone.
- Alternatives tested: shared-input row tiling and E2M1 lookup-table/E8M0 bit-scale decoding.
- Evidence: both variants preserved synthetic zero-weight output parity, but the 8-expert 1-token warm median increased from roughly 2.25--2.27 ms to roughly 3.00 ms for row tiling and 2.53--2.56 ms for lookup/bit-scale decoding over repeated 100-iteration runs.
- Rejected because: correctness alone is insufficient for the stated TPS goal, and the observed regressions would make a full 78-layer path slower.
- Revisit: only after a tensor-core or batched GEMM path changes the launch and memory balance.

## D-0007 -- Allow resident exact weights in expert-major batch verification

- Decision: permit CUDA expert-major verification to use either transient or resident weights, and make `mxfp4_situ_mlp_batch` acquire exact packed/scales through `ResidentWeightTable` when resident mode is selected.
- Alternatives: keep expert-major transient-only, rely on the host expert cache alone, or enable proxy/pruned weights.
- Evidence: the new CUDA regression passes exact CPU parity twice; the first call uploads 1,105 bytes in the tiny fixture and the second call uploads 0 while recording three resident cache hits. The GLM-shaped 8-expert/4-token benchmark uploads 160,432,128 bytes cold and 0 bytes in warm samples, with maximum absolute error 0.
- Accepted because: this reduces repeated VRAM transfer without changing routing, weights, or verification semantics.
- Revisit: after exact variable-union expert grouping and a tensor-core/dequantized resident kernel are measured.

## D-0008 -- Keep dequantized BF16 resident grid experimental

- Decision: add an opt-in `CudaMxfp4Execution::dequantized_bf16` path that dequantizes each exact MXFP4 expert once into resident BF16 storage and executes the expert grid through cublasLt. Keep native MXFP4 as the default and retain a capacity-bypass fallback to the native group path.
- Alternatives: replace native MXFP4 globally, keep only the per-expert BF16 batch path, or pre-store every expert in BF16.
- Evidence: on the RTX 5080 GLM-shaped 8-expert/4-token fixture, BF16 grid median was 2,582,527 ns/block versus 5,394,131 ns for native grid, with zero error on the zero-weight contract fixture. A nonzero deterministic packed pattern measured 0.95% maximum relative difference against a native GPU reference. Resident weight bytes increased from 160,432,128 to 603,979,776.
- Accepted because: the measured speedup is material and the exact native path remains available; the memory multiplier and missing quality benchmark make a default switch unjustified.
- Accepted guardrail: BF16 capacity is preflighted before admission, warm resident keys are excluded from the remaining-byte calculation, and insufficient capacity falls back to native before any partial BF16 admission. The released-dimension check reproduced a 14,319,240 ns fallback versus 6,333,866 ns native with zero oracle error.
- Revisit: after nonzero GLM shard parity, VRAM-bank pressure measurements, and end-to-end DSA/MoE quality tests.

## D-0009 -- Bind descriptor DSA metadata to a CPU reference state first

- Decision: add `GLM5XDSAConfig` and `GLM5XDSAState` as a CPU/reference bridge between descriptor index metadata, index keys, and the existing compressed KV cache. Keep exact per-query top-k refresh as the correctness path and expose stale refresh cadence only as an explicit experiment.
- Alternatives: leave TurboQuant as a standalone cache, implement a CUDA DSA kernel before a state contract, or infer the learned indexer from the model name without metadata validation.
- Evidence: the new tests pass exact top-k/attention selection with lossless KV, verify the `index_topk_freq` refresh boundary, and compute allocation-free 600k/1M state estimates of 201,637,504 and 336,062,512 bytes for the documented BF16-index/K6/V4 shape.
- Accepted because: it creates a testable state boundary without claiming that synthetic index keys are the learned GLM indexer. The reference and fast refresh semantics are visible and reversible.
- Revisit: when a real GLM shard or official indexer projection is available, before enabling any CUDA or stale-selection path by default.

## D-0010 -- Keep indexer projections explicit and weight-agnostic

- Decision: represent DSA query/key projections as explicit matrices in `GLM5XDSAIndexer` and feed projected keys into `GLM5XDSAState`; do not infer tensor names or silently synthesize official weights.
- Alternatives: hard-code a guessed GLM tensor layout, keep direct synthetic index keys only, or postpone the projection boundary until the full checkpoint is present.
- Evidence: the projection test passes exact projected-key top-k selection with a synthetic matrix, while the manifest still reports no downloaded GLM shard.
- Accepted because: the runtime contract can be tested now and swapped to official tensors later without changing cache or refresh semantics.
- Revisit: when the official GLM indexer tensor map and a nonzero shard are available.

## D-0011 -- Resolve shared indexer roles from the official metadata index

- Decision: parse `indexer_types` from the GLM descriptor and resolve a `shared` layer's indexer tensors to the nearest preceding `full` layer in `GLM5XTensorManifest`; accept only the observed `wk`, `wq_b`, `weights_proj`, and `k_norm` components.
- Alternatives: duplicate shared tensors into every layer, assume a fixed four-layer pattern, or postpone role resolution until payload download.
- Evidence: the local official metadata reports 59,585 tensors across 282 shards and the observed full/shared pattern resolves layer 3 -> 2, layer 7 -> 6, and layer 77 -> 74 with concrete shard names.
- Accepted because: role and shard selection can be verified from the small index file without downloading 1.5 TB, while tensor shapes remain a separate bounded-shard gate.
- Revisit: if opened shard headers or official loader code show a different sharing rule.

## D-0012 -- Use header-only safetensors inspection before conversion

- Decision: inspect bounded real shards with `safe_open().get_slice()` and compare their names against the manifest before any tensor conversion; do not call `get_tensor()` during the header gate.
- Alternatives: load the first shard tensors into RAM, trust the index without opening a shard, or start full-model conversion before shape validation.
- Evidence: the first 5,342,821,416-byte shard validated all 35 names and representative BF16 shapes without materializing payloads; the local machine has no complete checkpoint.
- Accepted because: it gives a real checkpoint boundary with bounded memory and catches index/header drift before conversion.
- Revisit: when a streaming converter can transform the same shard ranges into GLM5X extents.

## D-0013 -- Reuse K3X extents for the first real GLM shard

- Decision: use the existing aligned K3X extent/directory/checksum layout for a bounded raw-BF16 GLM shard, add `DType.BF16`, and keep tensor names in a GLM5X sidecar until the model-specific directory is complete.
- Alternatives: convert BF16 to FP32 immediately, invent a second binary container before a reader exists, or wait for the complete 1.5 TB checkpoint.
- Evidence: the first 5,342,821,416-byte shard converted with an 8 MiB maximum source read; Python checks passed and the WSL C++ reader returned exit 0 on the 5,342,863,616-byte artifact.
- Accepted because: it proves real shard streaming and reader compatibility without loading the full model, while keeping the native K3X storage core and a reversible experimental boundary.
- Revisit: when resumability, expert directories, and BF16 CUDA consumption are implemented.

## D-0014 -- Make GLM conversion independently resumable per shard

- Decision: keep one `.k3x` artifact, sidecar, and source/config-fingerprinted resume ledger per safetensors shard. Reuse is allowed only for the canonical prefix of tensor extents after validating expected IDs, aligned offsets, lengths, source CRCs, partial-file CRCs, and the ledger file UUID. `convert-shards` orchestrates these units and skips only finalized artifacts whose source digest and reader metadata still match.
- Alternatives: build one monolithic checkpoint writer, trust a JSON ledger's CRC values without recomputing source bytes, or delete partial work and restart a shard after interruption.
- Evidence: the new red-green tests resume a two-tensor shard after one completed extent, reject changed/corrupted state through the existing K3X error boundary, record complete same-shard expert triples in `EXPT`, and convert two manifest shards independently; the focused GLM suite passes 28 tests.
- Accepted because: Cloud Run/local worker preemption can retry a bounded shard without requiring full-model RAM/VRAM residency, while canonical validation prevents a syntactically valid but stale ledger from silently producing a wrong artifact.
- Revisit: when cross-shard expert bundles, resumable object-store uploads, and full GLM quality parity are implemented.

## D-0015 -- Keep the official GLM DSA indexer separate from the generic DSA state

- Decision: add `GLM5XOfficialDSAIndexer` as a separate reference boundary. It consumes `q_resid` through `wq_b`, projects hidden states through `wk` plus LayerNorm, applies the configured indexer RoPE convention, computes per-head ReLU scores and `weights_proj` aggregation, then applies causal masking and Top-K. The existing equal-width `GLM5XDSAIndexer/State` remains available for descriptor-shaped cache experiments.
- Alternatives: force official `[32,128]` queries and `[128]` keys into the old flattened index-width API, silently approximate the indexer with a single projection, or wait for the entire checkpoint before fixing the formula.
- Evidence: the official Transformers implementation documents the five indexer tensors and scoring order; independent synthetic parity passes, and a bounded run loaded only the five layer-0 indexer tensors from the real first shard with shapes `wq_b=(4096,2048)`, `wk=(128,6144)`, `weights_proj=(32,6144)`, and 128-element LayerNorm vectors.
- Accepted because: it preserves a correct, inspectable reference mode and avoids claiming that the existing generic projection matches GLM's learned indexer. It also keeps real-shard reads bounded and reversible.
- Revisit: when q-residual production weights, MLA latent projections, cache update semantics, and nonzero real-shard quality parity are connected.

## D-0016 -- Accept raw-BF16 expert directory records in the portable reader

- Decision: allow `EXPT` links to validated raw BF16 tensors (`dtype=BF16`, `quantization=NONE`, no auxiliary extent) in addition to native MXFP4 tensors. Keep the storage-slice expert loader strict until GLM BF16 execution exists.
- Alternatives: omit `EXPT` records for raw BF16, accept every dtype/quantization combination, or convert raw BF16 experts to MXFP4 before writing the staging artifact.
- Evidence: the second real GLM shard produced 70 complete raw-BF16 expert records. Python reader validation passed, while the C++ reader rejected the same artifact because its expert-directory check required `quantization=1`. After the narrow validator change, both 5.3 GB probe artifacts pass C++ metadata-only validation.
- Accepted because: the directory now describes the actual staging payload without weakening global tensor validation or falsely enabling the K3 MXFP4 loader.
- Revisit: when cross-shard expert bundles and the exact GLM BF16 execution path are connected; the native MXFP4 path remains the production reference.

## D-0017 -- Assemble cross-shard experts as a copy-free index

- Decision: keep each converted safetensors shard as an independent `.k3x` artifact and emit a separate `glm5x-expert-bundle-v1` JSON index that references complete `gate_proj/up_proj/down_proj` roles by artifact-relative path and exact extent metadata.
- Alternatives: concatenate all shard payloads into one new file, duplicate split roles into every shard, or make the runtime scan every sidecar on each expert request.
- Evidence: the two downloaded probe artifacts index in about 12 seconds and yield 70 complete experts across 247 tensors without a payload copy. Duplicate roles are rejected and incomplete groups are explicit.
- Accepted because: it preserves restartable shard ownership and avoids another multi-gigabyte write while giving the future runtime deterministic random access to all three roles.
- Revisit: when object-store URLs, bundle relocation, cross-process locking, and exact BF16 expert loading are implemented.

## D-0018 -- Validate bundle references before returning BF16 expert bytes

- Decision: make the reference bundle loader recheck artifact identity and every referenced tensor's dtype, quantization, shape, offset, length, logical length, and CRC before returning a role payload. The initial GLM path accepts raw BF16 with no auxiliary extent only.
- Alternatives: trust the JSON bundle after assembly, validate only the artifact root digest, or concatenate all three role payloads into a new file.
- Evidence: layer 10 expert 0 from the second real shard matched all three source safetensors role tensors byte-for-byte at 25,165,824 bytes per role. A tampered offset is rejected before payload return.
- Accepted because: the runtime can use copy-free random access while stale or manually edited bundle metadata cannot silently feed the model.
- Revisit: when a native C++/CUDA bundle reader exists and its validation cost is measured against prefetch deadlines.

## D-0019 -- Use canonical tensor IDs for the first C++ cross-shard loader

- Decision: let the bounded C++ loader search multiple validated `Reader` instances by the canonical GLM tensor-name FNV-1a ID, then validate raw BF16 metadata and CRC before returning the three role vectors. Keep JSON parsing outside this hot path.
- Alternatives: parse `glm5x-expert-bundle-v1` JSON in C++, concatenate all shard payloads, or require one reader per expert role from the caller.
- Evidence: the real two-shard host gate found layer 10 expert 0 across the second artifact and loaded 75,497,472 bytes with CRC checks in 465,087,758 ns under WSL. The Python bundle had already established the same bytes against safetensors.
- Accepted because: it keeps the C++ runtime dependency-free and deterministic while the bundle JSON remains the orchestration/index artifact. Duplicate or missing role IDs fail closed.
- Revisit: when hundreds of shard readers are opened concurrently and a parsed persistent bundle map measurably reduces lookup or I/O scheduling cost.

## D-0020 -- Keep FP32 resident conversion as the real-expert CUDA reference

- Decision: for the first real GLM expert CUDA bridge, decode exact raw BF16 payloads to host FP32 and use the existing resident dense SiTU path. Keep `bf16-rounded` resident execution experimental and do not make it the default.
- Alternatives: use the existing BF16-rounded cublasLt path immediately, upload raw BF16 bytes through a new kernel, or quantize the real shard to MXFP4 before the first execution gate.
- Evidence: layer 10 expert 0 on RTX 5080/WSL measured 275,473 ns warm median in FP32 resident mode with 150,994,944-byte resident weights and CPU max absolute error `8.38190317154e-09`. BF16-rounded used 75,497,472 resident bytes but measured 28,154,650 ns warm median and 0.1828% maximum relative CPU difference.
- Accepted because: FP32 provides a numerically tight, independently verifiable execution reference while the current BF16 plan is materially slower. The resident memory cost is bounded for one expert and can be revisited after a direct BF16/tensor-core path.
- Revisit: after direct BF16 storage views, pinned H2D, tensor-core cublasLt algorithm selection, and multi-expert resident pressure are benchmarked with nonzero real shards.

## D-0021 -- Cache repeated BF16 host conversion by tensor identity

- Decision: cache the FP32-to-BF16 host conversion used by `dense_situ_mlp` using tensor ID, source pointer/byte size, and matrix shape. Keep activation conversion per call because activations are mutable.
- Alternatives: leave conversion inside every call, require callers to pre-convert weights, or add a raw-BF16 `DenseWeightView` API before measuring the current bottleneck.
- Evidence: the real layer 10 expert BF16-rounded path fell from 28,154,650 ns to 236,593 ns warm median after caching, while resident bytes stayed 75,497,472 and GPU-vs-CPU relative error stayed 0.1828%. FP32 rerun was 271,493 ns.
- Accepted because: this removes a measured host-side cost without changing CUDA math or model weights. The cache invalidates when pointer, byte size, or shape changes.
- Revisit: when direct raw-BF16/tensor-core views can remove the remaining host decode and resident representation tradeoff.

## D-0022 -- Prefer BF16 residency for multi-expert real-shard probes

- Decision: use cached BF16-rounded resident weights as the bounded multi-expert candidate on the 16 GB target GPU; keep FP32 resident as the numerical reference and allow its capacity bypass to remain visible in telemetry.
- Alternatives: force FP32 residency and accept repeated H2D, evict experts aggressively, or quantize real GLM payloads before a quality gate.
- Evidence: eight real layer-10 experts used 603,979,776 BF16 resident bytes and measured 1,854,140 ns sequential warm median with zero warm H2D. FP32 required 1,207,959,552 bytes, exceeded the 1 GiB configured budget, transferred 3,019,898,880 warm bytes, and measured 13,153,048 ns.
- Accepted because: BF16 is the only tested representation that keeps an 8-expert bank resident under the current budget. Quality remains an explicit gate because the per-expert relative numerical difference is 0.1828%.
- Revisit: after expert-major batched GEMM, direct raw-BF16 storage, and full-layer quality comparison.

## D-0023 -- Use an opt-in dense BF16 grid for real candidate blocks

- Decision: add a dense `resident_grid` API for validated GLM raw-BF16 experts and use it for real-shard candidate-token blocks when `precision=bf16-rounded`. Keep FP32 and native MXFP4 on their existing scalar/exact reference paths until their own batched kernels are measured.
- Alternatives: keep real experts sequential, route FP32 through the BF16 grid, or convert the raw shard to MXFP4 before the first nonzero grid gate.
- Evidence: the two probe artifacts supplied 8 complete layer-10 experts. On the RTX 5080, 8 experts x 4 candidate tokens measured 1,758,739 ns/block with 603,979,776 resident bytes, zero warm weight H2D, and maximum relative CPU difference 0.00135118968 (0.135%). The two-token tiny CUDA regression and the full 26-test CTest suite passed.
- Accepted because: the grid amortizes one activation transfer and schedules the expert union as a single candidate block without changing routing or payload bytes. It remains opt-in because this is one FFN block and the BF16 quality gap is not a full-model quality result.
- Revisit: after direct raw-BF16 tensor-core storage, full natural Top-8 routing, nonzero layer parity, and end-to-end decode measurements.

## D-0024 -- Admit validated raw BF16 bytes directly

- Decision: add `RawBf16WeightView`/`RawBf16MlpView` and a CUDA grid entry point that passes validated `.k3x` BF16 role bytes directly to the resident table. Keep the dense FP32 view as the CPU/reference and compatibility path.
- Alternatives: decode every expert to FP32 then reconvert to BF16, require callers to preconvert a second BF16 file, or make the raw path replace the existing dense API.
- Evidence: on the same RTX 5080 real-shard probe, 8 experts x 4 tokens improved from 1,758,739 ns to 1,648,927 ns warm median; cold execution fell from 759,804,032 ns to 135,877,327 ns, with resident bytes 603,979,776, warm H2D 0, and unchanged 0.1351% maximum relative CPU difference. The CUDA dense/raw parity test and full 26-test CTest suite passed.
- Accepted because: the path removes seven unnecessary FP32 staging vectors from the multi-expert probe and makes the storage representation match the resident representation without changing the source bytes or routing semantics.
- Revisit: after pinned/asynchronous raw H2D, direct tensor-core algorithm selection, full natural Top-8 layer parity, and quality evaluation beyond the last-expert FFN output.

## D-0025 -- Batch multi-expert BF16 GEMMs with pointer-array layouts

- Decision: for `RawBf16MlpView` grids with more than one expert, use cublasLt pointer-array batch layouts so gate, up, and down projections are submitted as three GEMM calls. Keep the scalar-grid plan for one expert and retain a heuristic-unavailable fallback to the prior per-expert plan.
- Alternatives: keep one cublasLt call per expert, use a classic cuBLAS grouped handle, or pack all resident weights into a contiguous temporary matrix before GEMM.
- Evidence: the 8-expert/4-token real-shard probe on RTX 5080 measured 1,065,026 ns warm median with 4 resident-grid launches/call and 576 pointer-descriptor bytes/call, versus 1,648,927 ns for the direct raw per-expert path. CPU maximum relative error remained 0.00135860045. The nonzero CUDA regression and full CTest suite passed.
- Accepted because: it removes 24 expert-level projection launches from the hot block while preserving the same resident bytes, input/output layout, and raw source payloads. The one-expert branch avoids the measured pointer-array setup penalty.
- Revisit: after pinned pointer staging, direct tensor-core algorithm profiling, larger candidate blocks, and full-layer quality parity.

## D-0026 -- Keep BF16 resident-grid output as an opt-in experiment

- Decision: add `CudaBf16OutputMode::bf16` for the raw BF16 resident grid, but keep `fp32` output as the default and do not enable it in quality modes automatically.
- Alternatives: make BF16 output the default, retain FP32 intermediates and output only, or quantize the raw expert payloads further.
- Evidence: on the RTX 5080 with the two real probe artifacts, 8 experts x 4 tokens measured 1,034,950 ns warm median with BF16 output versus 1,091,122 ns in the paired FP32-output run. The maximum CPU-relative difference was 0.00316690677 (0.317%) for BF16 output versus 0.00135860045 (0.136%) for FP32 output. The physical final output transfer is halved.
- Accepted because: BF16 output removes a measured memory-traffic component without changing routing or source weights, and the reference mode remains one option away. It is not a quality-preserving default until a full GLM layer and model-level comparison exists.
- Revisit: after direct tensor-core algorithm profiling, full natural Top-8 routing, nonzero attention/trunk parity, and task-quality evaluation. Promote only if quality divergence stays within the selected quality-mode budget.

## D-0027 -- Expose cublasLt workspace as a runtime tuning knob

- Decision: allow the raw pointer-array grid to request a bounded cublasLt workspace through `cuda_cublas_workspace_bytes` and `--workspace-bytes`, but keep the default at zero and do not auto-tune during model startup.
- Alternatives: hard-code 64 MiB, tune every cublasLt algorithm at startup, or leave the heuristic preference unconfigurable.
- Evidence: on the RTX 5080 real 8-expert/4-token probe, a paired FP32-output run measured 994,529 ns with zero workspace and 967,790 ns with 64 MiB. An 8 MiB run measured 986,393 ns and a 16 MiB run 1,073,612 ns. The 64 MiB BF16-output run measured 1,080,469 ns versus 1,034,950 ns without workspace, so the result is mode- and shape-sensitive.
- Accepted because: the option enables measured per-workload tuning without changing the default path or routing/output semantics. The reusable workspace is bounded and accounted for by normal CUDA scratch allocation.
- Revisit: after a larger shape sweep, native tensor-core algorithm profiling, and full-layer scheduling. Do not infer end-to-end tok/s from this block-level knob.

## D-0028 -- Add a packed-input raw BF16 grid contract

- Decision: add `raw_bf16_situ_mlp_grid_packed` for expert-major schedulers. The input slab is `[expert][candidate][hidden]`, each pointer-array B operand selects one expert's slab, and the output remains one slab per expert for caller-owned route scatter. Keep the existing common-input grid API unchanged.
- Alternatives: continue broadcasting one input block to every expert, add a separate GEMM launch per expert assignment, or silently infer per-expert token counts from a ragged container.
- Evidence: `test_cuda_dense` now runs two nonzero experts with different one-token input slabs, checks CPU BF16-rounded parity, and verifies the packed activation/output byte counters. The test passed with the full WSL CUDA suite before documentation.
- Accepted because: natural MoE routing produces different assignment lists per expert, and this contract removes the forced common-input assumption without changing the exact weights or route decisions. Padding/scatter policy remains explicit in the future scheduler rather than hidden in the kernel.
- Revisit: when GLM DSA/router state is connected and a ragged expert-major benchmark can compare packed assignment counts against the common-input grid.

## D-0029 -- Keep expert-major packing model-neutral

- Decision: implement `build_expert_major_packed_plan` beside the existing route grouping utility. It copies token hidden states into per-expert assignment order and preserves token/router-slot/contribution metadata; it does not depend on GLM tensor names or CUDA.
- Alternatives: embed packing inside the CUDA backend, make the backend infer routing from raw scores, or add a GLM-specific scheduler before the exact router exists.
- Evidence: the C++ expert-major test covers a route where one expert receives one token and another receives two, and verifies both slab contents and assignment count.
- Accepted because: the same packed representation can feed raw-BF16 CUDA, CPU reference, or a future MTP verifier without weakening natural routing. Keeping route scatter outside the kernel makes quality/audit telemetry possible.
- Revisit: when the GLM runtime has real router scores and can bucket groups by assignment count for a measured ragged block.

## D-0030 -- Add a deterministic sparse-packed real-shard benchmark mode

- Decision: extend the bounded real-expert benchmark with `--input-mode sparse-packed`. For the explicit two-token probe, experts alternate between token 0 and token 1 and execute through `raw_bf16_situ_mlp_grid_packed`; `common` remains the default.
- Alternatives: infer a route pattern from expert IDs, overload `--tokens` with hidden semantics, or claim the synthetic assignment pattern represents learned GLM routing.
- Evidence: the RTX 5080 two-shard probe measured 1,040,559 ns warm median for common 8-expert/2-token input and 965,550 ns for sparse-packed one-token slabs per expert (about 7.2% lower block latency). CPU relative differences were 0.1777% and 0.1663%, respectively. BF16-output sparse-packed measured 995,611 ns with 0.3967% relative difference.
- Accepted because: the mode measures the new packed-addressing contract with real payload bytes while labeling the route pattern as deterministic and non-learned. It cannot alter the default benchmark or quality claims.
- Revisit: replace the deterministic pattern with exact GLM router assignments once DSA/MLA/trunk state is connected, then report real assignment-count distributions.

## D-0031 -- Bucket ragged expert-major slabs before CUDA dispatch

- Decision: add `bucket_expert_major_packed_plan` as a model-neutral CPU scheduler boundary. It groups packed expert records by assignment count in stable first-use order, concatenates each group's existing `[assignment][hidden]` slab without padding, and retains source group indices for later raw-BF16 view assembly and route scatter.
- Alternatives: pad every expert to the largest assignment count, issue one CUDA call per expert, or make the CUDA backend infer ragged lengths and route metadata.
- Evidence: `test_expert_major` covers separate assignment-count buckets, repeated-shape grouping, exact slab concatenation, and malformed payload rejection. The WSL CTest suite passed 26/26 and the focused GLM Python suite passed 35/35 at commit `46f2e8e`.
- Accepted because: the existing packed CUDA API is rectangular by construction, while real MoE routes are ragged. Keeping bucketing and scatter explicit avoids hidden padding cost and preserves exact routing/audit metadata.
- Revisit: when exact GLM router scores are connected and a real ragged assignment distribution can be executed through the bucket list with output scatter and full-layer parity.

## D-0032 -- Keep expert-major contribution scatter outside CUDA

- Decision: add `scatter_expert_major_outputs` as a CPU/reference helper that consumes group-order output slabs, validates their exact shape, multiplies each assignment by its retained router contribution, and accumulates into token-major output order.
- Alternatives: fuse contribution weighting and scatter into the CUDA grid, return only the last expert's output, or let callers rely on implicit group ordering without a validator.
- Evidence: `test_expert_major` verifies two groups with one- and two-assignment slabs, expected weighted token outputs, and short-output rejection. The WSL CTest suite passed 26/26 and the focused GLM Python suite passed 35/35 at commit `b777b1b`.
- Accepted because: explicit scatter preserves exact Top-K semantics and makes CPU/GPU parity inspectable. It also allows future ragged buckets to run independently without coupling router metadata to a kernel output layout.
- Revisit: after a real GLM router and CUDA bucket loop exist; a fused scatter can be evaluated only against the explicit reference result and its measured H2D/launch cost.

## D-0033 -- Gate migrated historical evidence without weakening new correctness tests

- Decision: when a test is explicitly tied to a historical B-0006 through B-0024 result artifact that is not shipped in GLM5X, skip that evidence check with a visible reason. Keep all synthetic, GLM5X, build, and cross-language tests active.
- Alternatives: copy the old K3X `results/` tree into the public repository, delete the historical tests, or mark the whole Python suite optional.
- Evidence: the first GLM5X push passed C++ and CodeQL but failed 50 Python tests with `FileNotFoundError` for absent historical JSON. After the targeted collection/fixture gate, GitHub Actions correctness passed on commit `a00beec` in 3m21s; CodeQL also passed.
- Accepted because: the repository does not contain the old benchmark artifacts by design, and copying them would mix K3X evidence into the GLM5X product. The skip is narrow, explicit, and reversible when artifacts are restored.
- Revisit: when GLM5X owns replacement benchmark artifacts, remove each historical skip and require the new raw/summary parity checks.

## D-0034 -- Keep the first real MoE reference layer lazy and exact

- Decision: add `GLM5XLayer10MoEReference` over the copy-free expert bundle. Load router/shared weights eagerly, load only selected expert raw-BF16 role triples on demand, cache them by `(layer, expert)`, and retain an uncached path for parity.
- Alternatives: materialize all layer-10 experts, decode every expert to FP32 before routing, or connect CUDA before the q-residual/MLA/DSA reference boundary exists.
- Evidence: the five-shard bundle contains 277 complete expert groups and a complete layer 10. A real layer-10 smoke selected 15 unique experts; cold and cached forwards produced identical `[2,6144]` BF16 outputs. No quality or tok/s claim is inferred.
- Accepted because: it proves the official sigmoid router/shared-SwiGLU boundary without requiring the full checkpoint in RAM/VRAM and provides the exact selected-expert set needed by the next packed CUDA integration.
- Revisit: after q-residual plus exact MLA/DSA and full-layer parity are connected; only then compare the lazy reference with the resident CUDA scheduler.

## D-0035 -- Make q-residual, MLA, DSA, and MoE one exact incremental reference boundary

- Decision: add `GLM5XMLAReference`, `GLM5XOfficialDSAState`, and `GLM5XDecoderLayerReference` as the CPU correctness path. Preserve compressed MLA state and DSA index keys across calls, pass the exact causal DSA Top-K mask into MLA, and keep the official natural Top-8 MoE route unchanged.
- Alternatives: connect CUDA kernels before a complete reference boundary, approximate DSA with a stale refresh in the default path, or test MLA and MoE as disconnected components only.
- Evidence: the focused GLM reference suite passed 42/42; full-vs-incremental synthetic layer output and final state lengths match, the bundle-backed loader reads real bounded layer-10 attention/indexer/norm tensors, and the real two-token smoke returned cached output maximum absolute difference `0.0`.
- Accepted because: every later CUDA, prefetch, and expert-major optimization can compare against one stateful layer contract without downloading the full checkpoint. The reference path remains CPU-only and is not a throughput claim.
- Revisit: when all 78 layers, final logits, MTP, and nonzero CUDA layer parity are connected.

## D-0036 -- Reuse one validated bundle reader for a layer construction

- Decision: `GLM5XDecoderLayerReference.from_bundle()` opens and root-verifies the cross-shard bundle once, shares its tensor readers with the attention/indexer/norm loader and lazy MoE loader, and retains the bundle through the expert-loader closure.
- Alternatives: let the attention and MoE constructors independently open the same bundle, disable root verification for the second open, or materialize all layer experts to avoid a shared reader.
- Evidence: a real five-artifact layer-10 smoke measured bundle construction at `250.637263 s` after reuse versus `491.483777 s` before reuse, a `49.0%` reduction. Cold forward was `5.969859 s`, warm cached forward `0.057331 s`, and cached output maximum absolute difference was `0.0`.
- Accepted because: it removes duplicate multi-gigabyte root hashing without weakening validation and keeps expert payloads lazy. The improvement is storage/open latency only; it does not establish model tok/s.
- Revisit: when the runtime switches to a persistent process-level artifact registry or direct asynchronous extent reads, while retaining one verified identity per artifact.

## D-0037 -- Connect ragged expert-major plans to the raw-BF16 CUDA grid

- Decision: add `CudaBackend::raw_bf16_situ_mlp_expert_major` as the first CUDA consumer of `ExpertMajorPackedPlan`. Bucket groups by assignment count, dispatch one packed raw-BF16 grid call per bucket, retain group offsets, and use the explicit contribution scatter helper for token-major output. Keep this path opt-in until learned GLM routing and full-layer parity are connected.
- Alternatives: issue one CUDA grid call per expert, pad every expert to the largest assignment count, or fuse route inference/scatter into the kernel before the exact GLM router boundary exists.
- Evidence: the nonzero CUDA regression passed in the full WSL CUDA suite. On five bounded real GLM-5.2 shard artifacts at layer 10, the deterministic two-token/8-expert expert-major mode measured `1,380,314 ns` warm median per block, compared with `1,651,193 ns` for common-input and `1,631,127 ns` for sparse-packed in paired runs. Maximum CPU-relative difference was `0.0014705552021`; resident bytes were `603,979,776` and warm weight H2D was `0`.
- Accepted because: the route metadata, ragged bucketing, raw-BF16 resident representation, and weighted scatter now form one executable boundary without changing source weights or natural-routing semantics. The measured block result is useful kernel evidence but is not a full-layer or tok/s claim.
- Revisit: after actual GLM router scores produce the plan, pinned/asynchronous staging is measured, and the complete q-residual/MLA/DSA/MoE layer has nonzero parity and quality results.

## D-0038 -- Use the official GLM router before expert-major CUDA dispatch

- Decision: add a `learned-expert-major` benchmark mode that reads `model.layers.<N>.mlp.gate.weight` and `e_score_correction_bias` from bounded K3X shards, computes float32 sigmoid scores, applies natural Top-8 selection and routed scale `2.5`, loads only the selected expert union, and executes the existing raw-BF16 bucket/scatter path. Make the VRAM admission budget explicit through `--resident-bytes`.
- Alternatives: retain the deterministic route pattern, load all 256 experts and filter after admission, or let CUDA infer routing from a hidden synthetic route.
- Evidence: official GLM-5.2 configuration uses 256 routed experts, natural Top-8, sigmoid scoring, `n_group=1`, `topk_group=1`, and routed scale 2.5. On five bounded real shards, two tokens selected 15 experts/16 assignments and measured `1,905,668 ns` warm median/block with FP32 output, `1,132,462,080` resident bytes, zero warm weight H2D, and `0.000865828245878` maximum CPU-relative difference. Four tokens selected 29 experts/32 assignments and measured `3,757,986 ns` per block with `0.000666717009153` maximum CPU-relative difference. The full WSL CUDA CTest suite remained 26/26.
- Accepted because: the benchmark now exercises real routing metadata and exact selected-expert residency without pretending that deterministic routing is learned GLM behavior. Loading only the selected union preserves the out-of-core traffic boundary and exposes capacity pressure instead of hiding it.
- Revisit: after q-residual/MLA/DSA produces the actual layer hidden state, the router runs on GPU or overlapped CPU, and the full layer output is compared against the Python reference.

## D-0039 -- Complete the bounded learned GLM MoE sublayer before attention CUDA

- Decision: add `learned-moe-layer` as an opt-in benchmark mode that reuses the exact learned Top-8 routed expert-major path and adds the real layer shared-expert raw-BF16 SwiGLU. Compare the combined routed-plus-shared output against the CPU dense reference.
- Alternatives: keep measuring routed FFN only, implement MLA/DSA CUDA before shared MoE parity, or silently treat the routed result as the complete GLM layer.
- Evidence: on five bounded real layer-10 shards, two tokens selected 15 routed experts and one shared expert, measured `2,155,188 ns` warm median/block with `0.000585675588809` maximum CPU-relative difference, and used `1,207,959,552` resident bytes. Four tokens selected 29 routed experts plus the shared expert and measured `3,968,243 ns` with `0.000429985491792` relative difference. The full WSL CUDA CTest remained 26/26.
- Accepted because: shared SwiGLU is part of the official MoE sublayer and omitting it hid both compute and residency pressure. The mode expands the measured boundary without changing natural routing or claiming full-layer throughput.
- Revisit: when the actual q-residual/MLA/DSA hidden state is connected and the complete layer output, final logits, and incremental state have nonzero parity.

## D-0040 -- Use a small CRC-checked activation artifact for the Python/C++ boundary

- Decision: define `GLM5XACT` v1 as a fixed 40-byte little-endian BF16 activation header plus a contiguous token-major payload. Make the Python writer atomic and make the C++ loader reject bad magic, dimensions, extent, dtype, or CRC before routing. Expose optional `--input-bf16` and `--expected-bf16` on the bounded real-expert benchmark for parity evidence.
- Alternatives: pass tensors through an in-process Python binding, reuse the full K3X tensor directory for every transient activation, or accept raw unframed bytes with an out-of-band shape.
- Evidence: the WSL CUDA build passed after adding the writer/loader and the C++ activation test; CTest passed 27/27 in 5.98 seconds. Commit `30bf5d4` exposes the exact `moe_input` field and extends the existing incremental parity assertion. Public correctness `31806277016` and CodeQL `31806277022` passed, including the focused Python header/CRC test; no end-to-end GLM tok/s or full-layer claim is made.
- Accepted because: the artifact is small enough for per-layer handoff, independently verifiable, crash-safe on the producer side, and does not require loading the checkpoint or changing the model graph. Expected-output comparison remains opt-in so the natural router path is unchanged.
- Revisit: when exact q-residual/MLA/DSA output is exported from the reference, add a real five-shard activation parity record and replace the bounded synthetic input path.

## D-0041 -- Give GLM an explicit SiLU activation path and compare at the BF16 boundary

- Decision: keep the inherited SiTU implementation for legacy callers, but route every GLM-5.2 MoE benchmark through an explicit SiLU gated MLP. Compare the CUDA result with the BF16-rounded expected artifact while retaining a separate FP32 GPU-versus-CPU diagnostic.
- Alternatives: reuse SiTU for all models, replace the shared activation API, or compare FP32 CUDA output directly against a BF16-only artifact.
- Evidence: the official GLM MoE implementation applies `act_fn(gate_proj(x)) * up_proj(x)` and the official configuration selects `hidden_act="silu"` ([modeling_glm_moe_dsa.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py), [configuration_glm_moe_dsa.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/configuration_glm_moe_dsa.py)). On the RTX 5080 bounded layer-10 probe, the corrected path measured `0.000452667358331` GPU-versus-CPU relative error. The expected artifact comparison after matching BF16 output precision measured `0.00152439018711` relative error and `0.00006103515625` maximum absolute error. The unrounded CPU-versus-Python diagnostic was `0.00262972200289`, attributable to FP32 accumulation order and the BF16 artifact boundary rather than route selection; route IDs and contributions matched for both tokens.
- Accepted because: GLM semantics are no longer coupled to K3X SiTU naming, the legacy default remains source-compatible, and the benchmark distinguishes model arithmetic error from storage-precision error.
- Revisit: after full-layer q-residual/MLA/DSA parity and model-level quality tests. The one-layer bounded result is not an end-to-end throughput or quality claim.

## D-0042 -- Reject prepared-bucket caching and shared-expert dispatch fusion as default optimizations

- Decision: do not retain either experimental optimization in the default CUDA path. Restore per-call bucket construction and the separate routed/shared dispatches until a device-side accumulation design has a positive paired result.
- Alternatives: keep the prepared bucket list inside `ExpertMajorPackedPlan`, fuse the one-token shared expert into the routed expert-major plan, or enable either change behind a runtime flag.
- Evidence: on the RTX 5080 with the same bounded layer-10 learned-MoE input, five-run token-1 medians were `1,307,995 ns` with both changes versus `1,265,441 ns` at the `f07d78c` baseline (about `3.4%` slower). For the real two-token activation, three-run medians were `2,191,291 ns` with bucket caching only versus `2,166,726 ns` at baseline (about `1.1%` slower). Shared fusion reduced the token-1 grid-call count but did not reduce the measured block latency. These are MoE-sublayer timings, not tok/s.
- Rejected because: neither change produced a stable positive latency result, and keeping them would expand the public API and benchmark semantics without a demonstrated benefit. The code was reverted and the WSL host/CUDA/Python regression remained green (`15/15`, `27/27`, `301 passed, 124 skipped`).
- Revisit: after implementing a device-side expert-output accumulation path that removes host copy/scatter overhead and passes exact CPU parity, then rerun paired multi-seed measurements.

## D-0043 -- Keep device-side ragged expert accumulation opt-in

- Decision: add a CUDA ragged expert-mix kernel and an explicit `BackendOptions::cuda_expert_major_device_accumulate`/`--device-accumulate` switch. The path keeps one token-major FP32 device accumulator, performs one final device-to-host copy, and remains disabled by default.
- Alternatives: keep the existing host output copy plus CPU scatter as the only path, make device accumulation the default immediately, or fuse routing and contribution construction into the CUDA kernel.
- Evidence: the new primitive passed direct and repeated-accumulation tests; a varied one-/two-assignment bucket regression passed CPU parity; WSL host CTest passed 15/15, CUDA CTest 27/27, and Python passed 301 with 124 explicit historical skips. On the bounded RTX 5080 layer-10 learned-MoE probe, three identical two-token runs measured baseline medians `2,198,145`, `2,736,064`, `2,492,351 ns` and device-accumulate medians `1,991,721`, `1,981,629`, `2,446,610 ns`; median-of-runs was `2,492,351 ns` versus `1,991,721 ns` (about 20.1% lower). GPU/CPU relative error stayed `0.000571510172449` in both modes and warm weight H2D stayed `0`.
- Accepted because: it removes the per-bucket device-to-host output copies and CPU scatter from the bounded MoE sublayer while preserving exact routing and an unchanged baseline fallback. The measured gain is encouraging but noisy and is not a full-layer or end-to-end tok/s result.
- Revisit: after exact q-residual/MLA/DSA, trunk residuals, final logits, incremental generation, and multi-seed/full-layer quality tests exist. Only then consider enabling it for a quality mode by default.

## D-0044 -- Keep shared-expert device fusion opt-in

- Decision: add `raw_bf16_situ_mlp_expert_major_with_shared` and the benchmark switch `--fuse-shared 1`. It is valid only with `--device-accumulate 1`, FP32 output, and the learned GLM MoE mode; the default path still performs separate routed and shared dispatches.
- Alternatives: make the fused shared path unconditional, fuse only the host-side result, or leave the shared expert as a separate device-to-host operation.
- Evidence: the CUDA synthetic regression passed exact routed-plus-shared parity and verified one final 16-byte D2H for the two-token fixture. On the exact two-token GLM5XACT layer-10 handoff, GPU/CPU maximum relative error remained `0.00045266628149` and expected BF16-artifact relative error remained `0.00152439018711`. With 100 warm iterations per run, baseline medians were `2,180,810`, `2,194,670`, `2,371,374 ns`; device accumulation without fusion was `2,326,186`, `2,590,515`, `2,098,680 ns`; fused shared accumulation was `1,984,222`, `1,986,460`, `2,090,547 ns`. Median-of-runs was `2,194,670 ns` baseline versus `1,986,460 ns` fused, approximately `9.49%` lower. The longer sweep did not reproduce the earlier standalone device-accumulation gain, so no universal speedup is claimed.
- Accepted because: it removes the second D2H and host addition while keeping natural routing and the separate reference mode intact. It is still a bounded MoE-sublayer result, so promotion waits for full-layer parity, quality, and multi-seed evidence.
- Revisit: after the exact MLA/DSA-to-CUDA handoff, pinned staging, and complete decoder-layer benchmark are available.

## D-0045 -- Make final logits and incremental state explicit in the CPU reference

- Decision: add `GLM5XDecoderModelReference` as a thin composition layer over exact decoder-layer references. It owns per-layer MLA/DSA states, final RMSNorm, the LM head, prompt prefill, one-token continuation, and greedy generation; it does not alter routing or enable CUDA fast paths.
- Alternatives: keep only isolated layer tests, add logits directly to the layer class, or connect CUDA before a model-level state contract exists.
- Evidence: the new synthetic two-layer test matched every incremental prompt logit to the corresponding prefill slice and matched an explicit greedy loop. WSL full Python passed `302 passed, 124 skipped` after the export.
- Accepted because: it gives CUDA, MTP, prefix/KDA caching, and quality comparisons one unambiguous model-level state boundary while preserving the existing exact layer reference.
- Revisit: when real all-layer GLM tensors and MTP metadata are available; batch support, tied embeddings, and CUDA final-logit placement must then be validated against the official checkpoint.

## D-0046 -- Keep decoder layer weights loadable per layer in the reference path

- Decision: add `GLM5XDecoderModelReference.from_layer_loader` with an explicit layer count and callable provider. The provider is invoked in layer order for each forward; only recurrent MLA/DSA states survive the call.
- Alternatives: materialize every decoder layer in a Python tuple, cache the entire model after first use, or make the CUDA benchmark own a separate untested layer order.
- Evidence: the two-layer loader test observed exactly `[0, 1]` requests and returned the same logits/state contract as the eager model. WSL passed `303 passed, 124 skipped`, host CTest `15/15`, and CUDA CTest `27/27`.
- Accepted because: a full GLM-5.2 checkpoint cannot be assumed to fit a single resident tier, and the loader boundary makes layer-at-a-time conversion/admission testable without changing natural routing.
- Revisit: when real all-layer tensors are available; add bounded layer cache, pinned staging, transfer deadlines, and a CUDA provider only after measurements show they reduce stalls without changing logits.

## D-0047 -- Reuse one verified bundle reader for layer providers

- Decision: expose `GLM5XDecoderLayerReference.bundle_layer_loader`, which opens the cross-shard bundle once and closes over its verified readers and tensor-reference map. Each call still constructs only the requested layer and keeps selected expert payloads lazy.
- Alternatives: call `from_bundle` for every layer, copy all layer tensors into a model-wide map, or bypass bundle identity/CRC checks in a fast path.
- Evidence: the bounded bundle test requested the same layer twice and observed exactly one `GLM5XExpertBundle.open` call; the full WSL Python suite passed `303 passed, 124 skipped`.
- Accepted because: repeated root scans would directly multiply startup and layer-provider overhead, while reusing the already validated readers preserves artifact identity and expert CRC checks.
- Revisit: after real 78-layer streaming measurements; add bounded reader lifetime and async prefetch only if they reduce deadline misses without increasing resident memory unexpectedly.

## D-0048 -- Keep lazy payload/root admission experimental and opt-in

- Decision: allow `K3XReader`/`GLM5XExpertBundle` to skip whole-file payload and root scans at open, then validate each selected tensor CRC on first read. Strict eager payload/root verification remains the default and the reference correctness path.
- Alternatives: always scan every shard before the first token, skip all CRC checks in fast mode, or verify only the bundle JSON without checking selected payloads.
- Evidence: on the real 5.34 GB first GLM shard, strict eager open took `49.816001 s`, while lazy directory open took `0.003153 s` and the first 1.90 GB tensor read/CRC took `8.989272 s`. A tampered lazy tensor raised `DATA_CRC_MISMATCH` on read; full Python passed `304 passed, 124 skipped`.
- Accepted because: out-of-core startup cannot afford rereading every cold shard before knowing which layer/expert is needed, while first-use CRC preserves selected-payload correctness.
- Revisit: after adding runtime recovery, optional deferred root verification, telemetry, and a measured 78-layer prefetch schedule. Do not promote `verify_root=False` to QUALITY mode without an integrity decision.

## D-0049 -- Add a bounded opt-in cache for reference trunk layers

- Decision: extend `GLM5XDecoderModelReference.from_layer_loader` with an explicit `layer_cache_capacity` LRU. A nonzero capacity retains validated decoder-layer objects and their non-expert trunk tensors between forwards; zero keeps the previous strict layer-at-a-time behavior. Expert payload residency remains a separate provider policy.
- Alternatives: cache every layer implicitly, cache nothing and reconstruct every token, or mix expert payloads into the model-level cache without a byte budget.
- Evidence: the two-layer correctness test produced identical logits. Across two forwards, capacity 2 invoked the loader once per layer (`[0, 1]`) while capacity 0 invoked it on every forward (`[0, 1, 0, 1]`). A single-threaded synthetic timing sample was `1.0073 ms` per forward without cache versus `1.0310 ms` with cache, so no synthetic speedup is claimed.
- Accepted because: real 78-layer execution must avoid rereading large attention/trunk tensors on every token, but the full trunk footprint and RAM pressure are not yet measured. The explicit capacity makes the tradeoff observable and reversible without changing logits or natural routing.
- Revisit: after a real all-layer provider exists; measure bytes, construction latency, RAM residency, NVMe traffic, and quality parity before selecting a default capacity or sharing the cache with expert payloads.

## D-0050 -- Model GLM-5.2 dense MLP layers explicitly

- Decision: add `GLM5XDenseMlpReference` and an explicit `mlp_type="dense"` branch to the bundle-backed decoder layer. Compute `silu(gate_proj(x)) * up_proj(x)` followed by `down_proj`, return the existing `GLM5XMoEForward` schema with empty routing, and keep sparse MoE as the default.
- Alternatives: treat every layer as sparse, special-case dense layers outside the decoder reference, or silently synthesize a zero-expert router record.
- Evidence: a direct BF16 SwiGLU parity test and a bundle-backed decoder-layer test passed; the full WSL Python suite passed `306 passed, 124 skipped` in `73.57 s`. This is a reference correctness boundary, not a full-checkpoint or throughput result.
- Accepted because: GLM-5.2 declares the first three layers as dense, so forcing a router/expert bundle there would make an all-layer loader incorrect. The explicit switch keeps the distinction visible and reversible.
- Revisit: when all 78 real layers are loaded, verify the official shared-indexer mapping and compare full-layer logits before connecting CUDA execution.

## D-0051 -- Add a configuration-driven GLM bundle model factory

- Decision: add `GLM5XDecoderModelReference.from_bundle`. It opens one cross-shard bundle, reads model-head tensors once, keeps decoder layers provider-owned, and resolves dense/sparse MLP type plus nearest preceding shared-indexer source from the official config. Permit explicit embedding/final-norm/LM-head overrides only for bounded partial probes.
- Alternatives: require callers to hand-build a 78-entry layer loader, assume every layer is sparse and every indexer is local, or materialize every decoder layer during factory construction.
- Evidence: the synthetic three-layer bundle test exercised dense layer 0, dense layer 1 with a shared indexer from layer 0, and sparse layer 2 with exact incremental-logit parity. Full WSL Python passed `307 passed, 124 skipped`; host CTest passed `15/15`. On five real probe artifacts, lazy factory setup took `0.0668 s`, real layer-0 admission took `4.2789 s`, and one-token CPU layer forward took `0.03685 s` with output `[1,1,6144]`.
- Accepted because: this is the first model-level boundary that can request real GLM layers in execution order without loading all decoder weights. Partial-head overrides are explicit and cannot silently be used for a full checkpoint.
- Revisit: when all 78 layers and final head tensors are available, replace probe overrides with exact bundle tensors, compare full logits, and hand the provider to CUDA/pinned staging.

## D-0052 -- Stage the reference on CUDA and materialize the checkpoint one shard at a time

- Decision: add an explicit `device` parameter to the Python reference bundle/model factories and use a resumable local stream that downloads, converts, verifies, and deletes one source shard before moving to the next.
- Alternatives: keep all reference tensors on CPU until the full checkpoint exists, download the entire safetensors repository before conversion, or keep source shards after conversion to simplify retries.
- Evidence: CPU-vs-CUDA layer/model parity tests passed; the complete WSL Python suite passed `311 passed, 124 skipped`, host CTest passed `15/15`, and the first two real GLM-5.2 shards finalized as `5,342,863,616` and `5,351,993,600` byte `.k3x` artifacts with source-deleted markers. The stream retains `.part` files for interruption-safe HTTP Range resume.
- Accepted because: direct device staging removes an unnecessary host-to-device copy at the reference boundary, while one-shard conversion keeps peak source residency bounded and makes local full-checkpoint progress durable without paid cloud resources. The CUDA path and stream are correctness/operational boundaries, not throughput claims.
- Revisit: after all 282 shards assemble into a verified bundle, exact final-head tensors and full-layer logits are available, and end-to-end CUDA decode measurements show whether direct staging and source deletion improve the actual bottleneck.

## D-0053 -- Reuse strict per-shard verification for final stream indexing

- Decision: keep `assemble_glm5x_expert_bundle` strict by default, but let the local stream call it with `verify_payloads=False, verify_root=False` after every shard has passed strict reader verification and received a source-deleted marker.
- Alternatives: rescan every completed payload during final indexing, weaken the per-shard conversion gate, or make lazy bundle assembly the public CLI default.
- Evidence: the focused bundle/stream regression passed `4/4`; strict verification still runs in `convert_glm5x_shards` before source deletion. A partial real assembly was observed to spend several minutes rereading seven multi-gigabyte artifacts, so the duplicate scan is a clear operational bottleneck, but no full-checkpoint assembly timing has been measured yet.
- Accepted because: the conversion marker is written only after the finalized artifact opens under strict checks, while the lazy final index still validates directory metadata, file UUID, source SHA, root SHA field, tensor IDs, and sidecar identity. The strict public bundle path remains available for independent revalidation.
- Revisit: after the 282-shard stream completes; compare lazy-index time and a separate strict post-build audit before enabling any runtime policy that trusts the assembled bundle without selected-tensor CRC checks.

## D-0054 -- Skip finalized source-deleted shards during stream resume

- Decision: before downloading a manifest shard, the local stream checks for its finalized `.k3x` artifact and source-deleted marker. If both exist, it invokes the existing converter resume path instead of issuing another HTTP request; incomplete `.part` files continue through Range resume.
- Alternatives: always redownload missing source files, trust the artifact without converter resume validation, or delete all partials on restart.
- Evidence: the regression test covers verified-artifact detection; focused bundle/stream/converter tests passed `7/7`. A live restart showed that the prior implementation began a redundant 2.8 GB shard-1 download, which was stopped before completion; the fixed process retained the partial only for the incomplete shard and did not redownload completed shard payloads.
- Accepted because: restart correctness remains anchored in the existing marker-aware converter validation, while avoiding needless network and disk traffic after interruption. Public Linux correctness and CodeQL for `db2cf37` passed.
- Revisit: after a full preemption/resume drill across a completed and an incomplete shard, add a measured restart-latency benchmark and consider lazy marker validation only if the strict restart scan becomes material.

## D-0055 -- Parallelize local shard conversion with disjoint ranges

- Decision: support multiple local stream processes, each owning a non-overlapping half-open shard range and running with `--no-assemble`; assemble the final bundle once after all workers finish.
- Alternatives: keep one sequential worker, let workers claim a shared queue with lock files, or run concurrent workers over overlapping ranges and rely on artifact skipping.
- Evidence: on the RTX 5080 PC's WSL2/NTFS workspace, a single worker averaged `232 s/shard` across the first ten artifacts. Three workers started at `04:39:53`; by `05:02:30` they had finalized 13 new artifacts in `22m37s`, an initial aggregate sample of approximately `34.5 shards/hour`. This demonstrates overlapping download/conversion and is consistent with a roughly `7.5 h` conditional estimate for the remaining 259 artifacts, but the sample is still too short for a final speedup or completion-time claim.
- Follow-up evidence: at `05:17:32`, the same three workers had finalized 22 additional artifacts after launch, 32/282 total, in `37m39s`, or approximately `35.1 shards/hour`. The longer sample remains consistent with a conditional `~7.1 h` for the remaining 250 artifacts; a `7.5–9 h` planning window is recorded because the rate can change with network, NTFS, and retries.
- Accepted because: independent artifacts and source-deletion markers make range ownership naturally restartable, and the shared final bundle is explicitly serialized. No routing or weight semantics change.
- Revisit: after at least ten completed shards per worker, record sustained shards/hour, CPU, NVMe traffic, and failure/retry behavior before choosing a default worker count.

## D-0056 -- Keep direct BF16-to-MXFP4 conversion experimental

- Decision: implement and retain a reference-only chunked encoder, but do not make it part of the GLM converter or default runtime until calibration and quality gates are available.
- Alternatives: immediately rewrite all raw-BF16 expert shards as native MXFP4, keep raw BF16 only, or add a mixed/outlier residual format before measuring a plain MXFP4 baseline.
- Evidence: on the real layer-10 expert 4 from the bounded bundle, three BF16 projections were `75,497,472` bytes and encoded native MXFP4 was `20,054,016` bytes (`26.5625%`). The max-abs encoder took `0.442 s` and produced `19.861969%` FFN relative L2 error on the four-token GLM5XACT input. MSE scale search took `7.439 s`, reduced weight relative L2 from roughly `11.7%` to `11.1%`, and reduced FFN relative L2 only to `19.069034%`. An offline four-outlier correction simulation reached `12.346%` FFN relative L2 but is not an implemented storage or runtime path.
- Accepted because: the storage reduction is materially useful for VRAM/NVMe/PCIe pressure, but the measured quality loss is too large to silently apply to coding/agentic workloads. The reference encoder and raw-BF16 path preserve a reversible comparison baseline.
- Revisit: after real-layer calibration, outlier-index/value storage, mixed 6/8-bit alternatives, or a model-quality suite shows a measured Pareto point that satisfies the quality contract.

## D-0057 -- Keep Python expert-major batching opt-in and default to the loop reference

- Decision: expose `execution_mode="expert_major"` through the GLM5X reference layer/model factories, but keep `execution_mode="loop"` as the default and do not treat the Python grouped path as the production CUDA scheduler.
- Alternatives: replace the reference loop globally, remove the experiment until the C++ backend is connected, or cache permanently stacked expert weights.
- Evidence: the focused parity test and full WSL Python suite passed (`319 passed, 124 skipped`). On the real layer-10 partial bundle and RTX 5080, direct four-token MoE warm median was `21.670 ms` for loop versus `18.652 ms` for expert-major, but one-token warm median was `5.584 ms` versus `7.359 ms`; the complete four-token layer was `18.676 ms` versus `20.082 ms`. The grouped four-token MoE forward temporarily allocated about `1.97 GB` for stacked weights, while the loop path added about `0.20 MB`.
- Accepted because: the switch provides a measured, parity-tested experiment without changing correctness mode or silently increasing VRAM pressure. The evidence does not justify enabling it for single-token decode or making it the production path; the existing C++ expert-major backend remains the intended optimization boundary.
- Revisit: after connecting C++ expert-major execution to the exact full-layer hidden-state handoff, add a resident-weight-aware grouped kernel and remeasure one-token decode, four-token verification, peak VRAM, and quality.

## D-0058 -- Add a standalone full-bundle reference benchmark gate

- Decision: measure a completed GLM5X bundle through `tools/benchmark_glm5x_reference.py` before changing production runtime policy. The CLI accepts explicit token IDs, config and bundle paths, strict or lazy admission, expert/layer cache settings, and loop or expert-major execution.
- Alternatives: infer full-model TPS from bounded layer/FFN timings, add tokenizer behavior before the model path is verified, or make the benchmark silently choose a fast mode.
- Evidence: the existing model reference already has synthetic full-bundle prefill/incremental/greedy parity, while real full-checkpoint logits and decode remain unmeasured. The new CLI compiles, prints help, and its model-reference regression remains green (`4 passed`).
- Accepted because: it creates one reproducible measured boundary for the first real full-model run and preserves explicit quality/reference settings. It does not change model semantics or claim a TPS result before execution.
- Revisit: immediately after all 282 shards assemble; add a native CUDA runtime benchmark once exact full-model logits and hidden-state handoff are validated.

## D-0059 -- Keep parallel exact expert reads opt-in

- Decision: add `expert_load_workers` to the reference bundle path so selected exact raw-BF16 expert payloads can be read concurrently before serial tensor decoding and device staging. Keep `1` as the default correctness path and expose higher values only through the benchmark gate.
- Alternatives: retain one-by-one reads, parallelize GPU copies as well, or prefetch speculative experts before the router result is known.
- Evidence: the batch loader preserves route/output/loaded-expert parity in `4/4` focused MoE tests, and the full layer/model reference regressions pass `8/8`. The real full-bundle I/O latency, NVMe contention, and H2D overlap are not measured yet.
- Accepted because: it changes only scheduling of exact reads, preserves natural routing and payloads, and gives the first full-model gate a controlled I/O-overlap knob without silently changing quality.
- Revisit: after the active 282-shard CUDA gate records cold/warm read latency, H2D bytes, and decode tok/s. Disable or retune if concurrent reads starve compute or increase residency pressure.

## D-0060 -- Keep exact host payload caching bounded and opt-in

- Decision: add a thread-safe byte-capacity cache inside `GLM5XExpertBundle` keyed by `(layer_id, expert_id)`. Cache raw exact BF16 role bytes across decoder-layer object lifetimes and token forwards, report hit/miss/eviction counts, and use capacity `0` as the default.
- Alternatives: cache decoded CUDA tensors, retain complete decoder layers, use an unbounded process cache, or rely only on NVMe readahead.
- Evidence: on the five-shard real layer-10 probe, the same eight experts fell from `3.070666336 s` cold to `0.094567934 s` on the second call with a 1,000,000,000-byte cache. Output maximum absolute difference was `0.0`; resident payload was `603,979,776` bytes and the cache reported 8 hits, 8 misses, and 0 evictions.
- Accepted because: it removes repeat NVMe reads without changing exact weights, routing, or GPU residency policy, while making memory cost explicit and observable.
- Revisit: after the 78-layer full-bundle cold/cached benchmark records decode tok/s, cache hit rate, host RAM, and NVMe GB/token. Increase, decrease, or disable the cache based on measured session locality and RAM pressure.

## D-0061 -- Keep decoded expert GPU residency bounded and opt-in

- Decision: add a shared exact decoded expert tensor cache keyed by `(layer_id, expert_id)` with an explicit byte capacity. It is passed through the bundle-backed reference layer loader, reports hit/miss/eviction statistics, and defaults to disabled.
- Alternatives: retain all decoded experts, cache only host bytes, or let every layer own an unbounded CUDA tensor cache.
- Evidence: on the five-shard real layer-10 probe with four parallel readers, the same eight experts took `3.124653389 s` cold and `0.003793419 s` on the second call with a 1,000,000,000-byte device cache. Output maximum absolute difference was `0.0`; peak CUDA allocation was `719,427,584` bytes and the device cache reported 8 hits, 8 misses, and 0 evictions.
- Accepted because: it removes repeat H2D and decode work without altering exact weights or routing, while capacity makes the 16 GB VRAM trade-off explicit.
- Revisit: after the full 78-layer cold/cached gate records VRAM peak, H2D GB/token, cache hit rate, and decode tok/s. The cached quality path must remain exact before any default promotion.

## D-0062 -- Reuse the decoder q-residual between DSA and MLA

- Decision: let `GLM5XMLAReference.__call__` accept an optional precomputed `q_residual`, and pass the value already produced for DSA from `GLM5XDecoderLayerReference`. Keep the standalone MLA path unchanged when the argument is absent.
- Alternatives: recompute the projection for the existing simple API, fuse DSA and MLA into a new kernel before full-layer parity, or cache a residual beyond the current forward.
- Evidence: the new parity test and the focused MLA/layer/model suite pass (`17 passed`). On the real five-shard layer-10 CUDA probe, the exact attention boundary measured `2.298938 ms` with the duplicate projection and `2.224939 ms` with reuse, a `3.22%` reduction in that bounded sample; output maximum absolute difference was `0.0`.
- Accepted because: it removes a verified duplicate GEMM without changing routing, attention state, logits, or the default standalone API. The result is a bounded layer optimization, not an end-to-end TPS claim.
- Revisit: after the complete 78-layer CUDA gate reports per-layer timing, decide whether the same residual should be carried into a fused C++/CUDA layer handoff.

## D-0063 -- Keep DSA sparse Top-K MLA attention opt-in

- Decision: add `use_sparse_topk` and the benchmark flag `--sparse-topk-attention`. When enabled and DSA indices are present, MLA gathers only the selected compressed KV positions before the `kv_b_proj`; the dense masked path remains the default reference.
- Alternatives: always project the full historical KV sequence, make sparse gather the default immediately, or change the DSA router/index selection itself.
- Evidence: synthetic prefill and incremental dense-vs-sparse parity tests pass, and the model-factory propagation test passes. On the real five-shard RTX 5080 layer-10 probe at context `16,385` with 128 selected positions, the attention boundary fell from `12.154 ms` to `2.040 ms` (`83.2%` lower); output maximum absolute difference was `0.000244140625` and relative L2 difference was `0.0278%`. A short context sample was slower, so no unconditional promotion is justified.
- Accepted because: it preserves natural DSA indices and exact dense fallback while removing full-context KV projection and attention work for long contexts. The measured BF16 drift is explicitly recorded and the quality mode remains controlled by the switch.
- Revisit: after the full 78-layer gate measures long-context quality, prefill/decode tok/s, and VRAM traffic. Promote only if the quality suite accepts the BF16 numerical drift and the full-model speedup is stable.

## D-0064 -- Index bundle tensor records at open time

- Decision: build one immutable `(artifact, tensor_id)` record index when `GLM5XExpertBundle` opens, then use O(1) lookup for each expert role read. Keep the existing artifact identity, shape, dtype, extent, and CRC checks unchanged.
- Alternatives: retain the per-read linear scan, add a separate index file to the bundle format, or skip record lookup validation after assembly.
- Evidence: the bundle reader previously scanned every shard's `tensor_records` for each of three expert roles. The new focused regression verifies an index entry for every opened record and the expert-bundle/MLA/layer/model/MoE suite passes `22/22`; no full-model timing is claimed before the 282-shard gate.
- Accepted because: the change is format-compatible, keeps strict validation, and removes repeated metadata scans from the cold exact-expert path without changing payload bytes or routing.
- Revisit: after the full-bundle gate records bundle-open time and per-token expert-read latency. If open-time memory is material, replace the Python dictionary with a compact sorted index while retaining the same validation contract.

## D-0065 -- Avoid a duplicate host copy while decoding large role payloads

- Decision: decode raw-BF16 and FP32 role bytes through a read-only `memoryview` for payloads larger than 4 KiB, with a writable `bytearray` fallback for tiny fixtures.
- Alternatives: keep `bytearray` for every payload, suppress PyTorch's warning globally, or move the conversion into a new C++ reader before the Python reference gate is complete.
- Evidence: the focused expert-bundle/MLA/layer/model/MoE suite passed `16/16` with no warnings after the fallback was added. The large-payload path remains a view over the reader bytes; no payload, route, or output-parity test changed. No full-model timing or host-memory measurement has been run yet.
- Accepted because: it removes one avoidable CPU copy on the real multi-megabyte expert/trunk payload path while keeping the synthetic reference suite warning-free and preserving the existing exact validation boundary.
- Revisit: after the 282-shard full-bundle gate records decode latency, host RSS, and H2D traffic. Replace the threshold or move the path to the native reader if Python view lifetime or allocator behavior becomes material.

## D-0066 -- Keep row-scaled FP8 expert execution experimental

- Decision: expose row-scaled E4M3 expert weights behind `expert_precision="fp8"` and `--expert-precision fp8`; keep raw BF16 exact execution as the default and correctness reference.
- Alternatives: promote FP8 globally, quantize only the router/trunk, use native MXFP4 immediately, or remove low-precision experiments until a CUDA full-model runtime exists.
- Evidence: the focused layer/model/reference suite passed `14/14`. On the real five-shard layer-10 RTX 5080 probe with one token, host-quantized FP8 measured `2.901 s` cold versus exact `2.752 s`, and `5.713 ms` warm versus exact `4.731 ms`; route IDs were identical but output relative L2 drift was `5.603%`. The path currently quantizes before device staging but has no packed on-disk FP8 artifact yet.
- Accepted because: it creates a reproducible Blackwell FP8 experiment without weakening exact mode or silently changing quality. The measured bounded result does not justify default promotion.
- Revisit: after a persistent packed FP8/MXFP4 artifact and full-model quality/traffic gate exist. Promote only if measured H2D/NVMe savings outweigh dequantization and the coding-quality suite accepts the divergence.

## D-0067 -- Read all roles of one exact expert under one artifact open

- Decision: group `gate_proj`, `up_proj`, and `down_proj` references by K3X artifact and read them through one `K3XReader.read_tensor_extents_many()` call. Keep the strict per-record metadata and CRC validation contract.
- Alternatives: keep one file open per role, mmap every full artifact, or use a process-wide descriptor pool before full-model I/O is measured.
- Evidence: the real five-shard layer-10 bundle places all three roles of every complete expert in the same shard. Focused bundle/reader/layer/model tests passed `20` cases (`4` capability skips). With four exact payload readers, the one-token real layer-10 cold sample fell from the earlier `2.752 s` baseline to `2.184 s`; one worker measured `4.979 s`, so parallel reads remain the larger knob. No output or route difference was observed.
- Accepted because: it is format-compatible, preserves exact bytes and validation, and reduces per-expert file-open overhead without changing cache policy or routing.
- Revisit: after the full 78-layer gate records NVMe GB/token, open/read latency, and I/O queue pressure. Replace with mmap or descriptor reuse only if the measured filesystem overhead remains material.

## D-0069 -- Reject artifact-wide expert batch reads for the current NVMe path

- Decision: keep `expert_load_workers` at the expert-task granularity. Do not replace concurrent per-expert reads with one sequential batch task per artifact.
- Alternatives: group all selected role records by artifact and read each artifact once, serialize all selected experts, or implement a direct-I/O queue before changing task granularity.
- Evidence: on the RTX 5080 WSL2 five-shard GLM-5.2 layer-10 probe with two tokens, 16 selected experts, four workers, lazy bundle admission, and no caches, the existing per-expert task path measured `4.175182 s` median over four samples. The artifact-wide batch variant measured `4.976070 s`, `16.09%` slower, while route count and output shape remained unchanged.
- Accepted because: the current storage stack benefits from multiple outstanding expert reads; sequentializing them by artifact loses useful NVMe/OS parallelism. The exact grouped three-role read remains accepted.
- Revisit: after a full-model trace with direct I/O or an asynchronous queue can prove that descriptor overhead, rather than storage parallelism, is dominant.

## D-0070 -- Use sixteen expert read workers for the local full-gate benchmark

- Decision: set `tools/monitor_glm5x_full_gate.sh` to default `EXPERT_LOAD_WORKERS=16`, while keeping the general benchmark and correctness defaults explicitly switchable and serial where already defined.
- Alternatives: keep the monitor at four workers, use eight workers, or promote sixteen workers globally for every correctness run.
- Evidence: the same RTX 5080 WSL2 five-shard layer-10 probe with two tokens and 16 selected experts measured medians of `7.635803 s` (1 worker), `5.390695 s` (2), `4.428789 s` (4), `3.704820 s` (8), and `3.159413 s` (16). The 16-worker sample was approximately `28.7%` below four workers; route count and output shape stayed unchanged. This is a bounded I/O result, not full-model tok/s.
- Accepted because: the monitor is an explicit performance gate, and sixteen is the maximum useful parallelism for this eight-way/top-k probe without adding duplicate work. The CLI's serial correctness path remains available.
- Revisit: after the 78-layer gate records NVMe queue depth, host RSS, H2D traffic, and end-to-end quality. Reduce the monitor default if storage contention or memory pressure appears.

## D-0068 -- Reject physical-offset sorting inside grouped role reads

- Decision: keep `read_tensor_extents_many()` in caller/request order. Do not sort the three role records by physical offset unless a future storage benchmark proves a stable benefit.
- Alternatives: sort every grouped read by `data_offset`, use an artifact-specific descriptor/mmap cache, or leave the grouped-open optimization unchanged.
- Evidence: on the RTX 5080 WSL2 five-shard GLM-5.2 layer-10 probe with two input tokens, 16 selected experts, four payload readers, lazy bundle admission, and no expert caches, four paired samples measured a sorted median of `4.745335 s` versus `4.565814 s` in the existing request order. The sorted variant was `3.93%` slower; route count and output shape were unchanged. Focused reader/bundle/CPP tests passed `24` cases with `4` capability skips after reverting the experiment.
- Accepted because: the proposed seek optimization did not improve measured end-to-end sublayer time. The existing one-open-per-artifact grouping remains accepted; no speculative physical-order behavior is added.
- Revisit: only with controlled direct-I/O or a full 78-layer NVMe trace showing seek latency as a dominant component.

## D-0071 -- Add a bounded C++ deadline expert-load worker pool

- Decision: allow `DeadlineExpertLoader` to run `1..64` workers and make `RuntimeSession` use eight workers for the deadline schedule. Move `HostExpertStore` payload I/O outside its global mutex while retaining per-key in-flight deduplication and exact cache accounting.
- Alternatives: keep one serial loader with the old lock held across I/O, submit one artifact-wide batch task, or expose unrestricted thread creation per request.
- Evidence: the pre-change scheduler test failed to compile for the new two-argument constructor, then the focused scheduler/store tests passed after implementation. The C++ full suite passed `15/15`, and the Python suite passed `325` with `124` capability skips. A tiny synthetic runtime sweep was compute-dominated and showed no meaningful throughput gain, so it is not a performance claim. The real layer-10 Python probe had already shown expert-read overlap improving from `7.635803 s` at one reader to `3.159413 s` at sixteen readers, but the full C++ model gate remains pending.
- Accepted because: the worker count is explicit and bounded, same-key loads still execute once, different experts can overlap, and serial correctness remains selectable. No quantization, routing, or payload semantics change.
- Revisit: after the complete bundle gate records end-to-end latency, NVMe/RAM/H2D traffic, host memory, and quality. Reduce or raise the default only from those measurements.

## D-0072 -- Record logical artifact reads before claiming NVMe traffic

- Decision: expose per-phase K3X artifact read calls and payload bytes in the Python full-bundle benchmark, but name them `storage_read_*` and explicitly avoid calling them physical NVMe traffic.
- Alternatives: infer NVMe bytes from payload sizes, omit storage telemetry until a kernel-level counter exists, or report OS file reads as NVMe unconditionally.
- Evidence: `K3XReader` owns every selected payload extent read and can count data plus auxiliary bytes without changing bytes, routing, or validation. The host page cache can satisfy those reads, so the counter is a logical storage request rather than a device-level measurement.
- Accepted because: it gives the first full-model gate an auditable traffic baseline while preserving the distinction needed for later `iostat`/ETW/NVML or direct-I/O measurements.
- Revisit: when the full gate runs, pair these counters with physical device counters and H2D telemetry before publishing NVMe GB/token.

## D-0073 -- Reuse the exact FP32 LM head between logits calls

- Decision: lazily convert `lm_head.weight` to FP32 once per model/device and reuse that matrix for later logits calls. Keep FP32 logits as the correctness path; do not switch to BF16 logits implicitly.
- Alternatives: convert on every forward, keep only BF16 and accept a fast-mode quality change, or keep the FP32 matrix on CPU and copy it for every token.
- Evidence: at the real GLM shape `(154880, 6144)` on the RTX 5080, a fresh BF16-to-FP32 conversion took a `0.061629 s` median and allocated a `3,806,330,880`-byte matrix. Reuse of the prepared transpose view measured `3.13 us` median. The model/reference parity suite passed `9/9` after the change, and the active head now replaces the BF16 source after preparation.
- Accepted because: output arithmetic and token routing are unchanged, while repeated full-vocabulary conversion is removed from the decode hot path and steady-state VRAM retains only the FP32 head. The temporary conversion peak and final approximately `3.81 GB` FP32 residency must be included in the full-model pressure result.
- Revisit: after the complete bundle gate records peak VRAM and decode tok/s. If 16 GB pressure is material, add an explicitly opt-in BF16 logits mode with a separate quality gate rather than silently changing default precision.

## D-0074 -- Treat the BF16 traffic model as a constraint, not a performance result

- Decision: keep the dimension-derived GLM-5.2 traffic model in a separate document and do not convert it into a TPS estimate. Use it to prioritize resident mixed-precision trunk and expert-major reuse work.
- Alternatives: report a theoretical TPS from PCIe specifications, defer all traffic reasoning until the full bundle, or assume expert streaming is the only dominant cost.
- Evidence: official dimensions imply `34,228,302,336` bytes (`31.88 GiB`) of non-routed BF16 trunk and `79,526,785,536` bytes (`74.07 GiB`) of one-token trunk plus Top-8 routed expert fetch if weights are reloaded each layer. No physical bandwidth or end-to-end latency has been measured yet.
- Accepted because: the bound rules out an unquantized reload-everything design for the 10 tok/s objective without fabricating a result, while leaving exact routing and quality gates intact.
- Revisit: after the complete bundle records actual H2D/NVMe traffic, cache residency, and quality for at least one exact and one mixed-precision mode.

## D-0075 -- Group decoder-layer trunk tensor reads by artifact

- Decision: when constructing one decoder layer from a validated bundle, collect its attention, indexer, norm, router, shared-expert, or dense-MLP records and read them through one `read_tensor_extents_many()` call per backing artifact. Keep the individual-read helper for one-off/global tensors and preserve the serial/exact reference semantics.
- Alternatives: keep one file open/read per tensor, group all selected expert payloads into artifact-wide tasks, or add mmap/direct-I/O before the full-model gate establishes the dominant cost.
- Evidence: the new regression test failed before the change with `19` single tensor reads during one synthetic layer construction. After the change, the same test observed `0` single reads and at least one grouped read; the focused layer/bundle/model suite passed `18/18`, and the complete Python suite passed `327 passed, 124 skipped`. No end-to-end throughput or quality number was inferred from this metadata/open-path result.
- Accepted because: grouped reads preserve record order, payload bytes, dtype/shape checks, and lazy CRC validation while removing repeated artifact-file opens. It is an exact, disable-free reference-path optimization with no routing or precision change.
- Revisit: after the 282-shard full bundle records construction latency, logical/physical storage traffic, host memory, and full-layer output parity. A full-model regression or storage contention can justify reverting or narrowing the grouping.

## D-0076 -- Treat exact full-model storage reload as the primary performance blocker

- Decision: keep the exact BF16 natural Top-16 path as the correctness baseline, but prioritize resident non-expert trunk reuse and layer-aware asynchronous staging before promoting quantization, proxy, adaptive Top-K, or speculation.
- Alternatives: tune CUDA kernels first, enlarge the bounded expert cache without changing trunk residency, or report the dimension-derived traffic model as a TPS estimate.
- Evidence: the first full `GLM-5.2` bundle gate completed over `78` layers and measured `0.0033037330949489767` decode tok/s with `79,763,152,896` logical artifact-read bytes per token. The cached two-token gate measured `0.0032712558027912816` tok/s, zero expert-cache hits, and `159,526,305,792` decode bytes because the current capacities evicted the working set and the trunk was not cached.
- Accepted because: the result is an exact, measured baseline that identifies storage reload as dominant while preserving routing and output semantics for future comparisons. It does not claim physical NVMe throughput or production quality.
- Revisit: after an exact resident-trunk/pinned-staging implementation records output parity, H2D bytes/token, physical device traffic, VRAM/RAM pressure, and a fresh end-to-end gate.

## D-0077 -- Make the local full-gate coordinator POSIX-sh and environment-aware

- Decision: declare `tools/monitor_glm5x_full_gate.sh` with a POSIX `sh` shebang, remove Bash-only syntax, and select the known CUDA Python venv first, with `K3X_PYTHON` override and `python3`/`python` fallback.
- Alternatives: require callers to invoke `bash` manually, keep Bash-only arithmetic and tests, or hard-code a Windows Python interpreter.
- Evidence: the prior invocation reached all `282/282` markers but exited before assembly with `line 27: syntax error near unexpected token 'then'` because Bash arithmetic syntax was parsed by `sh`; the WSL image also has no bare `python`, while `/home/jolib/.venvs/k3x-m1/bin/python` has CUDA 13.0 and RTX 5080 support. `bash -n` passes after the repair.
- Accepted because: the coordinator only needs POSIX tests and arithmetic; removing Bash-only `[[`, `(( ))`, `BASH_SOURCE`, and `pipefail` makes both direct and `sh script` invocation safe while changing no conversion or model semantics.
- Revisit: if the project adds a containerized Linux runner, make the Python path a documented container default and keep the explicit override.

## D-0078 -- Keep CUDA TinyGEMM INT4 experts experimental and cache-bound

- Decision: accept CUDA TinyGEMM INT4 packing as an opt-in representation, require an explicit CUDA target, and keep BF16 as the default/correctness representation. Treat packed device residency as the useful mode; do not promote cold per-token packing.
- Alternatives: make INT4 the default, quantize on CPU and upload packed bytes, or use the existing FP8 path for all experts.
- Evidence: focused INT4/reference coverage passed `33 passed, 6 skipped` in the current WSL environment. A real layer-10 probe measured `13.2817 s` for the first four-token INT4 MoE call and `0.00977 s` for the identical call after a 2 GiB packed device-cache fill. The first full-model cold INT4-expert probe measured `0.002830 tok/s`, `353.331 s` decode, `45,298,483,200` logical expert bytes/token, and `17,341,184,512` peak allocated VRAM bytes.
- Accepted because: the representation and cache are correctness-tested and the warm reuse benefit is directly observed, while the cold/full-model measurement explicitly rejects silent default promotion. GPU-side qparam/packing work removes a large CPU conversion pass without changing router semantics.
- Revisit: only after an on-disk packed expert sidecar or exact cache-residency policy reduces logical expert reads and a fresh full-model quality/VRAM gate passes on the 16 GiB target.

## D-0079 -- Do not claim 10--20 tok/s while expert traffic remains 45.3 GB/token

- Decision: block any 10--20 tok/s claim or quality-mode promotion until a full-model run records a materially lower expert-read bound and measured decode tok/s.
- Alternatives: extrapolate from the layer-10 warm cache probe, report CUDA kernel times as model tok/s, or enable adaptive Top-K/proxy routing without a quality gate.
- Evidence: the exact full-model baseline read `45,298,483,200` logical expert bytes per decode token even after trunk INT4 residency; the cold INT4 probe retained the same read volume and slowed to `0.002830 tok/s`. The layer-10 `0.00977 s` result is a four-token sublayer cache hit, not a decoder-token measurement.
- Accepted because: it keeps benchmark semantics honest and identifies the next bottleneck as storage-side expert residency/packing rather than another isolated kernel toggle.
- Revisit: after expert-major multi-token verification, packed sidecar conversion, or a measured cache trace demonstrates a lower full-model bytes/token value.

## D-0080 -- Make packed INT4 sidecars fingerprint-bound and crash-safe

- Decision: add an optional `.pi4` sidecar keyed by `(layer, expert)` with a source-layout digest, per-role shape/extent metadata, CRC32C checks, and atomic temporary-file replacement.
- Alternatives: cache only by path, reuse raw BF16 payloads on every process, or write sidecars directly in place.
- Evidence: the layer-10 repeat probe created 31 sidecars and a fresh process reused them with `0` bundle-read calls and `0` bundle-read bytes; the dedicated CUDA round-trip test and full Python suite passed.
- Accepted because: exact source identity prevents stale reuse, atomic rename limits torn artifacts after interruption, and the feature is opt-in without changing natural routing or BF16 correctness mode.
- Revisit: when multi-process locking, disk-budget eviction, and a complete 78-layer full-model gate are measured.

## D-0081 -- Keep packed sidecars opt-in until full-model residency is measured

- Decision: expose `--expert-packed-cache-dir` but do not enable it by default or promote INT4 to QUALITY/BALANCED.
- Alternatives: auto-create sidecars for all experts, replace the source `.k3x` artifacts, or claim the layer-10 warm result as model tok/s.
- Evidence: sidecar reuse is a measured sublayer improvement, while the only full-model INT4 gate remains `0.002830 tok/s` and `45,298,483,200` logical expert bytes/token.
- Accepted because: it captures the safe warm-path win while preserving an auditable cold path and avoiding unbounded local disk growth.
- Revisit: after a fresh full-model run records bytes/token, VRAM/RAM, physical NVMe traffic, and quality.

## D-0082 -- Keep reduced routing and shared cold-expert proxy experimental

- Decision: expose `routing_top_k` and `proxy_mode="shared"` as explicit reference/benchmark switches, but keep natural Top-8 routing and exact expert execution as the default.
- Alternatives: make Top-4 or Top-6 the default, hide dropped experts behind the normal route metadata, or promote the shared expert as a lossless cold-expert replacement.
- Evidence: on the real layer-10 four-token activation, natural Top-8 took `12.43729756900575 s` with 31 unique experts. The shared Top-4 proxy took `5.043440291978186 s` with 16 unique experts, but relative L2 drift was `0.8120684623718262` and maximum absolute error was `0.01171150803565979`.
- Rejected as a default because: the measured quality drift is far outside the current coding-quality budget even though expert admission and wall time improve. The route metadata remains available for diagnosis, and the exact path is unchanged.
- Revisit: only after a calibrated proxy, outlier residual, or task-specific quality gate demonstrates materially lower drift on real GLM layers and full-model logits.

## D-0083 -- Extend fingerprinted sidecars to row-scaled FP8, but keep them opt-in

- Decision: allow the existing fingerprint-bound sidecar container to persist either CUDA INT4 (`.pi4`) or row-scaled E4M3 FP8 (`.pf8`) expert roles. Keep BF16 exact loading as the default and require an explicit `expert_precision="fp8"` selection.
- Alternatives: quantize FP8 on every source read, make FP8 the default expert representation, or store only the already-tested INT4 sidecar.
- Evidence: on the real layer-10 four-token activation, a fresh FP8 sidecar process measured `4.820426017016871 s` versus BF16 `11.759381022013258 s`, with identical route IDs, `5.696592479944229%` relative L2 drift, and `0.0007408261299133301` maximum absolute error. First population was slower at `21.40642180899158 s` because it read BF16 roles, quantized, and wrote 31 sidecars; the 31 files totaled `1,171,511,902` bytes, `50.05%` of the raw BF16 role bytes.
- Accepted as experimental because: sidecar reuse removes repeated source decode/quantization and the quality drift is materially lower than the measured MXFP4/Top-K proxy paths, but no full-model logits or coding benchmark has passed.
- Revisit: after a full-model cold/warm gate records FP8 bytes/token, VRAM/RAM pressure, final-token quality, and task-level regression.

## D-0084 -- Make FP4 the optimization target and demote FP8 to a comparison baseline

- Decision: stop the long full-model FP8 sidecar population, retain the already measured FP8 sidecar only as an opt-in comparison/interoperability baseline, and prioritize an explicit MXFP4/NVFP4 path for GLM experts. If an official GLM FP8 artifact is supplied later, consume it rather than maintaining a competing local FP8 format.
- Alternatives: continue the full FP8 gate, make local FP8 the default, or directly promote uncalibrated FP4 to the runtime default.
- Evidence: FP8 was not the requested final precision and its full-model gate produced no completed result. The bounded reference MXFP4 path stored eight layer-10 routed experts in `160,440,156` bytes (`26.56%` of corresponding BF16 bytes) with unchanged routes but `0.16359105706214905` relative L2 error and `17.867729659978068 s` fresh sidecar decode versus `2.79652249100036 s` BF16. Existing real-layer MXFP4 quality evidence therefore requires calibration/native kernels before promotion.
- Accepted because: it prevents time being spent on a precision the project does not target, preserves a useful measured control, and keeps the correctness BF16 path unchanged while the FP4 storage and CUDA work proceeds.
- Revisit: after calibrated outlier/mixed FP4 residuals, native RTX 5080 FP4 execution, full-model final-logit parity, and a fresh bytes/quality/TPS gate.

## D-0085 -- Use the Blackwell NVFP4 contract for the first native FP4 path

- Decision: implement NVFP4 as the first native FP4 runtime/storage path on RTX 5080, using E2M1 payloads, FP8 E4M3 per-block scales, one FP32 global scale, cuBLAS blocked-scale layout, and `torch._scaled_mm`. Keep `mxfp4/.pm4` as the portable comparison/reference path and keep all FP4 modes default-off.
- Alternatives: continue the CPU-decoded `.pm4` path, build a local row-scaled FP8 final format, or quantize all three projections without a native Blackwell layout.
- Evidence: the synthetic CUDA NVFP4 scaled-GEMM probe matched the dequantized reference for the tested GLM-shaped dimensions. The real layer-10 paired gate measured all-NVFP4 `5.2946 s` versus BF16 `4.5003 s` with `0.18142111599445343` relative L2 error; routed gate/up-only NVFP4 measured `4.3504 s` versus BF16 `4.4513 s` with `0.12603828310966492` relative L2 error and equal routes.
- Accepted because: it exercises the official Blackwell hardware primitive directly and removes the CPU decode bottleneck from the FP4 experiment while preserving an exact BF16 fallback and source-bound sidecars.
- Rejected as a default because: current uncalibrated layer drift is too high for the project quality target, and the bounded layer result is not an end-to-end throughput result.
- Revisit: after calibrated outlier/residual storage, final-logit/coding tests, multi-layer residency, and a full-model bytes/quality/TPS gate.

## D-0086 -- Keep current NVFP4 gate-up mode experimental after full-model divergence

- Decision: do not promote `expert_precision=nvfp4_gate_up` to a quality mode or default. Keep exact BF16 as the production reference until calibrated FP4 residuals and final-logit parity are demonstrated.
- Alternatives: promote the measured mixed NVFP4 mode because it removes bundle expert reads, promote all-NVFP4, or trade token divergence for the nominal FP4 bandwidth reduction.
- Evidence: the full 78-layer RTX 5080 gate with a 40 GiB trunk cache measured NVFP4 prefill `0.002230757197422688 tok/s`, TTFT `631.1266474290169 s`, and decode `0.005469012467235659 tok/s`; the paired BF16 control measured `0.003236382626324253` prefill tok/s, TTFT `412.11868096899707 s`, and `0.00969633691172072` decode tok/s. NVFP4 generated `[154820]` while BF16 generated `[565]`. K3X bundle reads fell to `0` during decode and `33,396,272,640` bytes during prefill, but sidecar I/O was not included in those counters.
- Accepted because: it preserves the exact correctness contract and records the real result instead of mistaking lower logical reads for higher end-to-end speed or quality.
- Revisit: after outlier/residual calibration, sidecar-to-VRAM prefetch/device residency, multi-token reuse, and full-logit/coding-quality parity.

## D-0087 -- Do not rely on a larger plain LRU expert device cache

- Decision: keep the existing device cache opt-in, but do not treat a larger plain LRU budget as the next performance solution. Add layer-aware/protected residency and sidecar-I/O telemetry before another full gate.
- Alternatives: increase the cache to 4 GiB and accept the eviction pattern, disable the cache entirely, or pin every selected expert from the preceding token.
- Evidence: the 4 GiB NVFP4 two-token gate recorded `1,800` misses, `0` hits, `1,691` evictions, and `109` resident entries. Decode averaged `0.0029452347553256177 tok/s`, with sidecar misses/writes still occurring. The layer-10 repeated-call probe, where the eight-expert working set fits, recorded `8` hits and a second-call time of `0.00870082201436162 s`; the full-model working set does not fit.
- Accepted because: it separates the proven small-working-set resident reuse from the full-model cache-thrash result and avoids claiming that capacity alone solves route-stable residency.
- Revisit: after hot-bank scoring, per-layer quotas/protection, transition-aware prefetch, and clean warm-sidecar gates.

## D-0088 -- Share dynamic activation quantization for NVFP4 gate/up

- Decision: use `linear_nvfp4_pair` whenever gate and up projections are both NVFP4. Quantize the shared hidden activation once, then submit both scaled GEMMs; retain the independent path for mixed/non-NVFP4 inputs.
- Alternatives: keep two independent activation quantizations, prequantize all activations globally, or fuse gate/up with the full SiLU/down projection before a correctness boundary exists.
- Evidence: resident RTX 5080 GLM-shaped microbenchmark averaged `0.0007789301977027208 s` paired versus `0.0014070075005292893 s` independent (`1.8063332307297124x`); parity test passed and the full suite was `341 passed, 124 skipped`.
- Accepted because: it is local, correctness-preserving, and removes duplicated activation work without changing routing or quantized weights.
- Revisit: after a full clean NVFP4 gate; the microbenchmark is not an end-to-end TPS claim.

## D-0089 -- Keep grouped NVFP4 projection experimental

- Decision: add and test `nvfp4_batched.py` as a selectable kernel boundary, but do not silently replace the full-model loop scheduler with it yet.
- Alternatives: enable grouped gate/up for every NVFP4 token, discard the prototype because transfer dominates, or implement a larger fused expert-major scheduler first.
- Evidence: CUDA parity was bit-equal on the tested shapes. Real RTX 5080 layer-10 samples ranged from `0.783x` to `2.138x` versus sequential gate/up depending on expert count, while the full gate is dominated by roughly `90--100 ms` sidecar admission/H2D per expert and measured `0.0144836` decode tok/s.
- Accepted because: the primitive is small, independently tested, and preserves an exact fallback, but the data does not justify making a variable projection microbenchmark the default end-to-end path.
- Revisit: after pinned/asynchronous staging and a layer-window residency scheduler provide a clean full-layer and full-model comparison.

## D-0090 -- Make layer-balanced protection explicit

- Decision: represent protected `(layer, expert)` keys explicitly in `GLM5XExpertTensorCache`; a protected key is never selected as an eviction candidate while a non-protected overrepresented entry exists.
- Alternatives: keep the count-only heuristic, pin all entries from a layer, or increase the device-cache capacity without changing policy.
- Evidence: the RED regression reproduced eviction of `(0,0)` after `(0,0)`, `(0,1)`, `(1,0)`, and `(0,2)` were admitted with one protected entry per layer. A larger synthetic multi-layer trace also produced zero second-pass hits under the old heuristic.
- Accepted because: it fixes the stated policy invariant without changing LRU behavior or natural routing. It does not claim that a 4 GiB cache can hold the full expert working set.
- Revisit: after route-stable hot-bank selection and a full-model trace show how many protected entries per layer fit the 16 GiB VRAM budget.

## D-0091 -- Add a bounded verified packed-sidecar host tier

- Decision: add `--expert-packed-host-cache-bytes` as an opt-in RAM LRU for already validated packed sidecar payloads. Keep the default at zero, keep decoded GPU residency separate, and reject capacities that are not explicitly budgeted with the trunk cache.
- Alternatives: reread and CRC-check every sidecar request, cache all packed sidecars without a bound, or implement pinned H2D before eliminating repeated file/CRC work.
- Evidence: on 16 real `.pgu` sidecars with 16 readers and an RTX 5080, the first 2 GiB host-cache pass took `1.715 s` and the second pass `0.282 s`; `16` host hits retained `629,145,728` payload bytes. A 40 GiB host tier plus a 40 GiB trunk tier reached approximately `72 GiB` WSL RSS before a two-token full gate produced a result and was stopped safely.
- Accepted because: the cache preserves source digest, role metadata, CRC validation on first admission, exact decoder output, and a zero-capacity reference mode while removing repeated NVMe/JSON/CRC work for warm sessions.
- Revisit: after a route-stable multi-layer trace reports host hit rate, physical NVMe bytes, H2D bytes, RSS, and final-token parity. Do not treat the bounded sidecar result as full-model tok/s.

## D-0092 -- Repair the synthetic benchmark worker-option forwarding

- Decision: thread `l2_expert_workers` through `benchmark_once()` and every reference/diagnostic invocation so the CLI option no longer fails before running.
- Alternatives: remove the CLI option, silently ignore it, or leave the benchmark runner broken and rely on direct binary calls.
- Evidence: the CLI previously raised `TypeError: benchmark_once() got an unexpected keyword argument 'l2_expert_workers'`; the RED regression reproduced it, the focused test passed after the one-parameter forwarding fix, and the CUDA synthetic prefetch benchmark completed with its requested worker count.
- Accepted because: it restores an existing benchmark contract without changing runtime semantics. The synthetic prefetch measurement is recorded separately and is not a GLM full-model claim.
- Revisit: if benchmark schema generation changes again, add a direct CLI smoke test rather than relying only on function-level coverage.

## D-0093 -- Normalize the INT4 guard on CUDA-less CI

- Decision: reject INT4 quantization with `ValueError(GLM5X_INT4_CUDA_REQUIRED)` before entering the quantizer whenever the requested target is CPU or CUDA is unavailable.
- Alternatives: let the lower-level quantizer raise its environment-specific `RuntimeError`, skip the test on CPU-only runners, or change the test contract.
- Evidence: GitHub Linux run `31884496150` reproduced the mismatch as `RuntimeError: GLM5X_INT4_CUDA_UNAVAILABLE` for a test that requires the stable CPU-target `ValueError`. The focused regression passed after the guard, and the full local WSL suite remained `354 passed, 124 skipped`.
- Accepted because: callers receive one stable API error independent of whether the host has CUDA, while CUDA-enabled packing behavior is unchanged.
- Revisit: only if the public quantization API later adopts a distinct environment-error type across all precision paths.

## D-0094 -- Protect the current layer's resident expert set

- Decision: use an explicit per-layer access set in the C++ resident-weight table. Selected expert keys are protected before grouped execution, cache hits are touched, and only non-protected entries are eligible for byte-budget eviction. If every candidate is protected, admission bypasses rather than invalidating a live pointer.
- Alternatives: keep the previous capacity-only bypass, evict by plain LRU during a grouped launch, or pin the whole resident table. The first two either thrash or can invalidate pointers collected before a launch; the last one violates the VRAM budget.
- Evidence: the new CUDA unit test passes protected-first/second admission, all-protected bypass, explicit access protection, resident-byte accounting, and route/token parity. The synthetic 4 KiB and 1 MiB runs both produced exact token IDs; latency was statistically neutral (`17.018 ms` versus `17.116 ms` medians), so this is a safety/policy result rather than a speed claim.
- Accepted because: it preserves correctness under constrained residency without changing natural routing or default capacity behavior.
- Revisit: when a route-stable multi-layer trace supplies eviction counts, transition probabilities, and full-model H2D/quality measurements. A shared table still needs a per-forward context or session serialization before concurrent forwards are supported.

## D-0095 -- Keep pinned sidecar staging explicit and default-off

- Decision: add a bounded page-locked staging pool and non-blocking CUDA copy path to the packed sidecar cache, but require an explicit positive capacity and one expert reader. Reject pinned staging with BF16 precision or without a packed sidecar path.
- Alternatives: always allocate pinned buffers, use non-blocking copies from pageable memory, or enable multiple reader threads immediately. The first increases RAM pressure for every run; the second cannot provide asynchronous H2D; the third risks stream/event lifetime races in the current cache implementation.
- Evidence: the focused Python suite and full suite passed (`356 passed, 124 skipped`). On a real layer-10 bounded probe, the pinned path was slower on first use (`5.606441 s` versus `4.130233 s`) and slightly faster on the repeated forward (`3.370938 s` versus `3.469007 s`). A standalone RTX 5080 transport probe reduced GPU event time (`1.358 ms` to `0.588 ms`) but increased wall time when staging allocation was not pooled.
- Accepted because: the feature creates a measurable, correctness-preserving boundary for future overlap while keeping the exact synchronous reference path unchanged.
- Revisit: after pooled reuse across a layer window reports physical sidecar bytes, H2D bytes/time, host RSS, cache hits, and final-logit parity. It is not evidence for 10--20 tok/s by itself.

## D-0096 -- Keep packed-sidecar telemetry opt-in and separate from performance gates

- Decision: add exact additive counters for sidecar file reads, validated payload bytes, H2D bytes, host submission time, and CUDA-event transfer time, but enable them only through an explicit diagnostic flag.
- Alternatives: infer sidecar traffic from logical K3X reads, enable event timing unconditionally, or wait for a system-wide physical NVMe profiler before instrumenting the cache.
- Evidence: on one real layer-10 expert-48 `.pgu`, five warm paired samples moved `39,321,608` bytes per call with bit-exact tensor parity. Pageable H2D event median was `3,060,958 ns`; pooled pinned median was `1,033,468 ns`. Wall medians were `17,895,936 ns` and `11,424,774 ns` respectively.
- Accepted because: the counters identify the actual sidecar/H2D boundary without conflating it with logical artifact reads, while default execution and benchmark output remain unchanged.
- Revisit: replace forced event synchronization with asynchronous completion accounting when the layer-window scheduler exists, and add independent physical NVMe sampling before interpreting file bytes as device traffic.

## D-0097 -- Keep exact N+1 transition prefetch implemented but default-off

- Decision: retain deterministic transition-table prediction and exact ticket reuse behind `--transition-prefetch-candidates`, with zero as the default. Predictions may schedule payload work but never change router scores, selected K, expert weights, or output; a nonmatching selection always uses the exact load path.
- Alternatives: enable N+1 prediction by default, jump directly to an unmeasured learned/N+2 predictor, or discard lookahead entirely after the first slowdown.
- Evidence: the five-iteration standard synthetic gate kept token IDs `[43,32,28,49,9,28]` and the full routed-expert trace identical. Candidate counts `0/1/2` measured `150.141/129.338/113.328` decode tok/s. Candidate 2 reduced median exposed load wait from `50.024 ms` to `33.962 ms`, but only `17/36` submissions matched and `19/36` were unused, so added work outweighed hidden latency.
- Accepted because: the implementation establishes an exact, observable scheduling boundary for future real sidecar/H2D experiments while the default remains evidence-based and unchanged.
- Revisit: when the official full-model path can retain predicted packed experts in a pooled pinned/device layer window. Require higher recall, lower bytes per useful match, physical sidecar/H2D counters, final-logit parity, and positive end-to-end decode improvement before promotion.

## D-0098 -- Retain a stable per-layer exact hot bank, default-off

- Decision: add `stable_hot_bank` as an opt-in decoded-expert device-cache policy. Retain a fixed number of exact experts per layer, promote only a strictly more frequently observed same-layer candidate, and bypass transient admissions that would evict the retained bank.
- Alternatives: enlarge plain LRU, keep the first-admitted `layer_balanced` entry forever, or admit every routed expert and evict globally. Plain LRU already measured near-zero reuse under full-model churn; first-admitted protection cannot adapt; global admission recreates the same thrash.
- Evidence: on RTX 5080 with 128 digest-matched real `.pgu` experts across 16 layers and a 768 MiB cache, LRU warm passes had `0/128` hits and moved `5,033,165,824` H2D bytes. The stable bank retained one expert per layer, produced `16/128` hits, moved `4,404,020,096` bytes, and reduced three-pass median wall time from `3,656,221,967` to `3,289,269,276` ns (`10.04%`). Focused cache/model/CLI regression passed `35`, with `6` environment skips.
- Accepted because: the policy reduces measured repeated transfer and churn without changing routing or expert payload identity, while capacity and bypass behavior remain explicit.
- Revisit: keep default-off until a natural-router full-model trace confirms hit rate, byte reduction, final-token/logit parity, host RSS, and positive decode tok/s. Combine it with asynchronous pooled H2D rather than treating the bounded `10.04%` result as a token-throughput claim.
