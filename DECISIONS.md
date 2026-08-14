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
