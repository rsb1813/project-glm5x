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
