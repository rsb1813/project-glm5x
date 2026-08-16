# W8A16 Expert Runtime Design

## Goal

Cut official GLM-5.2 routed-expert weight traffic roughly in half without
quantizing decode activations, while preserving natural Top-8 routing and a
strict BF16 fallback.

## Evidence behind the choice

The bounded official layer-10, two-token probe compared the same natural
Top-8 routes against the BF16 reference.

| Representation | Ideal expert-weight bytes vs BF16 | Layer relative L2 |
|---|---:|---:|
| NVFP4 W4A4 | 28.125% | 12.60% in the paired layer gate |
| FP8 W8A8 | 50.054% | 4.657% |
| W6A16, group 32 | 40.625% | 2.722% |
| W8A16, group 128, BF16 scales | 50.781% | 0.8926% |

The router contributions were nearly uniform. Keeping seven of eight routed
experts in BF16 and only one in FP8 still produced 1.591% layer error while
retaining about 93.8% of BF16 weight bytes. Low-rank, row-outlier,
column-outlier, SmoothQuant, and dual-NVFP4 residual probes also failed the
one-percent boundary. W8A16-G128 is therefore the first measured candidate
that passes the bounded quality gate and materially reduces bytes.

## Packed representation

Each row-major matrix is split into groups of 128 input columns.

- Values are signed symmetric INT8 in `[-127, 127]`.
- One BF16 scale is stored per row and group.
- The quantized value is `round(weight / bf16(scale))`, clamped to the signed
  range. Dequantization is `int8_value * bf16(scale)`.
- The value slab precedes the scale slab. Both sizes are derived from shape and
  group size; no per-row offsets are stored.
- The format is fingerprinted and default-off until converter-side persistence
  and final-logit quality gates are complete.

The ideal payload ratio is `8/16 + 1/128 = 0.5078125` of BF16. Extent headers,
alignment, and checksums add a negligible fixed overhead outside this ratio.

## CUDA execution

Decode keeps activations in BF16 and uses an M=1 bandwidth-oriented CUDA path.

1. Convert the already BF16-rounded hidden state to a resident BF16 input slab.
2. Launch one fused gate/up expert-major kernel. Each warp owns one output row,
   reuses the input values, decodes signed INT8 weights with BF16 group scales,
   and stores the BF16-rounded SiLU-gated intermediate.
3. Launch one expert-major down kernel and store BF16-rounded per-assignment
   outputs.
4. Reuse the existing deterministic route metadata to accumulate natural
   router contributions by token.
5. Execute the shared expert through the existing exact resident BF16 path and
   add it to the routed result.

Routed W8 weights use the existing byte-bounded resident table under a distinct
representation key. A selected access set protects all gate/up/down slabs
until both kernels finish. Admission failure is explicit; the benchmark and
runtime retain the exact BF16 path rather than silently changing routing or K.

## Scope of the first implementation

- Host pack/decode contract and corruption/shape validation.
- CUDA resident W8A16 expert-major path for natural routed experts.
- Existing exact BF16 shared expert and decoder trunk.
- Opt-in `w8a16-g128` benchmark mode on the official layer-10 `.gxi` gate.
- Weight/H2D/residency and kernel/wall telemetry.

Persistent `.pw8` sidecars and full 78-layer manufacturing follow only after
the resident official-layer gate proves both quality and latency. This avoids
manufacturing hundreds of gigabytes for a kernel that has not passed the RTX
5080 gate.

## Acceptance gates

- Route expert IDs and router contributions remain unchanged.
- Official layer-10 relative L2 versus the BF16 control is at most 1.0%.
- The bounded greedy token must match once the full-logit path is connected.
- Warm routed-expert H2D is zero after resident admission.
- The first performance target is at most 1.0 ms per official layer token for
  the W8 routed MoE boundary. A complete decoder-layer result remains separate.
- Any failed gate leaves W8A16 experimental and the BF16 path authoritative.

## Rejected alternatives

- Native NVFP4 W4A4 is rejected for this quality mode because both activation
  and diffuse weight error remain far above the accepted boundary.
- Router-mass BF16 rescue is rejected because the official Top-8 contributions
  are too uniform to save enough bytes.
- FP8 W8A8 is rejected as the primary M=1 path because it misses the quality
  gate and its current scaled-matmul launch path was slower than resident BF16
  in the bounded probe.
- A general quantization framework is deferred. The first kernel and format are
  deliberately fixed to the official GLM dimensions and group size 128.
