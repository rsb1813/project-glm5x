# K3X Released-Dimension Resident MoE-Layer Benchmark Design

## Status

Accepted under the user's standing approval for non-billable work before Cloud Run. This design does not download the full Kimi K3 checkpoint, provision cloud resources, alter routing, or select a new runtime default.

## Purpose

B-0022 proves that the complete resident `moe-layer` boundary removes three synchronizations per call and lowers activation H2D, total H2D, and D2H on the executable synthetic graph. Its paired decode timing is mixed from -2.753% to +5.619%, so that graph is too small to choose CUDA Graph caching or a larger device-resident token boundary.

Milestone 22 will measure the same exact split-versus-layer boundary at the released Kimi K3 MoE dimensions without constructing a full checkpoint or claiming token throughput. The result is a bounded CUDA execution benchmark, not an executable model and not a quality benchmark.

## Released shape and memory contract

The benchmark fixes these released dimensions.

| Quantity | Value |
|---|---:|
| Hidden width | 7,168 |
| Routed latent width | 3,584 |
| Expert intermediate width | 3,072 |
| Native MXFP4 expert payload | 17,547,264 bytes |
| Natural routed expert count represented | 16 |

Six FP32 dense/vector members occupy 469,776,384 bytes.

- Routed down and up together occupy 205,520,896 bytes.
- Shared gate, up, and down together occupy 264,241,152 bytes.
- Routed RMSNorm occupies 14,336 bytes and exists in L0 only for the complete layer path.

At sixteen repeated-view experts, native expert residency occupies 280,756,224 bytes. The complete layer cold resident footprint is therefore 750,532,608 bytes before scratch and CUDA library resources. A 1 GiB hard capacity is sufficient while remaining far below the RTX 5080's 16 GB VRAM.

## Alternatives

### Selected: full released dense shape plus repeated-view experts

Generate deterministic FP32 dense weights in memory and load one existing released native-MXFP4 expert artifact. Present that payload under unique tensor IDs for 1, 4, and 16 expert slots. This exercises the real complete-layer backend dimensions, resident admission, expert grid, ordered mix, RMSNorm, shared branch, and final add.

The payload bytes are repeated, so the benchmark has no routing semantics and does not represent checkpoint diversity. Unique IDs intentionally force the same L0 capacity pressure as distinct expert weights.

### Rejected: another scaled synthetic sweep

A medium-width sweep would be cheaper but would not close the evidence gap identified by B-0022. It could locate a synthetic crossover without proving that the backend accepts and executes released dimensions.

### Deferred: released-dimension executable checkpoint

Materializing attention, router, embeddings, and every non-expert tensor would broaden the experiment from one dependency-closed MoE boundary into a multi-gigabyte model build. It would also require a much slower independent CPU graph oracle. That belongs to a later checkpoint-manufacturing milestone.

## Components

### `k3x_cuda_moe_layer_bench`

A CUDA-only C++ binary accepts:

```text
--model <released storage fixture>
--boundary ffn-block|moe-layer
--experts 1|4|16
--warmup <non-negative integer>
--iterations <positive integer>
```

It loads layer 1 expert 0 from the existing `STORAGE_FIXTURE`, generates deterministic FP32 hidden input and dense weights, and creates unique tensor IDs for every logical expert slot. It uses FP32, reused allocation, exact resident weights, resident-grid batching, synchronous transfer, and fusion none. Capacity is fixed at 1 GiB.

The split path executes routed-down, resident expert grid, CPU router-order mix, strict CPU RMSNorm, routed-up, shared SiTU MLP, and final vector addition. The layer path calls `resident_mxfp4_moe_layer` once. Both use identical input, weights, expert views, contributions, epsilon, SiTU parameters, layer, and phase.

### Numerical oracle

Every process builds a separate split CUDA backend and executes the split path once. The selected measurement backend then performs one untimed cold-admission execution. Its output must match the split output within `1e-5` maximum absolute error before warmup or measurement proceeds.

This oracle does not replace the existing independent CPU whole-layer oracle. It isolates numerical parity between two CUDA execution boundaries at dimensions where a full scalar CPU decode would dominate the benchmark setup.

### B-0023 runner

`tools/ablate_cuda_released_moe_layer.py` executes this canonical matrix.

```text
split-1
layer-1
split-4
layer-4
split-16
layer-16
```

The runner writes one raw JSON file per row plus `summary.json` and `summary.csv`. It hashes the released artifact, runner binary, every raw record, the canonical record aggregate, and the LF-stable summary CSV.

## Measurement phases

Each binary process has four phases.

1. Construct deterministic host views and the split oracle backend.
2. Execute one split oracle and one selected-backend cold admission.
3. Execute the requested warmups without recording latency samples.
4. Snapshot telemetry, execute measured iterations, and report deltas.

Cold admission and warm steady-state traffic are deliberately separate. The output records cold weight H2D and resident bytes after the admission execution. Measured iterations must report zero weight H2D for both paths.

## Required raw fields

Every row records:

- artifact kind and `routing_semantics=false`;
- boundary, expert count, released dimensions, warmups, and iterations;
- maximum absolute error;
- median boundary latency and aggregate kernel nanoseconds;
- measured activation H2D, D2H, weight H2D, and synchronization count;
- cold weight H2D, resident bytes, peak resident bytes, and peak VRAM;
- resident-grid calls, launches, and fallbacks;
- resident-MoE-layer calls, experts, launches, fallbacks, and contribution bytes.

## Pair gates

For each expert count, the runner rejects the result before summary publication unless all conditions hold.

- Both rows use the released dimensions, the same artifact, and `routing_semantics=false`.
- Maximum absolute error is at most `1e-5`.
- Both rows have zero residency bypass and zero grid/layer fallback.
- Warm measured weight H2D is zero.
- Split synchronization count equals `4 * iterations`.
- Layer synchronization count equals `iterations`.
- Layer calls equal `iterations` and layer launches equal `13 * iterations`.
- Layer activation H2D and D2H are lower than split.
- Layer-minus-split cold weight H2D equals 14,336 bytes.
- Layer-minus-split resident bytes equals the same 14,336 bytes.

Latency direction is recorded but never asserted. Kernel time, peak VRAM, and memory occupancy are evidence, not pass/fail speed claims.

## Failure behavior

Malformed CLI values return exit code 2. Reader, CUDA capability, resident capacity, validation, allocation, launch, numerical, or telemetry failures return exit code 4 with a typed message. No CUDA failure becomes CPU fallback. A hard-cap bypass is a benchmark failure because the 1 GiB identity is required to be full-fit.

## Evidence boundaries

B-0023 may establish released-dimension backend correctness, warm transfer reduction, synchronization reduction, memory footprint, and boundary latency. It may not claim decode or prefill tok/s, TTFT, model quality, coding quality, routing quality, full-model cache behavior, physical NVMe traffic, native-Linux performance, or a CUDA Graph decision.

The next architecture choice remains deferred until B-0023 is measured and reviewed.
