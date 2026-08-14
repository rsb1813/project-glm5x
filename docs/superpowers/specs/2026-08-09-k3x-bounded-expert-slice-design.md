# K3X Milestone 7 Full-Dimension Bounded Expert Slice Design

Milestone 7 adds one physically materialized routed-expert slice at released Kimi K3 dimensions. It exists to measure representative conversion and Reader traffic without downloading or pretending to execute the complete checkpoint.

## Scope

The default slice contains layer 1, expert 0 and exactly six source extents.

| Matrix | Logical shape | Packed E2M1 bytes | E8M0 scale bytes |
|---|---:|---:|---:|
| gate | 3,072 x 3,584 | 5,505,024 | 344,064 |
| up | 3,072 x 3,584 | 5,505,024 | 344,064 |
| down | 3,584 x 3,072 | 5,505,024 | 344,064 |
| total | 33,030,144 values | 16,515,072 | 1,032,192 |

The resulting expert payload is exactly 17,547,264 bytes before K3X alignment and directory overhead. No embedding, trunk, state, second expert, or generated token belongs to this fixture.

## Alternatives

### Sparse-file holes

Rejected. Holes would validate offsets but distort physical reads, checksums, page-cache behavior, and direct-I/O evidence.

### A scaled executable synthetic graph

Rejected. It would either retain non-representative expert extents or materialize a prohibitively large trunk and recurrent state. It also mixes storage evidence with CPU graph time.

### One exact-dimension, non-executable expert slice

Selected. It materializes every payload byte, reuses the existing streaming converter, and keeps the benchmark boundary limited to one exact expert load.

## Artifact identity

The source manifest uses `format: k3-storage-slice-v1`, the released text-model dimensions already recorded in `ARCHITECTURE.md`, and `artifact_kind: storage_fixture`. It contains only the selected expert tensor pairs.

K3X v1 reserves optional-feature bit 0 as `STORAGE_FIXTURE`. The converter sets this bit for the bounded slice and leaves it clear for executable checkpoints. Python and C++ Readers expose the bit and may inspect the artifact. Model execution rejects it before tensor lookup with `NON_EXECUTABLE_ARTIFACT`. Unknown optional bits remain ignorable as required by the existing optional-feature contract.

This is not a new K3X major version. It uses the existing tensor, layer, expert, extent, checksum, and configuration records.

## Streaming source generation

A focused source-fixture module writes a valid safetensors shard without building any full matrix in RAM.

1. Build the deterministic JSON header and exact offsets.
2. Write to a sibling `.partial` file.
3. Emit each packed or scale tensor in bounded chunks, updating a source digest while writing.
4. Flush and fsync the shard and manifest.
5. Atomically replace the final shard and manifest only after their lengths and digests match.

The default chunk size is 1 MiB and the recorded maximum generated chunk must not exceed the configured bound. Payload bytes are deterministic from the fixture seed, tensor identity, and absolute byte position. E8M0 scale bytes avoid reserved encodings.

The existing converter continues to read source tensors by chunk. Its peak source read remains bounded by `chunk_bytes`; the test uses a deliberately smaller value and verifies the reported maximum.

## Reader benchmark boundary

A dedicated `k3x_storage_bench` executable opens the artifact with the selected Reader options, resolves the expert's gate/up/down records, builds the six ordered data/auxiliary requests, and loads them as one batch. It never constructs `Model` and never reports token throughput.

For each measured expert load it records.

- engine, cache mode, queue depth, and direct-I/O alignments.
- exact logical, submitted, and completed bytes.
- calls, batches, completions, short reads, failures, and Reader storage time.
- wall-clock expert-load latency and expert loads per second.
- a deterministic digest over the six ordered returned extents.
- Linux process `rchar` and `read_bytes` deltas when available.

The benchmark runner crosses `pread|io_uring` and `buffered|direct`. It skips only explicit `STORAGE_UNAVAILABLE` capability failures, requires digest and logical-byte parity across supported cases, and writes raw JSON/CSV plus a compact B-0008 manifest. Warm/cold preparation is never automated because privileged global cache drops are outside scope.

## Correctness and failure policy

- Existing tiny-model layer, state, routing, and token tests remain the execution oracle.
- Source generation tests verify exact safetensors tensor shapes, lengths, deterministic bytes, manifest digests, and bounded chunk size.
- Converter tests verify six tensor records, one expert record, exact 17,547,264 logical payload bytes, optional-feature identity, checksums, and resumability.
- Python and C++ Reader tests verify the same ordered digest and exact byte accounting.
- Runtime tests verify that a storage fixture fails before graph execution and that ordinary synthetic checkpoints remain executable.
- Truncation, digest mismatch, invalid expert identity, missing matrix pairs, and accidental executable use fail closed.

## Acceptance

Milestone 7 is accepted when the bounded source and K3X artifact are deterministic, conversion stays within its configured chunk bound, all six full-dimension extents round-trip exactly, execution rejection is explicit, all existing correctness suites remain green, and B-0008 is recorded with its actual environment. No B-0008 value may be labeled token throughput, native P44 Pro traffic, or full-model performance.

## Explicit exclusions

- Full Kimi K3 checkpoint download.
- Full-dimension trunk or graph execution.
- Multiple experts, cache eviction, Least-Stale, or cross-layer scheduling.
- Default Reader-policy changes.
- Privileged cache dropping, Cloud Run, or paid cloud resources.
