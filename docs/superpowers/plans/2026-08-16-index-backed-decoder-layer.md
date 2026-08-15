# Index-backed GLM-5.2 Decoder Layer Plan

## Goal

Prove one complete official GLM-5.2 sparse decoder layer in C++ from the validated 282-shard `.gxi` source, using the existing RTX 5080 learned-MoE backend and one strong Python oracle boundary.

## Accepted first boundary

- Layer 10, because the official configuration marks it as a full DSA-indexer layer and it is already covered by the learned-MoE gate.
- Two BF16 input tokens, empty MLA/DSA state, official RoPE theta `8,000,000`, natural Top-8 routing, exact shared expert, and BF16-rounded output.
- Existing `GLM5XACT` files carry only full-layer input and output. The benchmark JSON records DSA top-k, MLA/DSA state lengths, source bytes, CUDA traffic, and numerical error.
- Trunk attention is orchestrated in C++ first. The existing resident raw-BF16 CUDA MoE remains the accelerator boundary. A device-resident attention API is deliberately deferred until the complete layer parity gate is green.

## Rejected shortcuts

- Python attention plus C++ MoE does not establish a C++ decoder-layer boundary.
- A new general device-resident attention API before parity would combine format, state, kernel, and ownership risks in one change.
- Exporting many intermediate test files is unnecessary for the first gate; one full-layer oracle plus route/state telemetry is the stronger minimal test.

## Tasks

1. Extend the existing activation exporter with an opt-in full-layer input/output mode while preserving the current MoE export behavior.
2. Add a RED CLI contract for the new C++ full-layer mode against the small fixture.
3. Implement layer-10 BF16 trunk loading from `.gxi`, RMSNorm, q residual, official full DSA indexer, MLA attention, residuals, and the existing learned-MoE handoff.
4. Compare the complete BF16-rounded layer output to the Python oracle and require unchanged natural routes.
5. Run focused Python tests, C++/CUDA CTest, then one bounded RTX 5080 measurement.
6. Record architecture, decision, benchmark, and project-state evidence. Design the incremental-state artifact only after this boundary passes.

## Acceptance checks

- Existing MoE-only command remains byte/route compatible.
- Full-layer command rejects non-index input and malformed/missing oracle artifacts.
- Official layer 10 output remains within the established BF16 numerical envelope and natural Top-8 routes match the Python export.
- JSON distinguishes full-layer latency from full-model token throughput and reports no projected TPS as measured TPS.
- All applicable focused tests and CUDA CTest pass before publication.
