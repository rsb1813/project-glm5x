# GLM5X Milestone Checklist

- [x] Verify and delete only the approved official K3 artifacts.
- [x] Bootstrap the independent GLM5X Git repository.
- [x] Preserve the tested K3X storage/cache/runtime compatibility core.
- [x] Add GLM-5.2 descriptor validation.
- [x] Add the GLM-5.2 tensor manifest boundary.
- [x] Add the synthetic GLM5X reference graph and greedy parity smoke tests.
- [x] Add the `glm5x-convert` user-facing wrapper.
- [x] Add CPU/reference TurboQuant KV compression and capacity arithmetic.
- [x] Record the TurboQuant correctness smoke result and non-performance caveat.
- [ ] Connect compressed blocks to the GLM DSA/indexer state.
- [x] Add a bounded GLM-5.2-shaped resident expert CUDA baseline and expert-major token batching benchmark.
- [x] Reuse exact resident MXFP4 weights in expert-major batch verification and record warm H2D telemetry.
- [ ] Implement packed paged-KV CUDA storage for RTX 5080.
- [ ] Add MTP/DSpark expert-major verification to the GLM path.
- [ ] Run real GLM-5.2 quality and throughput benchmarks.
