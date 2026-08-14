# GLM5X Project State

## Current milestone

GLM-5.2 shape/manifest boundary plus RTX 5080 resident native/BF16 expert-grid baselines.

## Completed

- Created the independent repository at `C:\Users\jolib\Documents\project-glm5x`.
- Migrated storage, converter, reference, runtime, test, and benchmark source directories without K3 official weight artifacts.
- Deleted the six verified official K3 artifact directories from the old K3X worktree. Synthetic fixtures remain.
- Added GLM-5.x descriptor validation and the `glm5x-convert` CLI wrapper.
- Added a tiny synthetic GLM5X reference fixture covering recurrent state, Top-K routing, and greedy generation parity.
- Added the GLM5X architecture/design and bootstrap plan documents.
- Added `TurboQuantConfig`, `QuantizedVector`, and `TurboQuantKVCache` with Hadamard rotation, 2/2.5/3/3.5/4/6/8/16-bit schedules, asymmetric K/V settings, incremental attention, and logical 1M-token capacity estimation.
- Added six focused TurboQuant correctness/capacity tests.
- Added GLM-5.2 descriptor fields for MoE intermediate width, DSA index shape, MTP sharing, and maximum position length.
- Added `GLM5XTensorManifest` validation for safetensors weight maps, shard names, tensor count, and source byte totals.
- Added and measured `k3x_cuda_glm5x_moe_bench` on the real RTX 5080 at GLM-5.2 expert dimensions.
- Added an expert-major candidate-token benchmark mode for 1/2/4/8 tokens.
- Added resident exact MXFP4 reuse to the CUDA expert-major batch backend and allowed resident weights in the CLI validation contract.
- Added opt-in resident BF16 dequantized expert-grid execution through cublasLt, with native MXFP4 fallback when resident capacity is insufficient.

## In progress

- Replace synthetic K3 source assumptions in the converter with model-specific GLM manifest extent roles.
- Build the tiny GLM-5.2-compatible reference graph and greedy parity tests.
- Rename user-facing runtime and benchmark commands where that does not break the inherited storage ABI.
- Connect compressed KV blocks to the GLM DSA/indexer state and add a reference-mode switch.
- Connect the now-resident expert-major batch backend to exact GLM DSA/MTP state and retain strict natural Top-8 verification.
- Validate the dequantized resident-GEMM path against nonzero GLM shard data and measure VRAM-bank pressure before considering a default.

## Known blockers

- No GLM-5.2 weights are present, so full checkpoint correctness and local TPS are not measured.
- The migrated C++ runtime still has K3-oriented names and graph assumptions in several files.
- CUDA build and bounded RTX 5080 kernel measurements are validated in WSL; native Linux end-to-end throughput has not been measured.
- The TurboQuant implementation is CPU/reference only; it does not yet contain a packed CUDA kernel or full PolarQuant/QJL production path.
- 600k/1M capacity is a formula estimate only until a real GLM-5.2 DSA state is allocated and restored.

## Hardware assumptions

- CPU: AMD Ryzen 7 9800X3D.
- GPU: NVIDIA RTX 5080 16 GB.
- RAM: 96 GB DDR5-4200.
- Storage: Solidigm P44 Pro 2 TB NVMe.
- Preferred execution environment: native Linux.

## Latest verified state

- Focused GLM descriptor, manifest, CLI, toy reference, and TurboQuant tests: 16 passed, 1 CUDA Python test skipped on Windows because the WSL ELF binary is not a Windows executable.
- CUDA CMake build: successful in WSL with CUDA 13.3 and RTX 5080 compute capability 12.0.
- CTest: 26/26 tests passed in WSL.
- Full inherited Python suite was not green because historical `results/` artifacts and a Windows `build/` executable path were intentionally not migrated; the focused GLM suite remains green.
- No end-to-end GLM decode tok/s or quality result exists yet.
- Bounded GLM-5.2-shaped CUDA result: 8 experts/1 token median 2,662,772 ns; 8 experts/4 tokens 1,344,816 ns per candidate token; maximum absolute error 0.
- Resident expert-major batch result: 8 groups x 4 candidates, 1,641,591 ns/candidate token, cold weight H2D 160,432,128 bytes and warm weight H2D 0 bytes.
- Resident BF16 grid result: 8 experts x 4 candidates, 2,582,527 ns/block versus native 5,394,131 ns/block; BF16 resident weight bytes 603,979,776 versus native 160,432,128; maximum absolute error 0 on the zero-weight fixture.
- Last known-good code HEAD: `5d1c636` (`feat: add resident BF16 expert grid path`).
- Next bottleneck: nonzero GLM shard parity, exact DSA/indexer graph, variable-union expert grouping, and VRAM-pressure-aware choice between native MXFP4 and BF16 resident execution; full weights remain intentionally absent.
