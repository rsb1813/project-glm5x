# GLM5X Project State

## Current milestone

Bootstrap after the K3X to GLM-5.x migration.

## Completed

- Created the independent repository at `C:\Users\jolib\Documents\project-glm5x`.
- Migrated storage, converter, reference, runtime, test, and benchmark source directories without K3 official weight artifacts.
- Deleted the six verified official K3 artifact directories from the old K3X worktree. Synthetic fixtures remain.
- Added GLM-5.x descriptor validation and the `glm5x-convert` CLI wrapper.
- Added the GLM5X architecture/design and bootstrap plan documents.

## In progress

- Replace synthetic K3 source assumptions in the converter with a model-neutral GLM manifest.
- Build the tiny GLM-5.2-compatible reference graph and greedy parity tests.
- Rename user-facing runtime and benchmark commands where that does not break the inherited storage ABI.

## Known blockers

- No GLM-5.2 weights are present, so full checkpoint correctness and local TPS are not measured.
- The migrated C++ runtime still has K3-oriented names and graph assumptions in several files.
- CUDA build and native Linux throughput have not been validated in this new repository.

## Hardware assumptions

- CPU: AMD Ryzen 7 9800X3D.
- GPU: NVIDIA RTX 5080 16 GB.
- RAM: 96 GB DDR5-4200.
- Storage: Solidigm P44 Pro 2 TB NVMe.
- Preferred execution environment: native Linux.

## Latest verified state

- Focused GLM descriptor and CLI tests: 4 passed.
- Portable C++ runtime: configured and built in WSL with 61 targets.
- CTest: 14/14 tests passed in WSL.
- Full inherited Python suite was not green because historical `results/` artifacts and a Windows `build/` executable path were intentionally not migrated; the focused GLM suite remains green.
- No GLM throughput result exists yet.
- Next bottleneck: model-neutral GLM manifest and reference graph correctness.
