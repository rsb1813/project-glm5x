# GLM5X Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean GLM-5.2-first repository that reuses K3X infrastructure without retaining Kimi-specific weight artifacts.

**Architecture:** Keep the existing K3X storage/cache/runtime interfaces as a compatibility core and add a model descriptor boundary for GLM DSA/MLA and Top-8 MoE. Keep model data, calibration, and profiles outside the runtime ABI so GLM-5.3 can replace GLM-5.2 later.

**Tech Stack:** Python 3.12, PyTorch reference path, C++20 runtime, optional CUDA 13.3, CMake, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-glm5x-design.md`

## Global Constraints

- Do not download GLM-5.2 or GLM-5.3 weights during bootstrap unless explicitly requested.
- Do not claim TPS without a recorded benchmark.
- Preserve a strict reference mode for every optimization.
- Do not delete synthetic fixtures from the old K3X repository.
- Do not provision paid cloud resources.

---

### Task 1: Repository and artifact boundary

**Files:**
- Create: `checklist.md`, `context-notes.md`
- Create: `docs/superpowers/specs/2026-08-14-glm5x-design.md`
- Create: `docs/superpowers/plans/2026-08-14-glm5x-bootstrap.md`
- Modify: `PROJECT_STATE.md`, `DECISIONS.md`

- [ ] Record the exact K3 artifact paths and byte total before deletion.
- [ ] Delete only the verified official K3 artifact directories from the old worktree.
- [ ] Verify each target is absent and synthetic K3 fixtures remain.
- [ ] Commit the clean repository bootstrap.

### Task 2: Model-neutral metadata

**Files:**
- Create: `reference/glm5x_model.py`
- Create: `reference/glm5x_config.py`
- Create: `tests/python/test_glm5x_model_descriptor.py`
- Modify: `pyproject.toml`

- [ ] Define a descriptor containing model family, layer count, hidden size, routed expert count, top-k, shared expert count, attention kind, and MTP metadata.
- [ ] Load descriptor values from a JSON config without embedding tensor file names in the runtime.
- [ ] Add tests for GLM-5.2 descriptor validation and rejection of K3-only assumptions.

### Task 3: Converter and runtime branding

**Files:**
- Modify: `converter/k3x_converter/*`, `CMakeLists.txt`, `pyproject.toml`
- Create: `converter/glm5x_converter/manifest.py`
- Create: `tests/python/test_glm5x_manifest.py`

- [ ] Keep K3X format compatibility while adding model family and descriptor metadata to the manifest.
- [ ] Verify source and tensor checksums before sealing an extent.
- [ ] Expose a `glm5x-convert` entry point with dry-run support.

### Task 4: Reference smoke path

**Files:**
- Create: `reference/glm5x_reference.py`
- Create: `tests/python/test_glm5x_reference_smoke.py`
- Modify: `tools/generate_synthetic.py`

- [ ] Add a small GLM-compatible dense plus routed-expert fixture.
- [ ] Test greedy incremental generation and exact routing IDs.
- [ ] Keep MTP verification as a separate opt-in path.

### Task 5: Documentation and verification

**Files:**
- Create: `README.md`, `ARCHITECTURE.md`, `PROJECT_STATE.md`, `DECISIONS.md`, `BENCHMARKS.md`
- Modify: `.gitignore`, `.github/*`

- [ ] Document implemented, experimental, proposed, and rejected components.
- [ ] Run the focused Python suite and CMake configure/build.
- [ ] Record only observed results and the next bottleneck.
- [ ] Commit one logical bootstrap change.

