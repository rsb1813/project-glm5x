# GLM5X Shape and CUDA Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline execution is selected for this session). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the official GLM-5.2 tensor shape at a model-neutral boundary and measure the existing RTX 5080 MoE path at that shape without downloading weights.

**Architecture:** Extend the Python descriptor with DSA/indexer and MoE dimensions, then add a small manifest reader that validates those fields from `config.json` and `model.safetensors.index.json`. Add a CUDA-only bounded benchmark that owns zero-filled GLM-shaped tensors, invokes the existing resident MoE layer, and reports latency and memory as shape evidence rather than end-to-end TPS.

**Tech Stack:** Python 3.12, pytest, C++20, CUDA 13.3, CMake/Ninja, existing K3X runtime interfaces.

**Spec:** `docs/superpowers/specs/2026-08-14-glm5x-design.md`

## Global Constraints

- Do not download GLM-5.2 or GLM-5.3 weights during this milestone.
- Do not claim end-to-end TPS from a bounded one-layer benchmark.
- Preserve a strict reference path and reject incomplete or inconsistent metadata.
- Keep all generated fixtures outside version control.
- Do not provision paid cloud resources.

---

### Task 1: Complete the GLM-5.2 shape descriptor

**Files:**
- Modify: `reference/glm5x_ref/model.py`
- Modify: `reference/glm5x_ref/__init__.py`
- Test: `tests/python/test_glm5x_model_descriptor.py`

**Interfaces:**
- Produces `GLM5XModelDescriptor.from_config(config)` with DSA index and MoE intermediate dimensions.
- Existing descriptor fields remain source-compatible.

- [ ] **Step 1: Write failing assertions for the official shape fields.**
- [ ] **Step 2: Run the focused descriptor tests and observe the missing attributes.**
- [ ] **Step 3: Add validated dataclass fields and config aliases.**
- [ ] **Step 4: Run the focused descriptor tests and the existing GLM5X smoke suite.**
- [ ] **Step 5: Commit the descriptor change.**

### Task 2: Add a bounded tensor manifest reader

**Files:**
- Create: `reference/glm5x_ref/manifest.py`
- Modify: `reference/glm5x_ref/__init__.py`
- Test: `tests/python/test_glm5x_manifest.py`

**Interfaces:**
- `GLM5XTensorManifest.from_json(config, index)` returns a validated manifest with shard count, total source bytes, and model descriptor.
- It rejects missing weight-map entries, duplicate tensor names, and invalid shard sizes.

- [ ] **Step 1: Write failing manifest tests for valid and invalid metadata.**
- [ ] **Step 2: Run the tests and confirm the module is absent.**
- [ ] **Step 3: Implement the minimal immutable manifest and checksum-neutral index validation.**
- [ ] **Step 4: Run focused Python tests.**
- [ ] **Step 5: Commit the manifest change.**

### Task 3: Measure GLM-shaped resident MoE CUDA work

**Files:**
- Create: `runtime/src/cuda_glm5x_moe_bench.cpp`
- Modify: `CMakeLists.txt`
- Test: `tests/python/test_cuda_glm5x_moe.py`

**Interfaces:**
- `k3x_cuda_glm5x_moe_bench --experts 8 --warmup N --iterations N` emits one JSON record containing GLM shape, median layer latency, validation error, and peak device bytes.
- The executable uses deterministic zero/constant synthetic tensors and labels output `artifact_kind=glm5.2_shaped_synthetic_layer`.

- [ ] **Step 1: Add the CLI contract test before the executable exists.**
- [ ] **Step 2: Run the test to record the expected missing-binary failure.**
- [ ] **Step 3: Implement the bounded benchmark using existing `resident_mxfp4_moe_layer`.**
- [ ] **Step 4: Build with CUDA and run the benchmark on the RTX 5080.**
- [ ] **Step 5: Run the CUDA CTest suite and commit the benchmark.**

### Task 4: Record evidence and next bottleneck

**Files:**
- Modify: `ARCHITECTURE.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `DECISIONS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] **Step 1: Record only observed GLM-shaped latency and memory values.**
- [ ] **Step 2: State explicitly that no end-to-end TPS or quality result exists yet.**
- [ ] **Step 3: Record the optimization candidate selected from the measured bottleneck.**
- [ ] **Step 4: Run repository status and final focused verification.**
- [ ] **Step 5: Commit and push the logical milestone.**
