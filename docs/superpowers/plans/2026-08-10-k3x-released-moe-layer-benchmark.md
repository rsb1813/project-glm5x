# Released-Dimension Resident MoE-Layer Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and measure a bounded released-dimension split-versus-resident-MoE-layer CUDA benchmark without a full checkpoint or token-throughput claim.

**Architecture:** A CUDA-only binary loads the existing released single-expert storage fixture, generates deterministic released-size FP32 dense weights, and presents the expert bytes under 1, 4, or 16 unique logical IDs. A separate split CUDA backend supplies the numerical oracle; the selected split or complete-layer backend records cold admission separately from warm steady-state latency and traffic. A six-row B-0023 runner validates every pair before writing digest-backed evidence.

**Tech Stack:** C++20, CUDA 13.3 native `sm_120`, cuBLASLt, K3X Reader/storage slice, Python 3.12, pytest, CSV/JSON/SHA-256.

## Global Constraints

- Preserve `operation`, CPU drafting, natural routing, and speculation-none defaults.
- Use exact FP32 dense weights and native K3 MXFP4 E2M1 plus E8M0/32 expert bytes.
- Do not download the full Kimi K3 checkpoint or provision cloud resources.
- Label every result `routing_semantics=false`; do not emit token, prefill, decode, or TTFT fields.
- Fix shapes to hidden 7,168, latent 3,584, and expert intermediate 3,072.
- Fix CUDA identity to reused allocation, resident weights, resident-grid batching, synchronous transfer, fusion none, and 1 GiB hard capacity.
- Treat any hard-cap bypass, layer fallback, CUDA error, or numerical error above `1e-5` as failure.
- Keep raw and summary CSV line endings LF-stable.

---

### Task 1: Released benchmark CLI and deterministic host fixture

**Files:**
- Create: `runtime/src/cuda_moe_layer_bench.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/python/test_cuda_released_moe_layer.py`

**Interfaces:**
- Consumes: `load_storage_expert(Reader&, 1, 0)`, `DenseWeightView`, `DenseMlpView`, `ResidentMoeLayerView`, and `Mxfp4MlpView`.
- Produces: `k3x_cuda_moe_layer_bench --model --boundary --experts --warmup --iterations`.
- Produces internally: `Fixture make_fixture(const LoadedStorageExpert&, std::size_t)` and `Result<std::vector<float>> execute_split(ComputeBackend&, const Fixture&)`.

- [x] **Step 1: Write CLI and capability RED tests**

Create a CUDA-gated Python test that generates the bounded storage fixture and invokes both boundaries with one expert, zero warmups, and one iteration.

```python
@pytest.mark.parametrize("boundary", ["ffn-block", "moe-layer"])
def test_released_moe_layer_bench_executes(boundary: str, tmp_path: Path) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released MoE-layer benchmark requires build-cuda")
    artifact = _released_artifact(tmp_path)
    result = subprocess.run([
        str(cpp_binary("k3x_cuda_moe_layer_bench")),
        "--model", str(artifact), "--boundary", boundary,
        "--experts", "1", "--warmup", "0", "--iterations", "1",
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
```

Add non-CUDA-independent argument cases for unknown boundary, experts 0/2/17, missing model, and zero iterations. Require exit 2 and exact messages.

- [x] **Step 2: Run RED and witness missing binary**

Run:

```bash
K3X_BUILD_DIR=build-cuda K3X_TEST_CUDA=1 \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_released_moe_layer.py -q
```

Expected: live cases fail because `k3x_cuda_moe_layer_bench` does not exist.

- [x] **Step 3: Add the CUDA-only target and strict parser**

Add the target only inside `if(K3X_ENABLE_CUDA)` and link `k3x_runtime`.

```cmake
add_executable(k3x_cuda_moe_layer_bench runtime/src/cuda_moe_layer_bench.cpp)
target_link_libraries(k3x_cuda_moe_layer_bench PRIVATE k3x_runtime)
```

Use constants and checked parsing.

```cpp
constexpr std::size_t kHidden = 7168;
constexpr std::size_t kLatent = 3584;
constexpr std::size_t kIntermediate = 3072;
constexpr std::uint64_t kResidentCapacity = 1ULL << 30;

bool valid_expert_count(std::size_t value) {
    return value == 1 || value == 4 || value == 16;
}
```

- [x] **Step 4: Build deterministic released-size views**

Define one owning fixture. Dense values use a bounded repeating pattern that is finite and nonzero but avoids overflow.

```cpp
struct Fixture {
    std::vector<float> input;
    std::vector<float> routed_down;
    std::vector<float> routed_norm;
    std::vector<float> routed_up;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_down;
    std::vector<Mxfp4MlpView> experts;
    std::vector<float> contributions;
    ResidentMoeLayerView layer;
};
```

Fill dense matrices with `((index % 17) - 8) * 1.0e-4F`, norm with `1.0F`, input with `((index % 19) - 9) * 0.01F`, and contributions with `1.0F / experts`. Reuse the loaded packed/scales spans but assign tensor IDs `1000 + expert * 3 + projection`; dense IDs occupy 100 through 105.

- [x] **Step 5: Implement exact split execution**

Execute in this order using the same backend.

```cpp
latent = backend.dense_matvec(input, layer.routed_down, 1, decode);
shared = backend.dense_situ_mlp(input, layer.shared, 4.0F, 25.0F, 1, decode);
expert_outputs = backend.mxfp4_situ_mlp_grid(
    latent.value(), 1, experts, 4.0F, 25.0F, 1, decode);
mixed = ordered_mix(expert_outputs.value(), contributions);
normalized = strict_rms_norm(mixed, routed_norm, 1.0e-6F);
routed = backend.dense_matvec(normalized, layer.routed_up, 1, decode);
return add(routed.value(), shared.value());
```

The host mix loops expert-first then row-second. RMSNorm accumulates squares in `double` and uses the same epsilon order as the CPU oracle.

- [x] **Step 6: Run focused GREEN**

Build the target and run the argument plus one-expert live tests. Expected: both boundaries execute; numerical and telemetry assertions are added in Task 2.

- [x] **Step 7: Commit**

```bash
git add CMakeLists.txt runtime/src/cuda_moe_layer_bench.cpp \
  tests/python/test_cuda_released_moe_layer.py
git commit -m "bench: add released MoE layer driver"
```

---

### Task 2: Cold admission, warm telemetry, and numerical parity

**Files:**
- Modify: `runtime/src/cuda_moe_layer_bench.cpp`
- Modify: `tests/python/test_cuda_released_moe_layer.py`

**Interfaces:**
- Consumes: Task 1 fixture and split executor.
- Produces: one JSON object with released identity, cold residency, warm traffic, calls, fallbacks, error, and median latency.

- [x] **Step 1: Add complete output-contract RED assertions**

For every boundary at expert counts 1 and 16, assert released dimensions, `routing_semantics is False`, error at most `1e-5`, zero bypass/fallback, positive cold weight H2D, zero measured weight H2D, positive kernel time, and exact iteration counters.

```python
if boundary == "ffn-block":
    assert payload["stream_synchronization_count"] == 4
    assert payload["resident_moe_layer_calls"] == 0
else:
    assert payload["stream_synchronization_count"] == 1
    assert payload["resident_moe_layer_calls"] == 1
    assert payload["resident_moe_layer_kernel_launches"] == 13
```

- [x] **Step 2: Run RED and witness missing fields**

Expected: JSON key failures name `cold_weight_h2d_bytes` or `resident_moe_layer_calls`.

- [x] **Step 3: Separate oracle, cold, warmup, and measurement phases**

Construct an `ffn-block` oracle backend and execute split once. Construct a second backend for the requested boundary, snapshot before cold execution, execute once, validate output, and record cold deltas. Run warmups, snapshot again, then collect requested iterations.

Use `std::nth_element` or sorted samples for an integer median. Reject a layer result with `executed=false` as `BACKEND_UNAVAILABLE: released MoE layer capacity bypass`.

- [x] **Step 4: Emit the exact JSON contract**

Include:

```text
artifact_kind, routing_semantics, boundary, experts, hidden_width,
latent_width, expert_intermediate_width, expert_payload_bytes,
resident_capacity_bytes, warmup, iterations, maximum_absolute_error,
latency_nanoseconds_median, kernel_nanoseconds, activation_h2d_bytes,
device_to_host_bytes, weight_h2d_bytes, stream_synchronization_count,
cold_weight_h2d_bytes, resident_weight_bytes, peak_resident_weight_bytes,
peak_vram_bytes, weight_cache_bypasses, resident_grid_calls,
resident_grid_kernel_launches, resident_grid_fallbacks,
resident_moe_layer_calls, resident_moe_layer_experts,
resident_moe_layer_kernel_launches, resident_moe_layer_fallbacks,
resident_moe_layer_contribution_h2d_bytes
```

- [x] **Step 5: Run GREEN for 1 and 16 experts**

Run the focused test with CUDA enabled. Expected: all direct cases pass without a speed-direction assertion.

- [x] **Step 6: Run Compute Sanitizer at one expert**

Generate one bounded artifact, then run:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  build-cuda/k3x_cuda_moe_layer_bench \
  --model <artifact> --boundary moe-layer --experts 1 \
  --warmup 0 --iterations 1
```

Expected: `ERROR SUMMARY: 0 errors`.

- [x] **Step 7: Commit**

```bash
git add runtime/src/cuda_moe_layer_bench.cpp \
  tests/python/test_cuda_released_moe_layer.py
git commit -m "bench: validate released MoE layer traffic"
```

---

### Task 3: B-0023 paired runner

**Files:**
- Create: `tools/ablate_cuda_released_moe_layer.py`
- Create: `tests/python/test_cuda_released_moe_layer_ablation.py`

**Interfaces:**
- Consumes: released K3X artifact and Task 2 binary.
- Produces: canonical six rows, three named pairs, digest-backed JSON/CSV summary.

- [x] **Step 1: Write matrix and gate RED tests**

Require this order.

```python
CASES = (
    ("split-1", "ffn-block", 1), ("layer-1", "moe-layer", 1),
    ("split-4", "ffn-block", 4), ("layer-4", "moe-layer", 4),
    ("split-16", "ffn-block", 16), ("layer-16", "moe-layer", 16),
)
```

Monkeypatch `_run_case` with deterministic records. Prove the runner rejects numerical divergence, fallback, nonzero warm weight H2D, incorrect sync counts, missing 14,336-byte cold/resident norm delta, or non-decreasing activation/D2H.

- [x] **Step 2: Run RED and witness missing module**

Expected: import failure for `tools.ablate_cuda_released_moe_layer`.

- [x] **Step 3: Implement the runner and pair validation**

Invoke the binary with no shell, parse one JSON object, and include raw record SHA-256. Before summary output, validate exact identity and compute:

```python
layer["paired_latency_delta_percent"] = (
    layer["latency_nanoseconds_median"]
    / split["latency_nanoseconds_median"] - 1.0
) * 100.0
layer["paired_activation_h2d_reduction_bytes"] = (
    split["activation_h2d_bytes"] - layer["activation_h2d_bytes"]
)
layer["paired_d2h_reduction_bytes"] = (
    split["device_to_host_bytes"] - layer["device_to_host_bytes"]
)
```

Do not assert latency direction. Write `summary.csv` with `lineterminator="\n"` and compute raw, aggregate, runner, artifact, and summary CSV hashes.

- [x] **Step 4: Run unit and one-sample live GREEN**

Run CPU unit tests first, then CUDA live tests with warmup 0/sample 1. Expected: three pairs pass every physical gate.

- [x] **Step 5: Commit**

```bash
git add tools/ablate_cuda_released_moe_layer.py \
  tests/python/test_cuda_released_moe_layer_ablation.py
git commit -m "bench: add released MoE layer ablation"
```

---

### Task 4: RTX 5080 B-0023 evidence

**Files:**
- Create: `results/b0023-cuda-released-moe-layer-wsl/*.json`
- Create: `results/b0023-cuda-released-moe-layer-wsl/summary.json`
- Create: `results/b0023-cuda-released-moe-layer-wsl/summary.csv`
- Modify: `tests/python/test_cuda_released_moe_layer_ablation.py`

**Interfaces:**
- Produces: measured released-dimension boundary evidence, not token throughput.

- [x] **Step 1: Generate one bounded released artifact**

Use `write_bounded_expert_source` and the streaming converter. Record its SHA-256 and reuse the same artifact for all six rows.

- [x] **Step 2: Run the formal matrix**

```bash
/home/jolib/.venvs/k3x-m1/bin/python \
  tools/ablate_cuda_released_moe_layer.py \
  --artifact build-fixtures/released-expert.k3x \
  --runner build-cuda/k3x_cuda_moe_layer_bench \
  --output-dir results/b0023-cuda-released-moe-layer-wsl \
  --warmup 3 --iterations 20
```

- [x] **Step 3: Add committed-evidence verification**

Recompute every raw digest, canonical aggregate, summary CSV digest, pair gate, and reported percentage directly from committed bytes. Require exactly six raw JSON files, one summary JSON, and one summary CSV.

- [x] **Step 4: Run evidence tests and commit**

```bash
git add results/b0023-cuda-released-moe-layer-wsl \
  tests/python/test_cuda_released_moe_layer_ablation.py
git commit -m "bench: measure released MoE layers"
```

---

### Task 5: Verification, ledger, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: implementation and B-0023 evidence.
- Produces: synchronized public state and the evidence-based next boundary decision.

- [x] **Step 1: Run complete verification**

Run CPU, liburing/direct, ASan/UBSan, CUDA CTest and pytest matrices plus Compute Sanitizer for the new binary. Record actual counts.

- [x] **Step 2: Cross-check evidence and defaults**

Run `git diff --check`, focused B-0023 digest tests, all `CudaBoundaryMode` searches, and verify that no token/TPS field appears in B-0023.

- [x] **Step 3: Synchronize the TITAN Ledger**

Record measured latency, cold/warm traffic, synchronization, resident/VRAM footprint, error, and missing metrics. Accept or reject the next CUDA Graph/device-token boundary only from evidence. Update `PROJECT_STATE.md` last.

- [x] **Step 4: Review, publish, and merge**

Commit docs, push the branch, open a ready PR, wait for correctness and CodeQL, rebase-merge, verify post-merge `main`, and reconcile public metadata in a small follow-up PR.

PR #36 was rebase-merged at public integration head `e4820a18`. Branch/PR correctness, CodeQL, and post-merge `main` correctness/CodeQL all passed; publication metadata is reconciled in the follow-up branch.

---

## Plan self-review result

- Every accepted design requirement maps to one task and one evidence gate.
- The binary owns only bounded released-shape execution; the Python runner owns pair comparison and publication artifacts.
- Cold admission and warm measured traffic are distinct fields.
- No task changes routing, target verification, precision, cache eviction, speculative policy, or defaults.
- No task claims token throughput, full-checkpoint behavior, native-Linux authority, or forced latency improvement.
