# K3X Milestone Zero Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic synthetic Kimi K3 text decoder, stream it into K3X v1, and prove an independent C++20 CPU runtime matches the PyTorch oracle through greedy token generation.

**Architecture:** Python owns the executable reference graph, fixture generation, source-shard parsing, and K3X writer. C++20 owns an independent little-endian K3X reader and deterministic CPU graph. A subprocess integration test compares primitive outputs, recurrent/KV state, layer outputs, logits, and token IDs without sharing production math between the two implementations.

**Tech Stack:** Python 3.12, PyTorch 2.13.0 CPU, NumPy 2.5.1, safetensors 0.8.0, pytest 9.1.1, google-crc32c 1.8.0, psutil 7.2.2, C++20, CMake, CTest, and Clang or MSVC on Windows with GCC or Clang on Linux.

## Global Constraints

- Correctness outranks throughput and every optimized path must retain a switchable reference path.
- Do not download the full Kimi K3 checkpoint.
- Do not provision Cloud Storage, Cloud Run Jobs, GPU VMs, or any other paid resource.
- The executable milestone covers the text decoder only; MoonViT-V2 is documented as an integration boundary.
- The synthetic topology is four layers in `KDA, KDA, KDA, MLA` order with dense layer 0 and MoE layers 1-3.
- The synthetic dimensions are vocabulary 64, hidden 64, KDA 4×16, MLA 4 heads, eight routed experts, Top-2, one shared expert, latent 32, and AttnRes block size 2.
- Native MXFP4 data and E8M0 scales must round-trip byte-for-byte; no dequantize/requantize converter path is allowed.
- New source files start with a one-line Korean comment stating the file's role, directly below a required shebang or language directive.
- Use repository-local dependency environments. Installing Python packages or a C++ toolchain requires the user's separate package-install approval before execution.
- Run tests before each completion claim and create one semantic commit per independently reviewable task.
- Update `checklist.md` and append decisions or discovered deviations to `context-notes.md` in the same commit that introduces them.

---

## File Map

### Project configuration

- `.gitignore` excludes `.venv/`, Python caches, CMake build trees, generated fixtures, partial artifacts, and benchmark scratch data.
- `pyproject.toml` pins the build and development dependencies, discovers packages under `reference/` and `converter/`, and gains the `k3x-convert` entry point when its callable exists.
- `CMakeLists.txt` builds `k3x_runtime`, `k3x_run`, unit tests, and CTest registrations without fetching dependencies.

### Python reference and converter

- `reference/k3x_ref/config.py` owns `SyntheticK3Config` and all shape validation.
- `reference/k3x_ref/mxfp4.py` owns E2M1 nibble decode, E8M0 block scales, packing validation, and reference matmul.
- `reference/k3x_ref/ops.py` owns RMSNorm, SiTU-GLU, and deterministic top-k helpers.
- `reference/k3x_ref/kda.py` owns ShortConv state and KDA recurrence.
- `reference/k3x_ref/mla.py` owns Gated MLA and its incremental cache.
- `reference/k3x_ref/moe.py` owns routing, routed experts, latent projections, and the shared branch.
- `reference/k3x_ref/attn_res.py` owns depth-wise residual mixing and block state.
- `reference/k3x_ref/model.py` composes decoder layers and exposes full and incremental generation.
- `reference/k3x_ref/fixtures.py` creates deterministic source shards and golden `.npz` files.
- `converter/k3x_converter/format.py` owns K3X constants and record dataclasses.
- `converter/k3x_converter/safetensors_reader.py` parses headers and yields bounded raw tensor chunks.
- `converter/k3x_converter/writer.py` writes aligned extents, directories, checksums, and finalized superblocks.
- `converter/k3x_converter/resume.py` owns the atomic JSON resume ledger.
- `converter/k3x_converter/reader.py` validates K3X artifacts for Python tests.
- `converter/k3x_converter/cli.py` exposes conversion, validation, and dry-run commands.

### C++ runtime

- `runtime/include/k3x/status.hpp` defines stable error codes and `Result<T>`.
- `runtime/include/k3x/format.hpp` defines decoded host-side records, never packed struct casts.
- `runtime/include/k3x/reader.hpp` defines positional reads and artifact validation.
- `runtime/include/k3x/tensor.hpp` defines owned FP32 tensors and byte views.
- `runtime/include/k3x/ops.hpp` declares deterministic primitives.
- `runtime/include/k3x/state.hpp` defines KDA convolution/recurrent state, MLA KV state, and AttnRes state.
- `runtime/include/k3x/model.hpp` declares graph loading, prefill, decode, and greedy generation.
- `runtime/src/crc32c.cpp` and `runtime/src/sha256.cpp` provide dependency-free checksum implementations.
- `runtime/src/reader.cpp`, `ops.cpp`, `kda.cpp`, `mla.cpp`, `moe.cpp`, and `model.cpp` implement the runtime.
- `runtime/src/main.cpp` emits one bounded JSON result for integration tests and benchmarks.

### Tests and tools

- `tests/python/test_config.py`, `test_ops.py`, `test_kda.py`, `test_mla.py`, `test_moe.py`, `test_attn_res.py`, and `test_model.py` gate the oracle.
- `tests/python/test_k3x_format.py`, `test_converter_resume.py`, and `test_cpp_parity.py` gate storage and cross-language behavior.
- `tests/cpp/test_checksums.cpp`, `test_reader.cpp`, and `test_ops.cpp` provide dependency-free CTest executables.
- `tools/generate_synthetic.py` creates source shards and independent golden fixtures.
- `tools/benchmark_synthetic.py` records latency, read bytes, and peak RSS as JSON and CSV.

---

### Task 1: Reproducible project skeleton and shape contract

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `CMakeLists.txt`
- Create: `reference/k3x_ref/__init__.py`
- Create: `reference/k3x_ref/config.py`
- Create: `converter/k3x_converter/__init__.py`
- Create: `tests/python/test_config.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `SyntheticK3Config.default() -> SyntheticK3Config`.
- Produces: `SyntheticK3Config.validate() -> None`, raising `ValueError` with a stable field name.
- Establishes package discovery with `setuptools==83.0.0`; no console entry point is registered until its callable exists.

- [ ] **Step 1: Record the toolchain gate without installing packages**

Run:

```powershell
rtk python --version
rtk where.exe cmake
rtk where.exe clang++
rtk where.exe cl
```

Expected on the current machine: system Python 3.14.6 is present while PyTorch, CMake, and a C++ compiler are absent. Ask for package-install approval before creating `.venv` or installing the missing toolchain.

- [ ] **Step 2: Add the failing config contract tests**

Create `tests/python/test_config.py` with literal expectations.

```python
# 합성 K3 설정의 실제 위상과 packing 제약을 검증하는 테스트
import pytest

from k3x_ref.config import SyntheticK3Config


def test_default_config_reproduces_minimal_k3_topology() -> None:
    cfg = SyntheticK3Config.default()
    assert cfg.layer_kinds == ("kda", "kda", "kda", "mla")
    assert cfg.dense_layers == (0,)
    assert cfg.hidden_size == 64
    assert cfg.num_experts == 8
    assert cfg.top_k == 2
    assert cfg.attn_res_block_size == 2


def test_config_rejects_mxfp4_incompatible_expert_width() -> None:
    cfg = SyntheticK3Config.default().replace(expert_intermediate_size=31)
    with pytest.raises(ValueError, match="expert_intermediate_size"):
        cfg.validate()
```

- [ ] **Step 3: Run the test and observe the intended RED failure**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_config.py -q
```

Expected: collection fails because `k3x_ref.config` does not exist. This is the intended missing-production-module failure.

- [ ] **Step 4: Add the minimal package and config implementation**

Use a frozen dataclass with the exact defaults below.

```python
# 합성 Kimi K3 그래프의 크기와 구조 계약을 정의하는 설정
from dataclasses import dataclass, replace as dataclass_replace


@dataclass(frozen=True)
class SyntheticK3Config:
    vocab_size: int = 64
    hidden_size: int = 64
    layer_kinds: tuple[str, ...] = ("kda", "kda", "kda", "mla")
    dense_layers: tuple[int, ...] = (0,)
    kda_heads: int = 4
    kda_head_dim: int = 16
    short_conv_kernel_size: int = 4
    mla_heads: int = 4
    q_lora_rank: int = 32
    kv_lora_rank: int = 32
    qk_nope_head_dim: int = 8
    qk_rope_head_dim: int = 8
    v_head_dim: int = 8
    mla_use_nope: bool = True
    mla_use_output_gate: bool = True
    num_experts: int = 8
    top_k: int = 2
    num_shared_experts: int = 1
    routed_latent_size: int = 32
    expert_intermediate_size: int = 32
    dense_intermediate_size: int = 96
    attn_res_block_size: int = 2
    rms_norm_eps: float = 1.0e-5
    kda_gate_lower_bound: float = -5.0
    activation_situ_beta: float = 4.0
    activation_situ_linear_beta: float = 25.0
    routed_scaling_factor: float = 1.0
    mxfp4_group_size: int = 32

    @classmethod
    def default(cls) -> "SyntheticK3Config":
        cfg = cls()
        cfg.validate()
        return cfg

    def replace(self, **changes: object) -> "SyntheticK3Config":
        return dataclass_replace(self, **changes)

    def validate(self) -> None:
        if self.layer_kinds != ("kda", "kda", "kda", "mla"):
            raise ValueError("layer_kinds must be KDA,KDA,KDA,MLA")
        if self.kda_heads * self.kda_head_dim != self.hidden_size:
            raise ValueError("kda_heads * kda_head_dim must equal hidden_size")
        if self.expert_intermediate_size % self.mxfp4_group_size:
            raise ValueError("expert_intermediate_size must align to MXFP4 group size")
        if not self.mla_use_nope:
            raise ValueError("mla_use_nope must be true for Kimi K3")
        if not 0 < self.top_k <= self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
```

Set `requires-python = ">=3.12,<3.14"`, pin `setuptools==83.0.0` in `build-system.requires`, and pin the package versions listed in the plan header. Configure package discovery with `where = ["reference", "converter"]` and pytest with `pythonpath = ["reference", "converter"]`.

- [ ] **Step 5: Run the config test and inspect the project metadata**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_config.py -q
rtk .\.venv\Scripts\python.exe -m pip check
rtk git diff --check
```

Expected: two tests pass, dependency check passes, and `git diff --check` emits nothing.

- [ ] **Step 6: Update tracking files and commit**

Mark the repository-tooling checklist item complete, append the resolved Python and compiler versions to `context-notes.md`, then run:

```powershell
rtk git add .gitignore pyproject.toml CMakeLists.txt reference/k3x_ref converter/k3x_converter tests/python/test_config.py checklist.md context-notes.md
rtk git commit -m "build: establish K3X milestone toolchain"
```

---

### Task 2: Independent numerical primitives and native MXFP4 oracle

**Files:**
- Create: `reference/k3x_ref/ops.py`
- Create: `reference/k3x_ref/mxfp4.py`
- Create: `tests/python/test_ops.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `rms_norm(x, weight, eps) -> torch.Tensor`.
- Produces: `situ_glu(gate, up, beta, linear_beta) -> torch.Tensor`.
- Produces: `decode_mxfp4(packed, scales, rows, cols, group_size=32) -> torch.Tensor`.
- Produces: `mxfp4_matmul(x, packed, scales, rows, cols) -> torch.Tensor`.

- [ ] **Step 1: Write literal primitive and MXFP4 tests**

The MXFP4 test must use a hand-authored nibble sequence and E8M0 bytes, not values produced by the decoder under test.

```python
# K3 reference 기본 연산과 MXFP4 해석을 독립 기대값으로 검증하는 테스트
import torch

from k3x_ref.mxfp4 import decode_mxfp4
from k3x_ref.ops import rms_norm, situ_glu


def test_rms_norm_uses_mean_square_without_centering() -> None:
    got = rms_norm(torch.tensor([[3.0, 4.0]]), torch.tensor([2.0, 0.5]), 0.0)
    assert torch.allclose(got, torch.tensor([[1.6970563, 0.5656854]]), atol=1e-6)


def test_situ_glu_multiplies_transformed_gate_by_up_branch() -> None:
    got = situ_glu(torch.tensor([[-1.0, 2.0]]), torch.tensor([[3.0, 4.0]]), 1.0, 1.0)
    expected = torch.tensor([[-0.20381131, 0.84854317]])
    assert torch.allclose(got, expected, atol=1e-6)


def test_mxfp4_decodes_e2m1_nibbles_with_one_e8m0_scale() -> None:
    packed = bytes([0x10, 0x32] + [0x00] * 14)
    scales = bytes([127])
    got = decode_mxfp4(packed, scales, rows=1, cols=32)
    assert torch.equal(got[0, :4], torch.tensor([0.0, 0.5, 1.0, 1.5]))
```

Before implementation, confirm the four literal E2M1 values against the released compressed-tensors mapping. If the upstream mapping differs, correct the literals and record the cited mapping in `context-notes.md`; do not change expectations to mirror implementation output.

- [ ] **Step 2: Run and observe RED**

Run `rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_ops.py -q`.

Expected: import failure for the missing `ops` and `mxfp4` modules.

- [ ] **Step 3: Implement the minimal explicit FP32 functions**

Use FP32 accumulation, reject E8M0 `0xFF`, require exactly one scale per 32 logical values, decode low nibble before high nibble, and return `[rows, cols]`. Keep MXFP4 decode and matmul separate so byte interpretation can be tested without GEMM.

- [ ] **Step 4: Run primitive tests and mutation checks**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_ops.py -q
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_config.py tests/python/test_ops.py -q
```

Temporarily swap low/high nibble order and confirm the MXFP4 literal test fails, then restore the implementation and rerun green.

- [ ] **Step 5: Commit the primitive contract**

```powershell
rtk git add reference/k3x_ref/ops.py reference/k3x_ref/mxfp4.py tests/python/test_ops.py checklist.md context-notes.md
rtk git commit -m "feat: add K3 numerical reference primitives"
```

---

### Task 3: Stateful KDA and Gated MLA reference paths

**Files:**
- Create: `reference/k3x_ref/kda.py`
- Create: `reference/k3x_ref/mla.py`
- Create: `tests/python/test_kda.py`
- Create: `tests/python/test_mla.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `ShortConvState(history: torch.Tensor)` and `KDAState(conv_q, conv_k, conv_v, recurrent)`.
- Produces: `kda_prefill(x, weights, state, cfg) -> tuple[torch.Tensor, KDAState]`.
- Produces: `kda_decode(x_one, weights, state, cfg) -> tuple[torch.Tensor, KDAState]`.
- Produces: `MLAState(keys, values, shared_keys, length)`.
- Produces: `mla_prefill(x, weights, positions, state, cfg) -> tuple[torch.Tensor, MLAState]`.
- Produces: `mla_decode(x_one, weights, position, state, cfg) -> tuple[torch.Tensor, MLAState]`.

- [ ] **Step 1: Add failing state equivalence tests**

Use small literal inputs for a one-head recurrence test and seeded weight fixtures for the complete modules.

```python
# KDA의 감쇠, delta 갱신, 출력 순서와 incremental state를 검증하는 테스트
import torch

from k3x_ref.kda import kda_step


def test_kda_step_reads_output_from_updated_state() -> None:
    state = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    q = torch.tensor([[1.0, 0.0]])
    k = torch.tensor([[1.0, 0.0]])
    v = torch.tensor([[3.0, 2.0]])
    alpha = torch.ones_like(k)
    out, updated = kda_step(state, q, k, v, alpha, torch.tensor([0.5]))
    assert torch.equal(updated, torch.tensor([[[2.0, 1.0], [0.0, 1.0]]]))
    assert torch.equal(out, torch.tensor([[2.0, 1.0]]))
```

Add parameterized sequence lengths 1, 2, and 5 asserting `prefill_output == concat(decode_output)` and exact equality of final convolution and recurrent states. Add the equivalent MLA test, comparing expanded cached K/V and shared extra-key slots as well as output.

- [ ] **Step 2: Run and observe RED for missing modules**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_kda.py tests/python/test_mla.py -q
```

- [ ] **Step 3: Implement KDA in the official operation order**

Project Q/K/V, apply causal depthwise convolution and SiLU, normalize Q/K per head, compute bounded forget gates and sigmoid beta, then perform decay, delta write, updated-state read, gated RMSNorm, and output projection. State tensors must be explicit function arguments and return values.

- [ ] **Step 4: Implement Gated MLA with explicit cache contents**

Use q-LoRA projection and norm, split the main and extra query subspaces, emit KV latent plus a shared extra key, normalize only the latent, and apply no positional rotation because K3 sets `mla_use_nope=true`. Expand the main K and V, apply causal softmax scaled by the complete query width, apply the sigmoid output gate before output projection, and append exact cache tensors.

- [ ] **Step 5: Run focused and cumulative tests**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_kda.py tests/python/test_mla.py -q
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_config.py tests/python/test_ops.py tests/python/test_kda.py tests/python/test_mla.py -q
```

Expected: all tests pass with no warnings. Mutating the KDA output to read the pre-update state must fail the literal recurrence test.

- [ ] **Step 6: Commit stateful attention**

```powershell
rtk git add reference/k3x_ref/kda.py reference/k3x_ref/mla.py tests/python/test_kda.py tests/python/test_mla.py checklist.md context-notes.md
rtk git commit -m "feat: add exact KDA and Gated MLA reference paths"
```

---

### Task 4: Router, Stable LatentMoE, Attention Residual, and greedy graph

**Files:**
- Create: `reference/k3x_ref/moe.py`
- Create: `reference/k3x_ref/attn_res.py`
- Create: `reference/k3x_ref/model.py`
- Create: `reference/k3x_ref/fixtures.py`
- Create: `tests/python/test_moe.py`
- Create: `tests/python/test_attn_res.py`
- Create: `tests/python/test_model.py`
- Create: `tests/python/conftest.py`
- Create: `tools/generate_synthetic.py`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `route(hidden, weight, correction_bias, top_k, routed_scale) -> RouterOutput`.
- Produces: `stable_latent_moe(hidden, weights, router_output, cfg) -> torch.Tensor`.
- Produces: `apply_attn_res(prefix_sum, block_sources, norm_weight, proj_weight, eps) -> torch.Tensor`.
- Produces: `SyntheticK3Model.prefill(token_ids) -> tuple[torch.Tensor, ModelState]`.
- Produces: `SyntheticK3Model.decode(token_id, state) -> tuple[torch.Tensor, ModelState]`.
- Produces: `SyntheticK3Model.generate_greedy(prompt_ids, count, incremental) -> list[int]`.
- Produces: `write_source_checkpoint(path, seed=20260808) -> SourceManifest` and `write_golden(path, model, prompt_ids) -> None`.
- Produces test fixtures `controlled_generation_model`, `seeded_model`, and `synthetic_source` from `tests/python/conftest.py`.

- [ ] **Step 1: Write router and AttnRes literal tests**

```python
# K3 router의 선택 점수와 실제 혼합 weight가 다름을 검증하는 테스트
import torch

from k3x_ref.moe import route


def test_router_bias_changes_selection_but_not_selected_weights() -> None:
    hidden = torch.tensor([[1.0, 0.0]])
    weight = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bias = torch.tensor([0.0, 0.0, 0.9])
    got = route(hidden, weight, bias, top_k=2, routed_scale=1.0)
    assert got.expert_ids.tolist() == [[2, 0]]
    expected = torch.tensor([[0.5, 0.8807971]])
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(got.weights, expected, atol=1e-6)
```

Add an AttnRes literal with two raw sources whose normalized keys give known softmax weights. Assert the values mixed are raw sources, not normalized keys.

- [ ] **Step 2: Write the failing full/incremental token test**

Use a controlled fixture whose transformer projections are zero, whose token embeddings are positive, and whose LM head makes token 5 the unique argmax. With prompt IDs `[1, 7, 3, 9]`, both full and incremental generation must equal the independent literal `[5, 5, 5, 5, 5, 5]`. Separately use weights generated with seed `20260808` to assert full versus incremental layer outputs, logits, and final states; the component literal tests remain the oracle for operation semantics.

- [ ] **Step 3: Run and observe RED**

Run `rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_moe.py tests/python/test_attn_res.py tests/python/test_model.py -q`.

Expected: imports fail because the three production modules are missing.

- [ ] **Step 4: Implement MoE and AttnRes minimally**

Keep router logits and sigmoid in FP32. Sort ties by lower expert ID. Apply correction bias only to selection. Down-project once, execute only selected routed experts with decoded MXFP4 weights, sum by normalized unbiased router weights, latent RMSNorm, up-project, and add the shared dense SiTU-GLU branch.

For AttnRes, concatenate prior block sources with the current prefix sum, RMS-normalize only for score keys, compute one learned score per source, softmax over depth, and mix raw sources. Append a block source at layer indices divisible by 2 and apply one final output mix.

- [ ] **Step 5: Compose the decoder and freeze independent golden data**

`ModelState` contains a tuple of per-layer KDA or MLA states and the current absolute position. AttnRes state is local to each forward call and is not persisted across tokens. `generate_greedy` runs the prompt once in incremental mode, then feeds each selected token with the returned state. The full path reruns the entire growing sequence.

Write source shards as two safetensors files plus `source-manifest.json`. Store packed expert bytes and E8M0 scales as separate `uint8` tensors. Golden output is a compressed NumPy archive containing named primitive outputs, every layer output, state tensors, final logits, and token IDs.

- [ ] **Step 6: Run the reference suite twice for determinism**

```powershell
rtk .\.venv\Scripts\python.exe tools/generate_synthetic.py --output build-fixtures/run-a
rtk .\.venv\Scripts\python.exe tools/generate_synthetic.py --output build-fixtures/run-b
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_moe.py tests/python/test_attn_res.py tests/python/test_model.py -q
rtk git diff --no-index build-fixtures/run-a/manifest.sha256 build-fixtures/run-b/manifest.sha256
```

Expected: tests pass and the two digest manifests are identical.

- [ ] **Step 7: Commit the complete PyTorch oracle**

```powershell
rtk git add reference/k3x_ref/moe.py reference/k3x_ref/attn_res.py reference/k3x_ref/model.py reference/k3x_ref/fixtures.py tests/python/conftest.py tests/python/test_moe.py tests/python/test_attn_res.py tests/python/test_model.py tools/generate_synthetic.py checklist.md context-notes.md
rtk git commit -m "feat: complete the synthetic K3 reference graph"
```

---

### Task 5: K3X v1 records, streaming conversion, and crash-safe resume

**Files:**
- Create: `K3X_FORMAT.md`
- Create: `converter/k3x_converter/format.py`
- Create: `converter/k3x_converter/safetensors_reader.py`
- Create: `converter/k3x_converter/resume.py`
- Create: `converter/k3x_converter/writer.py`
- Create: `converter/k3x_converter/reader.py`
- Create: `converter/k3x_converter/cli.py`
- Create: `tests/python/test_k3x_format.py`
- Create: `tests/python/test_converter_resume.py`
- Modify: `pyproject.toml`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `TensorRecord`, `LayerRecord`, `ExpertRecord`, `ModelConfigRecord`, and `Superblock` dataclasses with `encode()` and `decode()`.
- Produces: `iter_tensor_chunks(path, tensor_name, chunk_bytes) -> Iterator[bytes]`.
- Produces: `convert(source_dir, output_path, chunk_bytes, stop_after_extents=None) -> ConversionReport`.
- Produces: `K3XReader.open(path, verify_root=True) -> K3XReader`.
- Produces CLI commands `k3x-convert convert`, `validate`, and `dry-run`.

- [ ] **Step 1: Freeze the binary layout in failing encode/decode tests**

The first 232 superblock bytes have these exact offsets; bytes 232-4091 are zero-reserved and bytes 4092-4095 hold CRC32C over bytes 0-4091.

```text
0    char[8]  magic = K3XCHKPT
8    u16      major = 1
10   u16      minor = 0
12   u32      superblock_bytes = 4096
16   u32      alignment = 4096
20   u32      state = 0 partial, 1 finalized
24   u64      required_features
32   u64      optional_features
40   u8[16]   file_uuid
56   u8[32]   source_sha256
88   u64      tensor_directory_offset
96   u64      tensor_directory_length
104  u64      layer_directory_offset
112  u64      layer_directory_length
120  u64      expert_directory_offset
128  u64      expert_directory_length
136  u64      model_config_offset
144  u64      model_config_length
152  u64      payload_offset
160  u64      file_length
168  u8[32]   directory_sha256
200  u8[32]   file_sha256 with this and the CRC field zeroed during hashing
4092 u32      superblock_crc32c with this field excluded
```

Directory headers are 16 bytes: four-byte tag, `u32 record_size`, and `u64 count`. Freeze these record layouts in `K3X_FORMAT.md` before implementing encoding.

```text
Tensor record, 128 bytes
0   u64      tensor_id
8   u32      role
12  u16      dtype
14  u16      quantization
16  u8       rank
17  u8       flags
18  u16      reserved0
20  i32      layer_id, -1 for global
24  i32      expert_id, -1 for non-expert
28  u32      reserved1
32  u64[4]   dimensions, unused entries zero
64  u64      data_offset
72  u64      data_length
80  u64      logical_length
88  u64      auxiliary_offset
96  u64      auxiliary_length
104 u32      data_crc32c
108 u32      auxiliary_crc32c
112 u8[16]   reserved2

Layer record, 64 bytes
0   u32      layer_index
4   u16      attention_kind
6   u16      ffn_kind
8   u32      first_tensor_index
12  u32      tensor_count
16  u32      first_expert_index
20  u32      expert_count
24  i32      attention_residual_write_index, -1 when absent
28  u32      flags
32  u8[32]   reserved

Expert record, 64 bytes
0   u32      layer_index
4   u32      expert_id
8   u32      physical_order
12  u32      flags
16  u64      gate_tensor_id
24  u64      up_tensor_id
32  u64      down_tensor_id
40  u64      profile_frequency_q32
48  u8[16]   reserved

Model config record, 256 bytes
0   u32      vocabulary_size
4   u32      hidden_size
8   u32      layer_count
12  u32      kda_head_count
16  u32      kda_head_dimension
20  u32      short_conv_kernel_size
24  u32      mla_head_count
28  u32      q_lora_rank
32  u32      kv_lora_rank
36  u32      qk_nope_head_dimension
40  u32      qk_rope_head_dimension
44  u32      value_head_dimension
48  u32      expert_count
52  u32      top_k
56  u32      shared_expert_count
60  u32      routed_latent_size
64  u32      expert_intermediate_size
68  u32      dense_intermediate_size
72  u32      attention_residual_block_size
76  u32      mxfp4_group_size
80  f32      rms_norm_epsilon
84  f32      kda_gate_lower_bound
88  f32      routed_scaling_factor
92  f32      activation_situ_beta
96  f32      activation_situ_linear_beta
100 f32      absolute_tolerance
104 f32      relative_tolerance
108 u32      mla_flags, bit 0 use_nope and bit 1 output_gate
112 u8[144]  reserved
```

All enum numeric values, feature bits, and the canonical directory digest range are listed beside these layouts. Root SHA-256 is computed over the finalized file length with bytes 200-231 and 4092-4095 treated as zero, then written at 200; superblock CRC32C is computed last over bytes 0-4091.

Tests must assert literal offsets, record lengths, zero reserved bytes, rejection of unknown required feature bit 0, extent overlap, unaligned extent, truncated file, changed payload CRC, and changed root digest.

- [ ] **Step 2: Run format tests and observe RED**

Run `rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_k3x_format.py -q`.

Expected: missing `k3x_converter.format` import.

- [ ] **Step 3: Implement records and validating Python reader**

Use `struct.Struct` with explicit `<` little-endian formats and field-by-field decode. Never use native alignment. Validate all ranges with checked addition before reading. Verify per-extent CRC32C first, directory SHA-256 second, and file SHA-256 last.

- [ ] **Step 4: Write resume tests with an intentional interruption**

```python
# K3X 변환 중단 뒤 검증된 extent만 재사용하는지 확인하는 테스트
def test_conversion_resumes_without_rewriting_completed_extents(tmp_path):
    source = make_source_checkpoint(tmp_path / "source")
    output = tmp_path / "synthetic.k3x"
    first = convert(source, output, chunk_bytes=257, stop_after_extents=3)
    assert first.completed is False
    before = read_resume_manifest(output.with_suffix(".k3x.resume.json"))

    second = convert(source, output, chunk_bytes=257)
    assert second.completed is True
    assert second.reused_extent_ids == tuple(item.extent_id for item in before.completed)
    assert output.exists()
    assert not output.with_suffix(".k3x.partial").exists()
```

Add a second test that changes one source tensor byte and asserts resume refuses reuse with `SOURCE_FINGERPRINT_MISMATCH`.

- [ ] **Step 5: Implement bounded safetensors reads and writer ordering**

Parse the eight-byte header length and JSON header, validate non-overlapping data offsets, and issue positional reads no larger than `chunk_bytes`. Pack payloads in decoder execution order. Within an MoE layer, write router and latent/shared trunk tensors first, then expert IDs in physical order with packed weights adjacent to their scales.

After each extent, flush, read back, verify CRC32C, and atomically replace the resume JSON. Finalization writes directories, computes the directory digest, writes finalized metadata, computes the root digest with both digest and CRC fields zeroed, writes that digest, computes and writes superblock CRC32C, flushes, closes, and atomically renames `.partial` to the requested output.

Add `[project.scripts] k3x-convert = "k3x_converter.cli:main"` only after `cli.py` exists, then rerun `python -m pip install -e .` before invoking the command.

- [ ] **Step 6: Run storage and converter tests**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_k3x_format.py tests/python/test_converter_resume.py -q
rtk .\.venv\Scripts\k3x-convert.exe dry-run build-fixtures/run-a/source --output build-fixtures/synthetic.k3x
rtk .\.venv\Scripts\k3x-convert.exe convert build-fixtures/run-a/source --output build-fixtures/synthetic.k3x --chunk-bytes 257
rtk .\.venv\Scripts\k3x-convert.exe validate build-fixtures/synthetic.k3x
```

Expected: all tests pass, dry-run writes no artifact, conversion reports bounded maximum read size 257, and validation reports a finalized artifact with matching digests.

- [ ] **Step 7: Commit the K3X streaming round-trip**

```powershell
rtk git add K3X_FORMAT.md pyproject.toml converter/k3x_converter tests/python/test_k3x_format.py tests/python/test_converter_resume.py checklist.md context-notes.md
rtk git commit -m "feat: add crash-safe K3X streaming conversion"
```

---

### Task 6: Dependency-free C++20 K3X reader and primitive parity

**Files:**
- Create: `runtime/include/k3x/status.hpp`
- Create: `runtime/include/k3x/format.hpp`
- Create: `runtime/include/k3x/reader.hpp`
- Create: `runtime/include/k3x/tensor.hpp`
- Create: `runtime/include/k3x/ops.hpp`
- Create: `runtime/src/crc32c.cpp`
- Create: `runtime/src/sha256.cpp`
- Create: `runtime/src/reader.cpp`
- Create: `runtime/src/ops.cpp`
- Create: `tests/cpp/test_checksums.cpp`
- Create: `tests/cpp/test_reader.cpp`
- Create: `tests/cpp/test_ops.cpp`
- Modify: `CMakeLists.txt`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `enum class ErrorCode` values matching Python error names.
- Produces: `Result<Reader> Reader::open(const std::filesystem::path&, VerifyMode)`.
- Produces: `Result<std::vector<std::byte>> Reader::read_tensor(std::uint64_t tensor_id)`.
- Produces: `void rms_norm(std::span<float> out, std::span<const float> input, std::span<const float> weight, float epsilon)`.
- Produces: `void situ_glu(std::span<float> out, std::span<const float> gate, std::span<const float> up, float beta, std::optional<float> linear_beta)`.
- Produces: `Result<void> decode_mxfp4(std::span<float> out, std::span<const std::byte> packed, std::span<const std::byte> scales, std::size_t rows, std::size_t cols, std::size_t group_size)`.
- Produces: `Result<void> mxfp4_matmul(std::span<float> out, std::span<const float> input, std::span<const std::byte> packed, std::span<const std::byte> scales, std::size_t rows, std::size_t cols, std::size_t group_size)`.

- [ ] **Step 1: Write failing checksum tests from standard vectors**

Use ASCII `123456789` with CRC32C `0xe3069283` and SHA-256 `15e2b0d3c33891ebb0f1ef609ec419420c20e320ce94c65fbc8c3312448eb225`. Register the executable with CTest.

- [ ] **Step 2: Configure and observe the intended build failure**

```powershell
rtk cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
rtk cmake --build build --target test_checksums
```

Expected: compile or link failure because checksum implementations are missing.

- [ ] **Step 3: Implement checksums and run the focused CTest**

Use the Castagnoli polynomial `0x82f63b78` in a portable table implementation and a portable SHA-256 compression function. Do not add CPU intrinsics until a later measured optimization stage.

Run `rtk ctest --test-dir build -R checksums --output-on-failure` and expect one passing test.

- [ ] **Step 4: Add reader corruption and primitive parity tests**

The Python fixture command creates valid, truncated, bad-CRC, and bad-required-feature artifacts under `build-fixtures/cpp-reader/`. C++ tests open each path and assert the exact `ErrorCode`. Primitive tests read literal input/output tensors from the valid K3X artifact and compare FP32 with the per-operation tolerances stored in model config.

- [ ] **Step 5: Implement the reader without struct aliasing**

Decode little-endian integers from byte arrays, perform checked `offset + length`, validate alignment and extent non-overlap before allocating, and use positional reads. Instrument each read through `ReadCounters { calls, requested_bytes, completed_bytes }` owned by `Reader`.

- [ ] **Step 6: Run C++ and Python regression suites**

```powershell
rtk cmake --build build
rtk ctest --test-dir build --output-on-failure
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_k3x_format.py tests/python/test_converter_resume.py -q
```

- [ ] **Step 7: Commit the independent storage reader**

```powershell
rtk git add CMakeLists.txt runtime/include/k3x runtime/src/crc32c.cpp runtime/src/sha256.cpp runtime/src/reader.cpp runtime/src/ops.cpp tests/cpp checklist.md context-notes.md
rtk git commit -m "feat: add independent C++ K3X reader and primitives"
```

---

### Task 7: C++ stateful graph and end-to-end token parity

**Files:**
- Create: `runtime/include/k3x/state.hpp`
- Create: `runtime/include/k3x/model.hpp`
- Create: `runtime/src/kda.cpp`
- Create: `runtime/src/mla.cpp`
- Create: `runtime/src/moe.cpp`
- Create: `runtime/src/model.cpp`
- Create: `runtime/src/main.cpp`
- Create: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/conftest.py`
- Modify: `runtime/include/k3x/ops.hpp`
- Modify: `CMakeLists.txt`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `Result<Model> Model::load(Reader&)`.
- Produces: `Result<StepResult> Model::prefill(std::span<const std::uint32_t>)`.
- Produces: `Result<StepResult> Model::decode(std::uint32_t, ModelState&)`.
- Produces: `Result<std::vector<std::uint32_t>> Model::generate_greedy(prompt, count, incremental)`.
- Produces: `k3x_run --model PATH --prompt-ids 1,7,3,9 --generate 6 --mode full|incremental --json PATH`.
- Produces `K3XFixture(artifact, golden_npz, golden_token_ids, golden_state_sha256)` and `cpp_runner: pathlib.Path` session fixtures plus `run_checked(argv: Sequence[os.PathLike[str] | str]) -> subprocess.CompletedProcess[str]` in `tests/python/conftest.py`.

- [ ] **Step 1: Write the cross-language RED test**

```python
# 독립 C++ runtime이 PyTorch golden의 상태와 token을 재현하는지 검증하는 테스트
def test_cpp_incremental_generation_matches_python_golden(k3x_fixture, cpp_runner, tmp_path):
    output = tmp_path / "cpp-result.json"
    run_checked([
        cpp_runner,
        "--model", str(k3x_fixture.artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--json", str(output),
    ])
    got = json.loads(output.read_text(encoding="utf-8"))
    assert got["token_ids"] == k3x_fixture.golden_token_ids
        assert_numeric_state_close(got["state"], k3x_fixture.golden_state, atol=1e-6, rtol=1e-6)
```

Add parameterized checks for primitive, each of four layer outputs, final logits, full tokens, incremental tokens, KDA state tensors, and MLA state tensors. Numeric arrays use the tolerance stored in model config because different GEMM shapes and the independent C++ reduction order are not bit-identical. MXFP4 bytes, selected expert IDs, and token IDs remain exact. Optional state digests are compared only between repeated executions of the same runtime after numeric boundary checks.

- [ ] **Step 2: Build and observe RED**

Run:

```powershell
rtk cmake --build build
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_cpp_parity.py -q
```

Expected: `k3x_run` target or graph symbols are missing.

- [ ] **Step 3: Implement KDA and MLA from the public contract**

Port the operation order from Task 3 without calling Python or LibTorch. Preserve convolution tap order, per-key-channel decay, updated-state output, shared extra-key cache storage without rotary embedding, complete-head scaling, and pre-projection MLA output gate. Accumulate dot products in `double` and cast once to FP32 at the output boundary.

- [ ] **Step 4: Implement router, latent MoE, AttnRes, and model composition**

Use deterministic lower-ID tie breaks, selection-only correction bias, selected expert random access through `Reader`, and exact packed MXFP4 matmul. Keep routed latent and shared hidden branches distinct. Maintain AttnRes sources only for the active forward and persist only KDA/MLA state between tokens.

- [ ] **Step 5: Add diagnostic JSON and first-divergence reporting**

The JSON contains `token_ids`, `layer_max_abs_error` when a golden diagnostic extent is present, state digests, `read_calls`, `read_bytes`, `peak_rss_bytes`, and total/prefill/decode nanoseconds. It must not print tensor payloads or unbounded arrays.

- [ ] **Step 6: Run focused parity until the earliest boundary is green**

Run one parameterized boundary at a time using pytest node IDs. If two attempted fixes fail at the same boundary, stop patching, list verified facts and assumptions in `context-notes.md`, and run the cheapest discriminating primitive test before continuing.

- [ ] **Step 7: Run all suites and commit end-to-end parity**

```powershell
rtk cmake --build build --config Debug
rtk ctest --test-dir build --output-on-failure
rtk .\.venv\Scripts\python.exe -m pytest tests/python -q
rtk git diff --check
rtk git add runtime/include/k3x runtime/src tests/python/test_cpp_parity.py CMakeLists.txt checklist.md context-notes.md
rtk git commit -m "feat: match synthetic K3 generation in C++"
```

---

### Task 8: Architecture, performance model, and measured synthetic report

**Files:**
- Create: `ARCHITECTURE.md`
- Create: `PERFORMANCE_MODEL.md`
- Create: `docs/references.md`
- Create: `tools/benchmark_synthetic.py`
- Create: `tests/python/test_benchmark_schema.py`
- Modify: `README.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces: `benchmark_once(artifact, runner, warmup, iterations) -> BenchmarkRecord`.
- Produces JSON and CSV fields `prefill_tokens_per_second`, `decode_tokens_per_second`, `ttft_ms`, `peak_rss_bytes`, `file_read_bytes_per_token`, `kda_state_bytes`, `mla_kv_bytes`, and per-layer nanoseconds.

- [ ] **Step 1: Write the benchmark schema test before the runner**

The test constructs one `BenchmarkRecord` with literals and asserts JSON and CSV round-trip preserve field names, numeric types, units, and `scope == "synthetic-milestone-zero"`. It rejects a record that labels projected full-model values as measured.

- [ ] **Step 2: Run and observe RED**

Run `rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_benchmark_schema.py -q`.

Expected: missing `tools.benchmark_synthetic` module.

- [ ] **Step 3: Write source-backed architecture documentation**

`ARCHITECTURE.md` records the exact released 0-based layer map, KDA/MLA equations and operation order, AttnRes block flow, router bias semantics, Stable LatentMoE branch order, MXFP4 metadata, persistent state, MoonViT projector boundary, and proposed L2→L1→L0 data flow. Every materially source-dependent statement links to the pinned source ledger in `docs/references.md`.

- [ ] **Step 4: Calculate the performance model with explicit variables**

`PERFORMANCE_MODEL.md` separates confirmed config values, derived byte counts, target-hardware specifications, assumed bandwidth variables, and measured synthetic values. Include equations for trunk bytes/token, selected expert bytes/token, cache-hit-adjusted NVMe bytes/token, RAM→VRAM bytes/token, KDA state, MLA KV growth, and upper bounds from each memory tier. Do not insert a tok/s result unless produced by the benchmark command on the named machine.

- [ ] **Step 5: Implement and run the benchmark harness**

```powershell
rtk .\.venv\Scripts\python.exe tools/benchmark_synthetic.py --artifact build-fixtures/synthetic.k3x --runner build\k3x_run.exe --warmup 3 --iterations 20 --json build-results/milestone-zero.json --csv build-results/milestone-zero.csv
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_benchmark_schema.py -q
```

Run cold-open measurement in a separate process and label it OS-cache-dependent. Do not claim a physical cold NVMe measurement on Windows unless the OS cache was actually and safely controlled.

- [ ] **Step 6: Update README status using only measured evidence**

Mark PyTorch, K3X round-trip, C++ parity, and synthetic benchmark boxes complete only when their commands passed. Link the committed report schema and summarize measured values as synthetic-only.

- [ ] **Step 7: Commit documentation and measurement tooling**

```powershell
rtk git add ARCHITECTURE.md PERFORMANCE_MODEL.md docs/references.md tools/benchmark_synthetic.py tests/python/test_benchmark_schema.py README.md checklist.md context-notes.md
rtk git commit -m "docs: record K3X architecture and synthetic measurements"
```

---

### Task 9: Final verification, self-review, and publication

**Files:**
- Modify only files needed to correct verified milestone defects.
- Update: `checklist.md`
- Update: `context-notes.md`

**Interfaces:**
- Consumes all prior task outputs.
- Produces one clean, reproducible Milestone 0 verification result on the current machine.

- [ ] **Step 1: Run the complete clean verification**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python -q
rtk cmake --build build --config Release
rtk ctest --test-dir build -C Release --output-on-failure
rtk .\.venv\Scripts\k3x-convert.exe validate build-fixtures/synthetic.k3x
rtk build\Release\k3x_run.exe --model build-fixtures/synthetic.k3x --prompt-ids 1,7,3,9 --generate 6 --mode incremental --json build-results/final-run.json
rtk .\.venv\Scripts\python.exe tools/benchmark_synthetic.py --artifact build-fixtures/synthetic.k3x --runner build\Release\k3x_run.exe --warmup 3 --iterations 20 --json build-results/milestone-zero.json --csv build-results/milestone-zero.csv
```

Use the actual generator-specific executable path emitted by CMake when it differs from `build\Release\k3x_run.exe`; record that path in `context-notes.md`.

- [ ] **Step 2: Inspect the full diff and second-order effects**

```powershell
rtk git status --short --branch
rtk git diff --check
rtk git diff origin/main...HEAD --stat
rtk rg -n "T[O]DO|T[B]D|FIXME|print\(|std::cout" reference converter runtime tests tools
```

Review every caller of changed signatures, error paths, generated-file exclusions, source headers, debug output, artifact bounds checks, and README claims. Fix only verified milestone issues and rerun the affected test plus the complete suite once.

- [ ] **Step 3: Complete tracking documents and commit fixes if any**

Mark only evidenced checklist items complete. Append test counts, commands, measured values, limitations, and the next observed bottleneck to `context-notes.md`. If this changes tracked files, commit them as `docs: close K3X milestone zero evidence`.

- [ ] **Step 4: Push the reviewed branch**

```powershell
rtk git push -u origin HEAD
rtk git status --short --branch
```

Expected: the branch tracks its remote and the working tree is clean. Report exact test results and synthetic measurements, explicitly distinguishing them from projected full-model behavior.

---

## Plan Self-Review Result

- Every approved Milestone 0 requirement maps to at least one task and executable check.
- Python and C++ interfaces use the same state boundaries without sharing math implementations.
- Native MXFP4 byte preservation, full versus incremental parity, corruption rejection, and resume are explicit gates.
- CUDA, tiered caching, speculative decoding, cloud execution, and full weights remain outside this plan.
- Current machine toolchain gaps are acknowledged and package installation is not assumed authorized.
