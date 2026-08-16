# W8A16 Expert Runtime Implementation Plan

**Goal:** Add an opt-in resident W8A16-G128 routed-expert path and prove its
quality and warm latency on the official GLM-5.2 layer-10 RTX 5080 gate.

**Spec:** `docs/superpowers/specs/2026-08-16-w8a16-expert-runtime-design.md`

## Task 1: Packed host contract

- [ ] Add a RED unit contract for BF16-to-W8A16 packing, BF16 scale rounding,
  deterministic bytes, shape rejection, and reference decode error.
- [ ] Implement the smallest fixed G128 host pack/decode API.
- [ ] Run focused host tests and commit the contract.

## Task 2: Resident CUDA kernel

- [ ] Add a RED CUDA test for one small gate/up/down expert versus host decode.
- [ ] Add the W8 resident representation and fused gate/up plus down kernels.
- [ ] Preserve BF16 activation/intermediate/output rounding and validate kernel
  parity within the quantized-reference envelope.
- [ ] Run focused CUDA tests and commit the kernel.

## Task 3: Official expert-major integration

- [ ] Add opt-in W8A16-G128 routed experts to the existing `.gxi` learned MoE
  and decoder-layer benchmark while keeping the shared expert/trunk BF16.
- [ ] Preserve natural Top-8 IDs and contributions and report W8 bytes,
  resident bytes, H2D bytes, kernel launches, quality, and warm latency.
- [ ] Run the official two-token and one-token layer-10 gates.

## Task 4: Decision gate and publication

- [ ] Accept only if layer relative L2 is at most 1.0%, warm routed H2D is zero,
  and latency materially improves over B-0008.
- [ ] If accepted, add fingerprinted `.pw8` sidecar manufacturing and then run
  a final-logit/greedy-token gate before any full 78-layer artifact build.
- [ ] Run applicable CTest/Pytest, update benchmark evidence and continuity
  documents, commit semantically, push, and update `PROJECT_STATE.md` last.
