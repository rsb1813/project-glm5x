# Contributing to GLM5X

Thanks for your interest in GLM5X.

GLM5X is an experimental clean-room, out-of-core inference runtime for GLM-5.x. Correctness, reproducibility, and honest performance reporting take priority over headline throughput.

## Development principles

1. **Correctness before performance.**
2. **Optimizations retain a reference/comparison path whenever practical.**
3. **Performance claims require reproducible measurements.**
4. **Routing semantics must not change silently.**
5. **Residency or cache misses must never silently become expert pruning.**
6. **Synthetic and bounded-fixture results must not be presented as full GLM-5.x performance.**
7. **Measured values, estimates, and projections must be clearly distinguished.**
8. **WSL2 measurements are development evidence, not authoritative native-Linux performance.**

## Getting started

The primary development path is Linux with Python 3.12 and C++20. CUDA work targets NVIDIA RTX 5080-class hardware and the CUDA configuration documented in the repository.

Create a Python environment and install development dependencies using the instructions in `README.md`. Build the C++ runtime with CMake/Ninja and run the applicable CTest and Python suites before submitting a change.

## Choosing an issue

- Use **Bug Report** for reproducible correctness or runtime failures.
- Use **Performance Report** for regressions or reproducible performance results.
- Use **Feature Request** for new functionality or optimization proposals.
- Do **not** open a public issue for a suspected security vulnerability. Follow `SECURITY.md` and use GitHub Private Vulnerability Reporting.

For substantial architectural changes, opening an issue first is strongly encouraged so the intended correctness boundary, benchmark, and scope can be agreed before implementation.

## Pull requests

Keep pull requests focused. A PR should ideally change one coherent behavior or optimization boundary.

The repository PR template contains detailed checklists. Only items applicable to the change need to be checked; do not mark an item merely to make the checklist complete.

Before submitting a PR:

- build succeeds for applicable configurations;
- relevant CTest suites pass;
- relevant Python tests pass;
- new behavior has focused tests;
- failure paths are tested when applicable;
- documentation is updated when externally visible behavior changes;
- performance changes include a matched before/after measurement or explicitly state that no performance claim is being made.

## Correctness changes

When changing model execution, routing, cache behavior, speculative verification, checkpoint handling, or state management, preserve and report the strongest applicable parity evidence, such as:

- generated token IDs;
- logits or numerical error bounds;
- KDA/MLA recurrent state;
- routing and Top-K traces;
- expert ordering;
- Reader calls and logical bytes;
- L1 cache hits/misses;
- committed versus evaluated speculative state.

Lossy behavior must be explicit, opt-in unless separately justified, and documented as a quality/performance tradeoff.

## CUDA changes

For CUDA changes, include the applicable validation evidence:

- CPU/reference parity;
- CUDA capability and failure-path checks;
- Compute Sanitizer where the path actually executes CUDA work;
- memory ownership/lifetime review;
- H2D/D2H changes where relevant;
- peak/current resident VRAM where relevant;
- kernel/launch telemetry where relevant.

Do not report a sanitizer pass for a command that never executes instrumented CUDA work.

## Performance claims

Performance PRs should provide enough information for another contributor to reproduce the result.

Include, when applicable:

- exact commit SHA;
- benchmark ID and command;
- CPU, GPU, RAM, OS, CUDA/driver, and storage;
- warmup/sample counts;
- model identity: synthetic, bounded released-dimension fixture, or full model;
- decode/prefill throughput or latency;
- logical Reader bytes and clearly labeled physical I/O if actually measured;
- H2D/D2H;
- VRAM/RAM;
- cache hits/misses;
- kernel launches or synchronization counts.

A faster tiny synthetic result is evidence for that boundary only. It is not evidence of full GLM-5.x throughput.

## Documentation

Depending on the change, update the appropriate files:

- `README.md` for user-visible capabilities and milestone summaries;
- `ARCHITECTURE.md` for execution/runtime design;
- `BENCHMARKS.md` for measured evidence;
- `PROJECT_STATE.md` for current implementation state;
- `DECISIONS.md` for important design decisions;
- `ARCHITECTURE.md` for checkpoint-format changes.

Avoid claiming planned work as implemented work.

## Commit and PR style

Use clear, imperative commit messages when practical, for example:

- `feat: add ...`
- `fix: reject ...`
- `bench: measure ...`
- `docs: record ...`
- `test: cover ...`

PR descriptions should explain **what changed, why it changed, how correctness was checked, and what was actually measured**.

## Security

Please read `SECURITY.md`. Security-sensitive reports should be submitted privately rather than through public issues or discussions.

## License

By contributing to this repository, you agree that your contributions will be licensed under the repository's Apache License 2.0.
