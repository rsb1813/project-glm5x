## Summary

<!-- Briefly describe what this PR changes, what problem it solves, and why it is needed. -->

## Related Issue / Milestone

<!-- Examples: Closes #123 / Milestone 21 / Not applicable -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance optimization
- [ ] CUDA / GPU change
- [ ] Storage / I/O change
- [ ] Checkpoint / converter change
- [ ] Routing / MoE change
- [ ] AURORA / speculative decoding change
- [ ] Benchmark / profiling change
- [ ] Documentation
- [ ] Refactor / maintenance
- [ ] Other

## Correctness

- [ ] Existing reference behavior is preserved, or the intended behavior change is documented.
- [ ] Relevant tests were added or updated.
- [ ] CTest passes for applicable configurations.
- [ ] Python tests pass for applicable configurations.
- [ ] No silent fallback was introduced.
- [ ] Failure paths fail explicitly where required.

### Numerical / Behavioral Parity

<!-- If applicable: generated tokens, logits, recurrent state, routing trace, expert ordering, Reader traffic, etc. -->

Not applicable / Results:

## CUDA Changes

- [ ] This PR does not modify CUDA behavior.

If CUDA behavior is modified:

- [ ] CPU/reference parity was checked.
- [ ] CUDA errors are checked explicitly.
- [ ] Compute Sanitizer was run where applicable.
- [ ] Memory ownership and lifetime were reviewed.
- [ ] H2D/D2H changes were measured where relevant.
- [ ] Peak/resident VRAM changes were measured where relevant.

### CUDA Validation

<!-- Hardware, CUDA version, sanitizer results, etc. -->

Not applicable / Results:

## Performance

- [ ] This PR makes no performance claim.
- [ ] This PR includes a performance claim supported by measurements.

If performance is affected, include:

**Before:**

**After:**

**Environment:**

<!-- CPU / GPU / RAM / OS / CUDA / Storage / Commit -->

**Benchmark / workload:**

<!-- Exact command or benchmark ID -->

### Traffic / Resource Changes

<!-- When relevant: logical Reader bytes, physical I/O, H2D, D2H, VRAM, RAM, cache hits/misses, kernel launches. -->

Not applicable / Results:

## GLM5X Invariants

Please confirm applicable items:

- [ ] Correctness is not silently traded for throughput.
- [ ] Residency/cache misses do not silently become expert pruning.
- [ ] Routing semantics are unchanged unless explicitly documented.
- [ ] A high-scoring required cold expert can still be handled by the appropriate exact path where applicable.
- [ ] An optimization has a reference or comparison path where practical.
- [ ] Synthetic results are not presented as full GLM-5.x performance.
- [ ] WSL2 measurements are not presented as authoritative native-Linux performance.
- [ ] Estimated values are clearly distinguished from measured values.

## Documentation

- [ ] No documentation changes are required.
- [ ] README updated.
- [ ] ARCHITECTURE.md updated.
- [ ] BENCHMARKS.md updated.
- [ ] PROJECT_STATE.md updated.
- [ ] DECISIONS.md updated.
- [ ] Other relevant documentation updated.

## Testing

<!-- Paste the commands you ran and summarize the results. -->

```text

```

## Risks / Limitations

<!-- What could go wrong? What remains unsupported or unmeasured? -->

## Additional Notes

<!-- Anything reviewers should know. -->
