# K3X Milestone 16 AURORA Replay and Adaptive Scheduling Design

## Status and objective

Milestone 16 creates the first measured self-speculative K3 path without changing target correctness. A separate reduced-Top-K execution of the same K3X artifact proposes real candidate tokens. The existing natural-routing target then applies the unchanged strict greedy token-major or exact expert-major verifier. A deterministic adaptive scheduler chooses a draft prefix length from 1, 2, or 4 using observed prefix survival and measured expert-union cost.

The first AURORA implementation deliberately replays the committed prefix for every proposal. It is an exact, bounded acceptance-trace reference, not the final fast drafter. Persistent draft KDA/MLA state, reduced precision, resident-only drafting, learned DSpark checkpoints, EcoSpec path search, MoE-Spec budgets, and AcceptMoE verifier selection remain separate later work.

## Primary-source evidence

- DSpark paper arXiv `2607.05147` separates draft generation from target verification and schedules verification length from estimated prefix survival rather than using every proposed suffix position.
- DeepSpec commit `005e03b81cec38b7da6399833d609ee89a2587f2` forms an anchor-first proposal, crops draft cache after proposal, commits accepted draft prefix plus one target token, and updates draft context only from verified target state. Its `draft_ops.py` truncates at the first confidence below threshold; its confidence recorder evaluates cumulative prefix survival.
- EcoSpec arXiv `2607.12696` keeps the target verification rule unchanged while incorporating marginal expert activation cost into draft selection. Milestone 16 does not reproduce EcoSpec path search; it only establishes measured cost feedback required by a later reproduction.
- MoE-Spec arXiv `2602.16052` budgets verification experts and is therefore quality-changing. Milestone 16 does not drop, substitute, or budget target experts.
- K3X B-0015 and B-0016 show that perfect block-2 expert-major execution amortizes payload loads, while three evaluated and discarded positions make mixed blocks slower and increase traffic. Those scripted extremes are insufficient to select a production block size.

Pinned sources:

- <https://arxiv.org/abs/2607.05147>
- <https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2>
- <https://arxiv.org/abs/2607.12696>
- <https://arxiv.org/abs/2602.16052>

## Alternatives

### Accepted: replay-based K3 self draft plus exact target

Open a second Reader on the same K3X artifact and use a separate CPU backend and RuntimeSession configured for fixed reduced Top-K. For each proposal, concatenate the original prompt and currently committed generated tokens, run exact incremental generation for the scheduler-selected candidate count, and return those generated tokens as the proposal. This executes the real reduced-routing graph and produces a real acceptance trace while keeping draft Reader, profiler, cache, and routing telemetry separate from the target.

Replay is intentionally inefficient. Its role is to fix candidate, lifecycle, telemetry, and scheduling semantics before persistent draft state or CUDA drafting can hide correctness failures.

### Rejected: scripted traces as representative acceptance

Scripted perfect and mixed rows remain valuable edge cases, but their acceptance distributions are authored rather than produced by a drafter. They cannot determine whether reduced Top-K K3 candidates preserve coding tokens or whether block adaptation reacts to actual divergence.

### Deferred: trained DSpark draft checkpoint

DeepSpec requires a target-specific trained checkpoint and target hidden-state features. No Kimi K3 DSpark checkpoint or accepted tensor ABI exists. Introducing an unrelated checkpoint would not measure K3 self speculation and would add a training pipeline before the runtime contract is stable.

### Deferred: persistent reduced-state AURORA

A persistent drafter is the performance destination, but it must crop or advance KDA convolution history, recurrent matrices, and MLA KV state after every strict verification. Replay supplies the oracle against which that later stateful implementation will be tested.

## Runtime and provider contract

Add `AuroraReplayDraftProvider` as a normal `DraftProvider`. It owns references to a dedicated draft Reader and CPU backend, an independent draft RuntimeSession, the original prompt, an `AdaptiveDraftScheduler`, and cumulative draft telemetry.

The provider enforces one outstanding proposal. On the first request it adopts the target's current generated-token vector. On every later request it requires exact equality with the committed vector derived from the previous `DraftVerification`. It also requires `generated_position == generated_tokens.size()`, a nonempty generated sequence, and `generated_tokens.back() == anchor_token` before draft execution.

`propose` selects a candidate count no greater than `request.max_draft_tokens`, replays `prompt + generated_tokens` through reduced fixed Top-K, and returns the generated continuation. `update` is the existing no-fail observation hook: it advances the expected committed vector from the target-constructed verification and passes feedback to the scheduler. An impossible inconsistent update latches a provider lifecycle error, and the next `propose` fails before draft execution. Provider failure cannot mutate target state.

The first draft identity is restricted to the following.

- CPU backend with FP32 dense arithmetic and native exact MXFP4 decode.
- Incremental replay from the complete committed prefix.
- Fixed K4, K6, K8, or K12 strictly below the artifact's natural Top-K.
- Disabled L1, blocking `pread + buffered`, no runtime profile, no quality escalation, and no proxy or pruning.
- A separate Reader and profiler so draft bytes and time never appear as target bytes and time.

Ordinary greedy generation and `scripted-reference` remain unchanged. The public CLI identity is `--speculative-mode aurora-replay`; the component registry must say “experimental replay reference,” not “AURORA fast path implemented.”

## Verification feedback

Extend `DraftVerification` with physical feedback fields that default to zero.

```cpp
std::size_t target_positions_evaluated{};
std::size_t target_positions_discarded{};
std::size_t expert_major_payload_loads{};
std::size_t expert_major_assignments{};
```

The pure token verifier continues to decide only committed tokens. The runtime decorates a successful verification before `DraftProvider::update`.

- Token-major reports actual target forwards. It does not report unexecuted suffix positions as discarded work.
- Expert-major reports the complete block positions evaluated, positions beyond the committed state snapshot as discarded, and the existing payload-load and assignment counts.
- Scripted providers may ignore the new fields.

These fields are observations, not permission for a provider to alter target verification.

## Adaptive block scheduler

`AdaptiveDraftScheduler` is a pure, independently tested component. Its ladder is the ordered subset of `{1, 2, 4}` not exceeding the CLI maximum. It starts at the smallest rung and explores at most one unobserved rung beyond the largest observed proposal length.

For each draft position `i`, the scheduler records a trial when a proposal contains that position and a prefix success when `accepted_draft_tokens > i`. The smoothed prefix-survival estimate is `(successes + 1) / (trials + 2)`. A previously observed length is eligible only when its final-position survival estimate is at least 0.5.

For each observed proposal length with expert-major feedback, the scheduler records `expert_major_payload_loads / expert_major_assignments`. A length is cost-eligible when its cumulative ratio is no greater than 0.9. Zero-assignment token-major observations use acceptance only. An unobserved next rung is allowed once for exploration; it cannot skip a rung.

The scheduler selects the longest eligible rung within the request maximum. If no rung is eligible, it returns zero candidates and the unchanged verifier performs one ordinary target step. The resulting all-zero verification feedback is a state-preserving scheduler no-op. Any block with `accepted_draft_tokens < proposed_draft_tokens` immediately caps the next choice at the preceding rung. Rejection at length one therefore consumes a one-shot zero cap for the next positive request, then grants one smallest-rung retry even when cumulative survival is still below threshold. Repeated rejection alternates an ordinary target step with a bounded length-one retry instead of permanently disabling drafting. This rejection backoff takes precedence over exploration. Requests with maximum zero also return zero candidates without mutating scheduler state.

Fixed policy remains available and always requests the configured maximum. Adaptive scheduling changes draft work and target batch length only; it does not change target routing, logits, acceptance, or state commit.

## Telemetry

`DraftProviderStats` gains default-zero counters available through a virtual `stats()` method so existing providers remain source compatible.

- draft proposal calls and generated candidate tokens
- committed-prefix tokens replayed
- draft generation nanoseconds
- draft Reader calls and logical bytes
- draft routing decisions and selected experts
- scheduler selections at lengths 1, 2, and 4
- scheduler backoffs and exploratory growth steps

`GenerationResult` copies those values after successful generation. Existing target Reader, H2D, cache, routing, and expert-major counters retain their current meanings. JSON/CSV expose draft and target traffic separately; no combined value is mislabeled as physical NVMe traffic.

## CLI and failure boundaries

The CLI adds the following options.

- `--speculative-mode aurora-replay`
- `--aurora-draft-k 4|6|8|12`
- `--aurora-block-policy fixed|adaptive`

`--speculative-block-size` remains the maximum proposal length and must be 1, 2, or 4 for AURORA replay. The CLI rejects AURORA options in other speculative modes, requires incremental target execution and natural target routing, and rejects a draft K greater than or equal to the artifact natural K after metadata is available. It opens the draft Reader and creates the CPU draft backend before target prefill but after all option and artifact checks. Output files are not mutated on validation failure.

Target expert-major retains every Milestone 14/15 capability gate. AURORA replay may drive either token-major target verification or an already-supported CPU/CUDA expert-major target; its draft backend remains CPU-only in this milestone.

## Correctness and lifecycle tests

- Pure scheduler literals cover initial length, one-rung exploration, full-accept growth, first/middle rejection backoff, request caps, prefix-survival rejection, expert-cost rejection, zero-assignment token-major feedback, and invalid observations.
- Provider tests cover real reduced-Top-K candidates, request/anchor/history validation, one-outstanding-proposal enforcement, latched inconsistent-update rejection, update synchronization, zero-candidate requests, and draft Reader/route telemetry separation.
- Runtime tests compare AURORA token-major and CPU/CUDA expert-major outputs with natural greedy tokens, final KDA/MLA state, and committed natural routes.
- A reduced-K mismatch must prove that the target commits its own token and the next proposal replays the corrected committed history.
- CLI tests reject invalid mode, K, policy, block size, target routing, and unsupported expert-major combinations before output mutation.
- Existing scripted, greedy, routing, and B-0014 through B-0016 tests remain unchanged and passing.

## B-0017 measurement gate

B-0017 uses the deterministic four-layer, 24-expert, natural Top-16 executable artifact. It runs three warmups and twenty measured samples for natural greedy, AURORA K4 fixed block 1/2/4, AURORA K4 adaptive token-major, and AURORA K4 fixed/adaptive CPU expert-major. If the verified CUDA build is available, matched CUDA expert-major fixed/adaptive rows are a separate capability group rather than a requirement for the portable result.

Every row records target decode/prefill/TTFT, exact tokens/final state/committed routes, draft and target Reader calls/bytes, draft replay time/context tokens, proposal-length histogram, acceptance, evaluated/discarded target positions, unique experts, assignments, payload loads, H2D/D2H, kernel time, VRAM, and host RAM. Raw JSON/CSV and the summary carry checksums and are cross-validated.

No favorable throughput direction is required. Replay overhead is expected and must remain visible. Adaptive scheduling becomes preferable to a fixed block only if measured target waste falls without changing exact output; it does not become the default from synthetic WSL2 evidence.

## Follow-up boundary

After B-0017, replace replay with persistent AURORA draft state while using replay as the exact candidate oracle. The persistent path must update or crop KDA convolution/recurrent state and MLA KV state from strict verification, and it must demonstrate lower draft prefix work with identical proposals before reduced precision or resident-only expert constraints are added. EcoSpec, MoE-Spec, and AcceptMoE remain later independent experiments with separate quality gates.
