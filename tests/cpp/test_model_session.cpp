// 동일 runtime session의 연속 generation이 L1 expert residency를 재사용하는지 검증합니다.
#include "k3x/model.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void require(
    bool condition,
    const std::source_location location = std::source_location::current()) {
    if (!condition) {
        std::cerr << "session requirement failed at line "
                  << location.line() << '\n';
        throw std::runtime_error("session cache requirement failed");
    }
}

class ScriptedDraftProvider final : public k3x::DraftProvider {
public:
    explicit ScriptedDraftProvider(std::deque<k3x::DraftProposal> proposals)
        : proposals_(std::move(proposals)) {}

    k3x::Result<k3x::DraftProposal> propose(
        const k3x::DraftRequest& request) override {
        requests.push_back({request.anchor_token, request.max_draft_tokens,
                            request.generated_position,
                            std::vector<std::uint32_t>(
                                request.generated_tokens.begin(),
                                request.generated_tokens.end())});
        if (proposals_.empty()) {
            return k3x::Result<k3x::DraftProposal>::failure(
                k3x::ErrorCode::invalid_state, "script exhausted");
        }
        auto proposal = std::move(proposals_.front());
        proposals_.pop_front();
        return k3x::Result<k3x::DraftProposal>::success(std::move(proposal));
    }

    void update(const k3x::DraftVerification& verification) override {
        verifications.push_back(verification);
    }

    k3x::DraftProviderStats stats() const noexcept override {
        return {
            .proposal_calls = 7,
            .candidate_tokens = 9,
            .selected_length_2 = 3,
        };
    }

    struct RequestRecord {
        std::uint32_t anchor_token{};
        std::size_t max_draft_tokens{};
        std::size_t generated_position{};
        std::vector<std::uint32_t> generated_tokens;
    };
    std::vector<RequestRecord> requests;
    std::vector<k3x::DraftVerification> verifications;

private:
    std::deque<k3x::DraftProposal> proposals_;
};
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    k3x::ReaderOptions reader_options;
    if (argc == 3) {
        const auto mode = std::string_view(argv[2]);
        if (mode != "io-uring" && mode != "direct" &&
            mode != "io-uring-direct") {
            return 2;
        }
        if (mode != "direct") {
            reader_options.io_engine = k3x::L2IoEngine::io_uring;
        }
        if (mode != "io-uring") {
            reader_options.cache_mode = k3x::L2CacheMode::direct;
        }
    }
    auto reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    if (!reader) return 3;
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.diagnostics = true;
    options.l1_expert_cache = k3x::L1ExpertCacheMode::static_admission;
    options.l1_expert_cache_bytes = 65536;
    k3x::RuntimeSession session(options);
    const std::vector<std::uint32_t> prompt{1, 7, 3, 9};

    auto first = k3x::generate_greedy(
        reader.value(), *backend, prompt, 6, session);
    require(static_cast<bool>(first));
    require(first.value().draft_proposal_calls == 0);
    require(first.value().draft_candidate_tokens == 0);
    require(first.value().draft_reader_bytes == 0);
    require(first.value().draft_scheduler_growths == 0);
    require(first.value().draft_context_prefill_tokens == 0);
    require(first.value().draft_incremental_forward_calls == 0);
    require(first.value().draft_rollback_events == 0);
    require(first.value().draft_mla_positions_cropped == 0);
    require(first.value().draft_kda_checkpoint_bytes == 0);
    const auto first_reads = reader.value().counters();
    const auto first_cache = first.value().l1_expert_cache;
    require(first.value().expert_load_scheduler.submissions == 0);
    require(first_cache.hits == 36);
    require(first_cache.misses == 18);
    require(first_reads.calls - first_reads.batch_submissions ==
            first_cache.misses * 5);
    if (reader_options.cache_mode == k3x::L2CacheMode::direct) {
        require(first_reads.storage_submitted_bytes >
                first_reads.requested_bytes);
    }

    auto second = k3x::generate_greedy(
        reader.value(), *backend, prompt, 6, session);
    require(static_cast<bool>(second));
    const auto second_reads = reader.value().counters();
    const auto second_cache = second.value().l1_expert_cache;
    require(second.value().token_ids == first.value().token_ids);
    require(second.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(second_cache.misses == first_cache.misses);
    require(second_cache.hits == first_cache.hits + 54);
    require(second_cache.resident_bytes == first_cache.resident_bytes);
    require(second_reads.calls - first_reads.calls < first_reads.calls);
    require(second_reads.completed_bytes - first_reads.completed_bytes <
            first_reads.completed_bytes);
    require(second_reads.calls - first_reads.calls ==
            second_reads.batch_submissions - first_reads.batch_submissions);

    auto deadline_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(deadline_reader));
    auto deadline_backend = k3x::make_cpu_backend();
    auto deadline_options = options;
    deadline_options.l2_expert_schedule =
        k3x::L2ExpertScheduleMode::deadline;
    k3x::RuntimeSession deadline_session(deadline_options);
    auto deadline = k3x::generate_greedy(
        deadline_reader.value(), *deadline_backend, prompt, 6,
        deadline_session);
    require(static_cast<bool>(deadline));
    require(deadline.value().token_ids == first.value().token_ids);
    require(deadline.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(deadline.value().l1_expert_cache.hits == first_cache.hits);
    require(deadline.value().l1_expert_cache.misses == first_cache.misses);
    const auto scheduled = deadline.value().expert_load_scheduler;
    require(scheduled.submissions ==
            scheduled.inline_resident_hits + first_cache.misses);
    require(scheduled.inline_resident_hits == first_cache.hits);
    require(scheduled.completions == scheduled.submissions);
    require(scheduled.ready_before_use + scheduled.late_at_use ==
            scheduled.submissions);
    const auto deadline_reads = deadline_reader.value().counters();
    require(deadline_reads.calls == first_reads.calls);
    require(deadline_reads.requested_bytes == first_reads.requested_bytes);
    require(deadline_reads.completed_bytes == first_reads.completed_bytes);

    auto protected_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(protected_reader));
    auto protected_backend = k3x::make_cpu_backend();
    auto protected_options = options;
    protected_options.l1_expert_cache =
        k3x::L1ExpertCacheMode::least_stale;
    protected_options.l1_expert_cache_bytes = 1632;
    k3x::RuntimeSession protected_session(protected_options);
    auto protected_result = k3x::generate_greedy(
        protected_reader.value(), *protected_backend,
        std::vector<std::uint32_t>{1}, 0, protected_session);
    require(static_cast<bool>(protected_result));
    const auto& routed = protected_result.value().prefill_routed_experts;
    require(routed.size() >= 2);
    require(protected_session.expert_store().contains({3, routed[routed.size() - 2]}));
    require(!protected_session.expert_store().contains({3, routed.back()}));

    auto serialized_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(serialized_reader));
    auto serialized_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession serialized_session(options);
    auto generation_guard = serialized_session.acquire_generation_guard();
    auto concurrent_generation = std::async(std::launch::async, [&] {
        return k3x::generate_greedy(
            serialized_reader.value(), *serialized_backend,
            std::vector<std::uint32_t>{1}, 0, serialized_session);
    });
    require(concurrent_generation.wait_for(std::chrono::milliseconds(50)) ==
            std::future_status::timeout);
    generation_guard.unlock();
    require(static_cast<bool>(concurrent_generation.get()));

    k3x::RuntimeProfile runtime_profile;
    require(static_cast<bool>(runtime_profile.set_metadata("TASK", "coding")));
    require(static_cast<bool>(runtime_profile.set_metadata("REPO", "k3x")));
    auto profiled_options = options;
    profiled_options.l1_expert_cache = k3x::L1ExpertCacheMode::profiled;
    profiled_options.profile_prior_strength = 4;
    k3x::RuntimeSession profiled_session(
        profiled_options, std::move(runtime_profile));
    auto profiled = k3x::generate_greedy(
        serialized_reader.value(), *serialized_backend, prompt, 6,
        profiled_session);
    require(static_cast<bool>(profiled));
    require(profiled.value().token_ids == first.value().token_ids);
    require(profiled.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(profiled_session.profile().metadata().at("TASK") == "coding");
    require(profiled_session.profile().live_route_observations() > 0);

    auto run_speculative = [&](ScriptedDraftProvider& provider,
                               std::size_t count,
                               std::size_t block_size) {
        auto speculative_reader = k3x::Reader::open(
            std::filesystem::path(argv[1]), reader_options);
        require(static_cast<bool>(speculative_reader));
        auto speculative_backend = k3x::make_cpu_backend();
        k3x::RuntimeSession speculative_session(options);
        auto generated = k3x::generate_speculative(
            speculative_reader.value(), *speculative_backend, prompt, count,
            speculative_session, provider, block_size);
        return std::pair{std::move(generated),
                         speculative_reader.value().counters()};
    };

    const auto& greedy_tokens = first.value().token_ids;
    ScriptedDraftProvider perfect({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[3],
         .candidate_tokens = {greedy_tokens[4]}},
    });
    auto [perfect_result, perfect_reads] = run_speculative(perfect, 6, 2);
    require(static_cast<bool>(perfect_result));
    require(perfect_result.value().token_ids == greedy_tokens);
    require(perfect_result.value().prefill_state == first.value().prefill_state);
    require(perfect_result.value().final_state == first.value().final_state);
    require(perfect_result.value().routed_experts == first.value().routed_experts);
    require(perfect_result.value().routed_k == first.value().routed_k);
    require(perfect_result.value().speculative_verification_blocks == 2);
    require(perfect_result.value().speculative_proposed_draft_tokens == 3);
    require(perfect_result.value().speculative_accepted_draft_tokens == 3);
    require(perfect_result.value().speculative_committed_tokens == 5);
    require(perfect_result.value().speculative_max_proposal_tokens == 2);
    require(perfect_result.value().l1_expert_cache.hits == first_cache.hits);
    require(perfect_result.value().l1_expert_cache.misses == first_cache.misses);
    require(perfect_reads.calls == first_reads.calls);
    require(perfect_reads.requested_bytes == first_reads.requested_bytes);
    require(perfect.verifications.size() == 2);
    require(perfect.verifications[0].accepted_draft_tokens == 2);
    require(perfect.verifications[1].accepted_draft_tokens == 1);
    require(perfect.verifications[0].target_positions_evaluated == 3);
    require(perfect.verifications[0].target_positions_discarded == 0);
    require(perfect.verifications[1].target_positions_evaluated == 2);
    require(perfect_result.value().draft_proposal_calls == 7);
    require(perfect_result.value().draft_candidate_tokens == 9);
    require(perfect_result.value().draft_selected_length_2 == 3);

    ScriptedDraftProvider mixed({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1] ^ 1U, greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[1], .candidate_tokens = {}},
        {.anchor_token = greedy_tokens[2],
         .candidate_tokens = {greedy_tokens[3], greedy_tokens[4] ^ 1U}},
        {.anchor_token = greedy_tokens[4], .candidate_tokens = {}},
    });
    auto [mixed_result, mixed_reads] = run_speculative(mixed, 6, 2);
    require(static_cast<bool>(mixed_result));
    require(mixed_result.value().token_ids == greedy_tokens);
    require(mixed_result.value().final_state == first.value().final_state);
    require(mixed_result.value().routed_experts == first.value().routed_experts);
    require(mixed_result.value().routed_k == first.value().routed_k);
    require(mixed_result.value().speculative_verification_blocks == 4);
    require(mixed_result.value().speculative_proposed_draft_tokens == 4);
    require(mixed_result.value().speculative_accepted_draft_tokens == 1);
    require(mixed_result.value().speculative_committed_tokens == 5);
    require(mixed_reads.calls == first_reads.calls);
    require(mixed_reads.requested_bytes == first_reads.requested_bytes);
    require(mixed.verifications.size() == 4);
    require(mixed.verifications[0].target_positions_evaluated == 1);
    require(mixed.verifications[1].target_positions_evaluated == 1);
    require(mixed.verifications[2].target_positions_evaluated == 2);
    require(mixed.verifications[3].target_positions_evaluated == 1);
    for (const auto& verification : mixed.verifications) {
        require(verification.target_positions_discarded == 0);
    }

    auto exact_options = options;
    exact_options.l1_expert_cache = k3x::L1ExpertCacheMode::disabled;
    exact_options.l1_expert_cache_bytes = 0;
    auto exact_greedy_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(exact_greedy_reader));
    auto exact_greedy_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession exact_greedy_session(exact_options);
    auto exact_greedy = k3x::generate_greedy(
        exact_greedy_reader.value(), *exact_greedy_backend, prompt, 6,
        exact_greedy_session);
    require(static_cast<bool>(exact_greedy));

    auto run_exact_speculative = [&](ScriptedDraftProvider& provider,
                                     k3x::RuntimeOptions run_options) {
        auto exact_reader = k3x::Reader::open(
            std::filesystem::path(argv[1]), reader_options);
        require(static_cast<bool>(exact_reader));
        auto exact_backend = k3x::make_cpu_backend();
        k3x::RuntimeSession exact_session(run_options);
        auto generated = k3x::generate_speculative(
            exact_reader.value(), *exact_backend, prompt, 6,
            exact_session, provider, 2);
        return std::pair{std::move(generated),
                         exact_reader.value().counters()};
    };

    ScriptedDraftProvider token_major_perfect({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[3],
         .candidate_tokens = {greedy_tokens[4]}},
    });
    auto [token_major_perfect_result, token_major_perfect_reads] =
        run_exact_speculative(token_major_perfect, exact_options);
    require(static_cast<bool>(token_major_perfect_result));

    auto expert_major_options = exact_options;
    expert_major_options.speculative_verification =
        k3x::SpeculativeVerificationMode::expert_major;
    ScriptedDraftProvider expert_major_perfect({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[3],
         .candidate_tokens = {greedy_tokens[4]}},
    });
    auto [expert_major_perfect_result, expert_major_perfect_reads] =
        run_exact_speculative(expert_major_perfect, expert_major_options);
    require(static_cast<bool>(expert_major_perfect_result));
    require(expert_major_perfect_result.value().token_ids ==
            exact_greedy.value().token_ids);
    require(expert_major_perfect_result.value().final_state ==
            exact_greedy.value().final_state);
    require(expert_major_perfect_result.value().routed_experts ==
            exact_greedy.value().routed_experts);
    require(expert_major_perfect_result.value().routed_k ==
            exact_greedy.value().routed_k);
    require(expert_major_perfect_result.value().target_block_forward_calls == 2);
    require(expert_major_perfect_result.value().target_positions_evaluated == 5);
    require(expert_major_perfect_result.value().target_positions_discarded == 0);
    require(expert_major_perfect_result.value().expert_major_payload_loads ==
            expert_major_perfect_result.value().expert_major_unique_experts_sum);
    require(expert_major_perfect_result.value().expert_major_assignments >
            expert_major_perfect_result.value().expert_major_payload_loads);
    require(expert_major_perfect_result.value().expert_major_reused_assignments ==
            expert_major_perfect_result.value().expert_major_assignments -
                expert_major_perfect_result.value().expert_major_payload_loads);
    require(expert_major_perfect_reads.requested_bytes <
            token_major_perfect_reads.requested_bytes);
    require(expert_major_perfect.verifications.size() ==
            token_major_perfect.verifications.size());
    require(expert_major_perfect.verifications[0].committed_tokens ==
            token_major_perfect.verifications[0].committed_tokens);
    require(expert_major_perfect.verifications[1].committed_tokens ==
            token_major_perfect.verifications[1].committed_tokens);
    require(expert_major_perfect.verifications[0].target_positions_evaluated == 3);
    require(expert_major_perfect.verifications[1].target_positions_evaluated == 2);
    require(expert_major_perfect.verifications[0].target_positions_discarded == 0);
    require(expert_major_perfect.verifications[1].target_positions_discarded == 0);

    ScriptedDraftProvider token_major_mixed({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1] ^ 1U, greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[1], .candidate_tokens = {}},
        {.anchor_token = greedy_tokens[2],
         .candidate_tokens = {greedy_tokens[3], greedy_tokens[4] ^ 1U}},
        {.anchor_token = greedy_tokens[4], .candidate_tokens = {}},
    });
    auto [token_major_mixed_result, token_major_mixed_reads] =
        run_exact_speculative(token_major_mixed, exact_options);
    require(static_cast<bool>(token_major_mixed_result));
    ScriptedDraftProvider expert_major_mixed({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1] ^ 1U, greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[1], .candidate_tokens = {}},
        {.anchor_token = greedy_tokens[2],
         .candidate_tokens = {greedy_tokens[3], greedy_tokens[4] ^ 1U}},
        {.anchor_token = greedy_tokens[4], .candidate_tokens = {}},
    });
    auto [expert_major_mixed_result, expert_major_mixed_reads] =
        run_exact_speculative(expert_major_mixed, expert_major_options);
    require(static_cast<bool>(expert_major_mixed_result));
    require(expert_major_mixed_result.value().token_ids ==
            exact_greedy.value().token_ids);
    require(expert_major_mixed_result.value().final_state ==
            exact_greedy.value().final_state);
    require(expert_major_mixed_result.value().routed_experts ==
            exact_greedy.value().routed_experts);
    require(expert_major_mixed_result.value().routed_k ==
            exact_greedy.value().routed_k);
    require(expert_major_mixed_result.value().target_block_forward_calls == 4);
    require(expert_major_mixed_result.value().target_positions_evaluated == 8);
    require(expert_major_mixed_result.value().target_positions_discarded == 3);
    require(expert_major_mixed.verifications.size() ==
            token_major_mixed.verifications.size());
    for (std::size_t index = 0;
         index < expert_major_mixed.verifications.size(); ++index) {
        require(expert_major_mixed.verifications[index].committed_tokens ==
                token_major_mixed.verifications[index].committed_tokens);
        require(expert_major_mixed.verifications[index].accepted_draft_tokens ==
                token_major_mixed.verifications[index].accepted_draft_tokens);
    }
    require(expert_major_mixed.verifications[0].target_positions_evaluated == 3);
    require(expert_major_mixed.verifications[0].target_positions_discarded == 2);
    require(expert_major_mixed.verifications[1].target_positions_evaluated == 1);
    require(expert_major_mixed.verifications[1].target_positions_discarded == 0);
    require(expert_major_mixed.verifications[2].target_positions_evaluated == 3);
    require(expert_major_mixed.verifications[2].target_positions_discarded == 1);
    require(expert_major_mixed.verifications[3].target_positions_evaluated == 1);
    require(expert_major_mixed.verifications[3].target_positions_discarded == 0);
    std::uint64_t feedback_payload_loads = 0;
    std::uint64_t feedback_assignments = 0;
    for (const auto& verification : expert_major_mixed.verifications) {
        feedback_payload_loads += verification.expert_major_payload_loads;
        feedback_assignments += verification.expert_major_assignments;
    }
    require(feedback_payload_loads ==
            expert_major_mixed_result.value().expert_major_payload_loads);
    require(feedback_assignments ==
            expert_major_mixed_result.value().expert_major_assignments);

    auto require_expert_major_preflight_rejection =
        [&](k3x::RuntimeOptions rejected_options) {
            auto rejected_reader = k3x::Reader::open(
                std::filesystem::path(argv[1]), reader_options);
            require(static_cast<bool>(rejected_reader));
            const auto before = rejected_reader.value().counters();
            auto rejected_backend = k3x::make_cpu_backend();
            k3x::RuntimeSession rejected_session(rejected_options);
            ScriptedDraftProvider rejected_provider({});
            auto rejected = k3x::generate_speculative(
                rejected_reader.value(), *rejected_backend, prompt, 2,
                rejected_session, rejected_provider, 2);
            require(!rejected);
            require(rejected.error() == k3x::ErrorCode::invalid_state);
            require(rejected_provider.requests.empty());
            const auto after = rejected_reader.value().counters();
            require(after.calls == before.calls);
            require(after.requested_bytes == before.requested_bytes);
        };

    auto rejected_l1 = expert_major_options;
    rejected_l1.l1_expert_cache = k3x::L1ExpertCacheMode::static_admission;
    rejected_l1.l1_expert_cache_bytes = 65536;
    require_expert_major_preflight_rejection(rejected_l1);
    auto rejected_deadline = expert_major_options;
    rejected_deadline.l2_expert_schedule = k3x::L2ExpertScheduleMode::deadline;
    require_expert_major_preflight_rejection(rejected_deadline);
    auto rejected_routing = expert_major_options;
    rejected_routing.routing_policy.mode = k3x::RoutingMode::fixed;
    rejected_routing.routing_policy.fixed_k = 1;
    require_expert_major_preflight_rejection(rejected_routing);
    auto rejected_profile = expert_major_options;
    rejected_profile.profile_observation = true;
    require_expert_major_preflight_rejection(rejected_profile);

    ScriptedDraftProvider unused({});
    auto [single_result, single_reads] = run_speculative(unused, 1, 2);
    require(static_cast<bool>(single_result));
    require(single_result.value().token_ids ==
            std::vector<std::uint32_t>{greedy_tokens[0]});
    require(single_result.value().speculative_verification_blocks == 0);
    require(unused.requests.empty());

    ScriptedDraftProvider exhausted({});
    auto [exhausted_result, exhausted_reads] =
        run_speculative(exhausted, 2, 2);
    require(!exhausted_result);
    require(exhausted_result.error() == k3x::ErrorCode::invalid_state);
    require(exhausted.verifications.empty());

    ScriptedDraftProvider zero_block({});
    auto [zero_block_result, zero_block_reads] =
        run_speculative(zero_block, 2, 0);
    require(!zero_block_result);
    require(zero_block_result.error() == k3x::ErrorCode::invalid_state);
    require(zero_block.requests.empty());

    ScriptedDraftProvider wrong_anchor({
        {.anchor_token = greedy_tokens[0] ^ 1U, .candidate_tokens = {}},
    });
    auto [wrong_anchor_result, wrong_anchor_reads] =
        run_speculative(wrong_anchor, 2, 2);
    require(!wrong_anchor_result);
    require(wrong_anchor_result.error() == k3x::ErrorCode::invalid_state);
    require(wrong_anchor.verifications.empty());

    ScriptedDraftProvider oversized({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
    });
    auto [oversized_result, oversized_reads] =
        run_speculative(oversized, 2, 2);
    require(!oversized_result);
    require(oversized_result.error() == k3x::ErrorCode::invalid_extent);
    require(oversized.verifications.empty());

    auto nonincremental_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(nonincremental_reader));
    auto nonincremental_backend = k3x::make_cpu_backend();
    auto nonincremental_options = options;
    nonincremental_options.incremental = false;
    k3x::RuntimeSession nonincremental_session(nonincremental_options);
    ScriptedDraftProvider nonincremental_provider({});
    auto nonincremental = k3x::generate_speculative(
        nonincremental_reader.value(), *nonincremental_backend, prompt, 2,
        nonincremental_session, nonincremental_provider, 2);
    require(!nonincremental);
    require(nonincremental.error() == k3x::ErrorCode::invalid_state);
    require(nonincremental_provider.requests.empty());
    return 0;
}
