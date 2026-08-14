// 합성 K3 graph의 C++ greedy generation 인터페이스를 선언합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/expert_scheduler.hpp"
#include "k3x/host_expert_store.hpp"
#include "k3x/reader.hpp"
#include "k3x/routing_policy.hpp"
#include "k3x/runtime_profile.hpp"
#include "k3x/speculative.hpp"
#include "k3x/status.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace k3x {
enum class L2ExpertScheduleMode { blocking, deadline };
enum class SpeculativeVerificationMode { token_major, expert_major };

struct RuntimeOptions {
    bool incremental{true};
    bool diagnostics{};
    L1ExpertCacheMode l1_expert_cache{L1ExpertCacheMode::disabled};
    std::size_t l1_expert_cache_bytes{};
    std::uint64_t profile_prior_strength{64};
    bool profile_observation{};
    L2ExpertScheduleMode l2_expert_schedule{L2ExpertScheduleMode::blocking};
    RoutingPolicyConfig routing_policy{};
    SpeculativeVerificationMode speculative_verification{
        SpeculativeVerificationMode::token_major};
};

class RuntimeSession {
public:
    explicit RuntimeSession(RuntimeOptions options)
        : RuntimeSession(options, RuntimeProfile{}) {}

    RuntimeSession(RuntimeOptions options, RuntimeProfile profile)
        : options_(options), profile_(std::move(profile)),
          expert_store_(options.l1_expert_cache,
                        options.l1_expert_cache_bytes,
                        options.profile_observation ||
                                options.l1_expert_cache ==
                                    L1ExpertCacheMode::profiled
                            ? &profile_
                            : nullptr,
                        options.profile_prior_strength) {
        if (options.l2_expert_schedule == L2ExpertScheduleMode::deadline) {
            expert_loader_ = std::make_unique<DeadlineExpertLoader>(64);
        }
    }

    const RuntimeOptions& options() const noexcept { return options_; }
    HostExpertStore& expert_store() noexcept { return expert_store_; }
    RuntimeProfile& profile() noexcept { return profile_; }
    const RuntimeProfile& profile() const noexcept { return profile_; }
    L1ExpertCacheStats l1_expert_cache_stats() const {
        return expert_store_.stats();
    }
    std::uint64_t acquire_forward_cycle() noexcept {
        return next_forward_cycle_.fetch_add(1, std::memory_order_relaxed);
    }
    std::unique_lock<std::mutex> acquire_generation_guard() {
        return std::unique_lock(generation_mutex_);
    }
    DeadlineExpertLoader* expert_loader() noexcept {
        return expert_loader_.get();
    }
    ExpertLoadSchedulerStats expert_load_scheduler_stats() const {
        return expert_loader_ ? expert_loader_->stats()
                              : ExpertLoadSchedulerStats{};
    }

private:
    RuntimeOptions options_;
    RuntimeProfile profile_;
    HostExpertStore expert_store_;
    std::unique_ptr<DeadlineExpertLoader> expert_loader_;
    std::atomic<std::uint64_t> next_forward_cycle_{};
    std::mutex generation_mutex_;
};

struct GenerationResult {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint64_t> per_layer_nanoseconds;
    std::vector<std::vector<float>> prefill_layer_outputs;
    std::vector<float> prefill_logits;
    std::vector<float> prefill_state;
    std::vector<std::uint32_t> prefill_routed_experts;
    std::vector<std::uint32_t> prefill_routed_k;
    std::vector<float> final_state;
    std::vector<std::uint32_t> routed_experts;
    std::vector<std::uint32_t> routed_k;
    std::uint64_t prefill_nanoseconds{};
    std::uint64_t decode_nanoseconds{};
    std::uint64_t target_decode_forward_calls{};
    std::uint64_t speculative_verification_blocks{};
    std::uint64_t speculative_proposed_draft_tokens{};
    std::uint64_t speculative_accepted_draft_tokens{};
    std::uint64_t speculative_committed_tokens{};
    std::uint64_t speculative_max_proposal_tokens{};
    std::uint64_t draft_proposal_calls{};
    std::uint64_t draft_candidate_tokens{};
    std::uint64_t draft_replayed_context_tokens{};
    std::uint64_t draft_generation_nanoseconds{};
    std::uint64_t draft_reader_calls{};
    std::uint64_t draft_reader_bytes{};
    std::uint64_t draft_routing_decisions{};
    std::uint64_t draft_routing_selected_experts{};
    std::uint64_t draft_selected_length_1{};
    std::uint64_t draft_selected_length_2{};
    std::uint64_t draft_selected_length_4{};
    std::uint64_t draft_scheduler_growths{};
    std::uint64_t draft_scheduler_backoffs{};
    std::uint64_t draft_context_prefill_tokens{};
    std::uint64_t draft_incremental_forward_calls{};
    std::uint64_t draft_rollback_events{};
    std::uint64_t draft_mla_positions_cropped{};
    std::uint64_t draft_kda_checkpoint_bytes{};
    std::uint64_t target_block_forward_calls{};
    std::uint64_t target_positions_evaluated{};
    std::uint64_t target_positions_discarded{};
    std::uint64_t expert_major_unique_experts_sum{};
    std::uint64_t expert_major_unique_experts_max{};
    std::uint64_t expert_major_assignments{};
    std::uint64_t expert_major_reused_assignments{};
    std::uint64_t expert_major_payload_loads{};
    std::vector<std::uint32_t> evaluated_routed_experts;
    std::vector<std::uint32_t> evaluated_routed_k;
    L1ExpertCacheStats l1_expert_cache;
    ExpertLoadSchedulerStats expert_load_scheduler;
    std::uint64_t routing_decisions{};
    std::uint32_t routing_natural_top_k{};
    std::uint64_t routing_selected_experts{};
    std::uint64_t routing_quality_escalated_decisions{};
    std::uint64_t cold_rescue_count{};
    double routing_normalized_entropy_sum{};
    double routing_selected_mass_sum{};
    double routing_boundary_confidence_sum{};
};

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeSession& session);

Result<GenerationResult> generate_speculative(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt, std::size_t count,
    RuntimeSession& session, DraftProvider& draft_provider,
    std::size_t block_size);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics = false);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics = false);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options);
}
