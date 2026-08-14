// speculative draft proposal과 strict greedy target 검증 계약을 선언합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace k3x {
struct DraftRequest {
    std::uint32_t anchor_token{};
    std::size_t max_draft_tokens{};
    std::size_t generated_position{};
    std::span<const std::uint32_t> generated_tokens;
};

struct DraftProposal {
    std::uint32_t anchor_token{};
    std::vector<std::uint32_t> candidate_tokens;
};

struct DraftVerification {
    std::uint32_t anchor_token{};
    std::size_t proposed_draft_tokens{};
    std::size_t accepted_draft_tokens{};
    std::vector<std::uint32_t> committed_tokens;
    bool all_draft_tokens_accepted{};
    std::size_t target_positions_evaluated{};
    std::size_t target_positions_discarded{};
    std::size_t expert_major_payload_loads{};
    std::size_t expert_major_assignments{};
};

struct DraftProviderStats {
    std::uint64_t proposal_calls{};
    std::uint64_t candidate_tokens{};
    std::uint64_t replayed_context_tokens{};
    std::uint64_t generation_nanoseconds{};
    std::uint64_t reader_calls{};
    std::uint64_t reader_bytes{};
    std::uint64_t routing_decisions{};
    std::uint64_t routing_selected_experts{};
    std::uint64_t selected_length_1{};
    std::uint64_t selected_length_2{};
    std::uint64_t selected_length_4{};
    std::uint64_t scheduler_growths{};
    std::uint64_t scheduler_backoffs{};
    std::uint64_t context_prefill_tokens{};
    std::uint64_t incremental_forward_calls{};
    std::uint64_t rollback_events{};
    std::uint64_t mla_positions_cropped{};
    std::uint64_t kda_checkpoint_bytes{};
};

class DraftProvider {
public:
    virtual ~DraftProvider() = default;
    virtual Result<DraftProposal> propose(const DraftRequest& request) = 0;
    virtual void update(const DraftVerification& verification) = 0;
    virtual DraftProviderStats stats() const noexcept { return {}; }
};

using GreedyTargetStep =
    std::function<Result<std::uint32_t>(std::uint32_t input_token)>;

Result<DraftVerification> verify_greedy_draft(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size, const GreedyTargetStep& target_step);

Result<DraftVerification> verify_greedy_target_block(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size,
    std::span<const std::uint32_t> target_tokens);
}
