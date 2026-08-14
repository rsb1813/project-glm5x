// speculative draft prefix와 target bonus token의 exact 검증 계약을 검사합니다.
#include "k3x/speculative.hpp"

#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                         \
    do {                                                                          \
        if (!(condition)) {                                                        \
            throw std::runtime_error(                                              \
                "speculative requirement failed: " #condition);                  \
        }                                                                         \
    } while (false)

namespace {
k3x::Result<std::uint32_t> next_token(std::uint32_t input,
                                      std::size_t& calls) {
    ++calls;
    return k3x::Result<std::uint32_t>::success(input + 1);
}

class DefaultStatsProvider final : public k3x::DraftProvider {
public:
    k3x::Result<k3x::DraftProposal> propose(
        const k3x::DraftRequest& request) override {
        return k3x::Result<k3x::DraftProposal>::success(
            {.anchor_token = request.anchor_token});
    }

    void update(const k3x::DraftVerification&) override {}
};
}

int main() {
    using k3x::DraftProposal;

    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10, .candidate_tokens = {11, 12}},
            2, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(result);
        assert(result.value().accepted_draft_tokens == 2);
        assert(result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11, 12, 13}));
        assert(calls == 3);
        assert(result.value().target_positions_evaluated == 0);
        assert(result.value().target_positions_discarded == 0);
        assert(result.value().expert_major_payload_loads == 0);
        assert(result.value().expert_major_assignments == 0);
        const auto stats = DefaultStatsProvider{}.stats();
        assert(stats.proposal_calls == 0);
        assert(stats.selected_length_4 == 0);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10, .candidate_tokens = {99, 100}},
            2, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(result);
        assert(result.value().accepted_draft_tokens == 0);
        assert(!result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11}));
        assert(calls == 1);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10, .candidate_tokens = {11, 99}},
            2, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(result);
        assert(result.value().accepted_draft_tokens == 1);
        assert(!result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11, 12}));
        assert(calls == 2);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10}, 0, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(result);
        assert(result.value().proposed_draft_tokens == 0);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11}));
        assert(calls == 1);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10, .candidate_tokens = {11, 12}},
            1, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
        assert(calls == 0);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 128, .candidate_tokens = {}},
            0, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
        assert(calls == 0);
    }
    {
        std::size_t calls = 0;
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10, .candidate_tokens = {128}},
            1, 128,
            [&](std::uint32_t input) { return next_token(input, calls); });
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
        assert(calls == 0);
    }
    {
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10}, 0, 128,
            [](std::uint32_t) {
                return k3x::Result<std::uint32_t>::failure(
                    k3x::ErrorCode::backend_unavailable, "target failed");
            });
        assert(!result);
        assert(result.error() == k3x::ErrorCode::backend_unavailable);
        assert(result.message() == "target failed");
    }
    {
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10}, 0, 128,
            [](std::uint32_t) {
                return k3x::Result<std::uint32_t>::success(128);
            });
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_state);
    }
    {
        auto result = k3x::verify_greedy_draft(
            DraftProposal{.anchor_token = 10}, 0, 128,
            k3x::GreedyTargetStep{});
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_state);
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11, 12}},
            2, 128, std::vector<std::uint32_t>{11, 12, 13});
        assert(result);
        assert(result.value().accepted_draft_tokens == 2);
        assert(result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11, 12, 13}));
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {99, 100}},
            2, 128, std::vector<std::uint32_t>{11, 12, 13});
        assert(result);
        assert(result.value().accepted_draft_tokens == 0);
        assert(!result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11}));
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11, 99}},
            2, 128, std::vector<std::uint32_t>{11, 12, 13});
        assert(result);
        assert(result.value().accepted_draft_tokens == 1);
        assert(!result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11, 12}));
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11, 12, 99}},
            3, 128,
            std::vector<std::uint32_t>{11, 12, 13, 14});
        assert(result);
        assert(result.value().accepted_draft_tokens == 2);
        assert(!result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11, 12, 13}));
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10}, 0, 128,
            std::vector<std::uint32_t>{11});
        assert(result);
        assert(result.value().proposed_draft_tokens == 0);
        assert(result.value().all_draft_tokens_accepted);
        assert((result.value().committed_tokens ==
                std::vector<std::uint32_t>{11}));
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11, 12}},
            2, 128, std::vector<std::uint32_t>{11, 12});
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11, 12}},
            2, 128,
            std::vector<std::uint32_t>{11, 12, 13, 14});
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {128}},
            1, 128, std::vector<std::uint32_t>{11, 12});
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::verify_greedy_target_block(
            DraftProposal{.anchor_token = 10,
                          .candidate_tokens = {11}},
            1, 128, std::vector<std::uint32_t>{11, 128});
        assert(!result);
        assert(result.error() == k3x::ErrorCode::invalid_state);
    }
}
