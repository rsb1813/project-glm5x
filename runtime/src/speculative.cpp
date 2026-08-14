// speculative draft prefix를 target argmax와 비교하고 exact commit을 계산합니다.
#include "k3x/speculative.hpp"

#include <utility>

namespace k3x {
namespace {
ErrorCode validate_proposal(const DraftProposal& proposal,
                            std::size_t max_draft_tokens,
                            std::size_t vocabulary_size) {
    if (vocabulary_size == 0 || proposal.anchor_token >= vocabulary_size ||
        proposal.candidate_tokens.size() > max_draft_tokens) {
        return ErrorCode::invalid_extent;
    }
    for (const auto token : proposal.candidate_tokens) {
        if (token >= vocabulary_size) {
            return ErrorCode::invalid_extent;
        }
    }
    return ErrorCode::ok;
}
}

Result<DraftVerification> verify_greedy_draft(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size, const GreedyTargetStep& target_step) {
    if (!target_step) {
        return Result<DraftVerification>::failure(ErrorCode::invalid_state);
    }
    if (const auto error = validate_proposal(
            proposal, max_draft_tokens, vocabulary_size);
        error != ErrorCode::ok) {
        return Result<DraftVerification>::failure(error);
    }

    DraftVerification verification{
        .anchor_token = proposal.anchor_token,
        .proposed_draft_tokens = proposal.candidate_tokens.size(),
    };
    auto input_token = proposal.anchor_token;
    for (const auto candidate : proposal.candidate_tokens) {
        auto target = target_step(input_token);
        if (!target) {
            return Result<DraftVerification>::failure(
                target.error(), target.message());
        }
        if (target.value() >= vocabulary_size) {
            return Result<DraftVerification>::failure(
                ErrorCode::invalid_state);
        }
        if (candidate != target.value()) {
            verification.committed_tokens.push_back(target.value());
            return Result<DraftVerification>::success(
                std::move(verification));
        }
        verification.committed_tokens.push_back(candidate);
        ++verification.accepted_draft_tokens;
        input_token = candidate;
    }

    auto bonus = target_step(input_token);
    if (!bonus) {
        return Result<DraftVerification>::failure(
            bonus.error(), bonus.message());
    }
    if (bonus.value() >= vocabulary_size) {
        return Result<DraftVerification>::failure(ErrorCode::invalid_state);
    }
    verification.committed_tokens.push_back(bonus.value());
    verification.all_draft_tokens_accepted = true;
    return Result<DraftVerification>::success(std::move(verification));
}

Result<DraftVerification> verify_greedy_target_block(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size,
    std::span<const std::uint32_t> target_tokens) {
    if (const auto error = validate_proposal(
            proposal, max_draft_tokens, vocabulary_size);
        error != ErrorCode::ok) {
        return Result<DraftVerification>::failure(error);
    }
    if (target_tokens.empty() ||
        target_tokens.size() - 1 != proposal.candidate_tokens.size()) {
        return Result<DraftVerification>::failure(ErrorCode::invalid_extent);
    }
    for (const auto token : target_tokens) {
        if (token >= vocabulary_size) {
            return Result<DraftVerification>::failure(
                ErrorCode::invalid_state);
        }
    }

    DraftVerification verification{
        .anchor_token = proposal.anchor_token,
        .proposed_draft_tokens = proposal.candidate_tokens.size(),
    };
    for (std::size_t index = 0;
         index < proposal.candidate_tokens.size(); ++index) {
        if (proposal.candidate_tokens[index] != target_tokens[index]) {
            verification.committed_tokens.push_back(target_tokens[index]);
            return Result<DraftVerification>::success(
                std::move(verification));
        }
        verification.committed_tokens.push_back(
            proposal.candidate_tokens[index]);
        ++verification.accepted_draft_tokens;
    }
    verification.committed_tokens.push_back(target_tokens.back());
    verification.all_draft_tokens_accepted = true;
    return Result<DraftVerification>::success(std::move(verification));
}
}
