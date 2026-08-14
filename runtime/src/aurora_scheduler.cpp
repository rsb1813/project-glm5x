// AURORA draft scheduler의 보수적 rung 탐색과 backoff를 구현합니다.
#include "k3x/aurora_scheduler.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace k3x {
namespace {
constexpr std::array<std::size_t, 3> ladder{1, 2, 4};

std::optional<std::size_t> rung_index(std::size_t length) {
    const auto found = std::find(ladder.begin(), ladder.end(), length);
    if (found == ladder.end()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(found - ladder.begin());
}

void record_selection(DraftProviderStats& stats, std::size_t length) {
    if (length == 1) {
        ++stats.selected_length_1;
    } else if (length == 2) {
        ++stats.selected_length_2;
    } else if (length == 4) {
        ++stats.selected_length_4;
    }
}

bool add_would_overflow(std::uint64_t left, std::uint64_t right) {
    return right > std::numeric_limits<std::uint64_t>::max() - left;
}
}

Result<AdaptiveDraftScheduler> AdaptiveDraftScheduler::create(
    AuroraSchedulerConfig config) {
    if (!rung_index(config.maximum_length).has_value() ||
        !std::isfinite(config.minimum_prefix_survival) ||
        config.minimum_prefix_survival < 0.0 ||
        config.minimum_prefix_survival > 1.0 ||
        !std::isfinite(config.maximum_unique_load_ratio) ||
        config.maximum_unique_load_ratio < 0.0 ||
        config.maximum_unique_load_ratio > 1.0) {
        return Result<AdaptiveDraftScheduler>::failure(
            ErrorCode::invalid_extent);
    }
    return Result<AdaptiveDraftScheduler>::success(
        AdaptiveDraftScheduler(config));
}

Result<std::size_t> AdaptiveDraftScheduler::select(
    std::size_t request_maximum) {
    if (request_maximum == 0) {
        return Result<std::size_t>::success(0);
    }
    auto effective_maximum = std::min(request_maximum, config_.maximum_length);
    if (rejection_cap_.has_value()) {
        effective_maximum = std::min(effective_maximum,
                                     rejection_cap_.value());
    }
    if (effective_maximum == 0) {
        if (request_maximum != 0 && rejection_cap_.has_value() &&
            rejection_cap_.value() == 0) {
            rejection_cap_.reset();
            retry_smallest_rung_ = true;
        }
        return Result<std::size_t>::success(0);
    }

    if (config_.policy == AuroraBlockPolicy::fixed) {
        for (auto index = ladder.size(); index-- > 0;) {
            if (ladder[index] <= effective_maximum) {
                record_selection(stats_, ladder[index]);
                return Result<std::size_t>::success(ladder[index]);
            }
        }
        return Result<std::size_t>::success(0);
    }

    if (retry_smallest_rung_ && effective_maximum >= ladder.front()) {
        retry_smallest_rung_ = false;
        record_selection(stats_, ladder.front());
        return Result<std::size_t>::success(ladder.front());
    }

    const auto maximum_rung = rung_index(config_.maximum_length).value();
    const auto exploration_rung = largest_qualified_rung_.has_value()
        ? std::min(largest_qualified_rung_.value() + 1, maximum_rung)
        : 0;
    for (auto index = exploration_rung + 1; index-- > 0;) {
        const auto length = ladder[index];
        if (length > effective_maximum) {
            continue;
        }
        if (observed_lengths_[index]) {
            const auto final_position = length - 1;
            const auto survival =
                static_cast<double>(prefix_successes_[final_position] + 1) /
                static_cast<double>(prefix_trials_[final_position] + 2);
            if (survival < config_.minimum_prefix_survival) {
                continue;
            }
            if (cost_assignments_[index] != 0) {
                const auto cost_ratio =
                    static_cast<double>(cost_loads_[index]) /
                    static_cast<double>(cost_assignments_[index]);
                if (cost_ratio > config_.maximum_unique_load_ratio) {
                    continue;
                }
            }
        }
        record_selection(stats_, length);
        return Result<std::size_t>::success(length);
    }
    return Result<std::size_t>::success(0);
}

Result<bool> AdaptiveDraftScheduler::observe(
    const DraftVerification& verification) {
    if (verification.proposed_draft_tokens == 0) {
        if (verification.accepted_draft_tokens != 0 ||
            verification.expert_major_payload_loads != 0 ||
            verification.expert_major_assignments != 0 ||
            verification.target_positions_discarded >
                verification.target_positions_evaluated) {
            return Result<bool>::failure(ErrorCode::invalid_extent);
        }
        return Result<bool>::success(true);
    }
    const auto index = rung_index(verification.proposed_draft_tokens);
    if (!index.has_value() ||
        verification.proposed_draft_tokens > config_.maximum_length ||
        verification.accepted_draft_tokens >
            verification.proposed_draft_tokens ||
        verification.target_positions_discarded >
            verification.target_positions_evaluated ||
        (verification.expert_major_payload_loads != 0 &&
         verification.expert_major_assignments == 0) ||
        verification.expert_major_payload_loads >
            verification.expert_major_assignments) {
        return Result<bool>::failure(ErrorCode::invalid_extent);
    }
    for (std::size_t position = 0;
         position < verification.proposed_draft_tokens; ++position) {
        if (prefix_trials_[position] ==
            std::numeric_limits<std::uint64_t>::max()) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
    }
    const auto rung = index.value();
    if (add_would_overflow(cost_loads_[rung],
                           verification.expert_major_payload_loads) ||
        add_would_overflow(cost_assignments_[rung],
                           verification.expert_major_assignments)) {
        return Result<bool>::failure(ErrorCode::invalid_state);
    }

    for (std::size_t position = 0;
         position < verification.proposed_draft_tokens; ++position) {
        ++prefix_trials_[position];
        if (verification.accepted_draft_tokens > position) {
            ++prefix_successes_[position];
        }
    }
    cost_loads_[rung] += verification.expert_major_payload_loads;
    cost_assignments_[rung] += verification.expert_major_assignments;
    observed_lengths_[rung] = true;

    if (config_.policy == AuroraBlockPolicy::fixed) {
        return Result<bool>::success(true);
    }

    if (verification.accepted_draft_tokens <
        verification.proposed_draft_tokens) {
        rejection_cap_ = rung == 0 ? 0 : ladder[rung - 1];
        ++stats_.scheduler_backoffs;
    } else {
        rejection_cap_.reset();
        const auto final_position = verification.proposed_draft_tokens - 1;
        const auto survival =
            static_cast<double>(prefix_successes_[final_position] + 1) /
            static_cast<double>(prefix_trials_[final_position] + 2);
        const auto cost_qualifies = cost_assignments_[rung] == 0 ||
            static_cast<double>(cost_loads_[rung]) /
                    static_cast<double>(cost_assignments_[rung]) <=
                config_.maximum_unique_load_ratio;
        if (survival >= config_.minimum_prefix_survival && cost_qualifies &&
            (!largest_qualified_rung_.has_value() ||
             rung > largest_qualified_rung_.value())) {
            largest_qualified_rung_ = rung;
            if (rung + 1 < ladder.size()) {
                ++stats_.scheduler_growths;
            }
        }
    }
    return Result<bool>::success(true);
}
}
