// 전체 router 점수를 검증하고 결정론적 Top-K 정책을 적용합니다.
#include "k3x/routing_policy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

namespace k3x {
namespace {
constexpr std::array<std::size_t, 5> k_ladder{4, 6, 8, 12, 16};
constexpr float comparison_epsilon = 1.0e-5F;

bool allowed_k(std::size_t value) {
    return std::find(k_ladder.begin(), k_ladder.end(), value) !=
           k_ladder.end();
}

float boundary_confidence(std::span<const float> natural_probabilities,
                          std::size_t selected_k) {
    if (selected_k >= natural_probabilities.size()) return 1.0F;
    const auto gap = std::max(
        0.0F, natural_probabilities[selected_k - 1] -
                  natural_probabilities[selected_k]);
    return gap / std::max(natural_probabilities.front(),
                          std::numeric_limits<float>::min());
}
}

Result<RoutingDecision> select_routing(
    std::span<const float> scores,
    std::span<const float> correction_bias,
    std::size_t natural_top_k,
    const RoutingPolicyConfig& config) {
    const auto fail = [](const char* message) {
        return Result<RoutingDecision>::failure(
            ErrorCode::invalid_state, message);
    };
    if (scores.empty() || scores.size() != correction_bias.size() ||
        natural_top_k == 0 || natural_top_k > scores.size()) {
        return fail("invalid routing dimensions");
    }
    for (std::size_t index = 0; index < scores.size(); ++index) {
        if (!std::isfinite(scores[index]) || scores[index] < 0.0F ||
            scores[index] > 1.0F ||
            !std::isfinite(correction_bias[index])) {
            return fail("invalid routing score");
        }
    }
    if (!std::isfinite(config.mass_target) || config.mass_target <= 0.0F ||
        config.mass_target > 1.0F ||
        !std::isfinite(config.minimum_boundary_gap) ||
        config.minimum_boundary_gap < 0.0F ||
        config.minimum_boundary_gap > 1.0F) {
        return fail("invalid adaptive routing threshold");
    }
    if (config.quality_floor_k != 0 &&
        (!allowed_k(config.quality_floor_k) ||
         (config.mode != RoutingMode::natural &&
          config.quality_floor_k > natural_top_k))) {
        return fail("invalid routing quality floor");
    }
    if (config.mode == RoutingMode::fixed &&
        (!allowed_k(config.fixed_k) || config.fixed_k > natural_top_k)) {
        return fail("invalid fixed routing K");
    }
    if (config.mode == RoutingMode::adaptive && natural_top_k != 16) {
        return fail("adaptive routing requires natural Top-16");
    }

    RoutingDecision decision;
    decision.natural_top_k = natural_top_k;
    decision.full_order.resize(scores.size());
    std::vector<float> adjusted(scores.size());
    for (std::size_t index = 0; index < scores.size(); ++index) {
        adjusted[index] = scores[index] + correction_bias[index];
        if (!std::isfinite(adjusted[index])) {
            return fail("invalid adjusted routing score");
        }
    }
    std::iota(decision.full_order.begin(), decision.full_order.end(), 0);
    std::stable_sort(
        decision.full_order.begin(), decision.full_order.end(),
        [&](std::size_t left, std::size_t right) {
            return adjusted[left] > adjusted[right];
        });

    float natural_denominator = 0.0F;
    for (std::size_t slot = 0; slot < natural_top_k; ++slot) {
        natural_denominator += scores[decision.full_order[slot]];
    }
    if (!std::isfinite(natural_denominator) || natural_denominator <= 0.0F) {
        return fail("zero natural routing mass");
    }
    std::vector<float> natural_probabilities(natural_top_k);
    float entropy = 0.0F;
    for (std::size_t slot = 0; slot < natural_top_k; ++slot) {
        const auto probability =
            scores[decision.full_order[slot]] / natural_denominator;
        natural_probabilities[slot] = probability;
        if (probability > 0.0F) entropy -= probability * std::log(probability);
    }
    decision.normalized_entropy = natural_top_k == 1
        ? 0.0F
        : entropy / std::log(static_cast<float>(natural_top_k));
    decision.entropy_effective_support = std::exp(entropy);

    std::size_t selected_k = natural_top_k;
    std::size_t unfloored_selected_k = natural_top_k;
    if (config.mode == RoutingMode::fixed) {
        unfloored_selected_k = config.fixed_k;
        selected_k = std::max(config.fixed_k, config.quality_floor_k);
    } else if (config.mode == RoutingMode::adaptive) {
        const auto choose_adaptive = [&](std::size_t quality_floor) {
            float cumulative = 0.0F;
            for (std::size_t slot = 0; slot < natural_top_k; ++slot) {
                cumulative += natural_probabilities[slot];
                const auto candidate = slot + 1;
                if (!allowed_k(candidate) || candidate < quality_floor) continue;
                const auto confidence =
                    boundary_confidence(natural_probabilities, candidate);
                if (cumulative + comparison_epsilon >= config.mass_target &&
                    static_cast<float>(candidate) + comparison_epsilon >=
                        decision.entropy_effective_support &&
                    confidence + comparison_epsilon >=
                        config.minimum_boundary_gap) {
                    return candidate;
                }
            }
            return natural_top_k;
        };
        unfloored_selected_k = choose_adaptive(0);
        selected_k = choose_adaptive(config.quality_floor_k);
    }

    decision.selected_k = selected_k;
    decision.expert_ids.assign(decision.full_order.begin(),
                               decision.full_order.begin() + selected_k);
    decision.selected_cumulative_mass = std::accumulate(
        natural_probabilities.begin(),
        natural_probabilities.begin() + selected_k, 0.0F);
    decision.boundary_confidence =
        boundary_confidence(natural_probabilities, selected_k);
    decision.quality_floor_escalated = selected_k > unfloored_selected_k;
    float selected_denominator = 0.0F;
    for (const auto expert : decision.expert_ids) {
        selected_denominator += scores[expert];
    }
    if (!std::isfinite(selected_denominator) || selected_denominator <= 0.0F) {
        return fail("zero selected routing mass");
    }
    decision.normalized_weights.reserve(selected_k);
    for (const auto expert : decision.expert_ids) {
        decision.normalized_weights.push_back(scores[expert] /
                                              selected_denominator);
    }
    return Result<RoutingDecision>::success(std::move(decision));
}
}
