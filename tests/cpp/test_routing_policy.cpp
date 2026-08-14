// 자연 및 축소 Top-K 선택 정책의 결정론적 계약을 검증합니다.
#include "k3x/routing_policy.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                       \
            throw std::runtime_error("routing requirement failed: " #condition); \
        }                                                                        \
    } while (false)

namespace {
bool close(float left, float right, float tolerance = 1.0e-6F) {
    return std::abs(left - right) <= tolerance;
}
}

int main() {
    using k3x::RoutingMode;
    using k3x::RoutingPolicyConfig;

    const std::array natural_scores{0.8F, 0.8F, 0.4F, 0.7F, 0.2F, 0.1F};
    const std::array natural_bias{0.0F, 0.0F, 0.5F, 0.0F, 0.0F, 0.0F};
    auto natural = k3x::select_routing(
        natural_scores, natural_bias, 4,
        RoutingPolicyConfig{.mode = RoutingMode::natural});
    assert(natural);
    assert((natural.value().full_order ==
            std::vector<std::size_t>{2, 0, 1, 3, 4, 5}));
    assert((natural.value().expert_ids ==
            std::vector<std::size_t>{2, 0, 1, 3}));
    assert(natural.value().selected_k == 4);
    assert(natural.value().normalized_weights.size() == 4);
    assert(close(natural.value().normalized_weights[0], 0.4F / 2.7F));
    assert(close(natural.value().normalized_weights[1], 0.8F / 2.7F));
    assert(close(natural.value().normalized_weights[2], 0.8F / 2.7F));
    assert(close(natural.value().normalized_weights[3], 0.7F / 2.7F));
    assert(close(natural.value().selected_cumulative_mass, 1.0F));
    auto natural_with_floor = k3x::select_routing(
        natural_scores, natural_bias, 4,
        RoutingPolicyConfig{.mode = RoutingMode::natural,
                            .quality_floor_k = 16});
    assert(natural_with_floor);
    assert(natural_with_floor.value().selected_k == 4);
    assert(!natural_with_floor.value().quality_floor_escalated);

    std::array<float, 16> descending{};
    std::array<float, 16> zero_bias{};
    for (std::size_t index = 0; index < descending.size(); ++index) {
        descending[index] = 1.0F - static_cast<float>(index) * 0.04F;
    }
    auto fixed = k3x::select_routing(
        descending, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::fixed, .fixed_k = 4});
    assert(fixed);
    assert(fixed.value().selected_k == 4);
    assert((fixed.value().expert_ids ==
            std::vector<std::size_t>{0, 1, 2, 3}));
    float fixed_sum = 0.0F;
    for (const auto weight : fixed.value().normalized_weights) fixed_sum += weight;
    assert(close(fixed_sum, 1.0F));
    assert(fixed.value().selected_cumulative_mass < 1.0F);

    std::array<float, 16> eight_supported{};
    eight_supported.fill(1.0e-9F);
    for (std::size_t index = 0; index < 8; ++index) eight_supported[index] = 1.0F;
    auto adaptive = k3x::select_routing(
        eight_supported, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::adaptive,
                            .mass_target = 0.75F,
                            .minimum_boundary_gap = 0.1F});
    assert(adaptive);
    assert(adaptive.value().selected_k == 8);
    assert(adaptive.value().entropy_effective_support <= 8.0001F);
    assert(adaptive.value().boundary_confidence > 0.9F);

    auto escalated = k3x::select_routing(
        eight_supported, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::adaptive,
                            .mass_target = 0.75F,
                            .minimum_boundary_gap = 0.0F,
                            .quality_floor_k = 12});
    assert(escalated);
    assert(escalated.value().selected_k == 12);
    assert(escalated.value().quality_floor_escalated);

    std::array<float, 16> weak_boundary{};
    weak_boundary.fill(1.0e-9F);
    for (std::size_t index = 0; index < 8; ++index) weak_boundary[index] = 1.0F;
    for (std::size_t index = 8; index < 13; ++index) weak_boundary[index] = 0.1F;
    auto boundary_rejected = k3x::select_routing(
        weak_boundary, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::adaptive,
                            .mass_target = 0.8F,
                            .minimum_boundary_gap = 0.2F});
    assert(boundary_rejected);
    assert(boundary_rejected.value().selected_k == 16);

    assert(!k3x::select_routing(
        descending, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::fixed, .fixed_k = 2}));
    assert(!k3x::select_routing(
        descending, zero_bias, 12,
        RoutingPolicyConfig{.mode = RoutingMode::fixed, .fixed_k = 16}));
    assert(!k3x::select_routing(
        descending, std::span<const float>{zero_bias}.first(15), 16,
        RoutingPolicyConfig{.mode = RoutingMode::natural}));
    auto invalid_threshold = RoutingPolicyConfig{.mode = RoutingMode::adaptive};
    invalid_threshold.mass_target = std::numeric_limits<float>::quiet_NaN();
    assert(!k3x::select_routing(
        descending, zero_bias, 16, invalid_threshold));
    auto out_of_range_scores = descending;
    out_of_range_scores[0] = 1.1F;
    assert(!k3x::select_routing(
        out_of_range_scores, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::natural}));
    std::array<float, 16> zero_scores{};
    assert(!k3x::select_routing(
        zero_scores, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::natural}));
    assert(!k3x::select_routing(
        descending, zero_bias, 16,
        RoutingPolicyConfig{.mode = RoutingMode::adaptive,
                            .quality_floor_k = 5}));
    return 0;
}
