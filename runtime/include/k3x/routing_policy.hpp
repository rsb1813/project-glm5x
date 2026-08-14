// 전체 router 점수에서 자연 및 축소 Top-K 결정을 생성합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace k3x {
enum class RoutingMode { natural, fixed, adaptive };

struct RoutingPolicyConfig {
    RoutingMode mode{RoutingMode::natural};
    std::size_t fixed_k{};
    float mass_target{0.9F};
    float minimum_boundary_gap{};
    std::size_t quality_floor_k{};
};

struct RoutingDecision {
    std::vector<std::size_t> full_order;
    std::vector<std::size_t> expert_ids;
    std::vector<float> normalized_weights;
    std::size_t natural_top_k{};
    std::size_t selected_k{};
    float normalized_entropy{};
    float entropy_effective_support{};
    float selected_cumulative_mass{};
    float boundary_confidence{};
    bool quality_floor_escalated{};
};

Result<RoutingDecision> select_routing(
    std::span<const float> scores,
    std::span<const float> correction_bias,
    std::size_t natural_top_k,
    const RoutingPolicyConfig& config);
}
