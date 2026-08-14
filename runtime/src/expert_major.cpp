// expert-major 검증 블록의 stable first-use grouping을 구현합니다.
#include "k3x/expert_major.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace k3x {
Result<ExpertMajorPlan> build_expert_major_plan(
    std::span<const ExpertMajorTokenRoute> routes) {
    if (routes.empty()) {
        return Result<ExpertMajorPlan>::failure(ErrorCode::invalid_extent);
    }

    ExpertMajorPlan plan;
    std::unordered_map<std::uint32_t, std::size_t> group_indices;
    for (std::size_t token_index = 0; token_index < routes.size();
         ++token_index) {
        const auto& route = routes[token_index];
        if (route.expert_ids.empty() ||
            route.expert_ids.size() != route.contributions.size()) {
            return Result<ExpertMajorPlan>::failure(
                ErrorCode::invalid_extent);
        }

        std::unordered_set<std::uint32_t> token_experts;
        for (std::size_t router_slot = 0;
             router_slot < route.expert_ids.size(); ++router_slot) {
            const auto expert_id = route.expert_ids[router_slot];
            const auto contribution = route.contributions[router_slot];
            if (!std::isfinite(contribution) ||
                !token_experts.insert(expert_id).second) {
                return Result<ExpertMajorPlan>::failure(
                    ErrorCode::invalid_state);
            }

            auto [group, inserted] = group_indices.emplace(
                expert_id, plan.groups.size());
            if (inserted) {
                plan.groups.push_back(
                    ExpertMajorGroup{.expert_id = expert_id});
            }
            plan.groups[group->second].assignments.push_back(
                ExpertMajorAssignment{
                    .token_index = token_index,
                    .router_slot = router_slot,
                    .contribution = contribution,
                });
            ++plan.assignment_count;
        }
    }
    return Result<ExpertMajorPlan>::success(std::move(plan));
}

Result<ExpertMajorPackedPlan> build_expert_major_packed_plan(
    std::span<const float> token_hidden, std::size_t token_count,
    std::size_t hidden_size,
    std::span<const ExpertMajorTokenRoute> routes) {
    if (token_count == 0 || hidden_size == 0 || routes.size() != token_count ||
        hidden_size > std::numeric_limits<std::size_t>::max() / token_count ||
        token_hidden.size() != token_count * hidden_size) {
        return Result<ExpertMajorPackedPlan>::failure(
            ErrorCode::invalid_extent);
    }
    const auto plan = build_expert_major_plan(routes);
    if (!plan) {
        return Result<ExpertMajorPackedPlan>::failure(
            plan.error(), plan.message());
    }
    ExpertMajorPackedPlan packed{
        .hidden_size = hidden_size,
        .assignment_count = plan.value().assignment_count,
    };
    packed.groups.reserve(plan.value().groups.size());
    for (const auto& group : plan.value().groups) {
        if (group.assignments.size() >
            std::numeric_limits<std::size_t>::max() / hidden_size) {
            return Result<ExpertMajorPackedPlan>::failure(
                ErrorCode::invalid_extent);
        }
        ExpertMajorPackedGroup packed_group{
            .expert_id = group.expert_id,
            .assignments = group.assignments,
        };
        packed_group.inputs.reserve(group.assignments.size() * hidden_size);
        for (const auto& assignment : group.assignments) {
            const auto offset = assignment.token_index * hidden_size;
            packed_group.inputs.insert(
                packed_group.inputs.end(), token_hidden.begin() + offset,
                token_hidden.begin() + offset + hidden_size);
        }
        packed.groups.push_back(std::move(packed_group));
    }
    return Result<ExpertMajorPackedPlan>::success(std::move(packed));
}
}
