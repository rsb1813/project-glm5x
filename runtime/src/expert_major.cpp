// expert-major 검증 블록의 stable first-use grouping을 구현합니다.
#include "k3x/expert_major.hpp"

#include <cmath>
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
}
