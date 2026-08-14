// expert-major 검증 블록의 안정적인 expert grouping 계약을 선언합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace k3x {
struct ExpertMajorTokenRoute {
    std::vector<std::uint32_t> expert_ids;
    std::vector<float> contributions;
};

struct ExpertMajorAssignment {
    std::size_t token_index{};
    std::size_t router_slot{};
    float contribution{};
};

struct ExpertMajorGroup {
    std::uint32_t expert_id{};
    std::vector<ExpertMajorAssignment> assignments;
};

struct ExpertMajorPlan {
    std::vector<ExpertMajorGroup> groups;
    std::size_t assignment_count{};
};

Result<ExpertMajorPlan> build_expert_major_plan(
    std::span<const ExpertMajorTokenRoute> routes);
}
