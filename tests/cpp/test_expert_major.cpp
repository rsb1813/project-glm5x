// expert-major 검증 블록의 안정적인 expert grouping 계약을 검사합니다.
#include "k3x/expert_major.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                       \
            throw std::runtime_error(                                             \
                "expert-major requirement failed: " #condition);              \
        }                                                                        \
    } while (false)

int main() {
    using k3x::ErrorCode;
    using k3x::ExpertMajorTokenRoute;

    {
        const std::vector<ExpertMajorTokenRoute> routes{
            {{2, 1}, {0.6F, 0.4F}},
            {{1, 3}, {0.7F, 0.3F}},
        };
        const auto result = k3x::build_expert_major_plan(routes);
        assert(result);
        assert(result.value().assignment_count == 4);
        assert(result.value().groups.size() == 3);
        assert(result.value().groups[0].expert_id == 2);
        assert(result.value().groups[0].assignments.size() == 1);
        assert(result.value().groups[0].assignments[0].token_index == 0);
        assert(result.value().groups[0].assignments[0].router_slot == 0);
        assert(result.value().groups[0].assignments[0].contribution == 0.6F);
        assert(result.value().groups[1].expert_id == 1);
        assert(result.value().groups[1].assignments.size() == 2);
        assert(result.value().groups[1].assignments[0].token_index == 0);
        assert(result.value().groups[1].assignments[0].router_slot == 1);
        assert(result.value().groups[1].assignments[0].contribution == 0.4F);
        assert(result.value().groups[1].assignments[1].token_index == 1);
        assert(result.value().groups[1].assignments[1].router_slot == 0);
        assert(result.value().groups[1].assignments[1].contribution == 0.7F);
        assert(result.value().groups[2].expert_id == 3);
        assert(result.value().groups[2].assignments.size() == 1);
        assert(result.value().groups[2].assignments[0].token_index == 1);
        assert(result.value().groups[2].assignments[0].router_slot == 1);
        assert(result.value().groups[2].assignments[0].contribution == 0.3F);
    }
    {
        const std::vector<ExpertMajorTokenRoute> routes{
            {{4}, {1.0F}},
            {{5}, {1.0F}},
        };
        const auto result = k3x::build_expert_major_plan(routes);
        assert(result);
        assert(result.value().groups.size() == 2);
        assert(result.value().groups[0].expert_id == 4);
        assert(result.value().groups[1].expert_id == 5);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{{{}, {}}});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{{{1, 2}, {1.0F}}});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_extent);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{{{1, 1}, {0.5F, 0.5F}}});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_state);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{
                {{1}, {std::numeric_limits<float>::quiet_NaN()}}});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_state);
    }
    {
        const auto result = k3x::build_expert_major_plan(
            std::vector<ExpertMajorTokenRoute>{
                {{1}, {std::numeric_limits<float>::infinity()}}});
        assert(!result);
        assert(result.error() == ErrorCode::invalid_state);
    }
}
