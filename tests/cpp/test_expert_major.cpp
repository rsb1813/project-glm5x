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
    using k3x::ExpertMajorPackedBatch;
    using k3x::ExpertMajorPackedPlan;
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
    {
        const std::vector<float> hidden{
            1.0F, 2.0F, 3.0F,
            4.0F, 5.0F, 6.0F,
        };
        const std::vector<ExpertMajorTokenRoute> routes{
            {{2, 1}, {0.6F, 0.4F}},
            {{1}, {1.0F}},
        };
        const auto result = k3x::build_expert_major_packed_plan(
            hidden, 2, 3, routes);
        assert(result);
        const ExpertMajorPackedPlan& packed = result.value();
        assert(packed.hidden_size == 3);
        assert(packed.assignment_count == 3);
        assert(packed.groups.size() == 2);
        assert(packed.groups[0].expert_id == 2);
        assert(packed.groups[0].inputs == std::vector<float>({1.0F, 2.0F, 3.0F}));
        assert(packed.groups[1].expert_id == 1);
        assert(packed.groups[1].inputs ==
               std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

        const auto batches = k3x::bucket_expert_major_packed_plan(packed);
        assert(batches);
        assert(batches.value().size() == 2);
        const ExpertMajorPackedBatch& first = batches.value()[0];
        assert(first.token_count == 1);
        assert(first.group_indices == std::vector<std::size_t>({0}));
        assert(first.inputs == std::vector<float>({1.0F, 2.0F, 3.0F}));
        const ExpertMajorPackedBatch& second = batches.value()[1];
        assert(second.token_count == 2);
        assert(second.group_indices == std::vector<std::size_t>({1}));
        assert(second.inputs ==
               std::vector<float>({1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F}));

        auto repeated_shape = packed;
        repeated_shape.groups.push_back(packed.groups[0]);
        ++repeated_shape.assignment_count;
        const auto repeated =
            k3x::bucket_expert_major_packed_plan(repeated_shape);
        assert(repeated);
        assert(repeated.value().size() == 2);
        assert(repeated.value()[0].group_indices ==
               std::vector<std::size_t>({0, 2}));
        assert(repeated.value()[0].inputs ==
               std::vector<float>({1.0F, 2.0F, 3.0F,
                                   1.0F, 2.0F, 3.0F}));

        auto malformed = packed;
        malformed.groups[1].inputs.pop_back();
        const auto rejected = k3x::bucket_expert_major_packed_plan(malformed);
        assert(!rejected);
        assert(rejected.error() == ErrorCode::invalid_extent);
    }
}
