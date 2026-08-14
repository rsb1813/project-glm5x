// AURORA proposal 길이 scheduler의 acceptance와 expert-cost gate를 검증합니다.
#include "k3x/aurora_scheduler.hpp"

#include <source_location>
#include <stdexcept>

namespace {
void require(
    bool condition,
    const std::source_location location = std::source_location::current()) {
    if (!condition) {
        throw std::runtime_error(
            "AURORA scheduler requirement failed at line " +
            std::to_string(location.line()));
    }
}

k3x::AdaptiveDraftScheduler make_adaptive() {
    auto created = k3x::AdaptiveDraftScheduler::create({
        .policy = k3x::AuroraBlockPolicy::adaptive,
        .maximum_length = 4,
        .minimum_prefix_survival = 0.5,
        .maximum_unique_load_ratio = 0.9,
    });
    require(static_cast<bool>(created));
    return std::move(created.value());
}
}

int main() {
    {
        auto created = k3x::AdaptiveDraftScheduler::create({
            .policy = k3x::AuroraBlockPolicy::fixed,
            .maximum_length = 4,
        });
        require(static_cast<bool>(created));
        auto scheduler = std::move(created.value());
        require(scheduler.select(0).value() == 0);
        require(scheduler.select(1).value() == 1);
        require(scheduler.select(3).value() == 2);
        require(scheduler.select(4).value() == 4);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 4,
            .accepted_draft_tokens = 0,
        })));
        require(scheduler.select(4).value() == 4);
    }
    {
        auto scheduler = make_adaptive();
        require(scheduler.select(4).value() == 1);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 1,
            .expert_major_payload_loads = 8,
            .expert_major_assignments = 16,
        })));
        require(scheduler.select(4).value() == 2);
        require(scheduler.stats().scheduler_growths == 1);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 2,
            .accepted_draft_tokens = 2,
            .expert_major_payload_loads = 12,
            .expert_major_assignments = 32,
        })));
        require(scheduler.select(4).value() == 4);
        require(scheduler.stats().scheduler_growths == 2);
        require(scheduler.stats().selected_length_1 == 1);
        require(scheduler.stats().selected_length_2 == 1);
        require(scheduler.stats().selected_length_4 == 1);
    }
    {
        auto scheduler = make_adaptive();
        require(scheduler.select(4).value() == 1);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 0,
            .target_positions_evaluated = 2,
            .target_positions_discarded = 1,
        })));
        require(scheduler.select(4).value() == 0);
        require(scheduler.stats().scheduler_backoffs == 1);
        require(static_cast<bool>(scheduler.observe({})));
        require(scheduler.select(4).value() == 1);
    }
    {
        auto scheduler = make_adaptive();
        require(scheduler.select(4).value() == 1);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 1,
        })));
        require(scheduler.select(4).value() == 2);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 2,
            .accepted_draft_tokens = 1,
            .target_positions_evaluated = 3,
            .target_positions_discarded = 1,
            .expert_major_payload_loads = 20,
            .expert_major_assignments = 32,
        })));
        require(scheduler.select(4).value() == 1);
        require(scheduler.stats().scheduler_backoffs == 1);
    }
    {
        auto scheduler = make_adaptive();
        require(scheduler.select(4).value() == 1);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 1,
        })));
        require(scheduler.select(4).value() == 2);
        require(static_cast<bool>(scheduler.observe({
            .proposed_draft_tokens = 2,
            .accepted_draft_tokens = 2,
            .expert_major_payload_loads = 32,
            .expert_major_assignments = 32,
        })));
        require(scheduler.select(4).value() == 1);
    }
    {
        auto scheduler = make_adaptive();
        const auto before = scheduler.stats();
        require(!scheduler.observe({
            .proposed_draft_tokens = 2,
            .accepted_draft_tokens = 3,
        }));
        require(!scheduler.observe({
            .proposed_draft_tokens = 3,
            .accepted_draft_tokens = 0,
        }));
        require(!scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 1,
            .target_positions_evaluated = 1,
            .target_positions_discarded = 2,
        }));
        require(!scheduler.observe({
            .proposed_draft_tokens = 1,
            .accepted_draft_tokens = 1,
            .expert_major_payload_loads = 1,
        }));
        const auto after = scheduler.stats();
        require(before.scheduler_backoffs == after.scheduler_backoffs);
        require(before.scheduler_growths == after.scheduler_growths);
    }
    {
        require(!k3x::AdaptiveDraftScheduler::create({
            .policy = k3x::AuroraBlockPolicy::adaptive,
            .maximum_length = 3,
        }));
        require(!k3x::AdaptiveDraftScheduler::create({
            .policy = k3x::AuroraBlockPolicy::adaptive,
            .maximum_length = 4,
            .minimum_prefix_survival = 1.1,
        }));
        require(!k3x::AdaptiveDraftScheduler::create({
            .policy = k3x::AuroraBlockPolicy::adaptive,
            .maximum_length = 4,
            .maximum_unique_load_ratio = -0.1,
        }));
    }
    return 0;
}
