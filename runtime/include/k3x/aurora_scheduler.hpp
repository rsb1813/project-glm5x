// AURORA draft 길이를 acceptance와 expert 비용으로 선택합니다.
#pragma once

#include "k3x/speculative.hpp"
#include "k3x/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace k3x {
enum class AuroraBlockPolicy { fixed, adaptive };

struct AuroraSchedulerConfig {
    AuroraBlockPolicy policy{AuroraBlockPolicy::fixed};
    std::size_t maximum_length{1};
    double minimum_prefix_survival{0.5};
    double maximum_unique_load_ratio{0.9};
};

class AdaptiveDraftScheduler {
public:
    static Result<AdaptiveDraftScheduler> create(
        AuroraSchedulerConfig config);

    Result<std::size_t> select(std::size_t request_maximum);
    Result<bool> observe(const DraftVerification& verification);
    DraftProviderStats stats() const noexcept { return stats_; }

private:
    explicit AdaptiveDraftScheduler(AuroraSchedulerConfig config)
        : config_(config) {}

    AuroraSchedulerConfig config_;
    std::array<std::uint64_t, 4> prefix_trials_{};
    std::array<std::uint64_t, 4> prefix_successes_{};
    std::array<std::uint64_t, 3> cost_loads_{};
    std::array<std::uint64_t, 3> cost_assignments_{};
    std::array<bool, 3> observed_lengths_{};
    std::optional<std::size_t> largest_qualified_rung_;
    std::optional<std::size_t> rejection_cap_;
    bool retry_smallest_rung_{};
    DraftProviderStats stats_{};
};
}
