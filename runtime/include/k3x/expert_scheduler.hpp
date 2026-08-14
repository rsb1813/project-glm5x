// Expert payload load를 deadline 우선순위로 직렬 실행하는 scheduler를 선언합니다.
#pragma once

#include "k3x/host_expert_store.hpp"
#include "k3x/status.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace k3x {
struct ExpertLoadMetadata {
    std::chrono::steady_clock::time_point estimated_use_time;
    std::chrono::nanoseconds estimated_fetch_latency{};
    std::uint64_t payload_bytes{};
    bool resident{};
};

struct ExpertLoadSchedulerStats {
    std::uint64_t submissions{};
    std::uint64_t inline_resident_hits{};
    std::uint64_t completions{};
    std::uint64_t ready_before_use{};
    std::uint64_t late_at_use{};
    std::uint64_t estimated_deadline_misses{};
    std::uint64_t requested_bytes{};
    std::uint64_t queue_high_water{};
    std::uint64_t worker_nanoseconds{};
    std::uint64_t exposed_wait_nanoseconds{};
};

namespace detail {
struct ExpertLoadState;
struct ExpertLoadMetrics;
}

class ExpertLoadTicket {
public:
    ExpertLoadTicket() = default;
    ExpertLoadTicket(ExpertLoadTicket&&) noexcept = default;
    ExpertLoadTicket& operator=(ExpertLoadTicket&&) noexcept = default;
    ExpertLoadTicket(const ExpertLoadTicket&) = delete;
    ExpertLoadTicket& operator=(const ExpertLoadTicket&) = delete;

    bool ready() const;
    Result<ExpertPayloadHandle> wait();

private:
    friend class DeadlineExpertLoader;
    ExpertLoadTicket(std::shared_ptr<detail::ExpertLoadState> state,
                     std::shared_ptr<detail::ExpertLoadMetrics> metrics);
    std::shared_ptr<detail::ExpertLoadState> state_;
    std::shared_ptr<detail::ExpertLoadMetrics> metrics_;
};

using ExpertLoadFunction =
    std::function<Result<ExpertPayloadHandle>()>;

class DeadlineExpertLoader {
public:
    explicit DeadlineExpertLoader(std::size_t maximum_pending);
    ~DeadlineExpertLoader();
    DeadlineExpertLoader(const DeadlineExpertLoader&) = delete;
    DeadlineExpertLoader& operator=(const DeadlineExpertLoader&) = delete;
    DeadlineExpertLoader(DeadlineExpertLoader&&) = delete;
    DeadlineExpertLoader& operator=(DeadlineExpertLoader&&) = delete;

    Result<ExpertLoadTicket> submit(ExpertLoadMetadata metadata,
                                    ExpertLoadFunction loader);
    void wait_idle();
    ExpertLoadSchedulerStats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
