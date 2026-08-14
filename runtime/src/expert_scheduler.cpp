// Bounded expert load queue를 deadline slack 순서로 실행합니다.
#include "k3x/expert_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace k3x {
namespace detail {
struct ExpertLoadState {
    std::mutex mutex;
    std::condition_variable condition;
    std::optional<Result<ExpertPayloadHandle>> result;
    std::chrono::steady_clock::time_point estimated_use_time;
    bool consumed{};
};

struct ExpertLoadMetrics {
    std::atomic<std::uint64_t> submissions{};
    std::atomic<std::uint64_t> inline_resident_hits{};
    std::atomic<std::uint64_t> completions{};
    std::atomic<std::uint64_t> ready_before_use{};
    std::atomic<std::uint64_t> late_at_use{};
    std::atomic<std::uint64_t> estimated_deadline_misses{};
    std::atomic<std::uint64_t> requested_bytes{};
    std::atomic<std::uint64_t> queue_high_water{};
    std::atomic<std::uint64_t> worker_nanoseconds{};
    std::atomic<std::uint64_t> exposed_wait_nanoseconds{};
};
}

namespace {
using Clock = std::chrono::steady_clock;

Result<ExpertPayloadHandle> invoke_loader(const ExpertLoadFunction& loader) {
    try {
        return loader();
    } catch (const std::exception& error) {
        return Result<ExpertPayloadHandle>::failure(
            ErrorCode::io_error, error.what());
    } catch (...) {
        return Result<ExpertPayloadHandle>::failure(
            ErrorCode::io_error, "expert loader threw an unknown exception");
    }
}

void update_high_water(std::atomic<std::uint64_t>& target,
                       std::uint64_t candidate) {
    auto current = target.load();
    while (current < candidate &&
           !target.compare_exchange_weak(current, candidate)) {
    }
}
}

ExpertLoadTicket::ExpertLoadTicket(
    std::shared_ptr<detail::ExpertLoadState> state,
    std::shared_ptr<detail::ExpertLoadMetrics> metrics)
    : state_(std::move(state)), metrics_(std::move(metrics)) {}

bool ExpertLoadTicket::ready() const {
    if (!state_) return false;
    std::lock_guard lock(state_->mutex);
    return state_->result.has_value();
}

Result<ExpertPayloadHandle> ExpertLoadTicket::wait() {
    if (!state_ || !metrics_) {
        return Result<ExpertPayloadHandle>::failure(ErrorCode::invalid_state);
    }
    const auto start = Clock::now();
    std::unique_lock lock(state_->mutex);
    if (state_->consumed) {
        return Result<ExpertPayloadHandle>::failure(ErrorCode::invalid_state);
    }
    state_->consumed = true;
    const bool was_ready = state_->result.has_value();
    state_->condition.wait(lock, [&] { return state_->result.has_value(); });
    auto result = std::move(*state_->result);
    lock.unlock();

    if (was_ready) {
        ++metrics_->ready_before_use;
    } else {
        ++metrics_->late_at_use;
    }
    metrics_->exposed_wait_nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - start).count());
    return result;
}

struct DeadlineExpertLoader::Impl {
    struct Job {
        Clock::time_point latest_start;
        Clock::time_point estimated_use_time;
        std::uint64_t sequence{};
        ExpertLoadFunction loader;
        std::shared_ptr<detail::ExpertLoadState> state;
    };

    struct LaterDeadline {
        bool operator()(const Job& left, const Job& right) const {
            if (left.latest_start != right.latest_start) {
                return left.latest_start > right.latest_start;
            }
            return left.sequence > right.sequence;
        }
    };

    explicit Impl(std::size_t capacity)
        : maximum_pending(capacity),
          metrics(std::make_shared<detail::ExpertLoadMetrics>()),
          worker([this] { run(); }) {}

    ~Impl() {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        condition.notify_all();
        if (worker.joinable()) worker.join();
    }

    void run() {
        for (;;) {
            Job job;
            {
                std::unique_lock lock(mutex);
                condition.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                job = queue.top();
                queue.pop();
                active = true;
            }
            const auto start = Clock::now();
            auto result = invoke_loader(job.loader);
            const auto completed = Clock::now();
            metrics->worker_nanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    completed - start).count());
            if (completed > job.estimated_use_time) {
                ++metrics->estimated_deadline_misses;
            }
            {
                std::lock_guard lock(job.state->mutex);
                job.state->result.emplace(std::move(result));
            }
            ++metrics->completions;
            job.state->condition.notify_all();
            {
                std::lock_guard lock(mutex);
                active = false;
            }
            idle_condition.notify_all();
        }
    }

    std::size_t maximum_pending;
    std::shared_ptr<detail::ExpertLoadMetrics> metrics;
    std::mutex mutex;
    std::condition_variable condition;
    std::condition_variable idle_condition;
    std::priority_queue<Job, std::vector<Job>, LaterDeadline> queue;
    std::uint64_t next_sequence{};
    bool stopping{};
    bool active{};
    std::thread worker;
};

DeadlineExpertLoader::DeadlineExpertLoader(std::size_t maximum_pending)
    : impl_() {
    if (maximum_pending == 0) {
        throw std::invalid_argument("maximum pending loads must be positive");
    }
    impl_ = std::make_unique<Impl>(maximum_pending);
}

DeadlineExpertLoader::~DeadlineExpertLoader() = default;

Result<ExpertLoadTicket> DeadlineExpertLoader::submit(
    ExpertLoadMetadata metadata, ExpertLoadFunction loader) {
    if (!loader) {
        return Result<ExpertLoadTicket>::failure(ErrorCode::invalid_state);
    }
    auto state = std::make_shared<detail::ExpertLoadState>();
    state->estimated_use_time = metadata.estimated_use_time;

    if (metadata.resident) {
        ++impl_->metrics->submissions;
        ++impl_->metrics->inline_resident_hits;
        impl_->metrics->requested_bytes += metadata.payload_bytes;
        auto result = invoke_loader(loader);
        if (Clock::now() > metadata.estimated_use_time) {
            ++impl_->metrics->estimated_deadline_misses;
        }
        {
            std::lock_guard lock(state->mutex);
            state->result.emplace(std::move(result));
        }
        ++impl_->metrics->completions;
        state->condition.notify_all();
        return Result<ExpertLoadTicket>::success(
            ExpertLoadTicket(state, impl_->metrics));
    }

    {
        std::lock_guard lock(impl_->mutex);
        if (impl_->stopping || impl_->queue.size() >= impl_->maximum_pending) {
            return Result<ExpertLoadTicket>::failure(
                ErrorCode::storage_unavailable, "expert load queue is full");
        }
        const auto latest_start =
            metadata.estimated_use_time - metadata.estimated_fetch_latency;
        impl_->queue.push({latest_start, metadata.estimated_use_time,
                           impl_->next_sequence++, std::move(loader), state});
        ++impl_->metrics->submissions;
        impl_->metrics->requested_bytes += metadata.payload_bytes;
        update_high_water(impl_->metrics->queue_high_water,
                          static_cast<std::uint64_t>(impl_->queue.size()));
    }
    impl_->condition.notify_one();
    return Result<ExpertLoadTicket>::success(
        ExpertLoadTicket(state, impl_->metrics));
}

void DeadlineExpertLoader::wait_idle() {
    std::unique_lock lock(impl_->mutex);
    impl_->idle_condition.wait(
        lock, [&] { return impl_->queue.empty() && !impl_->active; });
}

ExpertLoadSchedulerStats DeadlineExpertLoader::stats() const {
    const auto& source = *impl_->metrics;
    return {
        source.submissions.load(),
        source.inline_resident_hits.load(),
        source.completions.load(),
        source.ready_before_use.load(),
        source.late_at_use.load(),
        source.estimated_deadline_misses.load(),
        source.requested_bytes.load(),
        source.queue_high_water.load(),
        source.worker_nanoseconds.load(),
        source.exposed_wait_nanoseconds.load(),
    };
}
}
