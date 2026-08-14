// Expert load scheduler의 deadline 순서와 실패 및 수명 계약을 검증합니다.
#include "k3x/expert_scheduler.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

void require(bool condition) {
    if (!condition) throw std::runtime_error("expert scheduler requirement failed");
}

k3x::ExpertPayloadHandle payload() {
    return std::make_shared<const k3x::ExpertMlpPayload>();
}
}

int main() {
    std::mutex mutex;
    std::condition_variable condition;
    bool blocker_started = false;
    bool release_blocker = false;
    std::vector<int> order;

    k3x::DeadlineExpertLoader scheduler(4);
    const auto now = Clock::now();
    auto blocker = scheduler.submit(
        {now, std::chrono::nanoseconds{0}, 11, false}, [&] {
            {
                std::lock_guard lock(mutex);
                blocker_started = true;
                order.push_back(0);
            }
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return release_blocker; });
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(blocker));
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return blocker_started; });
    }

    auto far = scheduler.submit(
        {now + std::chrono::seconds{10}, std::chrono::milliseconds{10}, 13,
         false},
        [&] {
            std::lock_guard lock(mutex);
            order.push_back(1);
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    auto near = scheduler.submit(
        {now + std::chrono::seconds{1}, std::chrono::milliseconds{10}, 17,
         false},
        [&] {
            std::lock_guard lock(mutex);
            order.push_back(2);
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(far) && static_cast<bool>(near));
    {
        std::lock_guard lock(mutex);
        release_blocker = true;
    }
    condition.notify_all();
    require(static_cast<bool>(blocker.value().wait()));
    require(static_cast<bool>(near.value().wait()));
    require(static_cast<bool>(far.value().wait()));
    require(order == std::vector<int>({0, 2, 1}));
    auto duplicate_wait = far.value().wait();
    require(!duplicate_wait &&
            duplicate_wait.error() == k3x::ErrorCode::invalid_state);

    auto failure = scheduler.submit(
        {Clock::now(), std::chrono::nanoseconds{0}, 19, false}, [] {
            return k3x::Result<k3x::ExpertPayloadHandle>::failure(
                k3x::ErrorCode::io_error, "injected failure");
        });
    require(static_cast<bool>(failure));
    auto failed = failure.value().wait();
    require(!failed && failed.error() == k3x::ErrorCode::io_error &&
            failed.message() == "injected failure");

    auto resident = scheduler.submit(
        {Clock::now(), std::chrono::nanoseconds{0}, 23, true}, [] {
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(resident));
    require(resident.value().ready());
    require(static_cast<bool>(resident.value().wait()));

    const auto stats = scheduler.stats();
    require(stats.submissions == 5);
    require(stats.inline_resident_hits == 1);
    require(stats.completions == 5);
    require(stats.requested_bytes == 83);
    require(stats.queue_high_water >= 2);
    require(stats.ready_before_use + stats.late_at_use == 5);

    std::vector<int> tied_order;
    bool tied_started = false;
    bool release_tied = false;
    k3x::DeadlineExpertLoader tied(2);
    auto tied_blocker = tied.submit(
        {Clock::now(), std::chrono::nanoseconds{0}, 1, false}, [&] {
            {
                std::lock_guard lock(mutex);
                tied_started = true;
                tied_order.push_back(0);
            }
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return release_tied; });
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(tied_blocker));
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return tied_started; });
    }
    const auto same_deadline = Clock::now() + std::chrono::seconds{1};
    auto tied_first = tied.submit(
        {same_deadline, std::chrono::milliseconds{1}, 1, false}, [&] {
            std::lock_guard lock(mutex);
            tied_order.push_back(1);
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(tied_first));
    auto tied_second = tied.submit(
        {same_deadline, std::chrono::milliseconds{1}, 1, false}, [&] {
            std::lock_guard lock(mutex);
            tied_order.push_back(2);
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(tied_second));
    {
        std::lock_guard lock(mutex);
        release_tied = true;
    }
    condition.notify_all();
    require(static_cast<bool>(tied_blocker.value().wait()));
    require(static_cast<bool>(tied_first.value().wait()));
    require(static_cast<bool>(tied_second.value().wait()));
    require(tied_order == std::vector<int>({0, 1, 2}));

    bool full_started = false;
    bool release_full = false;
    k3x::DeadlineExpertLoader bounded(1);
    auto full_blocker = bounded.submit(
        {Clock::now(), std::chrono::nanoseconds{0}, 1, false}, [&] {
            {
                std::lock_guard lock(mutex);
                full_started = true;
            }
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return release_full; });
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(full_blocker));
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return full_started; });
    }
    auto queued = bounded.submit(
        {same_deadline, std::chrono::milliseconds{1}, 1, false}, [] {
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(queued));
    auto queue_full = bounded.submit(
        {same_deadline, std::chrono::milliseconds{1}, 1, false}, [] {
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(!queue_full &&
            queue_full.error() == k3x::ErrorCode::storage_unavailable);
    {
        std::lock_guard lock(mutex);
        release_full = true;
    }
    condition.notify_all();
    require(static_cast<bool>(full_blocker.value().wait()));
    require(static_cast<bool>(queued.value().wait()));

    bool rejected_zero_capacity = false;
    try {
        k3x::DeadlineExpertLoader invalid(0);
    } catch (const std::invalid_argument&) {
        rejected_zero_capacity = true;
    }
    require(rejected_zero_capacity);

    std::atomic<bool> drained{false};
    {
        k3x::DeadlineExpertLoader draining(1);
        auto ticket = draining.submit(
            {Clock::now(), std::chrono::nanoseconds{0}, 1, false}, [&] {
                drained.store(true);
                return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
            });
        require(static_cast<bool>(ticket));
    }
    require(drained.load());

    std::atomic<bool> idle_started{false};
    std::atomic<bool> idle_returned{false};
    bool release_idle = false;
    k3x::DeadlineExpertLoader idle_scheduler(1);
    auto idle_ticket = idle_scheduler.submit(
        {Clock::now(), std::chrono::nanoseconds{0}, 1, false}, [&] {
            idle_started.store(true);
            condition.notify_all();
            std::unique_lock lock(mutex);
            condition.wait(lock, [&] { return release_idle; });
            return k3x::Result<k3x::ExpertPayloadHandle>::success(payload());
        });
    require(static_cast<bool>(idle_ticket));
    {
        std::unique_lock lock(mutex);
        condition.wait(lock, [&] { return idle_started.load(); });
    }
    std::thread idle_waiter([&] {
        idle_scheduler.wait_idle();
        idle_returned.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    require(!idle_returned.load());
    {
        std::lock_guard lock(mutex);
        release_idle = true;
    }
    condition.notify_all();
    idle_waiter.join();
    require(idle_returned.load());
    require(static_cast<bool>(idle_ticket.value().wait()));

    return 0;
}
