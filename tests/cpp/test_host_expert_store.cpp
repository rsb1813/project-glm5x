// 영속적 L1 expert store의 원자적 admission과 exact bypass 계약을 검증합니다.
#include "k3x/host_expert_store.hpp"
#include "k3x/runtime_profile.hpp"

#include <atomic>
#include <array>
#include <barrier>
#include <concepts>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                       \
            throw std::runtime_error("test requirement failed: " #condition);   \
        }                                                                        \
    } while (false)

namespace {
k3x::ExpertProjection projection(std::uint64_t id) {
    return {id, std::vector<std::byte>(512, std::byte{0x2a}),
            std::vector<std::byte>(32, std::byte{0x7f}), 32, 32};
}

k3x::ExpertMlpPayload payload(std::uint64_t base) {
    return {projection(base), projection(base + 1), projection(base + 2)};
}
}

int main() {
    static_assert(std::same_as<
                  decltype(std::declval<const k3x::HostExpertStore&>().stats()),
                  k3x::L1ExpertCacheStats>);
    using k3x::ErrorCode;
    using k3x::ExpertKey;
    using k3x::HostExpertStore;
    using k3x::L1ExpertCacheMode;
    using k3x::Result;

    HostExpertStore store(L1ExpertCacheMode::static_admission, 3264);
    std::size_t loads = 0;
    const auto loader = [&]() {
        ++loads;
        return Result<k3x::ExpertMlpPayload>::success(payload(10));
    };
    auto first = store.get_or_load({0, 3}, loader);
    assert(first);
    assert(loads == 1);
    assert(store.stats().hits == 0);
    assert(store.stats().misses == 1);
    assert(store.stats().resident_bytes == 1632);
    assert(store.stats().peak_resident_bytes == 1632);

    auto hit = store.get_or_load({0, 3}, loader);
    assert(hit);
    assert(loads == 1);
    assert(hit.value().get() == first.value().get());
    assert(store.stats().hits == 1);
    assert(store.stats().misses == 1);

    auto exact_fit = store.get_or_load({0, 4}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(20));
    });
    assert(exact_fit);
    assert(store.stats().resident_bytes == 3264);

    auto no_room = store.get_or_load({0, 5}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(30));
    });
    assert(no_room);
    assert(store.stats().misses == 3);
    assert(store.stats().bypasses == 1);
    assert(store.stats().resident_bytes == 3264);
    assert(no_room.value().get() != first.value().get());

    HostExpertStore oversized(L1ExpertCacheMode::static_admission, 1631);
    auto bypass = oversized.get_or_load({1, 0}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(40));
    });
    assert(bypass);
    assert(oversized.stats().misses == 1);
    assert(oversized.stats().bypasses == 1);
    assert(oversized.stats().resident_bytes == 0);

    HostExpertStore disabled(L1ExpertCacheMode::disabled, 0);
    auto transient_one = disabled.get_or_load({2, 0}, loader);
    auto transient_two = disabled.get_or_load({2, 0}, loader);
    assert(transient_one && transient_two);
    assert(transient_one.value().get() != transient_two.value().get());
    assert(disabled.stats().hits == 0);
    assert(disabled.stats().misses == 0);
    assert(disabled.stats().bypasses == 0);
    assert(disabled.stats().resident_bytes == 0);

    HostExpertStore failure_atomic(L1ExpertCacheMode::static_admission, 3264);
    auto failed = failure_atomic.get_or_load({3, 0}, []() {
        return Result<k3x::ExpertMlpPayload>::failure(ErrorCode::io_error);
    });
    assert(!failed);
    assert(failure_atomic.stats().hits == 0);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto malformed_payload = payload(50);
    malformed_payload.gate.packed.pop_back();
    auto malformed = failure_atomic.get_or_load(
        {3, 2}, [value = std::move(malformed_payload)]() mutable {
            return Result<k3x::ExpertMlpPayload>::success(std::move(value));
        });
    assert(!malformed);
    assert(malformed.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto reserved_payload = payload(60);
    reserved_payload.up.scales[0] = std::byte{0xff};
    auto reserved = failure_atomic.get_or_load(
        {3, 3}, [value = std::move(reserved_payload)]() mutable {
            return Result<k3x::ExpertMlpPayload>::success(std::move(value));
        });
    assert(!reserved);
    assert(reserved.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto invalid = failure_atomic.get_or_load({3, 1}, []() {
        return Result<k3x::ExpertMlpPayload>::success(k3x::ExpertMlpPayload{});
    });
    assert(!invalid);
    assert(invalid.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    constexpr std::size_t thread_count = 8;
    HostExpertStore concurrent(L1ExpertCacheMode::static_admission, 3264);
    std::atomic<std::size_t> concurrent_loads{};
    std::barrier start(static_cast<std::ptrdiff_t>(thread_count));
    std::vector<k3x::ExpertPayloadHandle> handles(thread_count);
    std::vector<std::thread> threads;
    for (std::size_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            auto result = concurrent.get_or_load({4, 7}, [&] {
                ++concurrent_loads;
                std::this_thread::sleep_for(std::chrono::milliseconds{20});
                return Result<k3x::ExpertMlpPayload>::success(payload(70));
            });
            assert(result);
            handles[index] = std::move(result.value());
        });
    }
    for (auto& thread : threads) thread.join();
    assert(concurrent_loads == 1);
    assert(concurrent.stats().misses == 1);
    assert(concurrent.stats().hits == thread_count - 1);
    for (const auto& handle : handles) {
        assert(handle.get() == handles.front().get());
    }

    const auto load_key = [](HostExpertStore& target, ExpertKey key,
                             std::uint64_t id) {
        auto result = target.get_or_load(key, [id] {
            return Result<k3x::ExpertMlpPayload>::success(payload(id));
        });
        assert(result);
    };

    HostExpertStore lru(L1ExpertCacheMode::lru, 3264);
    lru.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(lru, {0, 0}, 100);
    lru.begin_access_set(0, 1, std::array{ExpertKey{1, 0}});
    load_key(lru, {1, 0}, 110);
    lru.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(lru, {0, 0}, 100);
    lru.begin_access_set(0, 2, std::array{ExpertKey{2, 0}});
    load_key(lru, {2, 0}, 120);
    assert(lru.contains({0, 0}));
    assert(!lru.contains({1, 0}));
    assert(lru.contains({2, 0}));
    assert(lru.stats().evictions == 1);

    HostExpertStore lfu(L1ExpertCacheMode::lfu, 3264);
    lfu.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(lfu, {0, 0}, 130);
    load_key(lfu, {0, 0}, 130);
    lfu.begin_access_set(0, 1, std::array{ExpertKey{1, 0}});
    load_key(lfu, {1, 0}, 140);
    lfu.begin_access_set(0, 2, std::array{ExpertKey{2, 0}});
    load_key(lfu, {2, 0}, 150);
    assert(lfu.contains({0, 0}));
    assert(!lfu.contains({1, 0}));
    assert(lfu.contains({2, 0}));

    HostExpertStore least_stale(L1ExpertCacheMode::least_stale, 4896);
    for (std::size_t layer = 0; layer < 3; ++layer) {
        const ExpertKey key{layer, 0};
        least_stale.begin_access_set(0, layer, std::array{key});
        load_key(least_stale, key, 160 + layer * 10);
    }
    least_stale.begin_access_set(1, 1, std::array{ExpertKey{1, 1}});
    load_key(least_stale, {1, 1}, 190);
    assert(!least_stale.contains({0, 0}));
    assert(least_stale.contains({1, 0}));
    assert(least_stale.contains({2, 0}));
    assert(least_stale.contains({1, 1}));

    HostExpertStore protected_set(L1ExpertCacheMode::lru, 3264);
    protected_set.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(protected_set, {0, 0}, 200);
    protected_set.begin_access_set(0, 1, std::array{ExpertKey{1, 0}});
    load_key(protected_set, {1, 0}, 210);
    protected_set.begin_access_set(
        0, 2, std::array{ExpertKey{0, 0}, ExpertKey{2, 0}});
    load_key(protected_set, {2, 0}, 220);
    assert(protected_set.contains({0, 0}));
    assert(!protected_set.contains({1, 0}));

    HostExpertStore collision(L1ExpertCacheMode::lru, 3264);
    collision.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(collision, {0, 0}, 230);
    collision.begin_access_set(0, 1, std::array{ExpertKey{1, 0}});
    load_key(collision, {1, 0}, 240);
    collision.begin_access_set(0, 0, std::array{ExpertKey{0, 0}});
    load_key(collision, {0, 0}, 230);
    collision.begin_access_set(0, 2, std::array{ExpertKey{2, 0}});
    load_key(collision, {2, 0}, 250);
    collision.begin_access_set(0, 1, std::array{ExpertKey{1, 0}});
    load_key(collision, {1, 0}, 240);
    assert(collision.stats().collision_misses == 1);

    HostExpertStore lru_future(L1ExpertCacheMode::lru, 4896);
    HostExpertStore least_stale_future(
        L1ExpertCacheMode::least_stale, 4896);
    for (auto* target : {&lru_future, &least_stale_future}) {
        for (std::size_t layer = 1; layer <= 3; ++layer) {
            const ExpertKey key{layer, 0};
            target->begin_access_set(0, layer, std::array{key});
            load_key(*target, key, 260 + layer * 10);
        }
        target->begin_access_set(1, 0, std::array{ExpertKey{0, 0}});
        load_key(*target, {0, 0}, 300);
        target->begin_access_set(1, 1, std::array{ExpertKey{1, 0}});
        load_key(*target, {1, 0}, 270);
    }
    assert(lru_future.stats().collision_misses == 1);
    assert(least_stale_future.stats().collision_misses == 0);
    assert(least_stale_future.contains({1, 0}));
    assert(!least_stale_future.contains({3, 0}));

    k3x::RuntimeProfile prior_seed;
    for (std::uint64_t cycle = 0; cycle < 5; ++cycle) {
        prior_seed.observe(cycle, 0, std::array{ExpertKey{0, 0}});
    }
    prior_seed.observe(5, 0, std::array{ExpertKey{0, 1}});
    const auto prior_path = std::filesystem::temp_directory_path() /
        ("k3x-profiled-cache-prior-" +
         std::to_string(std::chrono::steady_clock::now()
                            .time_since_epoch().count()) +
         ".k3xp");
    assert(prior_seed.save(prior_path));
    auto helpful_profile = k3x::RuntimeProfile::load(prior_path);
    auto conflicting_profile = k3x::RuntimeProfile::load(prior_path);
    assert(helpful_profile && conflicting_profile);

    HostExpertStore profiled_helpful(
        L1ExpertCacheMode::profiled, 3264, &helpful_profile.value(), 1);
    load_key(profiled_helpful, {0, 0}, 310);
    load_key(profiled_helpful, {0, 1}, 320);
    profiled_helpful.begin_access_set(
        0, 1, std::array{ExpertKey{1, 0}});
    load_key(profiled_helpful, {1, 0}, 330);
    assert(profiled_helpful.contains({0, 0}));
    assert(!profiled_helpful.contains({0, 1}));

    for (std::uint64_t cycle = 0; cycle < 4; ++cycle) {
        conflicting_profile.value().observe(
            cycle, 0, std::array{ExpertKey{0, 1}});
    }
    HostExpertStore profiled_crossover(
        L1ExpertCacheMode::profiled, 3264,
        &conflicting_profile.value(), 1);
    load_key(profiled_crossover, {0, 0}, 340);
    load_key(profiled_crossover, {0, 1}, 350);
    profiled_crossover.begin_access_set(
        4, 1, std::array{ExpertKey{1, 0}});
    load_key(profiled_crossover, {1, 0}, 360);
    assert(!profiled_crossover.contains({0, 0}));
    assert(profiled_crossover.contains({0, 1}));
    std::filesystem::remove(prior_path);

    return 0;
}
