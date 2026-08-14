// 한 Reader의 병렬 호출이 payload와 누적 counter를 정확히 보존하는지 검증합니다.
#include "k3x/reader.hpp"

#include <atomic>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    static_assert(std::same_as<
                  decltype(std::declval<const k3x::Reader&>().counters()),
                  k3x::ReadCounters>);
    if (argc != 2 && argc != 3) return 2;
    k3x::ReaderOptions options;
    if (argc == 3) {
        const auto mode = std::string_view(argv[2]);
        if (mode != "io-uring") return 2;
        options.io_engine = k3x::L2IoEngine::io_uring;
    }
    auto reader = k3x::Reader::open(std::filesystem::path(argv[1]), options);
    if (!reader || reader.value().tensors().empty()) return 3;
    const auto& record = reader.value().tensors().front();
    constexpr std::size_t thread_count = 8;
    constexpr std::size_t reads_per_thread = 200;
    std::atomic<std::size_t> failures{};
    std::atomic<bool> reads_complete{};
    std::vector<std::thread> threads;
    std::thread observer([&] {
        while (!reads_complete.load()) {
            const auto snapshot = reader.value().counters();
            if (snapshot.completed_bytes > snapshot.requested_bytes ||
                snapshot.completions > snapshot.calls) {
                ++failures;
            }
        }
    });
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&] {
            for (std::size_t read = 0; read < reads_per_thread; ++read) {
                auto result = reader.value().read_tensor(record.tensor_id);
                if (!result || result.value().size() != record.data_length) {
                    ++failures;
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    reads_complete.store(true);
    observer.join();
    require(failures == 0, "concurrent payload read failed");
    const auto expected = thread_count * reads_per_thread;
    const auto counters = reader.value().counters();
    require(counters.calls == expected, "concurrent call count mismatch");
    require(counters.batch_submissions == expected,
            "concurrent batch count mismatch");
    require(counters.completions == expected,
            "concurrent completion count mismatch");
    require(counters.requested_bytes == expected * record.data_length,
            "concurrent requested byte count mismatch");
    require(counters.completed_bytes == expected * record.data_length,
            "concurrent completed byte count mismatch");
    require(counters.failures == 0, "unexpected concurrent read failure");
    require(counters.short_reads == 0, "unexpected concurrent short read");
    return 0;
}
