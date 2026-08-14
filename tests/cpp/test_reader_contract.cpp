// L2 reader의 기본 옵션과 ordered batch 공개 계약을 검증합니다.
#include "k3x/reader.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <type_traits>
#include <vector>

int main() {
    const k3x::ReaderOptions defaults;
    if (defaults.verify != k3x::VerifyMode::checksums ||
        defaults.io_engine != k3x::L2IoEngine::pread ||
        defaults.cache_mode != k3x::L2CacheMode::buffered ||
        defaults.queue_depth != 8 || k3x::maximum_l2_queue_depth != 1024) {
        return 1;
    }

    const k3x::ExtentRequest request{4096, 257};
    if (request.offset != 4096 || request.length != 257) return 2;

    const k3x::ReadCounters counters;
    if (counters.calls != 0 || counters.requested_bytes != 0 ||
        counters.completed_bytes != 0 || counters.batch_submissions != 0 ||
        counters.storage_submitted_bytes != 0 ||
        counters.storage_completed_bytes != 0 || counters.completions != 0 ||
        counters.short_reads != 0 || counters.failures != 0) {
        return 3;
    }

    using BatchResult = k3x::Result<std::vector<std::vector<std::byte>>>;
    static_assert(std::is_same_v<
        decltype(std::declval<const k3x::Reader&>().read_extents(
            std::declval<std::span<const k3x::ExtentRequest>>())),
        BatchResult>);
    static_assert(std::is_same_v<
        decltype(k3x::Reader::open(std::declval<const std::filesystem::path&>(),
                                   std::declval<k3x::ReaderOptions>())),
        k3x::Result<k3x::Reader>>);
    return 0;
}
