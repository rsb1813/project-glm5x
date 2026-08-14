// Linux direct I/O 정렬 조회와 exact bounce-buffer batch 계약을 정의합니다.
#pragma once

#include "k3x/reader.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#ifdef K3X_ENABLE_IO_URING
#include <liburing.h>
#endif

namespace k3x::detail {
struct DirectIoAlignment {
    std::uint64_t memory{};
    std::uint64_t offset{};
};

Result<DirectIoAlignment> query_direct_io_alignment(
    const std::filesystem::path& path);

Result<std::vector<std::vector<std::byte>>> read_direct_pread(
    int descriptor, std::span<const ExtentRequest> requests,
    DirectIoAlignment alignment, ReadCounters& counters);

#ifdef K3X_ENABLE_IO_URING
Result<std::vector<std::vector<std::byte>>> read_direct_io_uring(
    int descriptor, io_uring& ring, bool& ring_initialized,
    std::span<const ExtentRequest> requests, std::size_t queue_depth,
    DirectIoAlignment alignment,
    ReadCounters& counters);
#endif
}
