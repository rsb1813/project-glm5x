// 비순차 L2 I/O completion을 요청 identity와 정확히 대조합니다.
#pragma once

#include "k3x/reader.hpp"

#include <cstddef>
#include <span>

namespace k3x::detail {
inline ErrorCode record_io_completion(
    std::span<const ExtentRequest> requests, std::size_t index, int result,
    ReadCounters& counters) {
    if (index >= requests.size() || result < 0) return ErrorCode::io_error;
    if (result > 0) {
        counters.storage_completed_bytes +=
            static_cast<std::uint64_t>(result);
    }
    if (static_cast<std::uint64_t>(result) != requests[index].length) {
        return ErrorCode::truncated_file;
    }
    return ErrorCode::ok;
}
}
