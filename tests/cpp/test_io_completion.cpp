// io_uring completion의 순서 독립 매핑과 오류 회계를 검증합니다.
#include "io_completion.hpp"

#include <array>

int main() {
    const std::array requests{
        k3x::ExtentRequest{4096, 11},
        k3x::ExtentRequest{8192, 13},
        k3x::ExtentRequest{12288, 17},
    };
    k3x::ReadCounters counters;
    for (const auto [index, result] : {
             std::pair<std::size_t, int>{2, 17},
             {0, 11},
             {1, 13}}) {
        if (k3x::detail::record_io_completion(
                requests, index, result, counters) != k3x::ErrorCode::ok) {
            return 1;
        }
    }
    if (counters.storage_completed_bytes != 41) return 2;
    if (k3x::detail::record_io_completion(
            requests, 1, 12, counters) !=
        k3x::ErrorCode::truncated_file) {
        return 3;
    }
    if (counters.storage_completed_bytes != 53) return 4;
    if (k3x::detail::record_io_completion(
            requests, requests.size(), 1, counters) !=
        k3x::ErrorCode::io_error) {
        return 5;
    }
    if (k3x::detail::record_io_completion(
            requests, 0, -5, counters) !=
        k3x::ErrorCode::io_error) {
        return 6;
    }
    return 0;
}
