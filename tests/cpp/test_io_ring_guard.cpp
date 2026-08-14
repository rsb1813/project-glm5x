// io_uring 오류 경로가 버퍼 해제 전에 ring을 폐기하는지 검증합니다.
#include "io_ring_guard.hpp"

#include <liburing.h>

#include <array>
#include <cstddef>
#include <unistd.h>

namespace {
struct BufferLifetimeProbe {
    const bool& initialized;
    bool& released_safely;
    ~BufferLifetimeProbe() { released_safely = !initialized; }
};
}

int main() {
    io_uring ring{};
    bool initialized = io_uring_queue_init(2, &ring, 0) == 0;
    if (!initialized) return 1;
    int descriptors[2]{};
    if (::pipe(descriptors) != 0) return 2;
    bool released_safely = false;
    {
        std::array<std::byte, 16> pending_buffer{};
        BufferLifetimeProbe buffer{initialized, released_safely};
        k3x::detail::IoRingBatchGuard guard(ring, initialized);
        auto* submission = io_uring_get_sqe(&ring);
        if (submission == nullptr) return 3;
        io_uring_prep_read(
            submission, descriptors[0], pending_buffer.data(),
            static_cast<unsigned>(pending_buffer.size()), -1);
        if (io_uring_submit(&ring) != 1) return 4;
    }
    ::close(descriptors[0]);
    ::close(descriptors[1]);
    if (initialized || !released_safely) return 5;

    initialized = io_uring_queue_init(2, &ring, 0) == 0;
    if (!initialized) return 6;
    {
        k3x::detail::IoRingBatchGuard guard(ring, initialized);
        guard.release();
    }
    if (!initialized) return 7;
    io_uring_queue_exit(&ring);
    initialized = false;
    return 0;
}
