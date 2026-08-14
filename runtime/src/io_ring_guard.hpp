// io_uring batch 실패 시 버퍼 수명 안에서 ring 전체를 안전하게 폐기합니다.
#pragma once

#include <liburing.h>

namespace k3x::detail {
class IoRingBatchGuard {
public:
    IoRingBatchGuard(io_uring& ring, bool& initialized) noexcept
        : ring_(&ring), initialized_(&initialized) {}
    IoRingBatchGuard(const IoRingBatchGuard&) = delete;
    IoRingBatchGuard& operator=(const IoRingBatchGuard&) = delete;
    ~IoRingBatchGuard() { abandon(); }

    void release() noexcept { active_ = false; }

private:
    void abandon() noexcept {
        if (!active_ || !*initialized_) return;
        io_uring_queue_exit(ring_);
        *ring_ = {};
        *initialized_ = false;
        active_ = false;
    }

    io_uring* ring_{};
    bool* initialized_{};
    bool active_{true};
};
}
