// 비동기 전송용 고정 용량 CUDA pinned host buffer 계약을 정의합니다.
#pragma once

#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace k3x::cuda {

class PinnedBuffer {
public:
    explicit PinnedBuffer(BackendRuntimeStats* runtime) noexcept;
    ~PinnedBuffer();
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    cudaError_t allocate(std::size_t bytes) noexcept;
    void* get() const noexcept { return pointer_; }
    std::size_t size() const noexcept { return bytes_; }

private:
    BackendRuntimeStats* runtime_{};
    void* pointer_{};
    std::size_t bytes_{};
};

}  // namespace k3x::cuda
