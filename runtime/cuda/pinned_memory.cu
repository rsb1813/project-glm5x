// 고정 용량 CUDA pinned host buffer의 할당과 통계 수명을 구현합니다.
#include "pinned_memory.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace k3x::cuda {

PinnedBuffer::PinnedBuffer(BackendRuntimeStats* runtime) noexcept
    : runtime_(runtime) {}

PinnedBuffer::~PinnedBuffer() {
    if (!pointer_) return;
    if (cudaFreeHost(pointer_) == cudaSuccess && runtime_) {
        runtime_->pinned_host_bytes -= bytes_;
    }
}

cudaError_t PinnedBuffer::allocate(std::size_t bytes) noexcept {
    if (pointer_ || !runtime_ || bytes == 0 ||
        bytes > std::numeric_limits<std::uint64_t>::max() -
                    runtime_->pinned_host_bytes) {
        return cudaErrorInvalidValue;
    }
    void* pointer = nullptr;
    const auto status = cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault);
    if (status != cudaSuccess) return status;
    pointer_ = pointer;
    bytes_ = bytes;
    runtime_->pinned_host_bytes += bytes_;
    runtime_->peak_pinned_host_bytes = std::max(
        runtime_->peak_pinned_host_bytes, runtime_->pinned_host_bytes);
    return cudaSuccess;
}

}  // namespace k3x::cuda
