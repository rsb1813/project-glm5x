// CUDA device allocation과 grow-only scratch buffer를 안전하게 구현합니다.
#include "device_memory.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace k3x::cuda {
namespace {

cudaError_t allocate_device(void** pointer, std::size_t bytes) {
    return cudaMalloc(pointer, bytes);
}

cudaError_t free_device(void* pointer) {
    return cudaFree(pointer);
}

}  // namespace

AllocatorFunctions default_allocator() noexcept {
    return {allocate_device, free_device};
}

DeviceAllocation::DeviceAllocation(
    BackendMemoryStats* memory, BackendRuntimeStats* runtime,
    AllocatorFunctions allocator) noexcept
    : memory_(memory), runtime_(runtime), allocator_(allocator) {}

DeviceAllocation::~DeviceAllocation() {
    reset();
}

cudaError_t DeviceAllocation::allocate(std::size_t bytes) noexcept {
    if (pointer_ || !memory_ || !runtime_ || !allocator_.allocate ||
        !allocator_.free) {
        return cudaErrorInvalidValue;
    }
    if (bytes == 0) return cudaSuccess;
    if (bytes > std::numeric_limits<std::uint64_t>::max() -
                    memory_->current_device_bytes) {
        return cudaErrorInvalidValue;
    }
    void* pointer = nullptr;
    const auto status = allocator_.allocate(&pointer, bytes);
    if (status != cudaSuccess) return status;
    pointer_ = pointer;
    bytes_ = bytes;
    memory_->current_device_bytes += bytes_;
    memory_->peak_device_bytes = std::max(
        memory_->peak_device_bytes, memory_->current_device_bytes);
    ++runtime_->device_allocation_count;
    return cudaSuccess;
}

cudaError_t DeviceAllocation::reset() noexcept {
    if (!pointer_) return cudaSuccess;
    const auto status = allocator_.free(pointer_);
    if (status != cudaSuccess) return status;
    memory_->current_device_bytes -= bytes_;
    ++runtime_->device_free_count;
    pointer_ = nullptr;
    bytes_ = 0;
    return cudaSuccess;
}

void DeviceAllocation::swap(DeviceAllocation& other) noexcept {
    using std::swap;
    swap(pointer_, other.pointer_);
    swap(bytes_, other.bytes_);
    swap(memory_, other.memory_);
    swap(runtime_, other.runtime_);
    swap(allocator_, other.allocator_);
}

ScratchBuffer::ScratchBuffer(
    BackendMemoryStats* memory, BackendRuntimeStats* runtime,
    AllocatorFunctions allocator) noexcept
    : memory_(memory), runtime_(runtime), allocator_(allocator),
      allocation_(memory, runtime, allocator) {}

ScratchBuffer::~ScratchBuffer() {
    const auto capacity = allocation_.bytes();
    allocation_.reset();
    if (runtime_) runtime_->scratch_bytes -= capacity;
}

cudaError_t ScratchBuffer::reserve(std::size_t bytes) noexcept {
    if (bytes <= allocation_.bytes()) return cudaSuccess;
    DeviceAllocation replacement(memory_, runtime_, allocator_);
    const auto status = replacement.allocate(bytes);
    if (status != cudaSuccess) return status;
    const auto previous = allocation_.bytes();
    allocation_.swap(replacement);
    runtime_->scratch_bytes += bytes - previous;
    runtime_->peak_scratch_bytes = std::max(
        runtime_->peak_scratch_bytes, runtime_->scratch_bytes);
    return cudaSuccess;
}

}  // namespace k3x::cuda
