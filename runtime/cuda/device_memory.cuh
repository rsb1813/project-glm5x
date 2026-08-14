// CUDA device allocation과 재사용 scratch buffer의 소유권 및 계측 계약을 정의합니다.
#pragma once

#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace k3x::cuda {

using AllocateFunction = cudaError_t (*)(void**, std::size_t);
using FreeFunction = cudaError_t (*)(void*);

struct AllocatorFunctions {
    AllocateFunction allocate;
    FreeFunction free;
};

AllocatorFunctions default_allocator() noexcept;

class DeviceAllocation {
public:
    DeviceAllocation(BackendMemoryStats* memory, BackendRuntimeStats* runtime,
                     AllocatorFunctions allocator = default_allocator()) noexcept;
    ~DeviceAllocation();
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    cudaError_t allocate(std::size_t bytes) noexcept;
    cudaError_t reset() noexcept;
    void swap(DeviceAllocation& other) noexcept;
    void* get() const noexcept { return pointer_; }
    std::size_t bytes() const noexcept { return bytes_; }

private:
    void* pointer_{};
    std::size_t bytes_{};
    BackendMemoryStats* memory_{};
    BackendRuntimeStats* runtime_{};
    AllocatorFunctions allocator_{};
};

class ScratchBuffer {
public:
    ScratchBuffer(BackendMemoryStats* memory, BackendRuntimeStats* runtime,
                  AllocatorFunctions allocator = default_allocator()) noexcept;
    ~ScratchBuffer();
    ScratchBuffer(const ScratchBuffer&) = delete;
    ScratchBuffer& operator=(const ScratchBuffer&) = delete;

    cudaError_t reserve(std::size_t bytes) noexcept;
    void* get() const noexcept { return allocation_.get(); }
    std::size_t capacity() const noexcept { return allocation_.bytes(); }

private:
    BackendMemoryStats* memory_{};
    BackendRuntimeStats* runtime_{};
    AllocatorFunctions allocator_{};
    DeviceAllocation allocation_;
};

}  // namespace k3x::cuda
