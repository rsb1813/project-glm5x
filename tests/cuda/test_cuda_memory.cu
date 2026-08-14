// CUDA device allocation과 grow-only scratch의 계측 및 실패 원자성을 검증합니다.
#include "device_memory.cuh"

#include <cstdlib>

namespace {

int allocation_calls = 0;
int failure_call = 0;

cudaError_t fake_allocate(void** pointer, std::size_t bytes) {
    ++allocation_calls;
    if (allocation_calls == failure_call) return cudaErrorMemoryAllocation;
    *pointer = std::malloc(bytes);
    return *pointer ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t fake_free(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}

}  // namespace

int main() {
    k3x::BackendMemoryStats memory;
    k3x::BackendRuntimeStats runtime;
    {
        k3x::cuda::ScratchBuffer scratch(&memory, &runtime);
        if (scratch.reserve(64) != cudaSuccess) return 1;
        void* first = scratch.get();
        if (!first || scratch.capacity() != 64) return 2;
        if (scratch.reserve(32) != cudaSuccess || scratch.get() != first) return 3;
        if (runtime.device_allocation_count != 1 ||
            runtime.device_free_count != 0) return 4;
        if (scratch.reserve(128) != cudaSuccess) return 5;
        if (scratch.get() == first || scratch.capacity() != 128) return 6;
        if (runtime.device_allocation_count != 2 ||
            runtime.device_free_count != 1) return 7;
        if (runtime.scratch_bytes != 128 ||
            runtime.peak_scratch_bytes != 128) return 8;
    }
    if (memory.current_device_bytes != 0) return 9;
    if (runtime.device_free_count != 2 || runtime.scratch_bytes != 0) return 10;

    k3x::BackendMemoryStats failed_memory;
    k3x::BackendRuntimeStats failed_runtime;
    allocation_calls = 0;
    failure_call = 2;
    {
        const k3x::cuda::AllocatorFunctions allocator{
            fake_allocate, fake_free};
        k3x::cuda::ScratchBuffer scratch(
            &failed_memory, &failed_runtime, allocator);
        if (scratch.reserve(64) != cudaSuccess) return 11;
        void* first = scratch.get();
        if (scratch.reserve(128) != cudaErrorMemoryAllocation) return 12;
        if (scratch.get() != first || scratch.capacity() != 64) return 13;
        if (failed_memory.current_device_bytes != 64 ||
            failed_runtime.scratch_bytes != 64) return 14;
        if (failed_runtime.device_allocation_count != 1 ||
            failed_runtime.device_free_count != 0) return 15;
    }
    if (failed_memory.current_device_bytes != 0) return 16;
    if (failed_runtime.device_free_count != 1 ||
        failed_runtime.scratch_bytes != 0) return 17;
    return 0;
}
