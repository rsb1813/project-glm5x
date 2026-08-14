// 고정 용량 CUDA pinned host buffer의 소유권과 계측 계약을 검증합니다.
#include "pinned_memory.cuh"

int main() {
    k3x::BackendRuntimeStats runtime;
    {
        k3x::cuda::PinnedBuffer buffer(&runtime);
        if (buffer.allocate(64) != cudaSuccess || !buffer.get()) return 1;
        if (buffer.size() != 64 || runtime.pinned_host_bytes != 64 ||
            runtime.peak_pinned_host_bytes != 64) {
            return 2;
        }
        if (buffer.allocate(32) != cudaErrorInvalidValue) return 3;
        if (buffer.size() != 64 || runtime.pinned_host_bytes != 64 ||
            runtime.peak_pinned_host_bytes != 64) {
            return 4;
        }
    }
    if (runtime.pinned_host_bytes != 0 ||
        runtime.peak_pinned_host_bytes != 64) {
        return 5;
    }

    k3x::BackendRuntimeStats zero_runtime;
    k3x::cuda::PinnedBuffer zero(&zero_runtime);
    if (zero.allocate(0) != cudaErrorInvalidValue) return 6;
    if (zero.get() || zero.size() != 0 || zero_runtime.pinned_host_bytes != 0 ||
        zero_runtime.peak_pinned_host_bytes != 0) {
        return 7;
    }
    return 0;
}
