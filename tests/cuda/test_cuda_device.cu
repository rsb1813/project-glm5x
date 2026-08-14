// CUDA backend가 SM 12.0 이상 device를 선택하고 identity를 보존하는지 검증합니다.
#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 1;

    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess) return 2;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) return 3;
    if (major < 12) return 4;

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    const auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 5;
    if (backend.value()->kind() != k3x::BackendKind::cuda_dense) return 6;
    if (backend.value()->device_name().empty()) return 7;
    if (backend.value()->memory_stats().current_device_bytes != 0) return 8;
    if (backend.value()->memory_stats().peak_device_bytes != 0) return 9;
    return minor < 0 ? 10 : 0;
}
