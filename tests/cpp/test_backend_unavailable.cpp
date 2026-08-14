// CUDA 비활성 build가 명시적 CUDA 요청을 typed error로 거부하는지 검증합니다.
#include "k3x/backend.hpp"

int main() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::grouped;
    options.cuda_resident_bytes = 4096;

    const auto backend = k3x::make_cuda_backend(options);
    if (backend) return 1;
    if (backend.error() != k3x::ErrorCode::backend_unavailable) return 2;
    return 0;
}
