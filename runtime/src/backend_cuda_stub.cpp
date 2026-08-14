// CUDA 비활성 build에서 명시적 CUDA 요청을 typed unavailable 오류로 거부합니다.
#include "k3x/backend.hpp"

namespace k3x {

Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions&, Profiler*) {
    return Result<std::unique_ptr<ComputeBackend>>::failure(
        ErrorCode::backend_unavailable, "CUDA backend is disabled at build time");
}

}  // namespace k3x
