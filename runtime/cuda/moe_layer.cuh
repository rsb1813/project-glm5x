// resident MoE layer의 ordered mix, strict RMSNorm, 최종 합산 CUDA launcher를 선언합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <span>

namespace k3x::cuda {

cudaError_t launch_ordered_expert_mix(
    const float* expert_outputs, const float* device_contributions,
    std::span<const float> host_contributions, float* mixed,
    std::size_t width, cudaStream_t stream);

cudaError_t launch_strict_rms_norm(
    const float* input, const float* weight, float* output,
    std::size_t width, float epsilon, cudaStream_t stream);

cudaError_t launch_vector_add(
    const float* left, const float* right, float* output,
    std::size_t width, cudaStream_t stream);

}  // namespace k3x::cuda
