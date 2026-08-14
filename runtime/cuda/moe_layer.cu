// resident MoE layer의 ordered mix, strict RMSNorm, 최종 합산 CUDA kernel을 구현합니다.
#include "moe_layer.cuh"

#include <cmath>
#include <cstddef>

namespace k3x::cuda {
namespace {

__global__ void ordered_expert_mix_kernel(
    const float* outputs, const float* contributions, float* mixed,
    std::size_t experts, std::size_t width) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (row >= width) return;
    float value = 0.0F;
    for (std::size_t slot = 0; slot < experts; ++slot) {
        value += contributions[slot] * outputs[slot * width + row];
    }
    mixed[row] = value;
}

__global__ void strict_rms_norm_kernel(
    const float* input, const float* weight, float* output,
    std::size_t width, float epsilon) {
    __shared__ float inverse;
    if (threadIdx.x == 0) {
        double squares = 0.0;
        for (std::size_t row = 0; row < width; ++row) {
            squares += static_cast<double>(input[row]) * input[row];
        }
        inverse = 1.0F /
                  sqrtf(static_cast<float>(squares / width) + epsilon);
    }
    __syncthreads();
    for (std::size_t row = threadIdx.x; row < width; row += blockDim.x) {
        output[row] = input[row] * inverse * weight[row];
    }
}

__global__ void vector_add_kernel(
    const float* left, const float* right, float* output,
    std::size_t width) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (row < width) output[row] = left[row] + right[row];
}

}  // namespace

cudaError_t launch_ordered_expert_mix(
    const float* expert_outputs, const float* device_contributions,
    std::span<const float> host_contributions, float* mixed,
    std::size_t width, cudaStream_t stream) {
    if (!expert_outputs || !device_contributions || !mixed || width == 0 ||
        host_contributions.empty()) {
        return cudaErrorInvalidValue;
    }
    for (const auto contribution : host_contributions) {
        if (!std::isfinite(contribution)) return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>((width + threads - 1) / threads);
    ordered_expert_mix_kernel<<<blocks, threads, 0, stream>>>(
        expert_outputs, device_contributions, mixed,
        host_contributions.size(), width);
    return cudaGetLastError();
}

cudaError_t launch_strict_rms_norm(
    const float* input, const float* weight, float* output,
    std::size_t width, float epsilon, cudaStream_t stream) {
    if (!input || !weight || !output || width == 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return cudaErrorInvalidValue;
    }
    strict_rms_norm_kernel<<<1, 256, 0, stream>>>(
        input, weight, output, width, epsilon);
    return cudaGetLastError();
}

cudaError_t launch_vector_add(
    const float* left, const float* right, float* output,
    std::size_t width, cudaStream_t stream) {
    if (!left || !right || !output || width == 0) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>((width + threads - 1) / threads);
    vector_add_kernel<<<blocks, threads, 0, stream>>>(
        left, right, output, width);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
