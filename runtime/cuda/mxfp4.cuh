// native K3 MXFP4 바이트를 직접 계산하는 CUDA kernel launch 계약을 정의합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {

struct Mxfp4DeviceMatrix {
    const std::uint8_t* packed{};
    const std::uint8_t* scales{};
};

enum class ExpertGridInputLayout : std::uint8_t {
    shared_token_major,
    expert_token_major,
};

cudaError_t launch_mxfp4_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream);

cudaError_t launch_mxfp4_matvec_batch(
    const float* inputs, const std::uint8_t* packed,
    const std::uint8_t* scales, float* outputs,
    std::size_t rows, std::size_t cols, std::size_t batch_size,
    cudaStream_t stream);

cudaError_t launch_mxfp4_matvec_accumulate(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, float contribution,
    bool accumulate, cudaStream_t stream);

cudaError_t launch_mxfp4_matvec_grid(
    const float* inputs, const Mxfp4DeviceMatrix* descriptors,
    float* outputs, std::size_t rows, std::size_t cols,
    std::size_t expert_count, std::size_t token_count,
    ExpertGridInputLayout layout, cudaStream_t stream);

}  // namespace k3x::cuda
