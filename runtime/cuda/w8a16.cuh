// RTX 5080용 W8A16-G128 expert-major CUDA kernel 계약입니다.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {

struct W8A16DeviceMatrix {
    const std::int8_t* values;
    const __nv_bfloat16* scales;
};

cudaError_t launch_w8a16_gate_up(
    const __nv_bfloat16* inputs, const W8A16DeviceMatrix* gate,
    const W8A16DeviceMatrix* up, __nv_bfloat16* activated,
    std::size_t rows, std::size_t cols, std::size_t assignment_count,
    cudaStream_t stream);

cudaError_t launch_w8a16_down(
    const __nv_bfloat16* activated, const W8A16DeviceMatrix* down,
    float* outputs, std::size_t rows, std::size_t cols,
    std::size_t assignment_count, cudaStream_t stream);

}  // namespace k3x::cuda
