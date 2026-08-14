// CUDA FFN block에서 사용하는 SiTU-GLU launch 계약을 선언합니다.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace k3x::cuda {

cudaError_t launch_situ_glu(
    const float* gate, const float* up, void* output,
    std::size_t count, float beta, bool has_linear_beta,
    float linear_beta, bool output_bf16, cudaStream_t stream);

cudaError_t launch_situ_glu_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, std::size_t count, float beta,
    bool has_linear_beta, float linear_beta, cudaStream_t stream);

cudaError_t launch_silu_glu(
    const float* gate, const float* up, void* output,
    std::size_t count, bool output_bf16, cudaStream_t stream);

cudaError_t launch_silu_glu_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, std::size_t count, cudaStream_t stream);

}  // namespace k3x::cuda
