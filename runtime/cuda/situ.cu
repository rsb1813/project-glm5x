// gate와 up을 device에서 결합해 strict SiTU-GLU activation을 계산합니다.
#include "situ.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>

namespace k3x::cuda {
namespace {

template <bool OutputBf16>
__global__ void situ_glu_kernel(
    const float* gate, const float* up, void* output,
    std::size_t count, float beta, bool has_linear_beta,
    float linear_beta) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= count) return;
    const auto sigmoid = 1.0F / (1.0F + expf(-gate[index]));
    const auto bounded_gate = beta * tanhf(gate[index] / beta) * sigmoid;
    const auto bounded_up = has_linear_beta
        ? linear_beta * tanhf(up[index] / linear_beta)
        : up[index];
    const auto value = bounded_gate * bounded_up;
    if constexpr (OutputBf16) {
        static_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(value);
    } else {
        static_cast<float*>(output)[index] = value;
    }
}

}  // namespace

cudaError_t launch_situ_glu(
    const float* gate, const float* up, void* output,
    std::size_t count, float beta, bool has_linear_beta,
    float linear_beta, bool output_bf16, cudaStream_t stream) {
    if (!gate || !up || !output || count == 0 || !std::isfinite(beta) ||
        beta <= 0.0F ||
        (has_linear_beta &&
         (!std::isfinite(linear_beta) || linear_beta <= 0.0F))) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    if (output_bf16) {
        situ_glu_kernel<true><<<blocks, threads, 0, stream>>>(
            gate, up, output, count, beta, has_linear_beta, linear_beta);
    } else {
        situ_glu_kernel<false><<<blocks, threads, 0, stream>>>(
            gate, up, output, count, beta, has_linear_beta, linear_beta);
    }
    return cudaGetLastError();
}

}  // namespace k3x::cuda
