// W8A16-G128 가중치를 BF16 activation과 곱하는 fused expert-major kernel입니다.
#include "w8a16.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace k3x::cuda {
namespace {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 4;
constexpr unsigned kThreads = kWarpSize * kWarpsPerBlock;
constexpr std::size_t kGroupSize = 128;

__device__ float warp_sum(float value) {
    for (unsigned offset = kWarpSize / 2; offset != 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

__global__ void w8a16_gate_up_kernel(
    const __nv_bfloat16* inputs, const W8A16DeviceMatrix* gate_views,
    const W8A16DeviceMatrix* up_views, __nv_bfloat16* activated,
    std::size_t rows, std::size_t cols) {
    const auto assignment = static_cast<std::size_t>(blockIdx.y);
    const auto warp = threadIdx.x / kWarpSize;
    const auto lane = threadIdx.x % kWarpSize;
    const auto row = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock +
                     warp;
    if (row >= rows) return;
    const auto gate = gate_views[assignment];
    const auto up = up_views[assignment];
    const auto groups = cols / kGroupSize;
    const auto* input = inputs + assignment * cols;
    float gate_sum = 0.0F;
    float up_sum = 0.0F;
    for (std::size_t group = 0; group < groups; ++group) {
        float gate_scale = lane == 0
            ? __bfloat162float(gate.scales[row * groups + group]) : 0.0F;
        float up_scale = lane == 0
            ? __bfloat162float(up.scales[row * groups + group]) : 0.0F;
        gate_scale = __shfl_sync(0xffffffffU, gate_scale, 0);
        up_scale = __shfl_sync(0xffffffffU, up_scale, 0);
        const auto column = group * kGroupSize + lane * 4;
        const auto weight_offset = row * cols + column;
#pragma unroll
        for (unsigned item = 0; item < 4; ++item) {
            const auto value = __bfloat162float(input[column + item]);
            gate_sum += static_cast<float>(gate.values[weight_offset + item]) *
                        gate_scale * value;
            up_sum += static_cast<float>(up.values[weight_offset + item]) *
                      up_scale * value;
        }
    }
    gate_sum = warp_sum(gate_sum);
    up_sum = warp_sum(up_sum);
    if (lane == 0) {
        const auto gate_bf16 = __float2bfloat16_rn(gate_sum);
        const auto up_bf16 = __float2bfloat16_rn(up_sum);
        const auto gate_value = __bfloat162float(gate_bf16);
        const auto up_value = __bfloat162float(up_bf16);
        const auto silu = gate_value / (1.0F + expf(-gate_value));
        activated[assignment * rows + row] =
            __float2bfloat16_rn(silu * up_value);
    }
}

__global__ void w8a16_down_kernel(
    const __nv_bfloat16* activated, const W8A16DeviceMatrix* down_views,
    float* outputs, std::size_t rows, std::size_t cols) {
    const auto assignment = static_cast<std::size_t>(blockIdx.y);
    const auto warp = threadIdx.x / kWarpSize;
    const auto lane = threadIdx.x % kWarpSize;
    const auto row = static_cast<std::size_t>(blockIdx.x) * kWarpsPerBlock +
                     warp;
    if (row >= rows) return;
    const auto down = down_views[assignment];
    const auto groups = cols / kGroupSize;
    const auto* input = activated + assignment * cols;
    float sum = 0.0F;
    for (std::size_t group = 0; group < groups; ++group) {
        float scale = lane == 0
            ? __bfloat162float(down.scales[row * groups + group]) : 0.0F;
        scale = __shfl_sync(0xffffffffU, scale, 0);
        const auto column = group * kGroupSize + lane * 4;
        const auto weight_offset = row * cols + column;
#pragma unroll
        for (unsigned item = 0; item < 4; ++item) {
            sum += static_cast<float>(down.values[weight_offset + item]) *
                   scale * __bfloat162float(input[column + item]);
        }
    }
    sum = warp_sum(sum);
    if (lane == 0) {
        outputs[assignment * rows + row] =
            __bfloat162float(__float2bfloat16_rn(sum));
    }
}

bool valid_launch(const void* input, const W8A16DeviceMatrix* first,
                  const W8A16DeviceMatrix* second, const void* output,
                  std::size_t rows, std::size_t cols,
                  std::size_t assignment_count) {
    return input != nullptr && first != nullptr && second != nullptr &&
           output != nullptr && rows != 0 && cols != 0 &&
           cols % kGroupSize == 0 && assignment_count != 0 &&
           assignment_count <= 65535 &&
           rows <= static_cast<std::size_t>(
                       std::numeric_limits<unsigned>::max()) *
                       kWarpsPerBlock;
}

}  // namespace

cudaError_t launch_w8a16_gate_up(
    const __nv_bfloat16* inputs, const W8A16DeviceMatrix* gate,
    const W8A16DeviceMatrix* up, __nv_bfloat16* activated,
    std::size_t rows, std::size_t cols, std::size_t assignment_count,
    cudaStream_t stream) {
    if (!valid_launch(inputs, gate, up, activated, rows, cols,
                      assignment_count)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(
        static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock),
        static_cast<unsigned>(assignment_count));
    w8a16_gate_up_kernel<<<grid, kThreads, 0, stream>>>(
        inputs, gate, up, activated, rows, cols);
    return cudaGetLastError();
}

cudaError_t launch_w8a16_down(
    const __nv_bfloat16* activated, const W8A16DeviceMatrix* down,
    float* outputs, std::size_t rows, std::size_t cols,
    std::size_t assignment_count, cudaStream_t stream) {
    if (!valid_launch(activated, down, down, outputs, rows, cols,
                      assignment_count)) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid(
        static_cast<unsigned>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock),
        static_cast<unsigned>(assignment_count));
    w8a16_down_kernel<<<grid, kThreads, 0, stream>>>(
        activated, down, outputs, rows, cols);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
