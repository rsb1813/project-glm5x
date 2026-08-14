// K3 E2M1 값과 E8M0/32 scale을 재패킹 없이 decode하고 FP32로 누산합니다.
#include "mxfp4.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace k3x::cuda {
namespace {

__device__ __forceinline__ float decode_e2m1(std::uint8_t code) {
    float magnitude = 0.0F;
    switch (code & 0x07U) {
        case 1: magnitude = 0.5F; break;
        case 2: magnitude = 1.0F; break;
        case 3: magnitude = 1.5F; break;
        case 4: magnitude = 2.0F; break;
        case 5: magnitude = 3.0F; break;
        case 6: magnitude = 4.0F; break;
        case 7: magnitude = 6.0F; break;
        default: break;
    }
    return (code & 0x08U) != 0 ? -magnitude : magnitude;
}

__global__ void mxfp4_matvec_kernel(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, float contribution,
    bool accumulate) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    const auto token = static_cast<std::size_t>(blockIdx.y);
    if (row >= rows) return;
    input += token * cols;
    output += token * rows;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < cols;
         column += blockDim.x) {
        const auto index = row * cols + column;
        const auto pair = packed[index / 2];
        const auto code = static_cast<std::uint8_t>(
            index % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent = static_cast<int>(scales[index / 32]) - 127;
        const auto weight = ldexpf(decode_e2m1(code), exponent);
        sum = fmaf(input[column], weight, sum);
    }

    extern __shared__ float partials[];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = accumulate
            ? fmaf(contribution, partials[0], output[row])
            : contribution * partials[0];
    }
}

__global__ void mxfp4_matvec_grid_kernel(
    const float* inputs, const Mxfp4DeviceMatrix* descriptors,
    float* outputs, std::size_t rows, std::size_t cols,
    std::size_t token_count, ExpertGridInputLayout layout) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    const auto token = static_cast<std::size_t>(blockIdx.y);
    const auto expert = static_cast<std::size_t>(blockIdx.z);
    const auto& matrix = descriptors[expert];
    const auto input_base = layout == ExpertGridInputLayout::shared_token_major
        ? token * cols
        : (expert * token_count + token) * cols;
    const auto output_base = (expert * token_count + token) * rows;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < cols;
         column += blockDim.x) {
        const auto index = row * cols + column;
        const auto pair = matrix.packed[index / 2];
        const auto code = static_cast<std::uint8_t>(
            index % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent =
            static_cast<int>(matrix.scales[index / 32]) - 127;
        const auto weight = ldexpf(decode_e2m1(code), exponent);
        sum = fmaf(inputs[input_base + column], weight, sum);
    }

    extern __shared__ float partials[];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outputs[output_base + row] = partials[0];
    }
}

}  // namespace

cudaError_t launch_mxfp4_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<dim3(static_cast<unsigned>(rows), 1), threads,
                           threads * sizeof(float), stream>>>(
        input, packed, scales, output, rows, cols, 1.0F, false);
    return cudaGetLastError();
}

cudaError_t launch_mxfp4_matvec_batch(
    const float* inputs, const std::uint8_t* packed,
    const std::uint8_t* scales, float* outputs,
    std::size_t rows, std::size_t cols, std::size_t batch_size,
    cudaStream_t stream) {
    if (batch_size == 0 || batch_size > 65535) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<dim3(static_cast<unsigned>(rows),
                               static_cast<unsigned>(batch_size)),
                           threads, threads * sizeof(float), stream>>>(
        inputs, packed, scales, outputs, rows, cols, 1.0F, false);
    return cudaGetLastError();
}

cudaError_t launch_mxfp4_matvec_accumulate(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, float contribution,
    bool accumulate, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<dim3(static_cast<unsigned>(rows), 1), threads,
                           threads * sizeof(float), stream>>>(
        input, packed, scales, output, rows, cols, contribution, accumulate);
    return cudaGetLastError();
}

cudaError_t launch_mxfp4_matvec_grid(
    const float* inputs, const Mxfp4DeviceMatrix* descriptors,
    float* outputs, std::size_t rows, std::size_t cols,
    std::size_t expert_count, std::size_t token_count,
    ExpertGridInputLayout layout, cudaStream_t stream) {
    constexpr std::size_t maximum_grid_dimension = 65535;
    const auto valid_layout =
        layout == ExpertGridInputLayout::shared_token_major ||
        layout == ExpertGridInputLayout::expert_token_major;
    if (inputs == nullptr || descriptors == nullptr || outputs == nullptr ||
        rows == 0 || cols == 0 || cols % 32 != 0 ||
        rows > maximum_grid_dimension || expert_count == 0 ||
        expert_count > maximum_grid_dimension || token_count == 0 ||
        token_count > maximum_grid_dimension || !valid_layout) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = 256;
    mxfp4_matvec_grid_kernel<<<
        dim3(static_cast<unsigned>(rows), static_cast<unsigned>(token_count),
             static_cast<unsigned>(expert_count)),
        threads, threads * sizeof(float), stream>>>(
        inputs, descriptors, outputs, rows, cols, token_count, layout);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
