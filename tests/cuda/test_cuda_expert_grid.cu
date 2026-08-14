// 여러 토큰과 여러 MXFP4 전문가를 함께 계산하는 CUDA 그리드 계약을 검증합니다.
#include "mxfp4.cuh"
#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool close(float actual, float expected) {
    return std::abs(actual - expected) <= 1.0e-6F;
}

int run_grid(std::size_t expert_count, std::size_t token_count,
             k3x::cuda::ExpertGridInputLayout layout) {
    constexpr std::size_t rows = 1;
    constexpr std::size_t cols = 32;
    const std::array<std::uint8_t, 4> codes{1, 2, 3, 4};
    const std::array<float, 4> weights{0.5F, 1.0F, 1.5F, 2.0F};

    std::vector<std::uint8_t> packed(expert_count * cols / 2);
    std::vector<std::uint8_t> scales(expert_count, 127);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        packed[expert * cols / 2] =
            static_cast<std::uint8_t>(codes[expert] << 4U);
    }

    const auto input_sets = layout ==
            k3x::cuda::ExpertGridInputLayout::shared_token_major
        ? token_count
        : expert_count * token_count;
    std::vector<float> inputs(input_sets * cols);
    std::vector<float> expected(expert_count * token_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        for (std::size_t token = 0; token < token_count; ++token) {
            const auto input_set = layout ==
                    k3x::cuda::ExpertGridInputLayout::shared_token_major
                ? token
                : expert * token_count + token;
            const auto value = layout ==
                    k3x::cuda::ExpertGridInputLayout::shared_token_major
                ? static_cast<float>(token + 1) * (token % 2 == 0 ? 2.0F : -2.0F)
                : static_cast<float>(expert * 4 + token + 1);
            inputs[input_set * cols + 1] = value;
            expected[expert * token_count + token] = value * weights[expert];
        }
    }

    float* device_inputs = nullptr;
    float* device_outputs = nullptr;
    std::uint8_t* device_packed = nullptr;
    std::uint8_t* device_scales = nullptr;
    k3x::cuda::Mxfp4DeviceMatrix* device_descriptors = nullptr;
    if (cudaMalloc(&device_inputs, inputs.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_outputs, expected.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_packed, packed.size()) != cudaSuccess ||
        cudaMalloc(&device_scales, scales.size()) != cudaSuccess ||
        cudaMalloc(&device_descriptors,
                   expert_count * sizeof(k3x::cuda::Mxfp4DeviceMatrix)) !=
            cudaSuccess) {
        return 1;
    }

    std::vector<k3x::cuda::Mxfp4DeviceMatrix> descriptors(expert_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        descriptors[expert] = {
            device_packed + expert * cols / 2,
            device_scales + expert,
        };
    }
    if (cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_packed, packed.data(), packed.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_scales, scales.data(), scales.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_descriptors, descriptors.data(),
                   descriptors.size() * sizeof(descriptors.front()),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        k3x::cuda::launch_mxfp4_matvec_grid(
            device_inputs, device_descriptors, device_outputs, rows, cols,
            expert_count, token_count, layout, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return 2;
    }

    std::vector<float> actual(expected.size());
    if (cudaMemcpy(actual.data(), device_outputs,
                   actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 3;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!close(actual[index], expected[index])) return 4;
    }

    cudaFree(device_descriptors);
    cudaFree(device_scales);
    cudaFree(device_packed);
    cudaFree(device_outputs);
    cudaFree(device_inputs);
    return 0;
}

int test_backend_grid_and_fallback() {
    constexpr std::size_t expert_count = 4;
    constexpr std::size_t token_count = 4;
    constexpr std::size_t input_width = 32;
    constexpr std::size_t intermediate_width = 32;
    constexpr std::size_t output_width = 1;
    std::array<std::array<std::byte, intermediate_width * input_width / 2>,
               expert_count> gate_packed{};
    std::array<std::array<std::byte, intermediate_width * input_width / 2>,
               expert_count> up_packed{};
    std::array<std::array<std::byte, output_width * intermediate_width / 2>,
               expert_count> down_packed{};
    std::array<std::array<std::byte, intermediate_width * input_width / 32>,
               expert_count> gate_scales{};
    std::array<std::array<std::byte, intermediate_width * input_width / 32>,
               expert_count> up_scales{};
    std::array<std::array<std::byte, output_width * intermediate_width / 32>,
               expert_count> down_scales{};
    std::array<k3x::Mxfp4MlpView, expert_count> experts{};
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        gate_packed[expert][0] = std::byte{0x10};
        up_packed[expert][0] = std::byte{0x20};
        down_packed[expert][0] = std::byte{0x01};
        gate_scales[expert].fill(std::byte{127});
        up_scales[expert].fill(std::byte{127});
        down_scales[expert].fill(std::byte{127});
        experts[expert] = {
            {100 + expert * 3, gate_packed[expert], gate_scales[expert],
             intermediate_width, input_width, 32},
            {101 + expert * 3, up_packed[expert], up_scales[expert],
             intermediate_width, input_width, 32},
            {102 + expert * 3, down_packed[expert], down_scales[expert],
             output_width, intermediate_width, 32},
        };
    }
    std::array<float, token_count * input_width> inputs{};
    inputs[1] = 1.0F;
    inputs[input_width + 1] = -2.0F;
    inputs[2 * input_width + 1] = 3.0F;
    inputs[3 * input_width + 1] = -4.0F;

    auto cpu = k3x::make_cpu_backend();
    if (!cpu) return 10;
    const auto expected = cpu->mxfp4_situ_mlp_grid(
        inputs, token_count, experts, 1.0F, std::nullopt, 3,
        k3x::ProfilePhase::decode);
    if (!expected) return 11;

    const auto make_grid_backend = [](std::size_t capacity) {
        k3x::BackendOptions options;
        options.kind = k3x::BackendKind::cuda_custom;
        options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
        options.cuda_allocation = k3x::CudaAllocationMode::reused;
        options.cuda_weights = k3x::CudaWeightMode::resident;
        options.cuda_transfer = k3x::CudaTransferMode::synchronous;
        options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
        options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
        options.cuda_resident_bytes = capacity;
        return k3x::make_cuda_backend(options);
    };
    auto full = make_grid_backend(8U * 1024U * 1024U);
    auto bypass = make_grid_backend(1);
    if (!full || !bypass) return 12;
    const auto full_output = full.value()->mxfp4_situ_mlp_grid(
        inputs, token_count, experts, 1.0F, std::nullopt, 3,
        k3x::ProfilePhase::decode);
    const auto bypass_output = bypass.value()->mxfp4_situ_mlp_grid(
        inputs, token_count, experts, 1.0F, std::nullopt, 3,
        k3x::ProfilePhase::decode);
    if (!full_output || !bypass_output) {
        std::cerr << "full=" << static_cast<int>(full_output.error())
                  << " " << full_output.message()
                  << " bypass=" << static_cast<int>(bypass_output.error())
                  << " " << bypass_output.message() << '\n';
        return 13;
    }
    if (full_output.value().size() != expected.value().size() ||
        bypass_output.value().size() != expected.value().size()) return 13;
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        if (full_output.value()[expert].size() !=
                expected.value()[expert].size() ||
            bypass_output.value()[expert].size() !=
                expected.value()[expert].size()) {
            return 14;
        }
        for (std::size_t token = 0; token < token_count; ++token) {
            const auto wanted = expected.value()[expert][token];
            if (!close(full_output.value()[expert][token], wanted) ||
                !close(bypass_output.value()[expert][token], wanted)) {
                return 15;
            }
        }
    }
    const auto full_stats = full.value()->runtime_stats();
    const auto bypass_stats = bypass.value()->runtime_stats();
    if (full_stats.resident_grid_calls != 1 ||
        full_stats.resident_grid_experts != expert_count ||
        full_stats.resident_grid_tokens != token_count ||
        full_stats.resident_grid_expert_tokens != expert_count * token_count ||
        full_stats.resident_grid_kernel_launches != 4 ||
        full_stats.resident_grid_fallbacks != 0 ||
        full_stats.stream_synchronization_count != 1 ||
        full_stats.resident_grid_descriptor_h2d_bytes !=
            3 * expert_count * sizeof(k3x::cuda::Mxfp4DeviceMatrix) ||
        full_stats.weight_cache_misses != 3 * expert_count ||
        full_stats.weight_cache_hits != 0 ||
        bypass_stats.resident_grid_calls != 0 ||
        bypass_stats.resident_grid_fallbacks != 1 ||
        bypass_stats.weight_cache_bypasses == 0) {
        return 16;
    }
    return 0;
}

}  // namespace

int main() {
    for (const auto layout : {
             k3x::cuda::ExpertGridInputLayout::shared_token_major,
             k3x::cuda::ExpertGridInputLayout::expert_token_major}) {
        for (const auto dimensions : {
                 std::array<std::size_t, 2>{1, 1},
                 std::array<std::size_t, 2>{1, 4},
                 std::array<std::size_t, 2>{2, 2},
                 std::array<std::size_t, 2>{4, 4}}) {
            const auto result = run_grid(dimensions[0], dimensions[1], layout);
            if (result != 0) return result;
        }
    }

    if (k3x::cuda::launch_mxfp4_matvec_grid(
            nullptr, nullptr, nullptr, 1, 32, 0, 1,
            k3x::cuda::ExpertGridInputLayout::shared_token_major,
            nullptr) != cudaErrorInvalidValue ||
        k3x::cuda::launch_mxfp4_matvec_grid(
            nullptr, nullptr, nullptr, 1, 32, 1, 0,
            k3x::cuda::ExpertGridInputLayout::shared_token_major,
            nullptr) != cudaErrorInvalidValue ||
        k3x::cuda::launch_mxfp4_matvec_grid(
            nullptr, nullptr, nullptr, 1, 32, 65536, 1,
            k3x::cuda::ExpertGridInputLayout::shared_token_major,
            nullptr) != cudaErrorInvalidValue ||
        k3x::cuda::launch_mxfp4_matvec_grid(
            nullptr, nullptr, nullptr, 1, 32, 1, 65536,
            k3x::cuda::ExpertGridInputLayout::shared_token_major,
            nullptr) != cudaErrorInvalidValue ||
        k3x::cuda::launch_mxfp4_matvec_grid(
            nullptr, nullptr, nullptr, 1, 32, 1, 1,
            static_cast<k3x::cuda::ExpertGridInputLayout>(255),
            nullptr) != cudaErrorInvalidValue) {
        return 5;
    }
    return test_backend_grid_and_fallback();
}
