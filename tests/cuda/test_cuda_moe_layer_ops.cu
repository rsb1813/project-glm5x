// resident MoE layer의 ordered mix, strict RMSNorm, 최종 합산 CUDA primitive를 검증합니다.
#include "k3x/ops.hpp"
#include "moe_layer.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace {

bool close(float actual, float expected) {
    return std::abs(actual - expected) <= 1.0e-6F;
}

int test_values() {
    const std::array<float, 8> expert_outputs{
        1.0F, 2.0F, 3.0F, 4.0F,
        -2.0F, 1.0F, 0.5F, 8.0F,
    };
    const std::array<float, 2> contributions{0.75F, -0.25F};
    const std::array<float, 4> expected_mix{1.25F, 1.25F, 2.125F, 1.0F};
    const std::array<float, 4> norm_weight{1.0F, 0.5F, 2.0F, 1.5F};
    const std::array<float, 4> residual{0.25F, -0.5F, 1.0F, 2.0F};
    std::array<float, 4> expected_norm{};
    k3x::rms_norm(expected_norm, expected_mix, norm_weight, 1.0e-5F);

    float* device_outputs = nullptr;
    float* device_contributions = nullptr;
    float* device_mix = nullptr;
    float* device_weight = nullptr;
    float* device_norm = nullptr;
    float* device_residual = nullptr;
    float* device_sum = nullptr;
    if (cudaMalloc(&device_outputs, expert_outputs.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&device_contributions,
                   contributions.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_mix, expected_mix.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&device_weight, norm_weight.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&device_norm, expected_norm.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&device_residual, residual.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&device_sum, residual.size() * sizeof(float)) !=
            cudaSuccess) {
        return 1;
    }
    cudaMemcpy(device_outputs, expert_outputs.data(),
               expert_outputs.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_contributions, contributions.data(),
               contributions.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weight, norm_weight.data(),
               norm_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_residual, residual.data(),
               residual.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (k3x::cuda::launch_ordered_expert_mix(
            device_outputs, device_contributions, contributions, device_mix,
            expected_mix.size(), nullptr) != cudaSuccess) return 2;
    if (k3x::cuda::launch_strict_rms_norm(
            device_mix, device_weight, device_norm, expected_norm.size(),
            1.0e-5F, nullptr) != cudaSuccess) return 3;
    if (k3x::cuda::launch_vector_add(
            device_norm, device_residual, device_sum, residual.size(),
            nullptr) != cudaSuccess) return 4;
    if (cudaDeviceSynchronize() != cudaSuccess) return 5;

    std::array<float, 4> actual_mix{};
    std::array<float, 4> actual_norm{};
    std::array<float, 4> actual_sum{};
    cudaMemcpy(actual_mix.data(), device_mix,
               actual_mix.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_norm.data(), device_norm,
               actual_norm.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(actual_sum.data(), device_sum,
               actual_sum.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (std::size_t row = 0; row < expected_mix.size(); ++row) {
        if (!close(actual_mix[row], expected_mix[row]) ||
            !close(actual_norm[row], expected_norm[row]) ||
            !close(actual_sum[row], expected_norm[row] + residual[row])) {
            return 6;
        }
    }

    cudaFree(device_sum);
    cudaFree(device_residual);
    cudaFree(device_norm);
    cudaFree(device_weight);
    cudaFree(device_mix);
    cudaFree(device_contributions);
    cudaFree(device_outputs);
    return 0;
}

int test_validation() {
    float value = 0.0F;
    const std::array<float, 1> contribution{1.0F};
    const std::array<float, 1> nonfinite{
        std::numeric_limits<float>::infinity()};
    if (k3x::cuda::launch_ordered_expert_mix(
            nullptr, &value, contribution, &value, 1, nullptr) !=
        cudaErrorInvalidValue) return 10;
    if (k3x::cuda::launch_ordered_expert_mix(
            &value, &value, {}, &value, 1, nullptr) !=
        cudaErrorInvalidValue) return 11;
    if (k3x::cuda::launch_ordered_expert_mix(
            &value, &value, nonfinite, &value, 1, nullptr) !=
        cudaErrorInvalidValue) return 12;
    if (k3x::cuda::launch_ordered_expert_mix(
            &value, &value, contribution, &value, 0, nullptr) !=
        cudaErrorInvalidValue) return 13;
    if (k3x::cuda::launch_strict_rms_norm(
            nullptr, &value, &value, 1, 1.0e-5F, nullptr) !=
        cudaErrorInvalidValue) return 14;
    if (k3x::cuda::launch_strict_rms_norm(
            &value, &value, &value, 0, 1.0e-5F, nullptr) !=
        cudaErrorInvalidValue) return 15;
    if (k3x::cuda::launch_strict_rms_norm(
            &value, &value, &value, 1, 0.0F, nullptr) !=
        cudaErrorInvalidValue) return 16;
    if (k3x::cuda::launch_strict_rms_norm(
            &value, &value, &value, 1,
            std::numeric_limits<float>::quiet_NaN(), nullptr) !=
        cudaErrorInvalidValue) return 17;
    if (k3x::cuda::launch_vector_add(
            nullptr, &value, &value, 1, nullptr) != cudaErrorInvalidValue) {
        return 18;
    }
    if (k3x::cuda::launch_vector_add(
            &value, &value, &value, 0, nullptr) != cudaErrorInvalidValue) {
        return 19;
    }
    return 0;
}

}  // namespace

int main() {
    if (const auto result = test_values()) return result;
    return test_validation();
}
