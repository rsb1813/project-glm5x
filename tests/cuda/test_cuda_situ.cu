// GPU SiTU-GLU의 strict FP32 계산과 BF16 반올림 출력을 검증합니다.
#include "k3x/ops.hpp"
#include "situ.cuh"

#include <cuda_runtime_api.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

std::uint16_t round_to_bf16_bits(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

bool nearly_equal(float actual, float expected) {
    return std::abs(actual - expected) <= 1.0e-6F;
}

int test_fp32() {
    const std::array<float, 6> gate{0.0F, 1.0F, -1.0F, 20.0F, -20.0F, 0.37F};
    const std::array<float, 6> up{1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 0.83F};
    std::array<float, 6> expected{};
    k3x::situ_glu(expected, gate, up, 2.0F, 1.5F);

    float* device_gate = nullptr;
    float* device_up = nullptr;
    float* device_output = nullptr;
    if (cudaMalloc(&device_gate, gate.size() * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMalloc(&device_up, up.size() * sizeof(float)) != cudaSuccess) return 2;
    if (cudaMalloc(&device_output, (expected.size() + 1) * sizeof(float)) != cudaSuccess) return 3;
    const std::array<float, 7> initial{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1234.5F};
    if (cudaMemcpy(device_gate, gate.data(), gate.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 4;
    if (cudaMemcpy(device_up, up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 5;
    if (cudaMemcpy(device_output, initial.data(), initial.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) return 6;
    if (k3x::cuda::launch_situ_glu(device_gate, device_up, device_output,
            expected.size(), 2.0F, true, 1.5F, false, nullptr) != cudaSuccess) return 7;
    if (cudaDeviceSynchronize() != cudaSuccess) return 8;
    std::array<float, 7> actual{};
    if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return 9;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!nearly_equal(actual[index], expected[index])) return 10;
    }
    if (actual.back() != initial.back()) return 11;
    cudaFree(device_output);
    cudaFree(device_up);
    cudaFree(device_gate);
    return 0;
}

int test_bf16() {
    const std::array<float, 6> gate{0.0F, 1.0F, -1.0F, 20.0F, -20.0F, 0.37F};
    const std::array<float, 6> up{1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 0.83F};
    std::array<float, 6> expected{};
    k3x::situ_glu(expected, gate, up, 2.0F, std::nullopt);

    float* device_gate = nullptr;
    float* device_up = nullptr;
    std::uint16_t* device_output = nullptr;
    if (cudaMalloc(&device_gate, gate.size() * sizeof(float)) != cudaSuccess) return 20;
    if (cudaMalloc(&device_up, up.size() * sizeof(float)) != cudaSuccess) return 21;
    if (cudaMalloc(&device_output, (expected.size() + 1) * sizeof(std::uint16_t)) != cudaSuccess) return 22;
    const std::array<std::uint16_t, 7> initial{0, 0, 0, 0, 0, 0, 0x5A5A};
    cudaMemcpy(device_gate, gate.data(), gate.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_up, up.data(), up.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, initial.data(), initial.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice);
    if (k3x::cuda::launch_situ_glu(device_gate, device_up, device_output,
            expected.size(), 2.0F, false, 0.0F, true, nullptr) != cudaSuccess) return 23;
    if (cudaDeviceSynchronize() != cudaSuccess) return 24;
    std::array<std::uint16_t, 7> actual{};
    cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (actual[index] != round_to_bf16_bits(expected[index])) return 25;
    }
    if (actual.back() != initial.back()) return 26;
    cudaFree(device_output);
    cudaFree(device_up);
    cudaFree(device_gate);
    return 0;
}

int test_validation() {
    float value = 0.0F;
    if (k3x::cuda::launch_situ_glu(nullptr, &value, &value, 1, 1.0F,
            false, 0.0F, false, nullptr) != cudaErrorInvalidValue) return 30;
    if (k3x::cuda::launch_situ_glu(&value, &value, &value, 0, 1.0F,
            false, 0.0F, false, nullptr) != cudaErrorInvalidValue) return 31;
    if (k3x::cuda::launch_situ_glu(&value, &value, &value, 1, 0.0F,
            false, 0.0F, false, nullptr) != cudaErrorInvalidValue) return 32;
    if (k3x::cuda::launch_situ_glu(&value, &value, &value, 1, 1.0F,
            true, std::numeric_limits<float>::quiet_NaN(), false, nullptr) !=
        cudaErrorInvalidValue) return 33;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = test_fp32()) return result;
    if (const auto result = test_bf16()) return result;
    return test_validation();
}
