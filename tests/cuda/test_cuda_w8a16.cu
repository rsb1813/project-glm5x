// W8A16 expert-major CUDA kernel을 quantized CPU reference와 비교합니다.
#include "w8a16.cuh"

#include "k3x/w8a16.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

std::uint16_t to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float from_bf16(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::vector<std::byte> encode(std::span<const float> values) {
    std::vector<std::byte> result(values.size() * 2);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto value = to_bf16(values[index]);
        result[index * 2] = std::byte(value & 0xffU);
        result[index * 2 + 1] = std::byte(value >> 8U);
    }
    return result;
}

template <typename T>
bool allocate_copy(T*& device, std::span<const T> host) {
    return cudaMalloc(reinterpret_cast<void**>(&device), host.size_bytes()) ==
               cudaSuccess &&
           cudaMemcpy(device, host.data(), host.size_bytes(),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

int main() {
    constexpr std::size_t input_width = 128;
    constexpr std::size_t intermediate = 128;
    constexpr std::size_t output_width = 128;
    std::vector<float> input(input_width);
    std::vector<float> gate(intermediate * input_width);
    std::vector<float> up(intermediate * input_width);
    std::vector<float> down(output_width * intermediate);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) /
                       16.0F;
    }
    for (std::size_t index = 0; index < gate.size(); ++index) {
        gate[index] = static_cast<float>(static_cast<int>(index % 23) - 11) /
                      64.0F;
        up[index] = static_cast<float>(static_cast<int>(index % 29) - 14) /
                    72.0F;
        down[index] = static_cast<float>(static_cast<int>(index % 31) - 15) /
                      80.0F;
    }
    const auto gate_packed = k3x::pack_w8a16_bf16(
        encode(gate), intermediate, input_width);
    const auto up_packed = k3x::pack_w8a16_bf16(
        encode(up), intermediate, input_width);
    const auto down_packed = k3x::pack_w8a16_bf16(
        encode(down), output_width, intermediate);
    if (!gate_packed || !up_packed || !down_packed) return 1;
    const auto gate_decoded = k3x::decode_w8a16(gate_packed.value());
    const auto up_decoded = k3x::decode_w8a16(up_packed.value());
    const auto down_decoded = k3x::decode_w8a16(down_packed.value());
    if (!gate_decoded || !up_decoded || !down_decoded) return 2;

    std::vector<__nv_bfloat16> bf16_input(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        bf16_input[index] = __float2bfloat16_rn(input[index]);
    }
    __nv_bfloat16* device_input = nullptr;
    std::int8_t *device_gate = nullptr, *device_up = nullptr,
                *device_down = nullptr;
    __nv_bfloat16 *device_gate_scales = nullptr, *device_up_scales = nullptr,
                   *device_down_scales = nullptr;
    __nv_bfloat16* device_intermediate = nullptr;
    float* device_output = nullptr;
    k3x::cuda::W8A16DeviceMatrix *device_gate_view = nullptr,
                                 *device_up_view = nullptr,
                                 *device_down_view = nullptr;
    if (!allocate_copy(
            device_input, std::span<const __nv_bfloat16>(bf16_input)) ||
        !allocate_copy(
            device_gate,
            std::span<const std::int8_t>(gate_packed.value().values)) ||
        !allocate_copy(
            device_up,
            std::span<const std::int8_t>(up_packed.value().values)) ||
        !allocate_copy(
            device_down,
            std::span<const std::int8_t>(down_packed.value().values)) ||
        !allocate_copy(
            device_gate_scales,
            std::span(reinterpret_cast<const __nv_bfloat16*>(
                          gate_packed.value().scales.data()),
                      gate_packed.value().scales.size() / 2)) ||
        !allocate_copy(
            device_up_scales,
            std::span(reinterpret_cast<const __nv_bfloat16*>(
                          up_packed.value().scales.data()),
                      up_packed.value().scales.size() / 2)) ||
        !allocate_copy(
            device_down_scales,
            std::span(reinterpret_cast<const __nv_bfloat16*>(
                          down_packed.value().scales.data()),
                      down_packed.value().scales.size() / 2)) ||
        cudaMalloc(reinterpret_cast<void**>(&device_intermediate),
                   intermediate * sizeof(__nv_bfloat16)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output),
                   output_width * sizeof(float)) != cudaSuccess) {
        return 3;
    }
    const k3x::cuda::W8A16DeviceMatrix gate_view{
        device_gate, device_gate_scales};
    const k3x::cuda::W8A16DeviceMatrix up_view{device_up, device_up_scales};
    const k3x::cuda::W8A16DeviceMatrix down_view{
        device_down, device_down_scales};
    if (!allocate_copy(
            device_gate_view,
            std::span<const k3x::cuda::W8A16DeviceMatrix>(&gate_view, 1)) ||
        !allocate_copy(
            device_up_view,
            std::span<const k3x::cuda::W8A16DeviceMatrix>(&up_view, 1)) ||
        !allocate_copy(
            device_down_view,
            std::span<const k3x::cuda::W8A16DeviceMatrix>(&down_view, 1))) {
        return 4;
    }
    if (k3x::cuda::launch_w8a16_gate_up(
            device_input, device_gate_view, device_up_view,
            device_intermediate, intermediate, input_width, 1, nullptr) !=
            cudaSuccess ||
        k3x::cuda::launch_w8a16_down(
            device_intermediate, device_down_view, device_output,
            output_width, intermediate, 1, nullptr) != cudaSuccess) {
        return 5;
    }
    std::vector<float> actual(output_width);
    if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 6;
    }

    std::vector<float> activated(intermediate);
    for (std::size_t row = 0; row < intermediate; ++row) {
        float gate_sum = 0.0F;
        float up_sum = 0.0F;
        for (std::size_t column = 0; column < input_width; ++column) {
            const auto value = from_bf16(to_bf16(input[column]));
            gate_sum += gate_decoded.value()[row * input_width + column] * value;
            up_sum += up_decoded.value()[row * input_width + column] * value;
        }
        gate_sum = from_bf16(to_bf16(gate_sum));
        up_sum = from_bf16(to_bf16(up_sum));
        activated[row] = from_bf16(to_bf16(
            (gate_sum / (1.0F + std::exp(-gate_sum))) * up_sum));
    }
    float maximum_error = 0.0F;
    for (std::size_t row = 0; row < output_width; ++row) {
        float expected = 0.0F;
        for (std::size_t column = 0; column < intermediate; ++column) {
            expected += down_decoded.value()[row * intermediate + column] *
                        activated[column];
        }
        expected = from_bf16(to_bf16(expected));
        maximum_error = std::max(maximum_error, std::abs(actual[row] - expected));
    }
    if (!std::isfinite(maximum_error) || maximum_error > 0.02F) return 7;
    if (k3x::cuda::launch_w8a16_gate_up(
            device_input, device_gate_view, device_up_view,
            device_intermediate, intermediate, input_width - 1, 1,
            nullptr) != cudaErrorInvalidValue) {
        return 8;
    }
    cudaFree(device_gate_view);
    cudaFree(device_up_view);
    cudaFree(device_down_view);
    cudaFree(device_input);
    cudaFree(device_gate);
    cudaFree(device_up);
    cudaFree(device_down);
    cudaFree(device_gate_scales);
    cudaFree(device_up_scales);
    cudaFree(device_down_scales);
    cudaFree(device_intermediate);
    cudaFree(device_output);
    return 0;
}
