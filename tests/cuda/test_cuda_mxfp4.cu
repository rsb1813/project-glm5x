// native K3 MXFP4 CUDA matvec의 수치 결과와 프로파일 계약을 검증합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"
#include "mxfp4.cuh"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <span>

namespace {

bool nearly_equal(float actual, float expected, float tolerance = 1.0e-4F) {
    return std::abs(actual - expected) <= tolerance;
}

int test_rejects_incompatible_contracts() {
    std::array<float, 32> input{};
    input[1] = 2.0F;
    std::array<std::byte, 17> packed{};
    packed[0] = std::byte{0x10};
    std::array<std::byte, 2> scales{std::byte{127}, std::byte{127}};

    k3x::Profiler custom_profiler;
    k3x::BackendOptions custom_options;
    custom_options.kind = k3x::BackendKind::cuda_custom;
    auto custom = k3x::make_cuda_backend(custom_options, &custom_profiler);
    if (!custom) return 22;

    const auto wrong_group = custom.value()->mxfp4_matvec(
        input, std::span<const std::byte>(packed).first(16), scales,
        1, 32, 16, 1, k3x::ProfilePhase::decode);
    if (wrong_group || wrong_group.error() != k3x::ErrorCode::invalid_mxfp4) {
        std::cerr << "non-32 MXFP4 group size was not rejected\n";
        return 23;
    }

    const auto extra_packed = custom.value()->mxfp4_matvec(
        input, packed, std::span<const std::byte>(scales).first(1),
        1, 32, 32, 2, k3x::ProfilePhase::decode);
    if (extra_packed || extra_packed.error() != k3x::ErrorCode::invalid_mxfp4) return 24;

    const auto extra_scale = custom.value()->mxfp4_matvec(
        input, std::span<const std::byte>(packed).first(16), scales,
        1, 32, 32, 3, k3x::ProfilePhase::decode);
    if (extra_scale || extra_scale.error() != k3x::ErrorCode::invalid_mxfp4) return 25;

    scales[0] = std::byte{0xFF};
    const auto reserved_scale = custom.value()->mxfp4_matvec(
        input, std::span<const std::byte>(packed).first(16),
        std::span<const std::byte>(scales).first(1),
        1, 32, 32, 4, k3x::ProfilePhase::decode);
    if (reserved_scale || reserved_scale.error() != k3x::ErrorCode::invalid_mxfp4) return 26;

    const auto custom_summary = custom_profiler.summary();
    if (custom_summary.failed_operations != 4) return 27;
    if (custom_summary.host_to_device_bytes != 0 ||
        custom_summary.device_to_host_bytes != 0) return 28;

    k3x::Profiler dense_profiler;
    k3x::BackendOptions dense_options;
    dense_options.kind = k3x::BackendKind::cuda_dense;
    auto dense = k3x::make_cuda_backend(dense_options, &dense_profiler);
    if (!dense) return 29;
    scales[0] = std::byte{127};
    const auto incompatible_backend = dense.value()->mxfp4_matvec(
        input, std::span<const std::byte>(packed).first(16),
        std::span<const std::byte>(scales).first(1),
        1, 32, 32, 5, k3x::ProfilePhase::decode);
    if (!incompatible_backend || incompatible_backend.value().size() != 1 ||
        !nearly_equal(incompatible_backend.value()[0], 1.0F)) {
        std::cerr << "cuda_dense did not retain the CPU MXFP4 oracle\n";
        return 30;
    }
    const auto dense_summary = dense_profiler.summary();
    if (dense_summary.failed_operations != 0) return 31;
    if (dense_summary.host_to_device_bytes != 0 ||
        dense_summary.device_to_host_bytes != 0 ||
        dense_summary.device_nanoseconds != 0) return 32;
    std::size_t dense_mxfp4_events = 0;
    for (const auto& event : dense_profiler.events()) {
        if (event.operation == k3x::ProfileOperation::dense_matvec) return 33;
        if (event.operation != k3x::ProfileOperation::mxfp4_matvec) continue;
        ++dense_mxfp4_events;
        if (!event.success || event.device_nanoseconds != 0) return 34;
        if (event.precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 35;
    }
    if (dense_mxfp4_events != 1) return 36;
    return 0;
}

int test_strides_columns_beyond_one_block_width() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 37;

    std::array<float, 320> input{};
    input[256] = 2.0F;
    input[257] = 3.0F;
    std::array<std::byte, 160> packed{};
    packed[128] = std::byte{0x21};
    std::array<std::byte, 10> scales{};
    scales.fill(std::byte{127});

    const auto output = backend.value()->mxfp4_matvec(
        input, packed, scales, 1, 320, 32, 6, k3x::ProfilePhase::decode);
    if (!output) return 38;
    if (output.value().size() != 1 ||
        !nearly_equal(output.value()[0], 4.0F)) {
        std::cerr << "MXFP4 columns beyond one block width were skipped\n";
        return 39;
    }
    return 0;
}

int test_allocation_modes() {
    std::array<float, 64> input{};
    input[0] = 1.0F;
    std::array<std::byte, 128> packed{};
    packed[0] = std::byte{0x02};
    packed[32] = std::byte{0x02};
    packed[64] = std::byte{0x02};
    packed[96] = std::byte{0x02};
    std::array<std::byte, 8> scales{};
    scales.fill(std::byte{127});
    const k3x::Mxfp4WeightView view{
        201, std::span<const std::byte>(packed).first(96),
        std::span<const std::byte>(scales).first(6), 3, 64, 32};

    k3x::BackendOptions reference_options;
    reference_options.kind = k3x::BackendKind::cuda_custom;
    auto reference = k3x::make_cuda_backend(reference_options);
    if (!reference) return 40;
    if (!reference.value()->mxfp4_matvec(
            input, view, 7, k3x::ProfilePhase::decode)) return 41;
    const auto reference_stats = reference.value()->runtime_stats();
    if (reference_stats.device_allocation_count != 4 ||
        reference_stats.device_free_count != 4 ||
        reference_stats.stream_synchronization_count != 1) return 42;

    k3x::BackendOptions reused_options;
    reused_options.kind = k3x::BackendKind::cuda_custom;
    reused_options.cuda_allocation = k3x::CudaAllocationMode::reused;
    auto reused = k3x::make_cuda_backend(reused_options);
    if (!reused) return 43;
    const auto first = reused.value()->mxfp4_matvec(
        input, view, 7, k3x::ProfilePhase::decode);
    if (!first) return 44;
    const auto reused_first = reused.value()->runtime_stats();
    if (reused_first.device_allocation_count != 4 ||
        reused_first.device_free_count != 0 ||
        reused_first.stream_synchronization_count != 1 ||
        reused_first.scratch_bytes != 370) return 45;
    const auto second = reused.value()->mxfp4_matvec(
        input, view, 7, k3x::ProfilePhase::decode);
    if (!second || second.value() != first.value()) return 46;
    const auto reused_second = reused.value()->runtime_stats();
    if (reused_second.device_allocation_count !=
            reused_first.device_allocation_count ||
        reused_second.device_free_count != reused_first.device_free_count ||
        reused_second.stream_synchronization_count !=
            reused_first.stream_synchronization_count + 1) return 47;

    const auto larger = reused.value()->mxfp4_matvec(
        input, k3x::Mxfp4WeightView{202, packed, scales, 4, 64, 32}, 7,
        k3x::ProfilePhase::decode);
    if (!larger) return 48;
    const auto reused_larger = reused.value()->runtime_stats();
    if (reused_larger.device_allocation_count !=
            reused_second.device_allocation_count + 3 ||
        reused_larger.device_free_count !=
            reused_second.device_free_count + 3 ||
        reused_larger.scratch_bytes != 408 ||
        reused_larger.stream_synchronization_count !=
            reused_second.stream_synchronization_count + 1) return 49;
    return 0;
}

int test_grouped_resident_execution() {
    std::array<float, 64> input{};
    input[0] = 1.0F;
    input[1] = 2.0F;
    input[2] = 1.5F;
    input[3] = -0.5F;
    input[32] = 0.25F;
    input[33] = -0.5F;
    std::array<std::byte, 96> packed{};
    packed[0] = std::byte{0x10};
    packed[1] = std::byte{0x72};
    packed[16] = std::byte{0xD4};
    packed[32] = std::byte{0x96};
    packed[48] = std::byte{0x23};
    packed[64] = std::byte{0xF5};
    packed[80] = std::byte{0x4A};
    const std::array<std::byte, 6> scales{
        std::byte{127}, std::byte{128}, std::byte{126},
        std::byte{129}, std::byte{125}, std::byte{127},
    };
    const std::array<k3x::Mxfp4WeightView, 2> weights{{
        {401, packed, scales, 3, 64, 32},
        {402, packed, scales, 3, 64, 32},
    }};
    const std::vector<float> expected{3.5F, 1.0F, -3.5F};
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::grouped;
    options.cuda_resident_bytes = 204;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 60;
    for (const auto& weight : weights) {
        const auto warm = backend.value()->mxfp4_matvec(
            input, weight, 12, k3x::ProfilePhase::decode);
        if (!warm || warm.value() != expected) return 61;
    }
    const auto before = backend.value()->runtime_stats();
    const auto output = backend.value()->mxfp4_matvec_group(
        input, weights, 12, k3x::ProfilePhase::decode);
    if (!output || output.value().size() != 2 ||
        output.value()[0] != expected || output.value()[1] != expected) return 62;
    const auto after = backend.value()->runtime_stats();
    if (after.weight_cache_hits != before.weight_cache_hits + 2 ||
        after.weight_h2d_bytes != before.weight_h2d_bytes ||
        after.activation_h2d_bytes !=
            before.activation_h2d_bytes + input.size() * sizeof(float) ||
        after.stream_synchronization_count !=
            before.stream_synchronization_count + 1 ||
        after.grouped_projection_calls !=
            before.grouped_projection_calls + 1 ||
        after.grouped_projection_members !=
            before.grouped_projection_members + 2) return 63;
    return 0;
}

int test_scaled_ordered_accumulation() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 320;
    std::array<float, cols> input{};
    input[0] = 1.0F;
    input[1] = -2.0F;
    input[256] = 2.0F;
    input[257] = 3.0F;
    std::array<std::byte, rows * cols / 2> packed{};
    packed[0] = std::byte{0x71};
    packed[128] = std::byte{0x21};
    packed[cols / 2] = std::byte{0xD4};
    packed[cols / 2 + 128] = std::byte{0xF5};
    std::array<std::byte, rows * cols / 32> scales{};
    scales.fill(std::byte{127});
    scales[8] = std::byte{128};
    scales[rows * cols / 32 - 2] = std::byte{126};

    const auto oracle = k3x::mxfp4_matmul(
        input, packed, scales, rows, cols, 32);
    if (!oracle) return 70;

    float* device_input = nullptr;
    std::byte* device_packed = nullptr;
    std::byte* device_scales = nullptr;
    float* device_output = nullptr;
    if (cudaMalloc(&device_input, input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_packed, packed.size()) != cudaSuccess ||
        cudaMalloc(&device_scales, scales.size()) != cudaSuccess ||
        cudaMalloc(&device_output, rows * sizeof(float)) != cudaSuccess) {
        return 71;
    }
    if (cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_packed, packed.data(), packed.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_scales, scales.data(), scales.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return 72;
    }

    if (k3x::cuda::launch_mxfp4_matvec_accumulate(
            device_input,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_output, rows, cols, -0.5F, false, nullptr) != cudaSuccess ||
        k3x::cuda::launch_mxfp4_matvec_accumulate(
            device_input,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_output, rows, cols, 0.25F, true, nullptr) != cudaSuccess ||
        k3x::cuda::launch_mxfp4_matvec_accumulate(
            device_input,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_output, rows, cols, 0.0F, true, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return 73;
    }

    std::array<float, rows> actual{};
    if (cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 74;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        const auto expected = -0.25F * oracle.value()[row];
        if (!nearly_equal(actual[row], expected)) return 75;
    }

    cudaFree(device_output);
    cudaFree(device_scales);
    cudaFree(device_packed);
    cudaFree(device_input);
    return 0;
}

int test_batched_token_rows() {
    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 64;
    constexpr std::size_t batch_size = 3;
    std::array<float, batch_size * cols> inputs{};
    inputs[0] = 1.0F;
    inputs[1] = 2.0F;
    inputs[2] = 1.5F;
    inputs[3] = -0.5F;
    inputs[cols] = -2.0F;
    inputs[cols + 1] = 0.25F;
    inputs[2 * cols + 32] = 4.0F;
    inputs[2 * cols + 33] = -3.0F;

    std::array<std::byte, 96> packed{};
    packed[0] = std::byte{0x10};
    packed[1] = std::byte{0x72};
    packed[16] = std::byte{0xD4};
    packed[32] = std::byte{0x96};
    packed[48] = std::byte{0x23};
    packed[64] = std::byte{0xF5};
    packed[80] = std::byte{0x4A};
    const std::array<std::byte, 6> scales{
        std::byte{127}, std::byte{128}, std::byte{126},
        std::byte{129}, std::byte{125}, std::byte{127},
    };

    std::array<float, batch_size * rows> expected{};
    for (std::size_t token = 0; token < batch_size; ++token) {
        const auto oracle = k3x::mxfp4_matmul(
            std::span<const float>(inputs).subspan(token * cols, cols),
            packed, scales, rows, cols, 32);
        if (!oracle) return 80;
        std::copy(oracle.value().begin(), oracle.value().end(),
                  expected.begin() + token * rows);
    }

    float* device_inputs = nullptr;
    std::byte* device_packed = nullptr;
    std::byte* device_scales = nullptr;
    float* device_outputs = nullptr;
    if (cudaMalloc(&device_inputs, inputs.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_packed, packed.size()) != cudaSuccess ||
        cudaMalloc(&device_scales, scales.size()) != cudaSuccess ||
        cudaMalloc(&device_outputs, expected.size() * sizeof(float)) !=
            cudaSuccess) {
        return 81;
    }
    if (cudaMemcpy(device_inputs, inputs.data(), inputs.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_packed, packed.data(), packed.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_scales, scales.data(), scales.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return 82;
    }

    if (k3x::cuda::launch_mxfp4_matvec_batch(
            device_inputs,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_outputs, rows, cols, batch_size, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return 83;
    }
    std::array<float, batch_size * rows> actual{};
    if (cudaMemcpy(actual.data(), device_outputs, actual.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 84;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!nearly_equal(actual[index], expected[index])) return 85;
    }

    if (k3x::cuda::launch_mxfp4_matvec(
            device_inputs,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_outputs, rows, cols, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return 86;
    }
    std::array<float, rows> scalar{};
    if (cudaMemcpy(scalar.data(), device_outputs, scalar.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        k3x::cuda::launch_mxfp4_matvec_batch(
            device_inputs,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_outputs, rows, cols, 1, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        return 87;
    }
    std::array<float, rows> batch_one{};
    if (cudaMemcpy(batch_one.data(), device_outputs,
                   batch_one.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        scalar != batch_one ||
        k3x::cuda::launch_mxfp4_matvec_batch(
            device_inputs,
            reinterpret_cast<const std::uint8_t*>(device_packed),
            reinterpret_cast<const std::uint8_t*>(device_scales),
            device_outputs, rows, cols, 0, nullptr) != cudaErrorInvalidValue) {
        return 88;
    }

    cudaFree(device_outputs);
    cudaFree(device_scales);
    cudaFree(device_packed);
    cudaFree(device_inputs);
    return 0;
}

}  // namespace

int main() {
    k3x::Profiler profiler;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) {
        std::cerr << backend.message() << '\n';
        return 1;
    }
    if (backend.value()->kind() != k3x::BackendKind::cuda_custom) return 2;

    std::array<float, 64> input{};
    input[0] = 1.0F;
    input[1] = 2.0F;
    input[2] = 1.5F;
    input[3] = -0.5F;
    input[32] = 0.25F;
    input[33] = -0.5F;

    std::array<std::byte, 96> packed{};
    packed[0] = std::byte{0x10};
    packed[1] = std::byte{0x72};
    packed[16] = std::byte{0xD4};
    packed[32] = std::byte{0x96};
    packed[48] = std::byte{0x23};
    packed[64] = std::byte{0xF5};
    packed[80] = std::byte{0x4A};
    const std::array<std::byte, 6> scales{
        std::byte{127}, std::byte{128}, std::byte{126},
        std::byte{129}, std::byte{125}, std::byte{127},
    };

    const auto oracle = k3x::mxfp4_matmul(input, packed, scales, 3, 64, 32);
    if (!oracle) return 3;
    const std::array<float, 3> literal{3.5F, 1.0F, -3.5F};
    for (std::size_t row = 0; row < literal.size(); ++row) {
        if (!nearly_equal(oracle.value()[row], literal[row])) return 4;
    }

    const auto output = backend.value()->mxfp4_matvec(
        input, packed, scales, 3, 64, 32, 23, k3x::ProfilePhase::decode);
    if (!output) {
        std::cerr << output.message() << '\n';
        return 5;
    }
    if (output.value().size() != literal.size()) return 6;
    for (std::size_t row = 0; row < literal.size(); ++row) {
        if (!nearly_equal(output.value()[row], oracle.value()[row])) return 7;
    }

    std::size_t mxfp4_events = 0;
    for (const auto& event : profiler.events()) {
        if (event.operation == k3x::ProfileOperation::dense_matvec) return 8;
        if (event.operation != k3x::ProfileOperation::mxfp4_matvec) continue;
        ++mxfp4_events;
        if (!event.success) return 9;
        if (event.precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 10;
        if (event.layer != 23) return 11;
        if (event.phase != k3x::ProfilePhase::decode) return 12;
        if (event.device_nanoseconds == 0) return 13;
        if (event.logical_bytes != packed.size() + scales.size()) return 14;
    }
    if (mxfp4_events != 1) return 15;

    const auto summary = profiler.summary();
    if (summary.failed_operations != 0) return 16;
    if (summary.host_to_device_bytes !=
        input.size() * sizeof(float) + packed.size() + scales.size()) return 17;
    if (summary.device_to_host_bytes != literal.size() * sizeof(float)) return 18;
    if (summary.logical_bytes != packed.size() + scales.size()) return 19;
    if (backend.value()->memory_stats().current_device_bytes != 0) return 20;
    const auto stride_result = test_strides_columns_beyond_one_block_width();
    if (stride_result != 0) return stride_result;
    const auto contract_result = test_rejects_incompatible_contracts();
    if (contract_result != 0) return contract_result;
    const auto allocation_result = test_allocation_modes();
    if (allocation_result != 0) return allocation_result;
    const auto grouped_result = test_grouped_resident_execution();
    if (grouped_result != 0) return grouped_result;
    const auto accumulation_result = test_scaled_ordered_accumulation();
    if (accumulation_result != 0) return accumulation_result;
    return test_batched_token_rows();
}
