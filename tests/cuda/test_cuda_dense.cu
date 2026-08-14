// cuBLASLt FP32 dense matvec의 수치 결과와 전송 프로파일을 검증합니다.
#include "k3x/backend.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

bool nearly_equal(float actual, float expected, float tolerance) {
    return std::abs(actual - expected) <= tolerance;
}

float round_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF0000U);
}

int test_fp32() {
    k3x::Profiler profiler;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.dense_precision = k3x::DensePrecision::fp32;

    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) {
        std::cerr << backend.message() << '\n';
        return 1;
    }

    const std::vector<float> input{1.0F, 2.0F, 3.0F};
    const std::vector<float> weight{
        1.0F, 0.0F, -1.0F,
        0.5F, 2.0F, 1.0F,
    };
    const auto output = backend.value()->dense_matvec(
        input, weight, 2, 3, 7, k3x::ProfilePhase::decode);
    if (!output) {
        std::cerr << output.message() << '\n';
        return 2;
    }
    if (output.value().size() != 2) return 3;
    if (!nearly_equal(output.value()[0], -2.0F, 1.0e-5F)) return 4;
    if (!nearly_equal(output.value()[1], 7.5F, 1.0e-5F)) return 5;

    std::size_t dense_events = 0;
    for (const auto& event : profiler.events()) {
        if (event.operation != k3x::ProfileOperation::dense_matvec) continue;
        ++dense_events;
        if (!event.success) return 6;
        if (event.precision != k3x::NumericPrecision::fp32) return 7;
        if (event.layer != 7) return 8;
        if (event.phase != k3x::ProfilePhase::decode) return 9;
        if (event.device_nanoseconds == 0) return 10;
    }
    if (dense_events != 1) return 11;

    const auto summary = profiler.summary();
    if (summary.failed_operations != 0) return 12;
    if (summary.host_to_device_bytes != 36) return 13;
    if (summary.device_to_host_bytes != 8) return 14;
    if (backend.value()->memory_stats().current_device_bytes != 0) return 15;
    return 0;
}

int test_bf16_rounded() {
    k3x::Profiler profiler;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.dense_precision = k3x::DensePrecision::bf16_rounded;

    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 16;

    const std::vector<float> input{1.003F, -2.007F, 0.3333F};
    const std::vector<float> weight{
        0.1003F, -0.2007F, 0.3009F,
        -1.101F, 2.203F, 0.707F,
    };
    std::vector<float> expected(2);
    for (std::size_t row = 0; row < 2; ++row) {
        float sum = 0.0F;
        for (std::size_t column = 0; column < 3; ++column) {
            sum += round_to_bf16(weight[row * 3 + column]) *
                   round_to_bf16(input[column]);
        }
        expected[row] = sum;
    }

    const auto output = backend.value()->dense_matvec(
        input, weight, 2, 3, 11, k3x::ProfilePhase::prefill);
    if (!output) {
        std::cerr << output.message() << '\n';
        return 17;
    }
    if (output.value().size() != 2) return 18;
    if (!nearly_equal(output.value()[0], expected[0], 2.0e-2F)) return 19;
    if (!nearly_equal(output.value()[1], expected[1], 2.0e-2F)) return 20;

    std::size_t dense_events = 0;
    for (const auto& event : profiler.events()) {
        if (event.operation != k3x::ProfileOperation::dense_matvec) continue;
        ++dense_events;
        if (!event.success) return 21;
        if (event.precision != k3x::NumericPrecision::bf16_rounded) return 22;
        if (event.layer != 11) return 23;
        if (event.phase != k3x::ProfilePhase::prefill) return 24;
        if (event.device_nanoseconds == 0) return 25;
    }
    if (dense_events != 1) return 26;

    const auto summary = profiler.summary();
    if (summary.failed_operations != 0) return 27;
    if (summary.logical_bytes != 24) return 28;
    if (summary.host_to_device_bytes != 18) return 29;
    if (summary.device_to_host_bytes != 8) return 30;
    if (backend.value()->memory_stats().current_device_bytes != 0) return 31;
    return 0;
}

int test_allocation_modes() {
    const std::vector<float> input{1.0F, 2.0F, 3.0F};
    const std::vector<float> weight{
        1.0F, 0.0F, -1.0F,
        0.5F, 2.0F, 1.0F,
    };
    const k3x::DenseWeightView view{101, weight, 2, 3};

    k3x::BackendOptions reference_options;
    reference_options.kind = k3x::BackendKind::cuda_dense;
    auto reference = k3x::make_cuda_backend(reference_options);
    if (!reference) return 40;
    if (!reference.value()->dense_matvec(
            input, view, 4, k3x::ProfilePhase::decode)) return 41;
    const auto reference_first = reference.value()->runtime_stats();
    if (reference_first.device_allocation_count != 3 ||
        reference_first.device_free_count != 3 ||
        reference_first.stream_synchronization_count != 1 ||
        reference_first.scratch_bytes != 0) return 42;
    if (!reference.value()->dense_matvec(
            input, view, 4, k3x::ProfilePhase::decode)) return 43;
    const auto reference_second = reference.value()->runtime_stats();
    if (reference_second.device_allocation_count != 6 ||
        reference_second.device_free_count != 6 ||
        reference_second.stream_synchronization_count != 2) return 44;

    k3x::BackendOptions reused_options;
    reused_options.kind = k3x::BackendKind::cuda_dense;
    reused_options.cuda_allocation = k3x::CudaAllocationMode::reused;
    auto reused = k3x::make_cuda_backend(reused_options);
    if (!reused) return 45;
    const auto first = reused.value()->dense_matvec(
        input, view, 4, k3x::ProfilePhase::decode);
    if (!first) return 46;
    const auto reused_first = reused.value()->runtime_stats();
    if (reused_first.device_allocation_count != 3 ||
        reused_first.device_free_count != 0 ||
        reused_first.stream_synchronization_count != 1 ||
        reused_first.scratch_bytes != 44) return 47;
    const auto second = reused.value()->dense_matvec(
        input, view, 4, k3x::ProfilePhase::decode);
    if (!second || second.value() != first.value()) return 48;
    const auto reused_second = reused.value()->runtime_stats();
    if (reused_second.device_allocation_count !=
            reused_first.device_allocation_count ||
        reused_second.device_free_count != reused_first.device_free_count ||
        reused_second.stream_synchronization_count !=
            reused_first.stream_synchronization_count + 1) return 49;

    const std::vector<float> larger_weight{
        1.0F, 0.0F, -1.0F,
        0.5F, 2.0F, 1.0F,
        2.0F, 1.0F, 0.0F,
    };
    const auto larger = reused.value()->dense_matvec(
        input, k3x::DenseWeightView{102, larger_weight, 3, 3}, 4,
        k3x::ProfilePhase::decode);
    if (!larger) return 50;
    const auto reused_larger = reused.value()->runtime_stats();
    if (reused_larger.device_allocation_count !=
            reused_second.device_allocation_count + 2 ||
        reused_larger.device_free_count !=
            reused_second.device_free_count + 2 ||
        reused_larger.scratch_bytes != 60 ||
        reused_larger.stream_synchronization_count !=
            reused_second.stream_synchronization_count + 1) return 51;
    return 0;
}

int test_grouped_execution() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_batching = k3x::CudaBatchingMode::grouped;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 60;
    const std::array<float, 2> input{2.0F, -1.0F};
    const std::array<float, 4> first{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 2> second{3.0F, -2.0F};
    const std::array<k3x::DenseWeightView, 2> weights{{
        {301, first, 2, 2},
        {302, second, 1, 2},
    }};
    const auto output = backend.value()->dense_matvec_group(
        input, weights, 9, k3x::ProfilePhase::decode);
    if (!output || output.value().size() != 2) return 61;
    if (output.value()[0] != std::vector<float>{2.0F, -1.0F}) return 62;
    if (output.value()[1] != std::vector<float>{8.0F}) return 63;
    const auto stats = backend.value()->runtime_stats();
    if (stats.grouped_projection_calls != 1 ||
        stats.grouped_projection_members != 2 ||
        stats.activation_h2d_bytes != input.size() * sizeof(float) ||
        stats.weight_h2d_bytes !=
            (first.size() + second.size()) * sizeof(float) ||
        stats.stream_synchronization_count != 1) return 64;

    const std::array<k3x::DenseWeightView, 2> invalid{{
        {303, first, 2, 2},
        {304, second, 2, 2},
    }};
    const auto before = backend.value()->runtime_stats();
    const auto rejected = backend.value()->dense_matvec_group(
        input, invalid, 9, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_extent) return 65;
    const auto after = backend.value()->runtime_stats();
    if (after.grouped_projection_calls != before.grouped_projection_calls ||
        after.device_allocation_count != before.device_allocation_count ||
        after.activation_h2d_bytes != before.activation_h2d_bytes) return 66;
    return 0;
}

}  // namespace

int main() {
    const auto fp32_result = test_fp32();
    if (fp32_result != 0) return fp32_result;
    const auto bf16_result = test_bf16_rounded();
    if (bf16_result != 0) return bf16_result;
    const auto allocation_result = test_allocation_modes();
    if (allocation_result != 0) return allocation_result;
    return test_grouped_execution();
}
