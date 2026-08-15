// cuBLASLt FP32 dense matvec의 수치 결과와 전송 프로파일을 검증합니다.
#include "k3x/backend.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

int test_bf16_resident_hit_skips_host_weight_conversion() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.dense_precision = k3x::DensePrecision::bf16_rounded;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 1ULL << 20;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 102;

    const std::array<float, 3> input{1.0F, 2.0F, -1.0F};
    const std::array<float, 6> weight{
        0.5F, -0.25F, 2.0F,
        1.5F, 0.75F, -1.0F,
    };
    const k3x::DenseWeightView view{901, weight, 2, 3};
    const auto first = backend.value()->dense_matvec(
        input, view, 21, k3x::ProfilePhase::decode);
    if (!first) return 103;
    const auto first_stats = backend.value()->runtime_stats();
    if (first_stats.dense_bf16_host_conversion_calls != 1 ||
        first_stats.dense_bf16_host_conversion_bytes !=
            weight.size() * sizeof(std::uint16_t) ||
        first_stats.weight_cache_misses != 1 ||
        first_stats.weight_cache_hits != 0) {
        return 104;
    }

    const auto second = backend.value()->dense_matvec(
        input, view, 21, k3x::ProfilePhase::decode);
    if (!second || second.value() != first.value()) return 105;
    const auto second_stats = backend.value()->runtime_stats();
    if (second_stats.dense_bf16_host_conversion_calls !=
            first_stats.dense_bf16_host_conversion_calls ||
        second_stats.dense_bf16_host_conversion_bytes !=
            first_stats.dense_bf16_host_conversion_bytes ||
        second_stats.weight_cache_misses != first_stats.weight_cache_misses ||
        second_stats.weight_cache_hits != first_stats.weight_cache_hits + 1 ||
        second_stats.weight_h2d_bytes != first_stats.weight_h2d_bytes) {
        return 106;
    }
    return 0;
}

int test_bf16_resident_grid() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::bf16_rounded;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_resident_bytes = 1ULL << 20;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 70;

    const std::vector<float> input{
        0.31F, -0.72F, 1.13F,
        -0.44F, 0.27F, 0.91F,
    };
    const std::vector<float> gate0{
        0.20F, -0.40F, 0.70F,
        -0.80F, 0.50F, 0.10F,
    };
    const std::vector<float> up0{
        0.60F, 0.30F, -0.20F,
        0.10F, -0.90F, 0.40F,
    };
    const std::vector<float> down0{
        0.50F, -0.10F,
        0.20F, 0.80F,
    };
    const std::vector<float> gate1{
        -0.30F, 0.90F, 0.20F,
        0.70F, -0.60F, 0.40F,
    };
    const std::vector<float> up1{
        -0.50F, 0.20F, 0.80F,
        0.90F, 0.10F, -0.70F,
    };
    const std::vector<float> down1{
        -0.40F, 0.60F,
        0.30F, 0.20F,
    };
    const std::array<k3x::DenseMlpView, 2> experts{{
        {{701, gate0, 2, 3}, {702, up0, 2, 3}, {703, down0, 2, 2}},
        {{704, gate1, 2, 3}, {705, up1, 2, 3}, {706, down1, 2, 2}},
    }};
    const auto output = backend.value()->dense_situ_mlp_grid(
        input, 2, experts, 1.0F, std::nullopt, 17,
        k3x::ProfilePhase::decode);
    if (!output || output.value().size() != experts.size()) return 71;
    for (const auto& values : output.value()) {
        if (values.size() != 4) return 72;
    }

    std::vector<float> rounded_input(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        rounded_input[index] = round_to_bf16(input[index]);
    }
    std::array<std::vector<float>, 6> rounded_weights;
    const std::array<std::span<const float>, 6> source_weights{{
        gate0, up0, down0, gate1, up1, down1,
    }};
    for (std::size_t index = 0; index < source_weights.size(); ++index) {
        rounded_weights[index].resize(source_weights[index].size());
        for (std::size_t value = 0; value < source_weights[index].size();
             ++value) {
            rounded_weights[index][value] =
                round_to_bf16(source_weights[index][value]);
        }
    }
    const std::array<k3x::DenseMlpView, 2> rounded_experts{{
        {{701, rounded_weights[0], 2, 3},
         {702, rounded_weights[1], 2, 3},
         {703, rounded_weights[2], 2, 2}},
        {{704, rounded_weights[3], 2, 3},
         {705, rounded_weights[4], 2, 3},
         {706, rounded_weights[5], 2, 2}},
    }};
    std::array<std::vector<std::byte>, 6> raw_weights;
    for (std::size_t index = 0; index < rounded_weights.size(); ++index) {
        raw_weights[index].resize(rounded_weights[index].size() * 2);
        for (std::size_t value = 0; value < rounded_weights[index].size();
             ++value) {
            const auto bits = std::bit_cast<std::uint32_t>(
                rounded_weights[index][value]);
            const auto bf16_bits = static_cast<std::uint16_t>(bits >> 16U);
            std::memcpy(raw_weights[index].data() + value * 2, &bf16_bits, 2);
        }
    }
    const std::array<k3x::RawBf16MlpView, 2> raw_experts{{
        {{701, raw_weights[0], 2, 3},
         {702, raw_weights[1], 2, 3},
         {703, raw_weights[2], 2, 2}},
        {{704, raw_weights[3], 2, 3},
         {705, raw_weights[4], 2, 3},
         {706, raw_weights[5], 2, 2}},
    }};
    auto raw_backend = k3x::make_cuda_backend(options);
    if (!raw_backend) return 76;
    const auto raw_output = raw_backend.value()->raw_bf16_situ_mlp_grid(
        input, 2, raw_experts, 1.0F, std::nullopt, 17,
        k3x::ProfilePhase::decode);
    if (!raw_output || raw_output.value().size() != output.value().size()) {
        return 77;
    }
    for (std::size_t expert = 0; expert < output.value().size(); ++expert) {
        for (std::size_t value = 0; value < output.value()[expert].size();
             ++value) {
            if (!nearly_equal(raw_output.value()[expert][value],
                              output.value()[expert][value], 2.0e-2F)) {
                return 78;
            }
        }
    }
    auto cpu = k3x::make_cpu_backend();
    const std::vector<float> packed_inputs{
        input[0], input[1], input[2], input[3], input[4], input[5],
    };
    auto packed_backend = k3x::make_cuda_backend(options);
    if (!packed_backend) return 84;
    const auto packed_output =
        packed_backend.value()->raw_bf16_situ_mlp_grid_packed(
            packed_inputs, 1, raw_experts, 1.0F, std::nullopt, 17,
            k3x::ProfilePhase::decode);
    if (!packed_output || packed_output.value().size() != 2 ||
        packed_output.value()[0].size() != 2 ||
        packed_output.value()[1].size() != 2) {
        return 85;
    }
    for (std::size_t expert = 0; expert < rounded_experts.size(); ++expert) {
        const auto expected = cpu->dense_situ_mlp(
            std::span<const float>(rounded_input).subspan(expert * 3, 3),
            rounded_experts[expert], 1.0F, std::nullopt, 17,
            k3x::ProfilePhase::decode);
        if (!expected) return 86;
        for (std::size_t value = 0; value < expected.value().size(); ++value) {
            if (!nearly_equal(packed_output.value()[expert][value],
                              expected.value()[value], 2.0e-2F)) {
                return 87;
            }
        }
    }
    const auto packed_stats = packed_backend.value()->runtime_stats();
    if (packed_stats.activation_h2d_bytes != 12 ||
        packed_stats.device_to_host_bytes != 16 ||
        packed_stats.resident_grid_kernel_launches != 4) {
        return 88;
    }

    const std::array<k3x::ExpertMajorTokenRoute, 2> routes{{
        {{0}, {0.25F}},
        {{1}, {0.75F}},
    }};
    const auto packed_plan = k3x::build_expert_major_packed_plan(
        input, 2, 3, routes);
    if (!packed_plan) return 89;
    auto routed_backend = k3x::make_cuda_backend(options);
    if (!routed_backend) return 90;
    const auto routed_output =
        routed_backend.value()->raw_bf16_situ_mlp_expert_major(
            input, 2, packed_plan.value(), raw_experts, 1.0F, std::nullopt,
            17, k3x::ProfilePhase::decode);
    if (!routed_output || routed_output.value().size() != 4) return 91;
    for (std::size_t token = 0; token < 2; ++token) {
        const auto& expert = rounded_experts[token];
        const auto expected = cpu->dense_situ_mlp(
            std::span<const float>(rounded_input).subspan(token * 3, 3),
            expert, 1.0F, std::nullopt, 17, k3x::ProfilePhase::decode);
        if (!expected) return 92;
        const auto contribution = token == 0 ? 0.25F : 0.75F;
        for (std::size_t value = 0; value < expected.value().size(); ++value) {
            if (!nearly_equal(
                    routed_output.value()[token * 2 + value],
                    expected.value()[value] * contribution, 2.0e-2F)) {
                return 93;
            }
        }
    }
    const auto routed_stats = routed_backend.value()->runtime_stats();
    if (routed_stats.resident_grid_calls != 1 ||
        routed_stats.resident_grid_tokens != 1 ||
        routed_stats.resident_grid_expert_tokens != 2 ||
        routed_stats.activation_h2d_bytes != 12 ||
        routed_stats.device_to_host_bytes != 16) {
        return 94;
    }
    auto device_accumulate_options = options;
    device_accumulate_options.cuda_expert_major_device_accumulate = true;
    auto device_accumulate_backend =
        k3x::make_cuda_backend(device_accumulate_options);
    if (!device_accumulate_backend) return 95;
    const auto device_accumulate_output =
        device_accumulate_backend.value()->raw_bf16_situ_mlp_expert_major(
            input, 2, packed_plan.value(), raw_experts, 1.0F, std::nullopt,
            17, k3x::ProfilePhase::decode);
    if (!device_accumulate_output ||
        device_accumulate_output.value().size() != routed_output.value().size()) {
        return 96;
    }
    for (std::size_t index = 0; index < routed_output.value().size(); ++index) {
        if (!nearly_equal(device_accumulate_output.value()[index],
                          routed_output.value()[index], 2.0e-2F)) {
            std::cerr << "device expert accumulation mismatch at " << index
                      << ": " << device_accumulate_output.value()[index]
                      << " vs " << routed_output.value()[index] << '\n';
            return 97;
        }
    }
    const k3x::RawBf16MlpView shared_raw{
        {707, raw_weights[0], 2, 3},
        {708, raw_weights[1], 2, 3},
        {709, raw_weights[2], 2, 2},
    };
    auto fused_shared_backend =
        k3x::make_cuda_backend(device_accumulate_options);
    if (!fused_shared_backend) return 102;
    const auto fused_shared_output =
        fused_shared_backend.value()->raw_bf16_situ_mlp_expert_major_with_shared(
            input, 2, packed_plan.value(), raw_experts, shared_raw, 1.0F,
            std::nullopt, 17, k3x::ProfilePhase::decode);
    if (!fused_shared_output ||
        fused_shared_output.value().size() != routed_output.value().size()) {
        std::cerr << "fused shared output construction failed\n";
        return 103;
    }
    for (std::size_t token = 0; token < 2; ++token) {
        const auto shared_expected = cpu->dense_situ_mlp(
            std::span<const float>(rounded_input).subspan(token * 3, 3),
            rounded_experts[0], 1.0F, std::nullopt, 17,
            k3x::ProfilePhase::decode);
        if (!shared_expected) return 104;
        const auto routed_contribution = token == 0 ? 0.25F : 0.75F;
        const auto routed_expected = cpu->dense_situ_mlp(
            std::span<const float>(rounded_input).subspan(token * 3, 3),
            rounded_experts[token], 1.0F, std::nullopt, 17,
            k3x::ProfilePhase::decode);
        if (!routed_expected) return 105;
        for (std::size_t value = 0; value < shared_expected.value().size();
             ++value) {
            const auto combined = routed_contribution *
                    routed_expected.value()[value] +
                shared_expected.value()[value];
            if (!nearly_equal(
                    fused_shared_output.value()[token * 2 + value], combined,
                    2.0e-2F)) {
                std::cerr << "fused shared mismatch at " << token << ":"
                          << value << "\n";
                return 106;
            }
        }
    }
    const auto fused_shared_stats = fused_shared_backend.value()->runtime_stats();
    if (fused_shared_stats.device_to_host_bytes != 16) return 107;
    const std::array<k3x::ExpertMajorTokenRoute, 2> varied_routes{{
        {{0}, {0.25F}},
        {{0, 1}, {0.5F, 0.5F}},
    }};
    const auto varied_plan = k3x::build_expert_major_packed_plan(
        input, 2, 3, varied_routes);
    if (!varied_plan) {
        std::cerr << "varied plan construction failed\n";
        return 98;
    }
    auto varied_baseline_backend = k3x::make_cuda_backend(options);
    auto varied_device_backend =
        k3x::make_cuda_backend(device_accumulate_options);
    if (!varied_baseline_backend || !varied_device_backend) {
        std::cerr << "varied backend construction failed\n";
        return 99;
    }
    const auto varied_baseline =
        varied_baseline_backend.value()->raw_bf16_situ_mlp_expert_major(
            input, 2, varied_plan.value(), raw_experts, 1.0F, std::nullopt,
            17, k3x::ProfilePhase::decode);
    const auto varied_device =
        varied_device_backend.value()->raw_bf16_situ_mlp_expert_major(
            input, 2, varied_plan.value(), raw_experts, 1.0F, std::nullopt,
            17, k3x::ProfilePhase::decode);
    if (!varied_baseline || !varied_device ||
        varied_baseline.value().size() != varied_device.value().size()) {
        std::cerr << "varied output construction failed\n";
        return 100;
    }
    for (std::size_t index = 0; index < varied_baseline.value().size();
         ++index) {
        if (!nearly_equal(varied_baseline.value()[index],
                          varied_device.value()[index], 2.0e-2F)) {
            std::cerr << "varied device expert accumulation mismatch at "
                      << index << ": " << varied_device.value()[index]
                      << " vs " << varied_baseline.value()[index] << '\n';
            for (std::size_t dump = 0; dump < varied_baseline.value().size();
                 ++dump) {
                std::cerr << "  [" << dump << "] device="
                          << varied_device.value()[dump] << " baseline="
                          << varied_baseline.value()[dump] << '\n';
            }
            return 101;
        }
    }
    auto bf16_output_options = options;
    bf16_output_options.cuda_bf16_output = k3x::CudaBf16OutputMode::bf16;
    auto bf16_output_backend = k3x::make_cuda_backend(bf16_output_options);
    if (!bf16_output_backend) return 79;
    const auto bf16_grid_output =
        bf16_output_backend.value()->raw_bf16_situ_mlp_grid(
            input, 2, raw_experts, 1.0F, std::nullopt, 17,
            k3x::ProfilePhase::decode);
    if (!bf16_grid_output ||
        bf16_grid_output.value().size() != output.value().size()) {
        return 80;
    }
    for (std::size_t expert = 0; expert < rounded_experts.size(); ++expert) {
        for (std::size_t token = 0; token < 2; ++token) {
            const auto expected = cpu->dense_situ_mlp(
                std::span<const float>(rounded_input).subspan(token * 3, 3),
                rounded_experts[expert], 1.0F, std::nullopt, 17,
                k3x::ProfilePhase::decode);
            if (!expected) return 81;
            for (std::size_t value = 0; value < expected.value().size();
                 ++value) {
                if (!nearly_equal(
                        bf16_grid_output.value()[expert][token * 2 + value],
                        expected.value()[value], 4.0e-2F)) {
                    return 82;
                }
            }
        }
    }
    const auto bf16_output_stats = bf16_output_backend.value()->runtime_stats();
    if (bf16_output_stats.device_to_host_bytes != 16 ||
        bf16_output_stats.resident_grid_kernel_launches != 4) {
        return 83;
    }
    for (std::size_t expert = 0; expert < rounded_experts.size(); ++expert) {
        for (std::size_t token = 0; token < 2; ++token) {
            const auto expected = cpu->dense_situ_mlp(
                std::span<const float>(rounded_input).subspan(token * 3, 3),
                rounded_experts[expert], 1.0F, std::nullopt, 17,
                k3x::ProfilePhase::decode);
            if (!expected || expected.value().size() != 2) return 73;
            for (std::size_t value = 0; value < expected.value().size();
                 ++value) {
                if (!nearly_equal(output.value()[expert][token * 2 + value],
                                  expected.value()[value], 2.0e-2F)) {
                    return 74;
                }
            }
        }
    }
    const auto stats = backend.value()->runtime_stats();
    if (stats.resident_grid_calls != 1 ||
        stats.resident_grid_experts != experts.size() ||
        stats.resident_grid_tokens != 2 ||
        stats.resident_grid_expert_tokens != experts.size() * 2 ||
        stats.resident_grid_kernel_launches != 4 ||
        stats.resident_grid_descriptor_h2d_bytes != 144 ||
        stats.weight_h2d_bytes != 64 ||
        stats.activation_h2d_bytes != 12 ||
        stats.device_to_host_bytes != 32) {
        return 75;
    }
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
    const auto grouped_result = test_grouped_execution();
    if (grouped_result != 0) return grouped_result;
    const auto resident_hit_result =
        test_bf16_resident_hit_skips_host_weight_conversion();
    if (resident_hit_result != 0) return resident_hit_result;
    return test_bf16_resident_grid();
}
