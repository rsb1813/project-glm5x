// cuBLASLt와 strict SiTU를 연결한 dense FFN block의 전송과 출력을 검증합니다.
#include "k3x/backend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

float round_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF0000U);
}

bool nearly_equal(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance) return false;
    }
    return true;
}

struct Fixture {
    std::array<float, 3> input{1.25F, -0.75F, 0.5F};
    std::array<float, 12> gate{
        0.2F, -0.3F, 0.4F,
        -0.5F, 0.6F, 0.7F,
        0.8F, 0.1F, -0.2F,
        -0.4F, -0.9F, 0.3F,
    };
    std::array<float, 12> up{
        0.7F, -0.2F, 0.1F,
        0.3F, 0.8F, -0.6F,
        -0.5F, 0.4F, 0.9F,
        0.2F, -0.7F, 0.6F,
    };
    std::array<float, 8> down{
        0.4F, -0.2F, 0.7F, 0.1F,
        -0.6F, 0.5F, 0.3F, -0.8F,
    };

    k3x::DenseMlpView view(
        std::uint64_t gate_id = 501, std::uint64_t up_id = 502,
        std::uint64_t down_id = 503) const {
        return {
            {gate_id, gate, 4, 3},
            {up_id, up, 4, 3},
            {down_id, down, 2, 4},
        };
    }
};

std::vector<float> cpu_oracle(
    std::span<const float> input, k3x::DenseMlpView view) {
    auto cpu = k3x::make_cpu_backend();
    const auto result = cpu->dense_situ_mlp(
        input, view, 2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    return result ? result.value() : std::vector<float>{};
}

int test_fp32_and_residency() {
    const Fixture fixture;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 128;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 1;

    const auto weights = fixture.view();
    if (!backend.value()->dense_matvec(fixture.input, weights.gate, 7,
                                      k3x::ProfilePhase::decode)) return 2;
    if (!backend.value()->dense_matvec(fixture.input, weights.up, 7,
                                      k3x::ProfilePhase::decode)) return 3;
    const std::array<float, 4> down_input{};
    if (!backend.value()->dense_matvec(down_input, weights.down, 7,
                                      k3x::ProfilePhase::decode)) return 4;
    const auto before = backend.value()->runtime_stats();
    const auto before_profile = profiler.summary();
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, weights, 2.0F, 1.5F, 7,
        k3x::ProfilePhase::decode);
    if (!result || !nearly_equal(result.value(), cpu_oracle(fixture.input, weights),
                                 1.0e-5F)) return 5;
    const auto after = backend.value()->runtime_stats();
    const auto after_profile = profiler.summary();
    if (after.weight_cache_hits - before.weight_cache_hits != 3) return 6;
    if (after.weight_h2d_bytes != before.weight_h2d_bytes) return 7;
    if (after.activation_h2d_bytes - before.activation_h2d_bytes != 12) return 8;
    if (after.stream_synchronization_count -
            before.stream_synchronization_count != 1) return 9;
    if (after.ffn_block_calls - before.ffn_block_calls != 1) return 10;
    if (after.ffn_block_experts != before.ffn_block_experts) return 11;
    if (after_profile.device_to_host_bytes -
            before_profile.device_to_host_bytes != 8) return 12;
    return 0;
}

int test_bf16() {
    Fixture fixture;
    for (auto& value : fixture.input) value = round_to_bf16(value);
    for (auto& value : fixture.gate) value = round_to_bf16(value);
    for (auto& value : fixture.up) value = round_to_bf16(value);
    for (auto& value : fixture.down) value = round_to_bf16(value);
    const auto weights = fixture.view(601, 602, 603);
    const auto expected = cpu_oracle(fixture.input, weights);

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::bf16_rounded;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 20;
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, weights, 2.0F, 1.5F, 8,
        k3x::ProfilePhase::decode);
    if (!result || !nearly_equal(result.value(), expected, 2.0e-2F)) return 21;
    const auto stats = backend.value()->runtime_stats();
    if (stats.activation_h2d_bytes != 6 || stats.weight_h2d_bytes != 64 ||
        stats.stream_synchronization_count != 1 ||
        stats.ffn_block_calls != 1) return 22;
    return 0;
}

int test_validation() {
    const Fixture fixture;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 256;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 30;

    auto invalid = fixture.view(701, 702, 703);
    invalid.up.cols = 2;
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, invalid, 2.0F, 1.5F, 9,
        k3x::ProfilePhase::decode);
    if (result || result.error() != k3x::ErrorCode::invalid_extent) return 31;
    if (backend.value()->runtime_stats().ffn_block_calls != 0) return 32;

    const std::array<float, 3> collision_weight{1.0F, 2.0F, 3.0F};
    if (!backend.value()->dense_matvec(
            fixture.input, {704, collision_weight, 1, 3}, 9,
            k3x::ProfilePhase::decode)) return 33;
    const auto collision = backend.value()->dense_situ_mlp(
        fixture.input, fixture.view(704, 705, 706), 2.0F, 1.5F, 9,
        k3x::ProfilePhase::decode);
    if (collision || collision.error() != k3x::ErrorCode::invalid_extent) return 34;
    if (backend.value()->runtime_stats().ffn_block_calls != 0) return 35;
    return 0;
}

struct Mxfp4Fixture {
    std::array<float, 32> input{};
    std::array<std::byte, 512> gate{};
    std::array<std::byte, 512> up{};
    std::array<std::byte, 16> down_one{};
    std::array<std::byte, 16> down_two{};
    std::array<std::byte, 32> intermediate_scales{};
    std::array<std::byte, 1> down_scales{std::byte{127}};

    Mxfp4Fixture() {
        input[1] = 2.0F;
        gate[0] = std::byte{0x10};
        up[0] = std::byte{0x20};
        down_one[0] = std::byte{0x02};
        down_two[0] = std::byte{0x04};
        intermediate_scales.fill(std::byte{127});
    }

    std::array<k3x::Mxfp4MlpView, 2> views() const {
        return {{
            {
                {801, gate, intermediate_scales, 32, 32, 32},
                {802, up, intermediate_scales, 32, 32, 32},
                {803, down_one, down_scales, 1, 32, 32},
            },
            {
                {804, gate, intermediate_scales, 32, 32, 32},
                {805, up, intermediate_scales, 32, 32, 32},
                {806, down_two, down_scales, 1, 32, 32},
            },
        }};
    }
};

int warm_expert_weights(k3x::ComputeBackend& backend,
                        const Mxfp4Fixture& fixture,
                        std::span<const k3x::Mxfp4MlpView> experts) {
    const std::array<float, 32> down_input{};
    for (const auto& expert : experts) {
        if (!backend.mxfp4_matvec(fixture.input, expert.gate, 10,
                                  k3x::ProfilePhase::decode)) return 1;
        if (!backend.mxfp4_matvec(fixture.input, expert.up, 10,
                                  k3x::ProfilePhase::decode)) return 2;
        if (!backend.mxfp4_matvec(down_input, expert.down, 10,
                                  k3x::ProfilePhase::decode)) return 3;
    }
    return 0;
}

int test_exact_mxfp4_token_batch() {
    const Mxfp4Fixture fixture;
    const auto expert = fixture.views()[0];
    auto second_input = fixture.input;
    second_input[1] = -4.0F;
    std::array<float, 64> inputs{};
    std::copy(fixture.input.begin(), fixture.input.end(), inputs.begin());
    std::copy(second_input.begin(), second_input.end(), inputs.begin() + 32);

    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->mxfp4_situ_mlp_batch(
        inputs, 2, expert, 2.0F, 1.5F, 10,
        k3x::ProfilePhase::decode);
    if (!expected) return 90;

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::transient;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 91;
    const auto before = backend.value()->runtime_stats();
    const auto before_profile = profiler.summary();
    const auto actual = backend.value()->mxfp4_situ_mlp_batch(
        inputs, 2, expert, 2.0F, 1.5F, 10,
        k3x::ProfilePhase::decode);
    if (!actual || actual.value().size() != 2 ||
        !nearly_equal(actual.value()[0], expected.value()[0], 1.0e-6F) ||
        !nearly_equal(actual.value()[1], expected.value()[1], 1.0e-6F)) {
        return 92;
    }
    const auto after = backend.value()->runtime_stats();
    const auto after_profile = profiler.summary();
    constexpr std::uint64_t expert_weight_bytes = 512 + 32 + 512 + 32 + 16 + 1;
    if (after.weight_h2d_bytes - before.weight_h2d_bytes !=
            expert_weight_bytes ||
        after.activation_h2d_bytes - before.activation_h2d_bytes !=
            inputs.size() * sizeof(float) ||
        after.stream_synchronization_count -
                before.stream_synchronization_count != 1 ||
        after.ffn_block_calls - before.ffn_block_calls != 1 ||
        after.ffn_block_experts - before.ffn_block_experts != 1 ||
        after.batched_expert_ffn_calls -
                before.batched_expert_ffn_calls != 1 ||
        after.batched_expert_ffn_tokens -
                before.batched_expert_ffn_tokens != 2 ||
        after_profile.device_to_host_bytes -
                before_profile.device_to_host_bytes != 2 * sizeof(float)) {
        return 93;
    }

    const auto before_invalid = backend.value()->runtime_stats();
    const auto before_invalid_events = profiler.events().size();
    const auto rejected = backend.value()->mxfp4_situ_mlp_batch(
        std::span<const float>(inputs).first(63), 2, expert, 2.0F, 1.5F,
        10, k3x::ProfilePhase::decode);
    const auto after_invalid = backend.value()->runtime_stats();
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4 ||
        after_invalid.device_allocation_count !=
            before_invalid.device_allocation_count ||
        after_invalid.weight_h2d_bytes != before_invalid.weight_h2d_bytes ||
        after_invalid.activation_h2d_bytes !=
            before_invalid.activation_h2d_bytes ||
        after_invalid.batched_expert_ffn_calls !=
            before_invalid.batched_expert_ffn_calls ||
        profiler.events().size() != before_invalid_events) {
        return 94;
    }
    return 0;
}

int test_exact_mxfp4_group() {
    const Mxfp4Fixture fixture;
    const auto experts = fixture.views();
    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 10,
        k3x::ProfilePhase::decode);
    if (!expected) return 40;

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 2210;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 41;
    if (backend.value()->runtime_stats().pinned_host_bytes != 0) return 57;
    if (const auto result = warm_expert_weights(*backend.value(), fixture, experts)) {
        return 41 + result;
    }
    const auto before = backend.value()->runtime_stats();
    const auto before_profile = profiler.summary();
    const auto actual = backend.value()->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 10,
        k3x::ProfilePhase::decode);
    if (!actual || actual.value().size() != 2 ||
        !nearly_equal(actual.value()[0], expected.value()[0], 1.0e-6F) ||
        !nearly_equal(actual.value()[1], expected.value()[1], 1.0e-6F)) {
        return 45;
    }
    const auto after = backend.value()->runtime_stats();
    const auto after_profile = profiler.summary();
    if (after.weight_cache_hits - before.weight_cache_hits != 6) return 46;
    if (after.weight_h2d_bytes != before.weight_h2d_bytes) return 47;
    if (after.activation_h2d_bytes - before.activation_h2d_bytes != 128) return 48;
    if (after.stream_synchronization_count -
            before.stream_synchronization_count != 1) return 49;
    if (after.ffn_block_calls - before.ffn_block_calls != 1 ||
        after.ffn_block_experts - before.ffn_block_experts != 2) return 50;
    if (after_profile.device_to_host_bytes -
            before_profile.device_to_host_bytes != 8) return 51;

    auto invalid_experts = experts;
    auto invalid_scales = fixture.intermediate_scales;
    invalid_scales[0] = std::byte{0xFF};
    invalid_experts[1].up.scales = invalid_scales;
    const auto before_invalid = backend.value()->runtime_stats();
    const auto before_invalid_events = profiler.events().size();
    const auto rejected = backend.value()->mxfp4_situ_mlp_group(
        fixture.input, invalid_experts, 2.0F, 1.5F, 10,
        k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4) return 52;
    const auto after_invalid = backend.value()->runtime_stats();
    if (after_invalid.weight_cache_hits != before_invalid.weight_cache_hits ||
        after_invalid.weight_cache_misses != before_invalid.weight_cache_misses ||
        after_invalid.activation_h2d_bytes != before_invalid.activation_h2d_bytes ||
        after_invalid.ffn_block_calls != before_invalid.ffn_block_calls ||
        profiler.events().size() != before_invalid_events) return 53;
    return 0;
}

int test_exact_mxfp4_group_rejects_non_native_group_size() {
    std::array<float, 64> input{};
    std::array<std::byte, 2048> gate{};
    std::array<std::byte, 2048> up{};
    std::array<std::byte, 32> down{};
    std::array<std::byte, 256> scales{};
    std::array<std::byte, 4> down_scales{};
    scales.fill(std::byte{127});
    down_scales.fill(std::byte{127});

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 16384;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 54;

    for (const std::size_t group_size : {16U, 64U}) {
        const auto scale_count = 4096 / group_size;
        const std::array<k3x::Mxfp4MlpView, 1> experts{{{
            {901, gate, std::span<const std::byte>(scales).first(scale_count),
             64, 64, group_size},
            {902, up, std::span<const std::byte>(scales).first(scale_count),
             64, 64, group_size},
            {903, down,
             std::span<const std::byte>(down_scales).first(64 / group_size),
             1, 64, group_size},
        }}};
        const auto before = backend.value()->runtime_stats();
        const auto before_events = profiler.events().size();
        const auto rejected = backend.value()->mxfp4_situ_mlp_group(
            input, experts, 2.0F, 1.5F, 10, k3x::ProfilePhase::decode);
        if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4) {
            return 55;
        }
        const auto after = backend.value()->runtime_stats();
        if (after.device_allocation_count != before.device_allocation_count ||
            after.weight_cache_hits != before.weight_cache_hits ||
            after.weight_cache_misses != before.weight_cache_misses ||
            after.weight_h2d_bytes != before.weight_h2d_bytes ||
            after.activation_h2d_bytes != before.activation_h2d_bytes ||
            after.stream_synchronization_count !=
                before.stream_synchronization_count ||
            after.ffn_block_calls != before.ffn_block_calls ||
            after.ffn_block_experts != before.ffn_block_experts ||
            profiler.events().size() != before_events) {
            return 56;
        }
    }
    return 0;
}

int test_exact_mxfp4_capacity_bypass() {
    const Mxfp4Fixture fixture;
    const auto experts = fixture.views();
    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 11,
        k3x::ProfilePhase::decode);
    if (!expected) return 60;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 2209;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 61;
    const auto actual = backend.value()->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 11,
        k3x::ProfilePhase::decode);
    if (!actual || actual.value() != expected.value()) return 62;
    const auto stats = backend.value()->runtime_stats();
    if (stats.weight_cache_bypasses == 0 ||
        stats.resident_weight_bytes > options.cuda_resident_bytes ||
        stats.ffn_block_calls != 1 || stats.ffn_block_experts != 2) return 63;
    return 0;
}

int test_exact_mxfp4_fused_mix() {
    const Mxfp4Fixture fixture;
    const auto experts = fixture.views();
    const std::array<float, 2> contributions{0.25F, -0.5F};
    auto cpu = k3x::make_cpu_backend();
    const auto expert_outputs = cpu->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    if (!expert_outputs) return 70;
    std::vector<float> expected(expert_outputs.value().front().size(), 0.0F);
    for (std::size_t expert = 0; expert < contributions.size(); ++expert) {
        for (std::size_t row = 0; row < expected.size(); ++row) {
            expected[row] +=
                contributions[expert] * expert_outputs.value()[expert][row];
        }
    }
    const auto cpu_mixed = cpu->mxfp4_situ_moe(
        fixture.input, experts, contributions, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    if (!cpu_mixed || cpu_mixed.value() != expected) return 71;

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::routed_accumulate;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 2210;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 72;
    if (const auto result = warm_expert_weights(*backend.value(), fixture, experts)) {
        return 72 + result;
    }
    const auto before = backend.value()->runtime_stats();
    const auto before_profile = profiler.summary();
    const auto actual = backend.value()->mxfp4_situ_moe(
        fixture.input, experts, contributions, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    if (!actual || !nearly_equal(actual.value(), expected, 1.0e-6F)) return 76;
    const auto after = backend.value()->runtime_stats();
    const auto after_profile = profiler.summary();
    if (after.weight_cache_hits - before.weight_cache_hits != 6 ||
        after.weight_h2d_bytes != before.weight_h2d_bytes ||
        after.activation_h2d_bytes - before.activation_h2d_bytes != 128 ||
        after.stream_synchronization_count -
                before.stream_synchronization_count != 1 ||
        after.ffn_block_calls - before.ffn_block_calls != 1 ||
        after.ffn_block_experts - before.ffn_block_experts != 2 ||
        after.fused_moe_calls - before.fused_moe_calls != 1 ||
        after.fused_moe_experts - before.fused_moe_experts != 2) {
        return 77;
    }
    if (after_profile.device_to_host_bytes -
            before_profile.device_to_host_bytes != sizeof(float)) return 78;

    const auto before_invalid = backend.value()->runtime_stats();
    const auto before_invalid_events = profiler.events().size();
    const std::array<float, 1> wrong_count{1.0F};
    const auto rejected_count = backend.value()->mxfp4_situ_moe(
        fixture.input, experts, wrong_count, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    const std::array<float, 2> nonfinite{
        1.0F, std::numeric_limits<float>::quiet_NaN()};
    const auto rejected_nonfinite = backend.value()->mxfp4_situ_moe(
        fixture.input, experts, nonfinite, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    const auto after_invalid = backend.value()->runtime_stats();
    if (rejected_count ||
        rejected_count.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_nonfinite ||
        rejected_nonfinite.error() != k3x::ErrorCode::invalid_mxfp4 ||
        after_invalid.device_allocation_count !=
            before_invalid.device_allocation_count ||
        after_invalid.weight_cache_hits != before_invalid.weight_cache_hits ||
        after_invalid.activation_h2d_bytes !=
            before_invalid.activation_h2d_bytes ||
        after_invalid.ffn_block_calls != before_invalid.ffn_block_calls ||
        profiler.events().size() != before_invalid_events) {
        return 79;
    }
    return 0;
}

}  // namespace

int main() {
    if (const auto result = test_fp32_and_residency()) return result;
    if (const auto result = test_bf16()) return result;
    if (const auto result = test_validation()) return result;
    if (const auto result = test_exact_mxfp4_token_batch()) return result;
    if (const auto result = test_exact_mxfp4_group()) return result;
    if (const auto result = test_exact_mxfp4_group_rejects_non_native_group_size()) {
        return result;
    }
    if (const auto result = test_exact_mxfp4_capacity_bypass()) return result;
    return test_exact_mxfp4_fused_mix();
}
