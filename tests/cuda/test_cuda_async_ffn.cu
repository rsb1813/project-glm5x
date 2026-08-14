// prepared exact MXFP4 expert FFN backend의 독립 payload와 오류 원자성을 검증합니다.
#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

struct ExpertStorage {
    std::array<std::byte, 512> gate{};
    std::array<std::byte, 512> up{};
    std::array<std::byte, 16> down{};
    std::array<std::byte, 32> gate_scales{};
    std::array<std::byte, 32> up_scales{};
    std::array<std::byte, 1> down_scales{};
};

struct Fixture {
    std::array<float, 32> input{};
    ExpertStorage first;
    ExpertStorage second;

    Fixture() {
        input[1] = 2.0F;
        first.gate[0] = std::byte{0x10};
        first.up[0] = std::byte{0x20};
        first.down[0] = std::byte{0x02};
        second.gate[0] = std::byte{0x10};
        second.up[0] = std::byte{0x20};
        second.down[0] = std::byte{0x04};
        fill_scales(std::byte{127});
    }

    void fill_scales(std::byte value) {
        first.gate_scales.fill(value);
        first.up_scales.fill(value);
        first.down_scales.fill(value);
        second.gate_scales.fill(value);
        second.up_scales.fill(value);
        second.down_scales.fill(value);
    }

    std::array<k3x::Mxfp4MlpView, 2> views() {
        return {{
            {
                {1001, first.gate, first.gate_scales, 32, 32, 32},
                {1002, first.up, first.up_scales, 32, 32, 32},
                {1003, first.down, first.down_scales, 1, 32, 32},
            },
            {
                {1004, second.gate, second.gate_scales, 32, 32, 32},
                {1005, second.up, second.up_scales, 32, 32, 32},
                {1006, second.down, second.down_scales, 1, 32, 32},
            },
        }};
    }

    void mutate_payloads() {
        first.gate.fill(std::byte{0});
        first.up.fill(std::byte{0});
        first.down.fill(std::byte{0});
        second.gate.fill(std::byte{0});
        second.up.fill(std::byte{0});
        second.down.fill(std::byte{0});
        fill_scales(std::byte{126});
    }
};

bool nearly_equal(const std::vector<float>& actual,
                  const std::vector<float>& expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > 1.0e-6F) return false;
    }
    return true;
}

k3x::BackendOptions prefetch_options(std::uint64_t capacity = 4096) {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::transient;
    options.cuda_transfer = k3x::CudaTransferMode::prefetch;
    options.cuda_pinned_bytes = capacity;
    return options;
}

int test_owned_payload_and_accounting() {
    Fixture fixture;
    const auto experts = fixture.views();
    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->mxfp4_situ_mlp_group(
        fixture.input, experts, 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    if (!expected) return 1;

    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(prefetch_options(), &profiler);
    if (!backend) return 2;
    const auto token = backend.value()->prefetch_mxfp4_situ_mlp_group(
        experts, 1, 12, k3x::ProfilePhase::decode);
    if (!token) return 4;
    const auto initial = backend.value()->runtime_stats();
    cudaDeviceProp properties{};
    int device = -1;
    int device_overlap = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&device_overlap, cudaDevAttrGpuOverlap, device) !=
            cudaSuccess ||
        initial.pinned_host_bytes != 4096 ||
        initial.peak_pinned_host_bytes != 4096 ||
        initial.async_engine_count !=
            static_cast<std::uint64_t>(properties.asyncEngineCount) ||
        initial.device_overlap != (device_overlap != 0)) {
        return 3;
    }

    fixture.mutate_payloads();
    const auto actual = backend.value()->mxfp4_situ_mlp_group_prepared(
        fixture.input, token.value(), 2.0F, 1.5F, 12,
        k3x::ProfilePhase::decode);
    if (!actual || actual.value().size() != 2 ||
        !nearly_equal(actual.value()[0], expected.value()[0]) ||
        !nearly_equal(actual.value()[1], expected.value()[1])) {
        return 5;
    }
    const auto stats = backend.value()->runtime_stats();
    const auto profile = profiler.summary();
    if (stats.stream_synchronization_count != 1 ||
        stats.ffn_block_calls != 1 || stats.ffn_block_experts != 2 ||
        stats.fused_moe_calls != 0 || stats.fused_moe_experts != 0 ||
        stats.async_prefetch_calls != 1 || stats.async_prefetch_bytes != 2210 ||
        stats.weight_h2d_bytes != 2210 ||
        stats.transfer_stream_wait_count != 1 ||
        stats.async_prefetch_ready_before_use +
                stats.async_prefetch_late_at_use != 1 ||
        stats.weight_cache_hits != 0 || stats.weight_cache_misses != 0 ||
        stats.weight_cache_bypasses != 0 ||
        profile.weight_host_to_device_bytes != 2210 ||
        profile.device_to_host_bytes != 8) {
        return 6;
    }
    return 0;
}

int test_error_atomicity() {
    Fixture fixture;
    const auto experts = fixture.views();
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(prefetch_options(), &profiler);
    if (!backend) return 20;
    const auto token = backend.value()->prefetch_mxfp4_situ_mlp_group(
        experts, 1, 13, k3x::ProfilePhase::decode);
    if (!token) return 21;

    const auto before = backend.value()->runtime_stats();
    const auto before_memory = backend.value()->memory_stats();
    const auto before_profile = profiler.summary();
    const auto invalid_beta = backend.value()->mxfp4_situ_mlp_group_prepared(
        fixture.input, token.value(), 0.0F, std::nullopt, 13,
        k3x::ProfilePhase::decode);
    const auto foreign = backend.value()->mxfp4_situ_mlp_group_prepared(
        fixture.input, {token.value().value + 1, token.value().use_sequence},
        2.0F, 1.5F, 13,
        k3x::ProfilePhase::decode);
    const auto wrong_sequence =
        backend.value()->mxfp4_situ_mlp_group_prepared(
            fixture.input,
            {token.value().value, token.value().use_sequence + 1},
            2.0F, 1.5F, 13, k3x::ProfilePhase::decode);
    const auto wrong_input = backend.value()->mxfp4_situ_mlp_group_prepared(
        std::span<const float>(fixture.input).first(31), token.value(),
        2.0F, 1.5F, 13, k3x::ProfilePhase::decode);
    const auto after_invalid = backend.value()->runtime_stats();
    const auto after_invalid_memory = backend.value()->memory_stats();
    const auto after_invalid_profile = profiler.summary();
    if (invalid_beta || invalid_beta.error() != k3x::ErrorCode::invalid_mxfp4 ||
        foreign || foreign.error() != k3x::ErrorCode::invalid_state ||
        wrong_sequence ||
        wrong_sequence.error() != k3x::ErrorCode::invalid_state ||
        wrong_input || wrong_input.error() != k3x::ErrorCode::invalid_mxfp4 ||
        after_invalid.ffn_block_calls != before.ffn_block_calls ||
        after_invalid.stream_synchronization_count !=
            before.stream_synchronization_count ||
        after_invalid.device_allocation_count != before.device_allocation_count ||
        after_invalid.activation_h2d_bytes != before.activation_h2d_bytes ||
        after_invalid.transfer_stream_wait_count !=
            before.transfer_stream_wait_count ||
        after_invalid_memory.current_device_bytes !=
            before_memory.current_device_bytes ||
        after_invalid_memory.peak_device_bytes !=
            before_memory.peak_device_bytes ||
        after_invalid_profile.host_to_device_bytes !=
            before_profile.host_to_device_bytes) {
        return 22;
    }

    const auto valid = backend.value()->mxfp4_situ_mlp_group_prepared(
        fixture.input, token.value(), 2.0F, 1.5F, 13,
        k3x::ProfilePhase::decode);
    const auto repeated = backend.value()->mxfp4_situ_mlp_group_prepared(
        fixture.input, token.value(), 2.0F, 1.5F, 13,
        k3x::ProfilePhase::decode);
    if (!valid || repeated || repeated.error() != k3x::ErrorCode::invalid_state ||
        backend.value()->runtime_stats().ffn_block_calls != 1) {
        return 23;
    }

    auto too_small = k3x::make_cuda_backend(prefetch_options(2209));
    if (!too_small) return 24;
    const auto rejected = too_small.value()->prefetch_mxfp4_situ_mlp_group(
        experts, 1, 13, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_extent ||
        too_small.value()->runtime_stats().async_prefetch_calls != 0) {
        return 25;
    }
    return 0;
}

int test_fused_prepared_mix() {
    Fixture fixture;
    const auto experts = fixture.views();
    const std::array<float, 2> contributions{0.25F, -0.5F};
    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->mxfp4_situ_moe(
        fixture.input, experts, contributions, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!expected) return 30;

    k3x::Profiler profiler;
    auto options = prefetch_options();
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::routed_accumulate;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 31;
    const auto token = backend.value()->prefetch_mxfp4_situ_mlp_group(
        experts, 1, 14, k3x::ProfilePhase::decode);
    if (!token) return 32;
    fixture.mutate_payloads();

    const auto before_invalid = backend.value()->runtime_stats();
    const auto before_invalid_profile = profiler.summary();
    const std::array<float, 1> wrong_count{1.0F};
    const auto rejected_count =
        backend.value()->mxfp4_situ_moe_prepared(
            fixture.input, token.value(), wrong_count, 2.0F, 1.5F, 14,
            k3x::ProfilePhase::decode);
    const std::array<float, 2> nonfinite{
        1.0F, std::numeric_limits<float>::quiet_NaN()};
    const auto rejected_nonfinite =
        backend.value()->mxfp4_situ_moe_prepared(
            fixture.input, token.value(), nonfinite, 2.0F, 1.5F, 14,
            k3x::ProfilePhase::decode);
    const auto after_invalid = backend.value()->runtime_stats();
    const auto after_invalid_profile = profiler.summary();
    if (rejected_count ||
        rejected_count.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_nonfinite ||
        rejected_nonfinite.error() != k3x::ErrorCode::invalid_mxfp4 ||
        after_invalid.device_allocation_count !=
            before_invalid.device_allocation_count ||
        after_invalid.activation_h2d_bytes !=
            before_invalid.activation_h2d_bytes ||
        after_invalid.transfer_stream_wait_count !=
            before_invalid.transfer_stream_wait_count ||
        after_invalid_profile.host_to_device_bytes !=
            before_invalid_profile.host_to_device_bytes) {
        return 33;
    }

    const auto actual = backend.value()->mxfp4_situ_moe_prepared(
        fixture.input, token.value(), contributions, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!actual || !nearly_equal(actual.value(), expected.value())) return 34;
    const auto stats = backend.value()->runtime_stats();
    const auto profile = profiler.summary();
    if (stats.stream_synchronization_count != 1 ||
        stats.ffn_block_calls != 1 || stats.ffn_block_experts != 2 ||
        stats.fused_moe_calls != 1 || stats.fused_moe_experts != 2 ||
        stats.async_prefetch_calls != 1 || stats.async_prefetch_bytes != 2210 ||
        stats.transfer_stream_wait_count != 1 ||
        profile.device_to_host_bytes != sizeof(float)) {
        return 35;
    }
    const auto repeated = backend.value()->mxfp4_situ_moe_prepared(
        fixture.input, token.value(), contributions, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (repeated || repeated.error() != k3x::ErrorCode::invalid_state) {
        return 36;
    }
    return 0;
}

}  // namespace

int main() {
    if (const auto result = test_owned_payload_and_accounting()) return result;
    if (const auto result = test_error_atomicity()) return result;
    return test_fused_prepared_mix();
}
