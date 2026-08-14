// complete resident CUDA MoE layer의 exact 실행, hard-cap bypass, 선검증 계약을 검증합니다.
#include "k3x/backend.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

constexpr std::size_t hidden_width = 3;
constexpr std::size_t latent_width = 32;
constexpr std::size_t intermediate_width = 32;
constexpr std::size_t maximum_experts = 4;

struct Fixture {
    std::array<float, hidden_width> input{2.0F, -1.0F, 0.5F};
    std::array<float, latent_width * hidden_width> routed_down{};
    std::array<float, 1> routed_norm{1.25F};
    std::array<float, hidden_width> routed_up{1.0F, -0.5F, 2.0F};
    std::array<float, 2 * hidden_width> shared_gate{
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    std::array<float, 2 * hidden_width> shared_up{
        0.0F, 0.0F, 2.0F, 1.0F, 0.0F, 0.0F};
    std::array<float, hidden_width * 2> shared_down{
        1.0F, 2.0F, -1.0F, 0.5F, 0.25F, -0.75F};
    std::array<std::array<std::byte,
                          intermediate_width * latent_width / 2>,
               maximum_experts> gate_packed{};
    std::array<std::array<std::byte,
                          intermediate_width * latent_width / 2>,
               maximum_experts> up_packed{};
    std::array<std::array<std::byte, intermediate_width / 2>,
               maximum_experts> down_packed{};
    std::array<std::array<std::byte,
                          intermediate_width * latent_width / 32>,
               maximum_experts> gate_scales{};
    std::array<std::array<std::byte,
                          intermediate_width * latent_width / 32>,
               maximum_experts> up_scales{};
    std::array<std::array<std::byte, intermediate_width / 32>,
               maximum_experts> down_scales{};
    std::array<k3x::Mxfp4MlpView, maximum_experts> experts{};
    std::array<float, maximum_experts> contributions{
        0.5F, -0.25F, 0.125F, 0.625F};

    Fixture() {
        routed_down[hidden_width] = 1.0F;
        for (std::size_t expert = 0; expert < maximum_experts; ++expert) {
            gate_packed[expert][0] = std::byte{0x10};
            up_packed[expert][0] = std::byte{0x20};
            down_packed[expert][0] =
                static_cast<std::byte>(0x01U + expert);
            gate_scales[expert].fill(std::byte{127});
            up_scales[expert].fill(std::byte{127});
            down_scales[expert].fill(std::byte{127});
            experts[expert] = {
                {100 + expert * 3, gate_packed[expert], gate_scales[expert],
                 intermediate_width, latent_width, 32},
                {101 + expert * 3, up_packed[expert], up_scales[expert],
                 intermediate_width, latent_width, 32},
                {102 + expert * 3, down_packed[expert], down_scales[expert],
                 1, intermediate_width, 32},
            };
        }
    }

    k3x::ResidentMoeLayerView layer() const {
        return {
            {501, routed_down, latent_width, hidden_width},
            {502, routed_norm},
            {503, routed_up, hidden_width, 1},
            {
                {504, shared_gate, 2, hidden_width},
                {505, shared_up, 2, hidden_width},
                {506, shared_down, hidden_width, 2},
            },
        };
    }
};

k3x::BackendOptions options(std::size_t capacity) {
    k3x::BackendOptions result;
    result.kind = k3x::BackendKind::cuda_custom;
    result.dense_precision = k3x::DensePrecision::fp32;
    result.cuda_allocation = k3x::CudaAllocationMode::reused;
    result.cuda_weights = k3x::CudaWeightMode::resident;
    result.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    result.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    result.cuda_transfer = k3x::CudaTransferMode::synchronous;
    result.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    result.cuda_resident_bytes = capacity;
    return result;
}

constexpr std::uint64_t immutable_bytes =
    (latent_width * hidden_width + 1 + hidden_width +
     2 * hidden_width + 2 * hidden_width + hidden_width * 2) * sizeof(float);

bool close(float actual, float expected) {
    return std::abs(actual - expected) <= 1.0e-6F;
}

int run_full_fit(std::size_t expert_count) {
    const Fixture fixture;
    const auto experts = std::span(fixture.experts).first(expert_count);
    const auto contributions =
        std::span(fixture.contributions).first(expert_count);
    auto cpu = k3x::make_cpu_backend();
    const auto expected = cpu->resident_mxfp4_moe_layer(
        fixture.input, fixture.layer(), experts, contributions, 1.0e-5F,
        2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    if (!expected || !expected.value().executed) return 1;

    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(
        options(8U * 1024U * 1024U), &profiler);
    if (!backend) return 2;
    const auto actual = backend.value()->resident_mxfp4_moe_layer(
        fixture.input, fixture.layer(), experts, contributions, 1.0e-5F,
        2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    if (!actual || !actual.value().executed ||
        actual.value().output.size() != hidden_width) return 3;
    for (std::size_t row = 0; row < hidden_width; ++row) {
        if (!close(actual.value().output[row], expected.value().output[row])) {
            return 4;
        }
    }
    const auto stats = backend.value()->runtime_stats();
    if (stats.resident_moe_layer_calls != 1 ||
        stats.resident_moe_layer_experts != expert_count ||
        stats.resident_moe_layer_kernel_launches != 13 ||
        stats.resident_moe_layer_fallbacks != 0 ||
        stats.resident_moe_layer_contribution_h2d_bytes !=
            expert_count * sizeof(float) ||
        stats.resident_grid_calls != 1 ||
        stats.resident_grid_kernel_launches != 4 ||
        stats.stream_synchronization_count != 1 ||
        stats.activation_h2d_bytes == 0 ||
        stats.immutable_validation_scans != 6 ||
        stats.immutable_validation_hits != 0 ||
        stats.immutable_validation_bytes != immutable_bytes ||
        stats.immutable_validation_nanoseconds == 0) {
        return 5;
    }
    if (profiler.summary().device_to_host_bytes !=
        hidden_width * sizeof(float)) return 6;
    return 0;
}

int test_bypass_and_validation() {
    const Fixture fixture;
    auto bypass = k3x::make_cuda_backend(options(1));
    if (!bypass) return 10;
    const auto result = bypass.value()->resident_mxfp4_moe_layer(
        fixture.input, fixture.layer(), fixture.experts,
        fixture.contributions, 1.0e-5F, 2.0F, 1.5F, 7,
        k3x::ProfilePhase::decode);
    if (!result || result.value().executed || !result.value().output.empty()) {
        return 11;
    }
    const auto bypass_stats = bypass.value()->runtime_stats();
    if (bypass_stats.resident_moe_layer_fallbacks != 1 ||
        bypass_stats.resident_moe_layer_calls != 0 ||
        bypass_stats.resident_grid_calls != 0 ||
        bypass_stats.stream_synchronization_count != 0) return 12;

    auto invalid = k3x::make_cuda_backend(options(8U * 1024U * 1024U));
    if (!invalid) return 13;
    auto duplicate = fixture.layer();
    duplicate.routed_up.tensor_id = duplicate.routed_down.tensor_id;
    const auto rejected = invalid.value()->resident_mxfp4_moe_layer(
        fixture.input, duplicate, fixture.experts, fixture.contributions,
        1.0e-5F, 2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4) {
        return 14;
    }
    const auto invalid_stats = invalid.value()->runtime_stats();
    if (invalid_stats.resident_weight_bytes != 0 ||
        invalid_stats.resident_moe_layer_calls != 0 ||
        invalid_stats.resident_moe_layer_fallbacks != 0 ||
        invalid_stats.stream_synchronization_count != 0) return 15;
    return 0;
}

int test_admission_validation() {
    const Fixture fixture;
    auto admission_options = options(8U * 1024U * 1024U);
    admission_options.cuda_weight_validation =
        k3x::CudaWeightValidationMode::admission;
    auto backend = k3x::make_cuda_backend(admission_options);
    if (!backend) return 20;
    for (int call = 0; call < 2; ++call) {
        const auto result = backend.value()->resident_mxfp4_moe_layer(
            fixture.input, fixture.layer(), std::span(fixture.experts).first(1),
            std::span(fixture.contributions).first(1), 1.0e-5F, 2.0F, 1.5F,
            7, k3x::ProfilePhase::decode);
        if (!result || !result.value().executed) return 21 + call;
    }
    const auto stats = backend.value()->runtime_stats();
    if (stats.immutable_validation_scans != 6 ||
        stats.immutable_validation_hits != 6 ||
        stats.immutable_validation_bytes != immutable_bytes ||
        stats.immutable_validation_nanoseconds == 0) return 23;

    const Fixture different_allocation;
    const auto resident_before = stats.resident_weight_bytes;
    const auto rejected = backend.value()->resident_mxfp4_moe_layer(
        different_allocation.input, different_allocation.layer(),
        std::span(different_allocation.experts).first(1),
        std::span(different_allocation.contributions).first(1), 1.0e-5F,
        2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4 ||
        backend.value()->runtime_stats().resident_weight_bytes != resident_before) {
        return 24;
    }

    Fixture nonfinite;
    nonfinite.shared_down.back() = std::numeric_limits<float>::quiet_NaN();
    auto atomic = k3x::make_cuda_backend(admission_options);
    if (!atomic) return 25;
    const auto invalid = atomic.value()->resident_mxfp4_moe_layer(
        nonfinite.input, nonfinite.layer(), std::span(nonfinite.experts).first(1),
        std::span(nonfinite.contributions).first(1), 1.0e-5F, 2.0F, 1.5F,
        7, k3x::ProfilePhase::decode);
    const auto invalid_stats = atomic.value()->runtime_stats();
    if (invalid || invalid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        invalid_stats.resident_weight_bytes != 0 ||
        invalid_stats.immutable_validation_scans != 6 ||
        invalid_stats.immutable_validation_bytes != immutable_bytes) return 26;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run_full_fit(1)) return result;
    if (const auto result = run_full_fit(4)) return result + 20;
    if (const auto result = test_bypass_and_validation()) return result;
    return test_admission_validation();
}
