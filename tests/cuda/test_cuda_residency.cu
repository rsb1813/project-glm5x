// Stable tensor ID 기반 bounded CUDA weight residency와 exact bypass를 검증합니다.
#include "resident_weights.cuh"

#include "k3x/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

int test_representation_identity() {
    k3x::BackendMemoryStats memory;
    k3x::BackendRuntimeStats runtime;
    k3x::cuda::ResidentWeightTable table(36, &memory, &runtime, nullptr);
    const std::array<float, 6> fp32{1, 2, 3, 4, 5, 6};
    const std::array<std::uint16_t, 6> bf16{1, 2, 3, 4, 5, 6};
    const k3x::cuda::ResidentWeightKey fp32_key{
        7, k3x::cuda::WeightRepresentation::dense_fp32, 2, 3, 0};
    const k3x::cuda::ResidentWeightKey bf16_key{
        7, k3x::cuda::WeightRepresentation::dense_bf16, 2, 3, 0};
    const auto first = table.acquire(
        fp32_key, std::as_bytes(std::span(fp32)), {});
    if (!first || first.value().disposition !=
                      k3x::cuda::ResidentDisposition::admitted) return 1;
    const auto second = table.acquire(
        bf16_key, std::as_bytes(std::span(bf16)), {});
    if (!second || second.value().disposition !=
                       k3x::cuda::ResidentDisposition::admitted) return 2;
    if (runtime.weight_cache_misses != 2 ||
        runtime.weight_cache_hits != 0 ||
        runtime.resident_weight_bytes != 36 ||
        runtime.peak_resident_weight_bytes != 36) return 3;
    const auto hit = table.acquire(
        fp32_key, std::as_bytes(std::span(fp32)), {});
    if (!hit || hit.value().disposition !=
                    k3x::cuda::ResidentDisposition::hit) return 4;
    if (runtime.weight_cache_hits != 1 ||
        runtime.resident_weight_bytes != 36) return 5;
    return 0;
}

int test_dense_backend_residency() {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_dense;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 24;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 10;

    const std::array<float, 3> input{1.0F, 2.0F, 3.0F};
    const std::array<float, 6> first_weight{
        1.0F, 0.0F, -1.0F,
        0.5F, 2.0F, 1.0F,
    };
    const k3x::DenseWeightView first_view{101, first_weight, 2, 3};
    const auto first = backend.value()->dense_matvec(
        input, first_view, 4, k3x::ProfilePhase::decode);
    if (!first || first.value() != std::vector<float>{-2.0F, 7.5F}) return 11;
    const auto first_stats = backend.value()->runtime_stats();
    if (first_stats.weight_cache_misses != 1 ||
        first_stats.weight_cache_hits != 0 ||
        first_stats.weight_h2d_bytes != 24 ||
        first_stats.resident_weight_bytes != 24 ||
        first_stats.peak_resident_weight_bytes != 24) return 12;

    const auto hit = backend.value()->dense_matvec(
        input, first_view, 4, k3x::ProfilePhase::decode);
    if (!hit || hit.value() != first.value()) return 13;
    const auto hit_stats = backend.value()->runtime_stats();
    if (hit_stats.weight_cache_hits != 1 ||
        hit_stats.weight_cache_misses != 1 ||
        hit_stats.weight_h2d_bytes != first_stats.weight_h2d_bytes ||
        hit_stats.resident_weight_bytes != 24) return 14;

    const std::array<float, 3> incompatible_weight{1.0F, 2.0F, 3.0F};
    const auto incompatible = backend.value()->dense_matvec(
        input, k3x::DenseWeightView{101, incompatible_weight, 1, 3}, 4,
        k3x::ProfilePhase::decode);
    if (incompatible || incompatible.error() != k3x::ErrorCode::invalid_extent) {
        return 15;
    }

    const std::array<float, 6> second_weight{
        2.0F, 0.0F, 0.0F,
        0.0F, 3.0F, 0.0F,
    };
    const auto bypass = backend.value()->dense_matvec(
        input, k3x::DenseWeightView{102, second_weight, 2, 3}, 4,
        k3x::ProfilePhase::decode);
    if (!bypass || bypass.value() != std::vector<float>{2.0F, 6.0F}) return 16;
    const auto bypass_stats = backend.value()->runtime_stats();
    if (bypass_stats.weight_cache_misses != 2 ||
        bypass_stats.weight_cache_bypasses != 1 ||
        bypass_stats.weight_h2d_bytes != 48 ||
        bypass_stats.resident_weight_bytes != 24 ||
        bypass_stats.resident_weight_bytes > options.cuda_resident_bytes) return 17;
    return 0;
}

int test_mxfp4_backend_residency() {
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
    const k3x::Mxfp4WeightView view{201, packed, scales, 3, 64, 32};
    const std::vector<float> expected{3.5F, 1.0F, -3.5F};

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 102;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 20;
    const auto first = backend.value()->mxfp4_matvec(
        input, view, 8, k3x::ProfilePhase::decode);
    if (!first || first.value() != expected) return 21;
    const auto first_stats = backend.value()->runtime_stats();
    if (first_stats.weight_cache_misses != 1 ||
        first_stats.weight_cache_hits != 0 ||
        first_stats.weight_h2d_bytes != 102 ||
        first_stats.resident_weight_bytes != 102) return 22;
    const auto hit = backend.value()->mxfp4_matvec(
        input, view, 8, k3x::ProfilePhase::decode);
    if (!hit || hit.value() != expected) return 23;
    const auto hit_stats = backend.value()->runtime_stats();
    if (hit_stats.weight_cache_hits != 1 ||
        hit_stats.weight_cache_misses != 1 ||
        hit_stats.weight_h2d_bytes != 102 ||
        hit_stats.resident_weight_bytes != 102) return 24;

    const auto collision = backend.value()->mxfp4_matvec(
        std::span<const float>(input).first(32),
        k3x::Mxfp4WeightView{
            201, std::span<const std::byte>(packed).first(16),
            std::span<const std::byte>(scales).first(1), 1, 32, 32},
        8, k3x::ProfilePhase::decode);
    if (collision || collision.error() != k3x::ErrorCode::invalid_extent) {
        return 25;
    }

    k3x::BackendOptions bypass_options = options;
    bypass_options.cuda_resident_bytes = 101;
    auto bypass_backend = k3x::make_cuda_backend(bypass_options);
    if (!bypass_backend) return 26;
    const auto bypass = bypass_backend.value()->mxfp4_matvec(
        input, view, 8, k3x::ProfilePhase::decode);
    if (!bypass || bypass.value() != expected) return 27;
    const auto bypass_stats = bypass_backend.value()->runtime_stats();
    if (bypass_stats.weight_cache_misses != 1 ||
        bypass_stats.weight_cache_bypasses != 1 ||
        bypass_stats.weight_h2d_bytes != 102 ||
        bypass_stats.resident_weight_bytes != 0) return 28;

    k3x::BackendOptions dense_options = options;
    dense_options.kind = k3x::BackendKind::cuda_dense;
    auto dense = k3x::make_cuda_backend(dense_options);
    if (!dense) return 29;
    const auto oracle = dense.value()->mxfp4_matvec(
        input, view, 8, k3x::ProfilePhase::decode);
    if (!oracle || oracle.value() != expected) return 30;
    const auto dense_stats = dense.value()->runtime_stats();
    if (dense_stats.weight_cache_hits != 0 ||
        dense_stats.weight_cache_misses != 0 ||
        dense_stats.weight_cache_bypasses != 0 ||
        dense_stats.resident_weight_bytes != 0) return 31;
    return 0;
}

}  // namespace

int main() {
    const auto identity = test_representation_identity();
    if (identity != 0) return identity;
    const auto dense = test_dense_backend_residency();
    if (dense != 0) return dense;
    return test_mxfp4_backend_residency();
}
