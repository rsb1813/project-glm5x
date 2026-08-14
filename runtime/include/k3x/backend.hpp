// dense와 native MXFP4 연산을 실행하는 명시적 compute backend 계약을 정의합니다.
#pragma once

#include "k3x/profile.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace k3x {

enum class BackendKind { cpu, cuda_dense, cuda_custom };
enum class DensePrecision { fp32, bf16_rounded };
enum class CudaAllocationMode { per_operation, reused };
enum class CudaWeightMode { transient, resident };
enum class CudaBatchingMode { scalar, grouped, resident_grid };
enum class CudaBoundaryMode { operation, ffn_block, moe_layer };
enum class CudaTransferMode { synchronous, prefetch };
enum class CudaMoeFusionMode { none, routed_accumulate };
enum class CudaWeightValidationMode { per_call, admission };

struct BackendOptions {
    BackendKind kind{BackendKind::cpu};
    DensePrecision dense_precision{DensePrecision::fp32};
    CudaAllocationMode cuda_allocation{CudaAllocationMode::per_operation};
    CudaWeightMode cuda_weights{CudaWeightMode::transient};
    CudaBatchingMode cuda_batching{CudaBatchingMode::scalar};
    CudaBoundaryMode cuda_boundary{CudaBoundaryMode::operation};
    CudaTransferMode cuda_transfer{CudaTransferMode::synchronous};
    CudaMoeFusionMode cuda_moe_fusion{CudaMoeFusionMode::none};
    CudaWeightValidationMode cuda_weight_validation{
        CudaWeightValidationMode::per_call};
    std::uint64_t cuda_resident_bytes{};
    std::uint64_t cuda_pinned_bytes{};
};

struct BackendMemoryStats {
    std::uint64_t current_device_bytes{};
    std::uint64_t peak_device_bytes{};
};

struct BackendRuntimeStats {
    std::uint64_t device_allocation_count{};
    std::uint64_t device_free_count{};
    std::uint64_t stream_synchronization_count{};
    std::uint64_t weight_cache_hits{};
    std::uint64_t weight_cache_misses{};
    std::uint64_t weight_cache_bypasses{};
    std::uint64_t resident_weight_bytes{};
    std::uint64_t peak_resident_weight_bytes{};
    std::uint64_t immutable_validation_scans{};
    std::uint64_t immutable_validation_hits{};
    std::uint64_t immutable_validation_bytes{};
    std::uint64_t immutable_validation_nanoseconds{};
    std::uint64_t scratch_bytes{};
    std::uint64_t peak_scratch_bytes{};
    std::uint64_t weight_h2d_bytes{};
    std::uint64_t activation_h2d_bytes{};
    std::uint64_t device_to_host_bytes{};
    std::uint64_t grouped_projection_calls{};
    std::uint64_t grouped_projection_members{};
    std::uint64_t ffn_block_calls{};
    std::uint64_t ffn_block_experts{};
    std::uint64_t batched_expert_ffn_calls{};
    std::uint64_t batched_expert_ffn_tokens{};
    std::uint64_t resident_grid_calls{};
    std::uint64_t resident_grid_experts{};
    std::uint64_t resident_grid_tokens{};
    std::uint64_t resident_grid_expert_tokens{};
    std::uint64_t resident_grid_kernel_launches{};
    std::uint64_t resident_grid_fallbacks{};
    std::uint64_t resident_grid_descriptor_h2d_bytes{};
    std::uint64_t resident_moe_layer_calls{};
    std::uint64_t resident_moe_layer_experts{};
    std::uint64_t resident_moe_layer_kernel_launches{};
    std::uint64_t resident_moe_layer_fallbacks{};
    std::uint64_t resident_moe_layer_contribution_h2d_bytes{};
    std::uint64_t fused_moe_calls{};
    std::uint64_t fused_moe_experts{};
    std::uint64_t pinned_host_bytes{};
    std::uint64_t peak_pinned_host_bytes{};
    std::uint64_t async_prefetch_calls{};
    std::uint64_t async_prefetch_bytes{};
    std::uint64_t async_prefetch_ready_before_use{};
    std::uint64_t async_prefetch_late_at_use{};
    std::uint64_t transfer_stream_wait_count{};
    std::uint64_t pinned_staging_nanoseconds{};
    std::uint64_t transfer_device_nanoseconds{};
    std::uint64_t transfer_stall_nanoseconds{};
    std::uint64_t async_engine_count{};
    bool device_overlap{};
};

struct DenseWeightView {
    std::uint64_t tensor_id;
    std::span<const float> values;
    std::size_t rows;
    std::size_t cols;
};

struct Mxfp4WeightView {
    std::uint64_t tensor_id;
    std::span<const std::byte> packed;
    std::span<const std::byte> scales;
    std::size_t rows;
    std::size_t cols;
    std::size_t group_size;
};

struct DenseMlpView {
    DenseWeightView gate;
    DenseWeightView up;
    DenseWeightView down;
};

struct DenseVectorView {
    std::uint64_t tensor_id;
    std::span<const float> values;
};

struct ResidentMoeLayerView {
    DenseWeightView routed_down;
    DenseVectorView routed_norm;
    DenseWeightView routed_up;
    DenseMlpView shared;
};

struct ResidentMoeLayerResult {
    bool executed{};
    std::vector<float> output;
};

struct Mxfp4MlpView {
    Mxfp4WeightView gate;
    Mxfp4WeightView up;
    Mxfp4WeightView down;
};

struct Mxfp4PrefetchToken {
    std::uint64_t value{};
    std::uint64_t use_sequence{};
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual BackendKind kind() const noexcept = 0;
    virtual const BackendOptions& options() const noexcept = 0;
    virtual BackendRuntimeStats runtime_stats() const noexcept = 0;
    virtual Result<std::vector<float>> dense_matvec(
        std::span<const float> input, DenseWeightView weight,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, Mxfp4WeightView weight,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>> dense_matvec_group(
        std::span<const float> input, std::span<const DenseWeightView> weights,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>> mxfp4_matvec_group(
        std::span<const float> input, std::span<const Mxfp4WeightView> weights,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<float>> dense_situ_mlp(
        std::span<const float> input, DenseMlpView weights,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_batch(
        std::span<const float> inputs, std::size_t batch_size,
        Mxfp4MlpView expert, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_grid(
        std::span<const float>, std::size_t,
        std::span<const Mxfp4MlpView>, float,
        std::optional<float>, std::uint32_t, ProfilePhase) {
        return Result<std::vector<std::vector<float>>>::failure(
            ErrorCode::backend_unavailable);
    }
    virtual Result<std::vector<float>> mxfp4_situ_moe(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) = 0;
    virtual Result<ResidentMoeLayerResult> resident_mxfp4_moe_layer(
        std::span<const float>, ResidentMoeLayerView,
        std::span<const Mxfp4MlpView>, std::span<const float>, float, float,
        std::optional<float>, std::uint32_t, ProfilePhase) {
        return Result<ResidentMoeLayerResult>::failure(
            ErrorCode::backend_unavailable);
    }
    virtual Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
        std::span<const Mxfp4MlpView> experts, std::uint64_t use_sequence,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual Result<std::vector<float>> mxfp4_situ_moe_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) = 0;
    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, std::span<const float> values,
        std::size_t rows, std::size_t cols, std::uint32_t layer,
        ProfilePhase phase) {
        return dense_matvec(input, {0, values, rows, cols}, layer, phase);
    }
    Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, std::span<const std::byte> packed,
        std::span<const std::byte> scales, std::size_t rows,
        std::size_t cols, std::size_t group_size,
        std::uint32_t layer, ProfilePhase phase) {
        return mxfp4_matvec(
            input, {0, packed, scales, rows, cols, group_size}, layer, phase);
    }
    virtual BackendMemoryStats memory_stats() const noexcept = 0;
    virtual std::string_view device_name() const noexcept = 0;
};

std::unique_ptr<ComputeBackend> make_cpu_backend(Profiler* profiler = nullptr);
Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions& options, Profiler* profiler = nullptr);

}  // namespace k3x
