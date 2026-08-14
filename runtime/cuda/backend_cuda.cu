// RTX 5080용 CUDA stream과 cuBLASLt handle의 수명 및 device capability를 관리합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include "async_mxfp4_pipeline.cuh"
#include "device_memory.cuh"
#include "mxfp4.cuh"
#include "moe_layer.cuh"
#include "resident_weights.cuh"
#include "situ.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace k3x {
namespace {

class StreamOwner {
public:
    ~StreamOwner() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    StreamOwner() = default;

    cudaStream_t* out() { return &stream_; }
    cudaStream_t release() { return std::exchange(stream_, nullptr); }

private:
    cudaStream_t stream_{};
};

class CublasLtOwner {
public:
    ~CublasLtOwner() {
        if (handle_) cublasLtDestroy(handle_);
    }
    CublasLtOwner(const CublasLtOwner&) = delete;
    CublasLtOwner& operator=(const CublasLtOwner&) = delete;
    CublasLtOwner() = default;

    cublasLtHandle_t* out() { return &handle_; }
    cublasLtHandle_t release() { return std::exchange(handle_, nullptr); }

private:
    cublasLtHandle_t handle_{};
};

class MatmulDescOwner {
public:
    ~MatmulDescOwner() {
        if (descriptor_) cublasLtMatmulDescDestroy(descriptor_);
    }
    MatmulDescOwner(const MatmulDescOwner&) = delete;
    MatmulDescOwner& operator=(const MatmulDescOwner&) = delete;
    MatmulDescOwner() = default;
    cublasLtMatmulDesc_t* out() { return &descriptor_; }
    cublasLtMatmulDesc_t get() const { return descriptor_; }

private:
    cublasLtMatmulDesc_t descriptor_{};
};

class MatrixLayoutOwner {
public:
    ~MatrixLayoutOwner() {
        if (layout_) cublasLtMatrixLayoutDestroy(layout_);
    }
    MatrixLayoutOwner(const MatrixLayoutOwner&) = delete;
    MatrixLayoutOwner& operator=(const MatrixLayoutOwner&) = delete;
    MatrixLayoutOwner() = default;
    cublasLtMatrixLayout_t* out() { return &layout_; }
    cublasLtMatrixLayout_t get() const { return layout_; }

private:
    cublasLtMatrixLayout_t layout_{};
};

class MatmulPreferenceOwner {
public:
    ~MatmulPreferenceOwner() {
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    }
    MatmulPreferenceOwner(const MatmulPreferenceOwner&) = delete;
    MatmulPreferenceOwner& operator=(const MatmulPreferenceOwner&) = delete;
    MatmulPreferenceOwner() = default;
    cublasLtMatmulPreference_t* out() { return &preference_; }
    cublasLtMatmulPreference_t get() const { return preference_; }

private:
    cublasLtMatmulPreference_t preference_{};
};

class EventOwner {
public:
    ~EventOwner() {
        if (event_) cudaEventDestroy(event_);
    }
    EventOwner(const EventOwner&) = delete;
    EventOwner& operator=(const EventOwner&) = delete;
    EventOwner() = default;
    cudaError_t ensure() {
        return event_ ? cudaSuccess : cudaEventCreate(&event_);
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

struct DensePlan {
    MatmulDescOwner operation;
    MatrixLayoutOwner input_layout;
    MatrixLayoutOwner weight_layout;
    MatrixLayoutOwner output_layout;
    MatmulPreferenceOwner preference;
    cublasLtMatmulHeuristicResult_t heuristic{};
};

class CudaBackend final : public ComputeBackend {
public:
    CudaBackend(BackendOptions options, Profiler* profiler,
                cudaStream_t stream, cublasLtHandle_t handle,
                std::string device_name, std::uint64_t async_engine_count,
                bool device_overlap)
        : options_(options), profiler_(profiler), stream_(stream), handle_(handle),
          device_name_(std::move(device_name)),
          dense_input_scratch_(&memory_stats_, &runtime_stats_),
          dense_weight_scratch_(&memory_stats_, &runtime_stats_),
          dense_output_scratch_(&memory_stats_, &runtime_stats_),
          dense_group_output_scratch_(&memory_stats_, &runtime_stats_),
          ffn_input_scratch_(&memory_stats_, &runtime_stats_),
          ffn_weight_scratch_(&memory_stats_, &runtime_stats_),
          ffn_gate_scratch_(&memory_stats_, &runtime_stats_),
          ffn_up_scratch_(&memory_stats_, &runtime_stats_),
          ffn_activation_scratch_(&memory_stats_, &runtime_stats_),
          ffn_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_input_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_packed_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_scales_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_group_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_descriptor_scratch_(&memory_stats_, &runtime_stats_),
          layer_input_scratch_(&memory_stats_, &runtime_stats_),
          layer_routed_latent_scratch_(&memory_stats_, &runtime_stats_),
          layer_descriptor_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_gate_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_up_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_activation_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_output_scratch_(&memory_stats_, &runtime_stats_),
          layer_contribution_scratch_(&memory_stats_, &runtime_stats_),
          layer_mixed_scratch_(&memory_stats_, &runtime_stats_),
          layer_normalized_scratch_(&memory_stats_, &runtime_stats_),
          layer_routed_hidden_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_gate_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_up_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_activation_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_hidden_scratch_(&memory_stats_, &runtime_stats_),
          layer_final_hidden_scratch_(&memory_stats_, &runtime_stats_) {
        runtime_stats_.async_engine_count = async_engine_count;
        runtime_stats_.device_overlap = device_overlap;
        if (options_.cuda_weights == CudaWeightMode::resident) {
            resident_weights_ = std::make_unique<cuda::ResidentWeightTable>(
                options_.cuda_resident_bytes, &memory_stats_, &runtime_stats_,
                stream_);
        }
    }

    ~CudaBackend() override {
        if (handle_) cublasLtDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    BackendKind kind() const noexcept override { return options_.kind; }
    const BackendOptions& options() const noexcept override { return options_; }
    BackendRuntimeStats runtime_stats() const noexcept override {
        return runtime_stats_;
    }

    cudaError_t initialize_async_pipeline() {
        if (options_.cuda_transfer == CudaTransferMode::synchronous) {
            return cudaSuccess;
        }
        async_pipeline_ = std::make_unique<cuda::AsyncMxfp4Pipeline>(
            &memory_stats_, &runtime_stats_);
        return async_pipeline_->initialize(options_.cuda_pinned_bytes);
    }

    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, DenseWeightView weight_view,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto weight = weight_view.values;
        const auto rows = weight_view.rows;
        const auto cols = weight_view.cols;
        const auto operation_start = std::chrono::steady_clock::now();
        const auto precision = numeric_precision();
        if (input.size() != cols || rows > weight.size() ||
            (cols != 0 && rows > weight.size() / cols) ||
            weight.size() != rows * cols) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, weight.size_bytes(), 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
        std::vector<__nv_bfloat16> bf16_input;
        std::vector<__nv_bfloat16> bf16_weight;
        const auto logical_weight_bytes = weight.size_bytes();
        const void* host_input = input.data();
        const void* host_weight = weight.data();
        cudaDataType_t input_type = CUDA_R_32F;
        cudaDataType_t weight_type = CUDA_R_32F;
        std::size_t input_bytes = input.size_bytes();
        std::size_t weight_bytes = weight.size_bytes();
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            bf16_weight.reserve(weight.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            for (const auto value : weight) {
                bf16_weight.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            host_weight = bf16_weight.data();
            input_type = CUDA_R_16BF;
            weight_type = CUDA_R_16BF;
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
            weight_bytes = bf16_weight.size() * sizeof(__nv_bfloat16);
        }

        const auto output_bytes = rows * sizeof(float);
        bool has_resident_weight = false;
        std::uint64_t weight_transfer_bytes = weight_bytes;
        const void* resident_weight = nullptr;
        if (resident_weights_) {
            const auto representation =
                options_.dense_precision == DensePrecision::fp32
                    ? cuda::WeightRepresentation::dense_fp32
                    : cuda::WeightRepresentation::dense_bf16;
            const auto host_bytes = std::span(
                static_cast<const std::byte*>(host_weight), weight_bytes);
            const auto acquisition = resident_weights_->acquire(
                {weight_view.tensor_id, representation, rows, cols, 0},
                host_bytes, {});
            if (!acquisition) {
                record(phase, ProfileOperation::dense_matvec, precision, layer,
                       operation_start, logical_weight_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    acquisition.error(), acquisition.message());
            }
            has_resident_weight =
                acquisition.value().disposition !=
                cuda::ResidentDisposition::bypass;
            resident_weight = acquisition.value().primary;
            weight_transfer_bytes = acquisition.value().uploaded_bytes;
            if (!has_resident_weight) weight_transfer_bytes = weight_bytes;
        }
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* device_weight = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (dense_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                (!has_resident_weight &&
                 dense_weight_scratch_.reserve(weight_bytes) != cudaSuccess) ||
                dense_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                record(phase, ProfileOperation::dense_matvec, precision, layer,
                       operation_start, logical_weight_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA reusable device allocation failed");
            }
            device_input = dense_input_scratch_.get();
            device_weight = has_resident_weight
                                ? const_cast<void*>(resident_weight)
                                : dense_weight_scratch_.get();
            device_output = dense_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   (!has_resident_weight &&
                    local_weight.allocate(weight_bytes) != cudaSuccess) ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device allocation failed");
        } else {
            device_input = local_input.get();
            device_weight = has_resident_weight
                                ? const_cast<void*>(resident_weight)
                                : local_weight.get();
            device_output = local_output.get();
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_weight, host_weight, weight_bytes,
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess)) {
            record(phase, ProfileOperation::host_to_device, precision, layer,
                   h2d_start, 0, input_bytes + weight_transfer_bytes, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA host-to-device copy failed");
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               h2d_start, 0, weight_transfer_bytes, 0, true);
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;

        DensePlan local_plan;
        DensePlan* plan = &local_plan;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            const DensePlanKey key{rows, cols, static_cast<int>(input_type),
                                   static_cast<int>(weight_type)};
            const auto found = dense_plans_.find(key);
            if (found != dense_plans_.end()) {
                plan = found->second.get();
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(
                        *candidate, rows, cols, input_type, weight_type)) {
                    record(phase, ProfileOperation::dense_matvec, precision,
                           layer, operation_start, logical_weight_bytes, 0, 0,
                           false);
                    return Result<std::vector<float>>::failure(
                        ErrorCode::backend_unavailable,
                        "cuBLASLt plan creation failed");
                }
                plan = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
            }
        } else if (!initialize_dense_plan(
                       local_plan, rows, cols, input_type, weight_type)) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "cuBLASLt plan creation failed");
        }

        EventOwner local_event_start;
        EventOwner local_event_end;
        auto* event_start = &local_event_start;
        auto* event_end = &local_event_end;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            event_start = &dense_event_start_;
            event_end = &dense_event_end_;
        }
        if (event_start->ensure() != cudaSuccess ||
            event_end->ensure() != cudaSuccess ||
            cudaEventRecord(event_start->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event creation failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto matmul_status = cublasLtMatmul(
            handle_, plan->operation.get(), &alpha, device_weight,
            plan->weight_layout.get(), device_input, plan->input_layout.get(),
            &beta, device_output, plan->output_layout.get(), device_output,
            plan->output_layout.get(), &plan->heuristic.algo, nullptr, 0,
            stream_);
        if (matmul_status != CUBLAS_STATUS_SUCCESS ||
            cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "cuBLASLt dense matvec failed");
        }

        std::vector<float> output(rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            record(phase, ProfileOperation::device_to_host, precision, layer,
                   d2h_start, 0, output_bytes, 0, false);
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device-to-host copy failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.device_to_host_bytes += output_bytes;
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);

        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                 event_end->get()) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event timing failed");
        }
        const auto device_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(elapsed_milliseconds) * 1.0e6));
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, logical_weight_bytes, 0, device_nanoseconds, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, Mxfp4WeightView weight,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto packed = weight.packed;
        const auto scales = weight.scales;
        const auto rows = weight.rows;
        const auto cols = weight.cols;
        const auto group_size = weight.group_size;
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto logical_bytes = packed.size_bytes() + scales.size_bytes();
        if (options_.kind == BackendKind::cuda_dense) {
            auto result = k3x::mxfp4_matmul(
                input, packed, scales, rows, cols, group_size);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0,
                   static_cast<bool>(result));
            return result;
        }
        if (options_.kind != BackendKind::cuda_custom) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "native K3 MXFP4 requires the cuda_custom backend");
        }
        if (input.size() != cols || rows == 0 || cols == 0 ||
            group_size != 32 || cols % group_size != 0 || cols % 2 != 0 ||
            rows > std::numeric_limits<std::size_t>::max() / cols) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
        }
        const auto elements = rows * cols;
        if (packed.size() != elements / 2 ||
            scales.size() != elements / group_size ||
            std::any_of(scales.begin(), scales.end(), [](std::byte scale) {
                return std::to_integer<std::uint8_t>(scale) == 0xFFU;
            })) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
        }

        const auto input_bytes = input.size_bytes();
        const auto output_bytes = rows * sizeof(float);
        bool has_resident_weight = false;
        std::uint64_t weight_transfer_bytes = logical_bytes;
        const void* resident_packed = nullptr;
        const void* resident_scales = nullptr;
        if (resident_weights_) {
            const auto acquisition = resident_weights_->acquire(
                {weight.tensor_id, cuda::WeightRepresentation::mxfp4, rows,
                 cols, group_size},
                packed, scales);
            if (!acquisition) {
                record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                       operation_start, logical_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    acquisition.error(), acquisition.message());
            }
            has_resident_weight =
                acquisition.value().disposition !=
                cuda::ResidentDisposition::bypass;
            resident_packed = acquisition.value().primary;
            resident_scales = acquisition.value().secondary;
            weight_transfer_bytes = acquisition.value().uploaded_bytes;
            if (!has_resident_weight) weight_transfer_bytes = logical_bytes;
        }
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* device_packed = nullptr;
        void* device_scales = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (mxfp4_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                (!has_resident_weight &&
                 mxfp4_packed_scratch_.reserve(packed.size_bytes()) !=
                     cudaSuccess) ||
                (!has_resident_weight &&
                 mxfp4_scales_scratch_.reserve(scales.size_bytes()) !=
                     cudaSuccess) ||
                mxfp4_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                       operation_start, logical_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA reusable device allocation failed");
            }
            device_input = mxfp4_input_scratch_.get();
            device_packed = has_resident_weight
                                ? const_cast<void*>(resident_packed)
                                : mxfp4_packed_scratch_.get();
            device_scales = has_resident_weight
                                ? const_cast<void*>(resident_scales)
                                : mxfp4_scales_scratch_.get();
            device_output = mxfp4_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   (!has_resident_weight &&
                    local_packed.allocate(packed.size_bytes()) != cudaSuccess) ||
                   (!has_resident_weight &&
                    local_scales.allocate(scales.size_bytes()) != cudaSuccess) ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device allocation failed");
        } else {
            device_input = local_input.get();
            device_packed = has_resident_weight
                                ? const_cast<void*>(resident_packed)
                                : local_packed.get();
            device_scales = has_resident_weight
                                ? const_cast<void*>(resident_scales)
                                : local_scales.get();
            device_output = local_output.get();
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_packed, packed.data(), packed.size_bytes(),
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess) ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_scales, scales.data(), scales.size_bytes(),
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess)) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA host-to-device copy failed");
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               h2d_start, 0, weight_transfer_bytes, 0, true);
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;

        EventOwner local_event_start;
        EventOwner local_event_end;
        auto* event_start = &local_event_start;
        auto* event_end = &local_event_end;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            event_start = &mxfp4_event_start_;
            event_end = &mxfp4_event_end_;
        }
        if (event_start->ensure() != cudaSuccess ||
            event_end->ensure() != cudaSuccess ||
            cudaEventRecord(event_start->get(), stream_) != cudaSuccess ||
            cuda::launch_mxfp4_matvec(
                static_cast<const float*>(device_input),
                static_cast<const std::uint8_t*>(device_packed),
                static_cast<const std::uint8_t*>(device_scales),
                static_cast<float*>(device_output), rows, cols, stream_) !=
                cudaSuccess ||
            cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA MXFP4 kernel launch failed");
        }

        std::vector<float> output(rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA MXFP4 execution or device-to-host copy failed");
        }
        ++runtime_stats_.stream_synchronization_count;

        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                 event_end->get()) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event timing failed");
        }
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        const auto device_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(elapsed_milliseconds) * 1.0e6));
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_bytes, 0, device_nanoseconds, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<std::vector<float>>> dense_matvec_group(
        std::span<const float> input, std::span<const DenseWeightView> weights,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& weight : weights) {
            if (!valid_dense(input, weight)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_extent);
            }
        }
        if (options_.cuda_batching != CudaBatchingMode::grouped) {
            std::vector<std::vector<float>> outputs;
            outputs.reserve(weights.size());
            for (const auto& weight : weights) {
                auto output = dense_matvec(input, weight, layer, phase);
                if (!output) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        output.error(), output.message());
                }
                outputs.push_back(std::move(output.value()));
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }
        if (weights.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        const auto precision = numeric_precision();
        std::vector<__nv_bfloat16> bf16_input;
        const void* host_input = input.data();
        std::size_t input_bytes = input.size_bytes();
        cudaDataType_t input_type = CUDA_R_32F;
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
            input_type = CUDA_R_16BF;
        }

        struct GroupMember {
            const void* host_weight{};
            std::size_t weight_bytes{};
            void* device_weight{};
            std::size_t output_offset{};
            std::uint64_t transfer_bytes{};
            DensePlan* plan{};
        };
        std::vector<std::vector<__nv_bfloat16>> bf16_weights(weights.size());
        std::vector<GroupMember> members(weights.size());
        std::size_t maximum_weight_bytes = 0;
        std::size_t total_output_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        const auto weight_type =
            options_.dense_precision == DensePrecision::fp32
                ? CUDA_R_32F
                : CUDA_R_16BF;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto& member = members[index];
            const auto& weight = weights[index];
            member.host_weight = weight.values.data();
            member.weight_bytes = weight.values.size_bytes();
            if (options_.dense_precision == DensePrecision::bf16_rounded) {
                auto& converted = bf16_weights[index];
                converted.reserve(weight.values.size());
                for (const auto value : weight.values) {
                    converted.push_back(__float2bfloat16_rn(value));
                }
                member.host_weight = converted.data();
                member.weight_bytes = converted.size() * sizeof(__nv_bfloat16);
            }
            member.output_offset = total_output_bytes;
            total_output_bytes += weight.rows * sizeof(float);
            maximum_weight_bytes =
                std::max(maximum_weight_bytes, member.weight_bytes);
            member.transfer_bytes = member.weight_bytes;
            if (resident_weights_) {
                const auto representation =
                    options_.dense_precision == DensePrecision::fp32
                        ? cuda::WeightRepresentation::dense_fp32
                        : cuda::WeightRepresentation::dense_bf16;
                const auto host_bytes = std::span(
                    static_cast<const std::byte*>(member.host_weight),
                    member.weight_bytes);
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, representation, weight.rows,
                     weight.cols, 0},
                    host_bytes, {});
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device_weight =
                        const_cast<void*>(acquisition.value().primary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_weight = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (dense_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                dense_weight_scratch_.reserve(maximum_weight_bytes) !=
                    cudaSuccess ||
                dense_group_output_scratch_.reserve(total_output_bytes) !=
                    cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped dense allocation failed");
            }
            device_input = dense_input_scratch_.get();
            transient_weight = dense_weight_scratch_.get();
            device_output = dense_group_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_weight.allocate(maximum_weight_bytes) != cudaSuccess ||
                   local_output.allocate(total_output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped dense allocation failed");
        } else {
            device_input = local_input.get();
            transient_weight = local_weight.get();
            device_output = local_output.get();
        }

        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped activation upload failed");
        }
        std::vector<std::vector<float>> outputs(weights.size());
        std::vector<std::unique_ptr<DensePlan>> local_plans;
        std::vector<std::unique_ptr<EventOwner>> local_starts;
        std::vector<std::unique_ptr<EventOwner>> local_ends;
        local_plans.reserve(weights.size());
        local_starts.reserve(weights.size());
        local_ends.reserve(weights.size());
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            while (dense_group_event_starts_.size() < weights.size()) {
                dense_group_event_starts_.push_back(
                    std::make_unique<EventOwner>());
                dense_group_event_ends_.push_back(
                    std::make_unique<EventOwner>());
            }
        }
        const auto d2h_start = std::chrono::steady_clock::now();
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            if (!member.device_weight) {
                member.device_weight = transient_weight;
                if (cudaMemcpyAsync(member.device_weight, member.host_weight,
                                    member.weight_bytes,
                                    cudaMemcpyHostToDevice, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA grouped weight upload failed");
                }
            }

            const DensePlanKey key{
                weight.rows, weight.cols, static_cast<int>(input_type),
                static_cast<int>(weight_type)};
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                const auto found = dense_plans_.find(key);
                if (found != dense_plans_.end()) {
                    member.plan = found->second.get();
                } else {
                    auto candidate = std::make_unique<DensePlan>();
                    if (!initialize_dense_plan(*candidate, weight.rows,
                                               weight.cols, input_type,
                                               weight_type)) {
                        return Result<std::vector<std::vector<float>>>::failure(
                            ErrorCode::backend_unavailable,
                            "cuBLASLt grouped plan creation failed");
                    }
                    member.plan = candidate.get();
                    dense_plans_.emplace(key, std::move(candidate));
                }
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, weight.rows, weight.cols,
                                           input_type, weight_type)) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "cuBLASLt grouped plan creation failed");
                }
                member.plan = candidate.get();
                local_plans.push_back(std::move(candidate));
            }

            EventOwner* event_start = nullptr;
            EventOwner* event_end = nullptr;
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                event_start = dense_group_event_starts_[index].get();
                event_end = dense_group_event_ends_[index].get();
            } else {
                local_starts.push_back(std::make_unique<EventOwner>());
                local_ends.push_back(std::make_unique<EventOwner>());
                event_start = local_starts.back().get();
                event_end = local_ends.back().get();
            }
            if (event_start->ensure() != cudaSuccess ||
                event_end->ensure() != cudaSuccess ||
                cudaEventRecord(event_start->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped event creation failed");
            }
            auto* member_output = static_cast<std::byte*>(device_output) +
                                  member.output_offset;
            const auto status = cublasLtMatmul(
                handle_, member.plan->operation.get(), &alpha,
                member.device_weight, member.plan->weight_layout.get(),
                device_input, member.plan->input_layout.get(), &beta,
                member_output, member.plan->output_layout.get(), member_output,
                member.plan->output_layout.get(), &member.plan->heuristic.algo,
                nullptr, 0, stream_);
            if (status != CUBLAS_STATUS_SUCCESS ||
                cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "cuBLASLt grouped dense matvec failed");
            }
            outputs[index].resize(weight.rows);
            if (cudaMemcpyAsync(outputs[index].data(), member_output,
                                weight.rows * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.grouped_projection_calls;
        runtime_stats_.grouped_projection_members += weights.size();
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, d2h_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               d2h_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto* event_start =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? dense_group_event_starts_[index].get()
                    : local_starts[index].get();
            auto* event_end =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? dense_group_event_ends_[index].get()
                    : local_ends[index].get();
            float elapsed_milliseconds = 0.0F;
            if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                     event_end->get()) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped event timing failed");
            }
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   d2h_start, weights[index].values.size_bytes(), 0,
                   static_cast<std::uint64_t>(std::llround(
                       static_cast<double>(elapsed_milliseconds) * 1.0e6)),
                   true);
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));

    }

    Result<std::vector<std::vector<float>>> mxfp4_matvec_group(
        std::span<const float> input, std::span<const Mxfp4WeightView> weights,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& weight : weights) {
            if (!valid_mxfp4(input, weight) ||
                (options_.kind == BackendKind::cuda_custom &&
                 weight.group_size != 32)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_batching != CudaBatchingMode::grouped) {
            std::vector<std::vector<float>> outputs;
            outputs.reserve(weights.size());
            for (const auto& weight : weights) {
                auto output = mxfp4_matvec(input, weight, layer, phase);
                if (!output) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        output.error(), output.message());
                }
                outputs.push_back(std::move(output.value()));
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }
        if (weights.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        struct GroupMember {
            void* device_packed{};
            void* device_scales{};
            std::size_t output_offset{};
            std::uint64_t transfer_bytes{};
        };
        std::vector<GroupMember> members(weights.size());
        std::size_t maximum_packed_bytes = 0;
        std::size_t maximum_scale_bytes = 0;
        std::size_t total_output_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            member.output_offset = total_output_bytes;
            total_output_bytes += weight.rows * sizeof(float);
            maximum_packed_bytes =
                std::max(maximum_packed_bytes, weight.packed.size_bytes());
            maximum_scale_bytes =
                std::max(maximum_scale_bytes, weight.scales.size_bytes());
            member.transfer_bytes =
                weight.packed.size_bytes() + weight.scales.size_bytes();
            if (resident_weights_) {
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, cuda::WeightRepresentation::mxfp4,
                     weight.rows, weight.cols, weight.group_size},
                    weight.packed, weight.scales);
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device_packed =
                        const_cast<void*>(acquisition.value().primary);
                    member.device_scales =
                        const_cast<void*>(acquisition.value().secondary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        const auto input_bytes = input.size_bytes();
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_packed = nullptr;
        void* transient_scales = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (mxfp4_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                mxfp4_packed_scratch_.reserve(maximum_packed_bytes) !=
                    cudaSuccess ||
                mxfp4_scales_scratch_.reserve(maximum_scale_bytes) !=
                    cudaSuccess ||
                mxfp4_group_output_scratch_.reserve(total_output_bytes) !=
                    cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 allocation failed");
            }
            device_input = mxfp4_input_scratch_.get();
            transient_packed = mxfp4_packed_scratch_.get();
            transient_scales = mxfp4_scales_scratch_.get();
            device_output = mxfp4_group_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_packed.allocate(maximum_packed_bytes) != cudaSuccess ||
                   local_scales.allocate(maximum_scale_bytes) != cudaSuccess ||
                   local_output.allocate(total_output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 allocation failed");
        } else {
            device_input = local_input.get();
            transient_packed = local_packed.get();
            transient_scales = local_scales.get();
            device_output = local_output.get();
        }
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 activation upload failed");
        }

        std::vector<std::vector<float>> outputs;
        outputs.resize(weights.size());
        std::vector<std::unique_ptr<EventOwner>> local_starts;
        std::vector<std::unique_ptr<EventOwner>> local_ends;
        local_starts.reserve(weights.size());
        local_ends.reserve(weights.size());
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            while (mxfp4_group_event_starts_.size() < weights.size()) {
                mxfp4_group_event_starts_.push_back(
                    std::make_unique<EventOwner>());
                mxfp4_group_event_ends_.push_back(
                    std::make_unique<EventOwner>());
            }
        }
        const auto group_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            if (!member.device_packed) {
                member.device_packed = transient_packed;
                member.device_scales = transient_scales;
                if (cudaMemcpyAsync(
                        member.device_packed, weight.packed.data(),
                        weight.packed.size_bytes(), cudaMemcpyHostToDevice,
                        stream_) != cudaSuccess ||
                    cudaMemcpyAsync(
                        member.device_scales, weight.scales.data(),
                        weight.scales.size_bytes(), cudaMemcpyHostToDevice,
                        stream_) != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA grouped MXFP4 weight upload failed");
                }
            }
            EventOwner* event_start = nullptr;
            EventOwner* event_end = nullptr;
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                event_start = mxfp4_group_event_starts_[index].get();
                event_end = mxfp4_group_event_ends_[index].get();
            } else {
                local_starts.push_back(std::make_unique<EventOwner>());
                local_ends.push_back(std::make_unique<EventOwner>());
                event_start = local_starts.back().get();
                event_end = local_ends.back().get();
            }
            auto* member_output = static_cast<std::byte*>(device_output) +
                                  member.output_offset;
            if (event_start->ensure() != cudaSuccess ||
                event_end->ensure() != cudaSuccess ||
                cudaEventRecord(event_start->get(), stream_) != cudaSuccess ||
                cuda::launch_mxfp4_matvec(
                    static_cast<const float*>(device_input),
                    static_cast<const std::uint8_t*>(member.device_packed),
                    static_cast<const std::uint8_t*>(member.device_scales),
                    static_cast<float*>(static_cast<void*>(member_output)),
                    weight.rows, weight.cols, stream_) != cudaSuccess ||
                cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 launch failed");
            }
            outputs[index].resize(weight.rows);
            if (cudaMemcpyAsync(outputs[index].data(), member_output,
                                weight.rows * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.grouped_projection_calls;
        runtime_stats_.grouped_projection_members += weights.size();
        record(phase, ProfileOperation::activation_host_to_device,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               total_output_bytes, 0, true);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto* event_start =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? mxfp4_group_event_starts_[index].get()
                    : local_starts[index].get();
            auto* event_end =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? mxfp4_group_event_ends_[index].get()
                    : local_ends[index].get();
            float elapsed_milliseconds = 0.0F;
            if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                     event_end->get()) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 event timing failed");
            }
            record(phase, ProfileOperation::mxfp4_matvec,
                   NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start,
                   weights[index].packed.size_bytes() +
                       weights[index].scales.size_bytes(),
                   0,
                   static_cast<std::uint64_t>(std::llround(
                       static_cast<double>(elapsed_milliseconds) * 1.0e6)),
                   true);
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<float>> dense_situ_mlp(
        std::span<const float> input, DenseMlpView weights, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        const auto precision = numeric_precision();
        if (options_.kind != BackendKind::cuda_custom ||
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
            !valid_dense_mlp(input, weights) || !std::isfinite(situ_beta) ||
            situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }

        std::vector<__nv_bfloat16> bf16_input;
        std::array<std::vector<__nv_bfloat16>, 3> bf16_weights;
        const void* host_input = input.data();
        std::size_t input_bytes = input.size_bytes();
        const auto input_type =
            options_.dense_precision == DensePrecision::fp32
                ? CUDA_R_32F
                : CUDA_R_16BF;
        const auto weight_type = input_type;
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
        }

        struct WeightMember {
            DenseWeightView view;
            const void* host{};
            std::size_t bytes{};
            void* device{};
            std::uint64_t transfer_bytes{};
        };
        std::array<WeightMember, 3> members{{
            {weights.gate}, {weights.up}, {weights.down}}};
        std::size_t maximum_weight_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        for (std::size_t index = 0; index < members.size(); ++index) {
            auto& member = members[index];
            member.host = member.view.values.data();
            member.bytes = member.view.values.size_bytes();
            if (options_.dense_precision == DensePrecision::bf16_rounded) {
                auto& converted = bf16_weights[index];
                converted.reserve(member.view.values.size());
                for (const auto value : member.view.values) {
                    converted.push_back(__float2bfloat16_rn(value));
                }
                member.host = converted.data();
                member.bytes = converted.size() * sizeof(__nv_bfloat16);
            }
            member.transfer_bytes = member.bytes;
            maximum_weight_bytes = std::max(maximum_weight_bytes, member.bytes);
            if (resident_weights_) {
                const auto representation =
                    options_.dense_precision == DensePrecision::fp32
                        ? cuda::WeightRepresentation::dense_fp32
                        : cuda::WeightRepresentation::dense_bf16;
                const auto acquisition = resident_weights_->acquire(
                    {member.view.tensor_id, representation, member.view.rows,
                     member.view.cols, 0},
                    std::span(static_cast<const std::byte*>(member.host),
                              member.bytes),
                    {});
                if (!acquisition) {
                    return Result<std::vector<float>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device =
                        const_cast<void*>(acquisition.value().primary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        const auto intermediate_count = weights.gate.rows;
        const auto gate_bytes = intermediate_count * sizeof(float);
        const auto activation_bytes = intermediate_count *
            (options_.dense_precision == DensePrecision::fp32
                 ? sizeof(float)
                 : sizeof(__nv_bfloat16));
        const auto output_bytes = weights.down.rows * sizeof(float);
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_gate(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_up(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_activation(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_weight = nullptr;
        void* device_gate = nullptr;
        void* device_up = nullptr;
        void* device_activation = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                ffn_weight_scratch_.reserve(maximum_weight_bytes) != cudaSuccess ||
                ffn_gate_scratch_.reserve(gate_bytes) != cudaSuccess ||
                ffn_up_scratch_.reserve(gate_bytes) != cudaSuccess ||
                ffn_activation_scratch_.reserve(activation_bytes) != cudaSuccess ||
                ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA FFN reusable allocation failed");
            }
            device_input = ffn_input_scratch_.get();
            transient_weight = ffn_weight_scratch_.get();
            device_gate = ffn_gate_scratch_.get();
            device_up = ffn_up_scratch_.get();
            device_activation = ffn_activation_scratch_.get();
            device_output = ffn_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_weight.allocate(maximum_weight_bytes) != cudaSuccess ||
                   local_gate.allocate(gate_bytes) != cudaSuccess ||
                   local_up.allocate(gate_bytes) != cudaSuccess ||
                   local_activation.allocate(activation_bytes) != cudaSuccess ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN allocation failed");
        } else {
            device_input = local_input.get();
            transient_weight = local_weight.get();
            device_gate = local_gate.get();
            device_up = local_up.get();
            device_activation = local_activation.get();
            device_output = local_output.get();
        }

        std::vector<std::unique_ptr<DensePlan>> local_plans;
        const auto resolve_plan = [&](std::size_t rows, std::size_t cols)
            -> DensePlan* {
            const DensePlanKey key{rows, cols, static_cast<int>(input_type),
                                   static_cast<int>(weight_type)};
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                const auto found = dense_plans_.find(key);
                if (found != dense_plans_.end()) return found->second.get();
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, rows, cols, input_type,
                                           weight_type)) return nullptr;
                auto* result = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
                return result;
            }
            auto candidate = std::make_unique<DensePlan>();
            if (!initialize_dense_plan(*candidate, rows, cols, input_type,
                                       weight_type)) return nullptr;
            auto* result = candidate.get();
            local_plans.push_back(std::move(candidate));
            return result;
        };
        auto* gate_plan = resolve_plan(weights.gate.rows, weights.gate.cols);
        auto* up_plan = resolve_plan(weights.up.rows, weights.up.cols);
        auto* down_plan = resolve_plan(weights.down.rows, weights.down.cols);
        if (!gate_plan || !up_plan || !down_plan) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN plan creation failed");
        }

        EventOwner gate_start, gate_end, up_start, up_end;
        EventOwner situ_start, situ_end, down_start, down_end;
        for (auto* event : {&gate_start, &gate_end, &up_start, &up_end,
                            &situ_start, &situ_end, &down_start, &down_end}) {
            if (event->ensure() != cudaSuccess) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA FFN event creation failed");
            }
        }
        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA FFN activation upload failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto run_matmul = [&](WeightMember& member, DensePlan* plan,
                                    const void* matmul_input, void* output,
                                    EventOwner& start, EventOwner& end) {
            if (!member.device) {
                member.device = transient_weight;
                if (cudaMemcpyAsync(member.device, member.host, member.bytes,
                                    cudaMemcpyHostToDevice, stream_) !=
                    cudaSuccess) return false;
            }
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cublasLtMatmul(
                       handle_, plan->operation.get(), &alpha, member.device,
                       plan->weight_layout.get(), matmul_input,
                       plan->input_layout.get(), &beta, output,
                       plan->output_layout.get(), output,
                       plan->output_layout.get(), &plan->heuristic.algo,
                       nullptr, 0, stream_) == CUBLAS_STATUS_SUCCESS &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        if (!run_matmul(members[0], gate_plan, device_input, device_gate,
                        gate_start, gate_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN gate failed");
        }
        if (!run_matmul(members[1], up_plan, device_input, device_up,
                        up_start, up_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN up failed");
        }
        if (cudaEventRecord(situ_start.get(), stream_) != cudaSuccess ||
            cuda::launch_situ_glu(
                static_cast<const float*>(device_gate),
                static_cast<const float*>(device_up), device_activation,
                intermediate_count, situ_beta, situ_linear.has_value(),
                situ_linear.value_or(0.0F),
                options_.dense_precision == DensePrecision::bf16_rounded,
                stream_) != cudaSuccess ||
            cudaEventRecord(situ_end.get(), stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN SiTU failed");
        }
        if (!run_matmul(members[2], down_plan, device_activation, device_output,
                        down_start, down_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN down failed");
        }

        std::vector<float> output(weights.down.rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA FFN output copy or synchronization failed");
        }

        const auto elapsed = [](EventOwner& start, EventOwner& end,
                                std::uint64_t& nanoseconds) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, start.get(), end.get()) !=
                cudaSuccess) return false;
            nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
            return true;
        };
        std::uint64_t gate_ns = 0, up_ns = 0, situ_ns = 0, down_ns = 0;
        if (!elapsed(gate_start, gate_end, gate_ns) ||
            !elapsed(up_start, up_end, up_ns) ||
            !elapsed(situ_start, situ_end, situ_ns) ||
            !elapsed(down_start, down_end, down_ns)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN event timing failed");
        }

        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        runtime_stats_.device_to_host_bytes += output_bytes;
        ++runtime_stats_.ffn_block_calls;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.gate.values.size_bytes(), 0, gate_ns,
               true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.up.values.size_bytes(), 0, up_ns, true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.down.values.size_bytes(), 0, down_ns,
               true);
        record(phase, ProfileOperation::situ_glu, precision, layer,
               operation_start, intermediate_count * sizeof(float), 0,
               situ_ns, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        return mxfp4_situ_mlp_group_impl(
            input, experts, {}, situ_beta, situ_linear, layer, phase);
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_grid(
        std::span<const float> inputs, std::size_t token_count,
        std::span<const Mxfp4MlpView> experts, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto operation_start = std::chrono::steady_clock::now();
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::resident ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            options_.cuda_resident_bytes == 0 || resident_weights_ == nullptr ||
            token_count == 0 ||
            token_count > 65535 || experts.empty() || experts.size() > 65535 ||
            experts.front().gate.cols == 0 ||
            !multiply_fits(token_count, experts.front().gate.cols) ||
            inputs.size() != token_count * experts.front().gate.cols ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto input_width = experts.front().gate.cols;
        const auto intermediate_width = experts.front().gate.rows;
        const auto output_width = experts.front().down.rows;
        std::unordered_set<std::uint64_t> tensor_ids;
        for (const auto& expert : experts) {
            if (expert.gate.cols != input_width ||
                expert.gate.rows != intermediate_width ||
                expert.up.cols != input_width ||
                expert.up.rows != intermediate_width ||
                expert.down.cols != intermediate_width ||
                expert.down.rows != output_width ||
                expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(input_width, expert.gate) ||
                !valid_mxfp4_size(input_width, expert.up) ||
                !valid_mxfp4_size(intermediate_width, expert.down)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            for (const auto tensor_id : {
                     expert.gate.tensor_id, expert.up.tensor_id,
                     expert.down.tensor_id}) {
                if (tensor_id == 0 || !tensor_ids.insert(tensor_id).second) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::invalid_mxfp4);
                }
            }
        }
        if (!multiply_fits(experts.size(), token_count) ||
            !multiply_fits(experts.size() * token_count, intermediate_width) ||
            !multiply_fits(experts.size() * token_count, output_width) ||
            intermediate_width > 65535 || output_width > 65535) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::vector<cuda::Mxfp4DeviceMatrix> descriptors(experts.size() * 3);
        std::uint64_t weight_transfer_bytes = 0;
        bool bypass = false;
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            const auto weights = std::array{
                experts[expert_index].gate,
                experts[expert_index].up,
                experts[expert_index].down,
            };
            for (std::size_t projection = 0; projection < weights.size();
                 ++projection) {
                const auto& weight = weights[projection];
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, cuda::WeightRepresentation::mxfp4,
                     weight.rows, weight.cols, weight.group_size},
                    weight.packed, weight.scales);
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition ==
                    cuda::ResidentDisposition::bypass) {
                    bypass = true;
                    continue;
                }
                descriptors[projection * experts.size() + expert_index] = {
                    static_cast<const std::uint8_t*>(
                        acquisition.value().primary),
                    static_cast<const std::uint8_t*>(
                        acquisition.value().secondary),
                };
                weight_transfer_bytes += acquisition.value().uploaded_bytes;
            }
        }
        if (bypass) {
            ++runtime_stats_.resident_grid_fallbacks;
            if (weight_transfer_bytes != 0) {
                runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
                record(phase, ProfileOperation::weight_host_to_device,
                       precision, layer, operation_start, 0,
                       weight_transfer_bytes, 0, true);
            }
            std::vector<std::vector<float>> outputs(experts.size());
            for (std::size_t token = 0; token < token_count; ++token) {
                const auto token_input = inputs.subspan(
                    token * input_width, input_width);
                auto token_outputs = mxfp4_situ_mlp_group_impl(
                    token_input, experts, {}, situ_beta, situ_linear, layer,
                    phase);
                if (!token_outputs) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        token_outputs.error(), token_outputs.message());
                }
                for (std::size_t expert = 0; expert < experts.size(); ++expert) {
                    outputs[expert].insert(outputs[expert].end(),
                                           token_outputs.value()[expert].begin(),
                                           token_outputs.value()[expert].end());
                }
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }

        const auto expert_tokens = experts.size() * token_count;
        const auto input_bytes = inputs.size_bytes();
        const auto descriptor_bytes =
            descriptors.size() * sizeof(cuda::Mxfp4DeviceMatrix);
        const auto intermediate_count = expert_tokens * intermediate_width;
        const auto intermediate_bytes = intermediate_count * sizeof(float);
        const auto output_count = expert_tokens * output_width;
        const auto output_bytes = output_count * sizeof(float);
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            mxfp4_descriptor_scratch_.reserve(descriptor_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid allocation failed");
        }
        for (auto& event : resident_grid_events_) {
            if (event.ensure() != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA resident expert grid event creation failed");
            }
        }

        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_descriptors = static_cast<cuda::Mxfp4DeviceMatrix*>(
            mxfp4_descriptor_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation = static_cast<float*>(
            ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, inputs.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_descriptors, descriptors.data(),
                            descriptor_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid upload failed");
        }
        const auto launch_grid = [&](std::size_t event_pair,
                                     std::size_t descriptor_projection,
                                     const float* grid_input, float* grid_output,
                                     std::size_t rows, std::size_t cols,
                                     cuda::ExpertGridInputLayout layout) {
            auto& start = resident_grid_events_[event_pair * 2];
            auto& end = resident_grid_events_[event_pair * 2 + 1];
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cuda::launch_mxfp4_matvec_grid(
                       grid_input,
                       device_descriptors +
                           descriptor_projection * experts.size(),
                       grid_output, rows, cols, experts.size(), token_count,
                       layout, stream_) == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        if (!launch_grid(0, 0, device_input, device_gate, intermediate_width,
                         input_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_grid(1, 1, device_input, device_up, intermediate_width,
                         input_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            cudaEventRecord(resident_grid_events_[4].get(), stream_) !=
                cudaSuccess ||
            cuda::launch_situ_glu(
                device_gate, device_up, device_activation, intermediate_count,
                situ_beta, situ_linear.has_value(), situ_linear.value_or(0.0F),
                false, stream_) != cudaSuccess ||
            cudaEventRecord(resident_grid_events_[5].get(), stream_) !=
                cudaSuccess ||
            !launch_grid(3, 2, device_activation, device_output, output_width,
                         intermediate_width,
                         cuda::ExpertGridInputLayout::expert_token_major)) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid launch failed");
        }

        std::vector<float> flat_output(output_count);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(flat_output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid output or synchronization failed");
        }
        std::array<std::uint64_t, 4> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds,
                                     resident_grid_events_[pair * 2].get(),
                                     resident_grid_events_[pair * 2 + 1].get()) !=
                cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA resident expert grid timing failed");
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }

        std::vector<std::vector<float>> outputs(experts.size());
        const auto values_per_expert = token_count * output_width;
        for (std::size_t expert = 0; expert < experts.size(); ++expert) {
            outputs[expert].assign(
                flat_output.begin() + expert * values_per_expert,
                flat_output.begin() + (expert + 1) * values_per_expert);
        }
        std::array<std::uint64_t, 3> logical_projection_bytes{};
        for (const auto& expert : experts) {
            const auto weights = std::array{expert.gate, expert.up, expert.down};
            for (std::size_t projection = 0; projection < weights.size();
                 ++projection) {
                logical_projection_bytes[projection] +=
                    weights[projection].packed.size_bytes() +
                    weights[projection].scales.size_bytes();
            }
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
        runtime_stats_.device_to_host_bytes += output_bytes;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        ++runtime_stats_.resident_grid_calls;
        runtime_stats_.resident_grid_experts += experts.size();
        runtime_stats_.resident_grid_tokens += token_count;
        runtime_stats_.resident_grid_expert_tokens += expert_tokens;
        runtime_stats_.resident_grid_kernel_launches += 4;
        runtime_stats_.resident_grid_descriptor_h2d_bytes += descriptor_bytes;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes + descriptor_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, weight_transfer_bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[0], 0,
               durations[0], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[1], 0,
               durations[1], true);
        record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
               layer, operation_start, intermediate_bytes, 0, durations[2],
               true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[2], 0,
               durations[3], true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<ResidentMoeLayerResult> resident_mxfp4_moe_layer(
        std::span<const float> input, ResidentMoeLayerView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto dense_precision = NumericPrecision::fp32;
        constexpr auto expert_precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto all_finite = [](std::span<const float> values) {
            return std::all_of(values.begin(), values.end(),
                               [](float value) { return std::isfinite(value); });
        };
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.dense_precision != DensePrecision::fp32 ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::resident ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            options_.cuda_resident_bytes == 0 || resident_weights_ == nullptr ||
            input.empty() || experts.empty() || experts.size() > 65535 ||
            experts.size() != contributions.size() ||
            !std::isfinite(epsilon) || epsilon <= 0.0F ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F)) ||
            !all_finite(input) || !all_finite(contributions) ||
            !valid_dense(input, weights.routed_down) ||
            !valid_dense_mlp(input, weights.shared)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto hidden_width = input.size();
        const auto latent_width = weights.routed_down.rows;
        const auto routed_width = experts.front().down.rows;
        const auto intermediate_width = experts.front().gate.rows;
        const auto shared_width = weights.shared.gate.rows;
        if (latent_width == 0 || routed_width == 0 ||
            intermediate_width == 0 || shared_width == 0 ||
            latent_width > 65535 || routed_width > 65535 ||
            intermediate_width > 65535 ||
            weights.routed_norm.values.size() != routed_width ||
            !valid_dense_size(routed_width, weights.routed_up) ||
            weights.routed_up.rows != hidden_width ||
            weights.shared.down.rows != hidden_width ||
            !multiply_fits(experts.size(), intermediate_width) ||
            !multiply_fits(experts.size(), routed_width)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::unordered_set<std::uint64_t> tensor_ids;
        const auto insert_id = [&tensor_ids](std::uint64_t tensor_id) {
            return tensor_id != 0 && tensor_ids.insert(tensor_id).second;
        };
        if (!insert_id(weights.routed_down.tensor_id) ||
            !insert_id(weights.routed_norm.tensor_id) ||
            !insert_id(weights.routed_up.tensor_id) ||
            !insert_id(weights.shared.gate.tensor_id) ||
            !insert_id(weights.shared.up.tensor_id) ||
            !insert_id(weights.shared.down.tensor_id)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        for (const auto& expert : experts) {
            if (expert.gate.cols != latent_width ||
                expert.gate.rows != intermediate_width ||
                expert.up.cols != latent_width ||
                expert.up.rows != intermediate_width ||
                expert.down.cols != intermediate_width ||
                expert.down.rows != routed_width ||
                expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(latent_width, expert.gate) ||
                !valid_mxfp4_size(latent_width, expert.up) ||
                !valid_mxfp4_size(intermediate_width, expert.down) ||
                !insert_id(expert.gate.tensor_id) ||
                !insert_id(expert.up.tensor_id) ||
                !insert_id(expert.down.tensor_id)) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }

        const std::array dense_views{
            weights.routed_down,
            DenseWeightView{weights.routed_norm.tensor_id,
                            weights.routed_norm.values, 1, routed_width},
            weights.routed_up,
            weights.shared.gate,
            weights.shared.up,
            weights.shared.down,
        };
        std::array<bool, 6> validation_hits{};
        std::array<bool, 6> validation_scans{};
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < dense_views.size(); ++index) {
                const auto& view = dense_views[index];
                const ImmutableWeightIdentity identity{
                    view.values.data(), view.values.size_bytes(),
                    view.rows, view.cols};
                const auto found = immutable_weights_.find(view.tensor_id);
                if (found == immutable_weights_.end()) {
                    validation_scans[index] = true;
                } else if (found->second == identity) {
                    validation_hits[index] = true;
                } else {
                    return Result<ResidentMoeLayerResult>::failure(
                        ErrorCode::invalid_mxfp4);
                }
            }
        } else {
            validation_scans.fill(true);
        }
        const auto validation_start = std::chrono::steady_clock::now();
        bool immutable_finite = true;
        for (std::size_t index = 0; index < dense_views.size(); ++index) {
            if (!validation_scans[index]) continue;
            ++runtime_stats_.immutable_validation_scans;
            runtime_stats_.immutable_validation_bytes +=
                dense_views[index].values.size_bytes();
            if (!all_finite(dense_views[index].values)) immutable_finite = false;
        }
        runtime_stats_.immutable_validation_nanoseconds +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                         validation_start).count());
        if (!immutable_finite) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < dense_views.size(); ++index) {
                if (validation_hits[index]) {
                    ++runtime_stats_.immutable_validation_hits;
                } else {
                    const auto& view = dense_views[index];
                    immutable_weights_.emplace(
                        view.tensor_id,
                        ImmutableWeightIdentity{
                            view.values.data(), view.values.size_bytes(),
                            view.rows, view.cols});
                }
            }
        }
        std::array<cuda::ResidentAcquisition, 6> dense_members{};
        std::uint64_t weight_transfer_bytes = 0;
        bool bypass = false;
        for (std::size_t index = 0; index < dense_views.size(); ++index) {
            const auto& view = dense_views[index];
            const auto acquired = resident_weights_->acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_fp32,
                 view.rows, view.cols, 0},
                std::as_bytes(view.values), {});
            if (!acquired) {
                return Result<ResidentMoeLayerResult>::failure(
                    acquired.error(), acquired.message());
            }
            dense_members[index] = acquired.value();
            bypass |= acquired.value().disposition ==
                      cuda::ResidentDisposition::bypass;
            weight_transfer_bytes += acquired.value().uploaded_bytes;
        }

        std::vector<cuda::Mxfp4DeviceMatrix> descriptors(experts.size() * 3);
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            const std::array expert_weights{
                experts[expert_index].gate,
                experts[expert_index].up,
                experts[expert_index].down,
            };
            for (std::size_t projection = 0;
                 projection < expert_weights.size(); ++projection) {
                const auto& view = expert_weights[projection];
                const auto acquired = resident_weights_->acquire(
                    {view.tensor_id, cuda::WeightRepresentation::mxfp4,
                     view.rows, view.cols, view.group_size},
                    view.packed, view.scales);
                if (!acquired) {
                    return Result<ResidentMoeLayerResult>::failure(
                        acquired.error(), acquired.message());
                }
                bypass |= acquired.value().disposition ==
                          cuda::ResidentDisposition::bypass;
                weight_transfer_bytes += acquired.value().uploaded_bytes;
                if (acquired.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    descriptors[projection * experts.size() + expert_index] = {
                        static_cast<const std::uint8_t*>(
                            acquired.value().primary),
                        static_cast<const std::uint8_t*>(
                            acquired.value().secondary),
                    };
                }
            }
        }
        if (bypass) {
            ++runtime_stats_.resident_moe_layer_fallbacks;
            if (weight_transfer_bytes != 0) {
                runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
                record(phase, ProfileOperation::weight_host_to_device,
                       dense_precision, layer, operation_start, 0,
                       weight_transfer_bytes, 0, true);
            }
            return Result<ResidentMoeLayerResult>::success({false, {}});
        }

        std::array<DensePlan*, 5> plans{};
        const std::array plan_views{
            dense_views[0], dense_views[2], dense_views[3], dense_views[4],
            dense_views[5],
        };
        for (std::size_t index = 0; index < plan_views.size(); ++index) {
            const auto& view = plan_views[index];
            const DensePlanKey key{view.rows, view.cols,
                                   static_cast<int>(CUDA_R_32F),
                                   static_cast<int>(CUDA_R_32F)};
            const auto found = dense_plans_.find(key);
            if (found != dense_plans_.end()) {
                plans[index] = found->second.get();
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, view.rows, view.cols,
                                           CUDA_R_32F, CUDA_R_32F)) {
                    return Result<ResidentMoeLayerResult>::failure(
                        ErrorCode::backend_unavailable,
                        "resident MoE layer dense plan creation failed");
                }
                plans[index] = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
            }
        }

        const auto input_bytes = input.size_bytes();
        const auto latent_bytes = latent_width * sizeof(float);
        const auto descriptor_bytes =
            descriptors.size() * sizeof(cuda::Mxfp4DeviceMatrix);
        const auto expert_intermediate_count =
            experts.size() * intermediate_width;
        const auto expert_intermediate_bytes =
            expert_intermediate_count * sizeof(float);
        const auto expert_output_count = experts.size() * routed_width;
        const auto expert_output_bytes = expert_output_count * sizeof(float);
        const auto contribution_bytes = contributions.size_bytes();
        const auto routed_bytes = routed_width * sizeof(float);
        const auto hidden_bytes = hidden_width * sizeof(float);
        const auto shared_bytes = shared_width * sizeof(float);
        if (layer_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            layer_routed_latent_scratch_.reserve(latent_bytes) != cudaSuccess ||
            layer_descriptor_scratch_.reserve(descriptor_bytes) != cudaSuccess ||
            layer_expert_gate_scratch_.reserve(expert_intermediate_bytes) !=
                cudaSuccess ||
            layer_expert_up_scratch_.reserve(expert_intermediate_bytes) !=
                cudaSuccess ||
            layer_expert_activation_scratch_.reserve(
                expert_intermediate_bytes) != cudaSuccess ||
            layer_expert_output_scratch_.reserve(expert_output_bytes) !=
                cudaSuccess ||
            layer_contribution_scratch_.reserve(contribution_bytes) !=
                cudaSuccess ||
            layer_mixed_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_normalized_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_routed_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_shared_gate_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_up_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_activation_scratch_.reserve(shared_bytes) !=
                cudaSuccess ||
            layer_shared_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_final_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer scratch allocation failed");
        }
        for (auto& event : resident_moe_layer_events_) {
            if (event.ensure() != cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    "resident MoE layer event creation failed");
            }
        }

        auto* device_input = static_cast<float*>(layer_input_scratch_.get());
        auto* device_latent =
            static_cast<float*>(layer_routed_latent_scratch_.get());
        auto* device_descriptors = static_cast<cuda::Mxfp4DeviceMatrix*>(
            layer_descriptor_scratch_.get());
        auto* device_expert_gate =
            static_cast<float*>(layer_expert_gate_scratch_.get());
        auto* device_expert_up =
            static_cast<float*>(layer_expert_up_scratch_.get());
        auto* device_expert_activation =
            static_cast<float*>(layer_expert_activation_scratch_.get());
        auto* device_expert_output =
            static_cast<float*>(layer_expert_output_scratch_.get());
        auto* device_contributions =
            static_cast<float*>(layer_contribution_scratch_.get());
        auto* device_mixed = static_cast<float*>(layer_mixed_scratch_.get());
        auto* device_normalized =
            static_cast<float*>(layer_normalized_scratch_.get());
        auto* device_routed =
            static_cast<float*>(layer_routed_hidden_scratch_.get());
        auto* device_shared_gate =
            static_cast<float*>(layer_shared_gate_scratch_.get());
        auto* device_shared_up =
            static_cast<float*>(layer_shared_up_scratch_.get());
        auto* device_shared_activation =
            static_cast<float*>(layer_shared_activation_scratch_.get());
        auto* device_shared =
            static_cast<float*>(layer_shared_hidden_scratch_.get());
        auto* device_final =
            static_cast<float*>(layer_final_hidden_scratch_.get());
        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_contributions, contributions.data(),
                            contribution_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_descriptors, descriptors.data(),
                            descriptor_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer upload failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto launch_dense = [&](std::size_t pair, DensePlan* plan,
                                      const void* weight, const float* source,
                                      float* output) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cublasLtMatmul(
                       handle_, plan->operation.get(), &alpha, weight,
                       plan->weight_layout.get(), source,
                       plan->input_layout.get(), &beta, output,
                       plan->output_layout.get(), output,
                       plan->output_layout.get(), &plan->heuristic.algo,
                       nullptr, 0, stream_) == CUBLAS_STATUS_SUCCESS &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        const auto launch_grid = [&](std::size_t pair,
                                     std::size_t projection,
                                     const float* source, float* output,
                                     std::size_t rows, std::size_t cols,
                                     cuda::ExpertGridInputLayout layout) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cuda::launch_mxfp4_matvec_grid(
                       source,
                       device_descriptors + projection * experts.size(),
                       output, rows, cols, experts.size(), 1, layout,
                       stream_) == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        const auto launch_simple = [&](std::size_t pair, auto&& launch) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   launch() == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        if (!launch_dense(0, plans[0], dense_members[0].primary,
                          device_input, device_latent) ||
            !launch_grid(1, 0, device_latent, device_expert_gate,
                         intermediate_width, latent_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_grid(2, 1, device_latent, device_expert_up,
                         intermediate_width, latent_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_simple(3, [&] {
                return cuda::launch_situ_glu(
                    device_expert_gate, device_expert_up,
                    device_expert_activation, expert_intermediate_count,
                    situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_);
            }) ||
            !launch_grid(4, 2, device_expert_activation,
                         device_expert_output, routed_width,
                         intermediate_width,
                         cuda::ExpertGridInputLayout::expert_token_major) ||
            !launch_simple(5, [&] {
                return cuda::launch_ordered_expert_mix(
                    device_expert_output, device_contributions,
                    contributions, device_mixed, routed_width, stream_);
            }) ||
            !launch_simple(6, [&] {
                return cuda::launch_strict_rms_norm(
                    device_mixed,
                    static_cast<const float*>(dense_members[1].primary),
                    device_normalized, routed_width, epsilon, stream_);
            }) ||
            !launch_dense(7, plans[1], dense_members[2].primary,
                          device_normalized, device_routed) ||
            !launch_dense(8, plans[2], dense_members[3].primary,
                          device_input, device_shared_gate) ||
            !launch_dense(9, plans[3], dense_members[4].primary,
                          device_input, device_shared_up) ||
            !launch_simple(10, [&] {
                return cuda::launch_situ_glu(
                    device_shared_gate, device_shared_up,
                    device_shared_activation, shared_width, situ_beta,
                    situ_linear.has_value(), situ_linear.value_or(0.0F),
                    false, stream_);
            }) ||
            !launch_dense(11, plans[4], dense_members[5].primary,
                          device_shared_activation, device_shared) ||
            !launch_simple(12, [&] {
                return cuda::launch_vector_add(
                    device_routed, device_shared, device_final,
                    hidden_width, stream_);
            })) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer launch failed");
        }

        std::vector<float> output(hidden_width);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_final, hidden_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer output or synchronization failed");
        }
        std::array<std::uint64_t, 13> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(
                    &milliseconds,
                    resident_moe_layer_events_[pair * 2].get(),
                    resident_moe_layer_events_[pair * 2 + 1].get()) !=
                cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    "resident MoE layer timing failed");
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }

        std::array<std::uint64_t, 3> expert_logical_bytes{};
        for (const auto& expert : experts) {
            const std::array expert_weights{
                expert.gate, expert.up, expert.down};
            for (std::size_t projection = 0; projection < 3; ++projection) {
                expert_logical_bytes[projection] +=
                    expert_weights[projection].packed.size_bytes() +
                    expert_weights[projection].scales.size_bytes();
            }
        }
        const auto activation_transfer_bytes =
            input_bytes + contribution_bytes + descriptor_bytes;
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += activation_transfer_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
        runtime_stats_.device_to_host_bytes += hidden_bytes;
        ++runtime_stats_.resident_moe_layer_calls;
        runtime_stats_.resident_moe_layer_experts += experts.size();
        runtime_stats_.resident_moe_layer_kernel_launches += 13;
        runtime_stats_.resident_moe_layer_contribution_h2d_bytes +=
            contribution_bytes;
        ++runtime_stats_.resident_grid_calls;
        runtime_stats_.resident_grid_experts += experts.size();
        ++runtime_stats_.resident_grid_tokens;
        runtime_stats_.resident_grid_expert_tokens += experts.size();
        runtime_stats_.resident_grid_kernel_launches += 4;
        runtime_stats_.resident_grid_descriptor_h2d_bytes += descriptor_bytes;
        record(phase, ProfileOperation::activation_host_to_device,
               dense_precision, layer, h2d_start, 0,
               activation_transfer_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device,
               dense_precision, layer, operation_start, 0,
               weight_transfer_bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, dense_precision, layer,
               d2h_start, 0, hidden_bytes, 0, true);
        const std::array operations{
            ProfileOperation::dense_matvec,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::situ_glu,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::moe_mix,
            ProfileOperation::rms_norm,
            ProfileOperation::dense_matvec,
            ProfileOperation::dense_matvec,
            ProfileOperation::dense_matvec,
            ProfileOperation::situ_glu,
            ProfileOperation::dense_matvec,
            ProfileOperation::residual_add,
        };
        const std::array logical_bytes{
            dense_views[0].values.size_bytes(), expert_logical_bytes[0],
            expert_logical_bytes[1], expert_intermediate_bytes,
            expert_logical_bytes[2], expert_output_bytes + contribution_bytes,
            routed_bytes + dense_views[1].values.size_bytes(),
            dense_views[2].values.size_bytes(),
            dense_views[3].values.size_bytes(),
            dense_views[4].values.size_bytes(), shared_bytes,
            dense_views[5].values.size_bytes(), hidden_bytes * 2,
        };
        for (std::size_t index = 0; index < operations.size(); ++index) {
            const auto precision = index >= 1 && index <= 4 && index != 3
                                       ? expert_precision
                                       : dense_precision;
            record(phase, operations[index], precision, layer,
                   operation_start, logical_bytes[index], 0,
                   durations[index], true);
        }
        return Result<ResidentMoeLayerResult>::success(
            {true, std::move(output)});
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_batch(
        std::span<const float> inputs, std::size_t batch_size,
        Mxfp4MlpView expert, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto operation_start = std::chrono::steady_clock::now();
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::ffn_block ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::transient ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            batch_size == 0 || batch_size > 65535 ||
            expert.gate.group_size != 32 || expert.up.group_size != 32 ||
            expert.down.group_size != 32 ||
            !multiply_fits(batch_size, expert.gate.cols) ||
            !multiply_fits(batch_size, expert.gate.rows) ||
            !multiply_fits(batch_size, expert.down.rows) ||
            inputs.size() != batch_size * expert.gate.cols ||
            !valid_mxfp4_size(expert.gate.cols, expert.gate) ||
            !valid_mxfp4_size(expert.gate.cols, expert.up) ||
            expert.gate.rows != expert.up.rows ||
            !valid_mxfp4_size(expert.gate.rows, expert.down) ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto input_bytes = inputs.size_bytes();
        const auto intermediate_count = batch_size * expert.gate.rows;
        const auto intermediate_bytes = intermediate_count * sizeof(float);
        const auto output_count = batch_size * expert.down.rows;
        const auto output_bytes = output_count * sizeof(float);
        const auto maximum_packed_bytes = std::max({
            expert.gate.packed.size_bytes(), expert.up.packed.size_bytes(),
            expert.down.packed.size_bytes()});
        const auto maximum_scale_bytes = std::max({
            expert.gate.scales.size_bytes(), expert.up.scales.size_bytes(),
            expert.down.scales.size_bytes()});
        const auto total_weight_transfer =
            expert.gate.packed.size_bytes() + expert.gate.scales.size_bytes() +
            expert.up.packed.size_bytes() + expert.up.scales.size_bytes() +
            expert.down.packed.size_bytes() + expert.down.scales.size_bytes();

        std::array<EventOwner, 8> events;
        for (auto& event : events) {
            if (event.ensure() != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA batched expert event creation failed");
            }
        }
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            mxfp4_packed_scratch_.reserve(maximum_packed_bytes) != cudaSuccess ||
            mxfp4_scales_scratch_.reserve(maximum_scale_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert reusable allocation failed");
        }

        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_packed = static_cast<std::uint8_t*>(
            mxfp4_packed_scratch_.get());
        auto* device_scales = static_cast<std::uint8_t*>(
            mxfp4_scales_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation = static_cast<float*>(
            ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        if (cudaMemcpyAsync(device_input, inputs.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert activation upload failed");
        }

        const auto launch_weight = [&](Mxfp4WeightView weight,
                                       const float* input, float* output,
                                       EventOwner& start, EventOwner& end) {
            if (cudaMemcpyAsync(device_packed, weight.packed.data(),
                                weight.packed.size_bytes(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaMemcpyAsync(device_scales, weight.scales.data(),
                                weight.scales.size_bytes(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            return cuda::launch_mxfp4_matvec_batch(
                       input, device_packed, device_scales, output,
                       weight.rows, weight.cols, batch_size, stream_) ==
                       cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        if (!launch_weight(expert.gate, device_input, device_gate,
                           events[0], events[1]) ||
            !launch_weight(expert.up, device_input, device_up,
                           events[2], events[3])) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert gate or up failed");
        }
        if (cudaEventRecord(events[4].get(), stream_) != cudaSuccess ||
            cuda::launch_situ_glu(
                device_gate, device_up, device_activation,
                intermediate_count, situ_beta, situ_linear.has_value(),
                situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
            cudaEventRecord(events[5].get(), stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert SiTU failed");
        }
        if (!launch_weight(expert.down, device_activation, device_output,
                           events[6], events[7])) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert down failed");
        }

        std::vector<float> flat_output(output_count);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(flat_output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert output copy or synchronization failed");
        }

        std::array<std::uint64_t, 4> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, events[pair * 2].get(),
                                     events[pair * 2 + 1].get()) !=
                cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA batched expert event timing failed");
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }

        std::vector<std::vector<float>> outputs(batch_size);
        for (std::size_t token = 0; token < batch_size; ++token) {
            outputs[token].assign(
                flat_output.begin() + token * expert.down.rows,
                flat_output.begin() + (token + 1) * expert.down.rows);
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.ffn_block_calls;
        ++runtime_stats_.ffn_block_experts;
        ++runtime_stats_.batched_expert_ffn_calls;
        runtime_stats_.batched_expert_ffn_tokens += batch_size;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision,
               layer, operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.gate.packed.size_bytes() +
                   expert.gate.scales.size_bytes(),
               0, durations[0], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.up.packed.size_bytes() + expert.up.scales.size_bytes(),
               0, durations[1], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.down.packed.size_bytes() +
                   expert.down.scales.size_bytes(),
               0, durations[3], true);
        record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
               layer, operation_start, intermediate_bytes, 0, durations[2],
               true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<float>> mxfp4_situ_moe(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (options_.cuda_moe_fusion !=
                CudaMoeFusionMode::routed_accumulate ||
            experts.empty() || experts.size() != contributions.size()) {
            return Result<std::vector<float>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        auto result = mxfp4_situ_mlp_group_impl(
            input, experts, contributions, situ_beta, situ_linear, layer,
            phase);
        if (!result) {
            return Result<std::vector<float>>::failure(
                result.error(), result.message());
        }
        return Result<std::vector<float>>::success(
            std::move(result.value().front()));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group_impl(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const bool fuse_outputs = !contributions.empty();
        if (options_.kind != BackendKind::cuda_custom ||
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
            (fuse_outputs && contributions.size() != experts.size()) ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        const auto output_rows = experts.empty() ? 0 : experts.front().down.rows;
        for (std::size_t index = 0; index < experts.size(); ++index) {
            const auto& expert = experts[index];
            if (expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(input.size(), expert.gate) ||
                !valid_mxfp4_size(input.size(), expert.up) ||
                expert.gate.rows != expert.up.rows ||
                !valid_mxfp4_size(expert.gate.rows, expert.down) ||
                (fuse_outputs && expert.down.rows != output_rows) ||
                (fuse_outputs && !std::isfinite(contributions[index]))) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        if (experts.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        struct WeightMember {
            Mxfp4WeightView view;
            void* packed{};
            void* scales{};
            std::uint64_t transfer_bytes{};
        };
        struct ExpertMember {
            std::array<WeightMember, 3> weights;
            std::array<std::unique_ptr<EventOwner>, 8> events;
        };
        std::vector<ExpertMember> members;
        members.reserve(experts.size());
        std::size_t maximum_packed_bytes = 0;
        std::size_t maximum_scale_bytes = 0;
        std::size_t maximum_intermediate = 0;
        std::size_t maximum_output = 0;
        std::uint64_t total_weight_transfer = 0;
        for (const auto& expert : experts) {
            ExpertMember member{{{{expert.gate}, {expert.up}, {expert.down}}}};
            for (auto& weight : member.weights) {
                weight.transfer_bytes =
                    weight.view.packed.size_bytes() + weight.view.scales.size_bytes();
                maximum_packed_bytes = std::max(
                    maximum_packed_bytes, weight.view.packed.size_bytes());
                maximum_scale_bytes = std::max(
                    maximum_scale_bytes, weight.view.scales.size_bytes());
                if (resident_weights_) {
                    const auto acquisition = resident_weights_->acquire(
                        {weight.view.tensor_id,
                         cuda::WeightRepresentation::mxfp4,
                         weight.view.rows, weight.view.cols,
                         weight.view.group_size},
                        weight.view.packed, weight.view.scales);
                    if (!acquisition) {
                        return Result<std::vector<std::vector<float>>>::failure(
                            acquisition.error(), acquisition.message());
                    }
                    if (acquisition.value().disposition !=
                        cuda::ResidentDisposition::bypass) {
                        weight.packed =
                            const_cast<void*>(acquisition.value().primary);
                        weight.scales =
                            const_cast<void*>(acquisition.value().secondary);
                        weight.transfer_bytes =
                            acquisition.value().uploaded_bytes;
                    }
                }
                total_weight_transfer += weight.transfer_bytes;
            }
            maximum_intermediate =
                std::max(maximum_intermediate, expert.gate.rows);
            maximum_output = std::max(maximum_output, expert.down.rows);
            for (auto& event : member.events) {
                event = std::make_unique<EventOwner>();
                if (event->ensure() != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN event creation failed");
                }
            }
            members.push_back(std::move(member));
        }

        const auto input_bytes = input.size_bytes();
        const auto intermediate_bytes = maximum_intermediate * sizeof(float);
        const auto output_bytes = maximum_output * sizeof(float);
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_gate(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_up(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_activation(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_packed = nullptr;
        void* transient_scales = nullptr;
        void* device_gate = nullptr;
        void* device_up = nullptr;
        void* device_activation = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                mxfp4_packed_scratch_.reserve(maximum_packed_bytes) !=
                    cudaSuccess ||
                mxfp4_scales_scratch_.reserve(maximum_scale_bytes) !=
                    cudaSuccess ||
                ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
                ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
                ffn_activation_scratch_.reserve(intermediate_bytes) !=
                    cudaSuccess ||
                ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN reusable allocation failed");
            }
            device_input = ffn_input_scratch_.get();
            transient_packed = mxfp4_packed_scratch_.get();
            transient_scales = mxfp4_scales_scratch_.get();
            device_gate = ffn_gate_scratch_.get();
            device_up = ffn_up_scratch_.get();
            device_activation = ffn_activation_scratch_.get();
            device_output = ffn_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_packed.allocate(maximum_packed_bytes) != cudaSuccess ||
                   local_scales.allocate(maximum_scale_bytes) != cudaSuccess ||
                   local_gate.allocate(intermediate_bytes) != cudaSuccess ||
                   local_up.allocate(intermediate_bytes) != cudaSuccess ||
                   local_activation.allocate(intermediate_bytes) != cudaSuccess ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN allocation failed");
        } else {
            device_input = local_input.get();
            transient_packed = local_packed.get();
            transient_scales = local_scales.get();
            device_gate = local_gate.get();
            device_up = local_up.get();
            device_activation = local_activation.get();
            device_output = local_output.get();
        }
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN activation upload failed");
        }

        const auto launch_weight = [&](WeightMember& weight,
                                       const float* matvec_input,
                                       float* output, EventOwner& start,
                                       EventOwner& end, bool scaled = false,
                                       float contribution = 1.0F,
                                       bool accumulate = false) {
            if (!weight.packed) {
                weight.packed = transient_packed;
                weight.scales = transient_scales;
                if (cudaMemcpyAsync(weight.packed, weight.view.packed.data(),
                                    weight.view.packed.size_bytes(),
                                    cudaMemcpyHostToDevice, stream_) !=
                        cudaSuccess ||
                    cudaMemcpyAsync(weight.scales, weight.view.scales.data(),
                                    weight.view.scales.size_bytes(),
                                    cudaMemcpyHostToDevice, stream_) !=
                        cudaSuccess) return false;
            }
            if (cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            const auto launched = scaled
                ? cuda::launch_mxfp4_matvec_accumulate(
                      matvec_input,
                      static_cast<const std::uint8_t*>(weight.packed),
                      static_cast<const std::uint8_t*>(weight.scales), output,
                      weight.view.rows, weight.view.cols, contribution,
                      accumulate, stream_)
                : cuda::launch_mxfp4_matvec(
                      matvec_input,
                      static_cast<const std::uint8_t*>(weight.packed),
                      static_cast<const std::uint8_t*>(weight.scales), output,
                      weight.view.rows, weight.view.cols, stream_);
            return launched == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        std::vector<std::vector<float>> outputs(
            fuse_outputs ? 1 : experts.size());
        std::uint64_t total_output_bytes = 0;
        for (std::size_t index = 0; index < experts.size(); ++index) {
            auto& member = members[index];
            const auto& expert = experts[index];
            if (!launch_weight(member.weights[0],
                               static_cast<const float*>(device_input),
                               static_cast<float*>(device_gate),
                               *member.events[0], *member.events[1]) ||
                !launch_weight(member.weights[1],
                               static_cast<const float*>(device_input),
                               static_cast<float*>(device_up),
                               *member.events[2], *member.events[3])) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN gate or up failed");
            }
            if (cudaEventRecord(member.events[4]->get(), stream_) != cudaSuccess ||
                cuda::launch_situ_glu(
                    static_cast<const float*>(device_gate),
                    static_cast<const float*>(device_up), device_activation,
                    expert.gate.rows, situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
                cudaEventRecord(member.events[5]->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN SiTU failed");
            }
            if (!launch_weight(member.weights[2],
                               static_cast<const float*>(device_activation),
                               static_cast<float*>(device_output),
                               *member.events[6], *member.events[7],
                               fuse_outputs,
                               fuse_outputs ? contributions[index] : 1.0F,
                               fuse_outputs && index != 0)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN down failed");
            }
            if (!fuse_outputs) {
                outputs[index].resize(expert.down.rows);
                const auto bytes = expert.down.rows * sizeof(float);
                if (cudaMemcpyAsync(outputs[index].data(), device_output, bytes,
                                    cudaMemcpyDeviceToHost, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN output copy failed");
                }
                total_output_bytes += bytes;
            }
        }
        if (fuse_outputs) {
            outputs.front().resize(output_rows);
            total_output_bytes = output_rows * sizeof(float);
            if (cudaMemcpyAsync(outputs.front().data(), device_output,
                                total_output_bytes, cudaMemcpyDeviceToHost,
                                stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA fused expert output copy failed");
            }
        }
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN synchronization failed");
        }

        const auto elapsed = [](EventOwner& start, EventOwner& end,
                                std::uint64_t& nanoseconds) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, start.get(), end.get()) !=
                cudaSuccess) return false;
            nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
            return true;
        };
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            std::array<std::uint64_t, 4> durations{};
            auto& events = members[expert_index].events;
            for (std::size_t pair = 0; pair < durations.size(); ++pair) {
                if (!elapsed(*events[pair * 2], *events[pair * 2 + 1],
                             durations[pair])) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN event timing failed");
                }
            }
            const auto& expert = experts[expert_index];
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.gate.packed.size_bytes() +
                       expert.gate.scales.size_bytes(),
                   0, durations[0], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.up.packed.size_bytes() + expert.up.scales.size_bytes(),
                   0, durations[1], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.down.packed.size_bytes() +
                       expert.down.scales.size_bytes(),
                   0, durations[3], true);
            record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
                   layer, operation_start,
                   expert.gate.rows * sizeof(float), 0, durations[2], true);
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        if (fuse_outputs) {
            ++runtime_stats_.fused_moe_calls;
            runtime_stats_.fused_moe_experts += experts.size();
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
        std::span<const Mxfp4MlpView> experts, std::uint64_t use_sequence,
        std::uint32_t layer, ProfilePhase phase) override {
        if (!async_pipeline_ || experts.empty()) {
            return Result<Mxfp4PrefetchToken>::failure(
                async_pipeline_ ? ErrorCode::invalid_extent
                                : ErrorCode::backend_unavailable);
        }
        const auto input_cols = experts.front().gate.cols;
        const auto output_rows = experts.front().down.rows;
        std::size_t maximum_intermediate = 0;
        std::size_t maximum_output = 0;
        for (const auto& expert : experts) {
            if (expert.gate.cols != input_cols ||
                expert.down.rows != output_rows) {
                return Result<Mxfp4PrefetchToken>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            maximum_intermediate =
                std::max(maximum_intermediate, expert.gate.rows);
            maximum_output = std::max(maximum_output, expert.down.rows);
        }
        auto token = async_pipeline_->prepare(
            experts, use_sequence, layer, phase);
        if (token) {
            prepared_mxfp4_ = PreparedMxfp4Metadata{
                token.value(), layer, phase, input_cols, maximum_intermediate,
                maximum_output, experts.size()};
        }
        return token;
    }

    Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        return mxfp4_situ_mlp_group_prepared_impl(
            input, token, {}, situ_beta, situ_linear, layer, phase);
    }

    Result<std::vector<float>> mxfp4_situ_moe_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (options_.cuda_moe_fusion !=
                CudaMoeFusionMode::routed_accumulate ||
            contributions.empty()) {
            return Result<std::vector<float>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        auto result = mxfp4_situ_mlp_group_prepared_impl(
            input, token, contributions, situ_beta, situ_linear, layer,
            phase);
        if (!result) {
            return Result<std::vector<float>>::failure(
                result.error(), result.message());
        }
        return Result<std::vector<float>>::success(
            std::move(result.value().front()));
    }

    Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared_impl(
        std::span<const float> input, Mxfp4PrefetchToken token,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const bool fuse_outputs = !contributions.empty();
        if (!async_pipeline_) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable);
        }
        if (!prepared_mxfp4_ || token.value == 0 || token.use_sequence == 0 ||
            token.value != prepared_mxfp4_->token.value ||
            token.use_sequence != prepared_mxfp4_->token.use_sequence ||
            layer != prepared_mxfp4_->layer || phase != prepared_mxfp4_->phase) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_state);
        }
        if ((fuse_outputs &&
             contributions.size() != prepared_mxfp4_->expert_count) ||
            (fuse_outputs &&
             std::any_of(contributions.begin(), contributions.end(),
                         [](float value) { return !std::isfinite(value); })) ||
            input.size() != prepared_mxfp4_->input_cols ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        struct PreparedMember {
            std::array<std::unique_ptr<EventOwner>, 8> events;
        };
        std::vector<PreparedMember> members(prepared_mxfp4_->expert_count);
        for (auto& member : members) {
            for (auto& event : member.events) {
                event = std::make_unique<EventOwner>();
                if (event->ensure() != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert FFN event creation failed");
                }
            }
        }

        const auto input_bytes = input.size_bytes();
        const auto intermediate_bytes =
            prepared_mxfp4_->maximum_intermediate * sizeof(float);
        const auto output_bytes =
            prepared_mxfp4_->maximum_output * sizeof(float);
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert FFN reusable allocation failed");
        }
        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation =
            static_cast<float*>(ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert activation upload failed");
        }

        const auto consumed = async_pipeline_->consume(
            token, layer, phase, stream_);
        if (!consumed ||
            consumed.value().size() != prepared_mxfp4_->expert_count) {
            return Result<std::vector<std::vector<float>>>::failure(
                consumed ? ErrorCode::backend_unavailable : consumed.error(),
                consumed ? "CUDA prepared expert count changed"
                         : consumed.message());
        }
        const auto experts = consumed.value();
        const auto launch_weight = [&](const cuda::DeviceMxfp4WeightView& weight,
                                       const float* matvec_input, float* output,
                                       EventOwner& start, EventOwner& end,
                                       bool scaled = false,
                                       float contribution = 1.0F,
                                       bool accumulate = false) {
            if (cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            const auto launched = scaled
                ? cuda::launch_mxfp4_matvec_accumulate(
                      matvec_input, weight.packed, weight.scales, output,
                      weight.rows, weight.cols, contribution, accumulate,
                      stream_)
                : cuda::launch_mxfp4_matvec(
                      matvec_input, weight.packed, weight.scales, output,
                      weight.rows, weight.cols, stream_);
            return launched == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        std::vector<std::vector<float>> outputs(
            fuse_outputs ? 1 : experts.size());
        std::uint64_t total_output_bytes = 0;
        const auto d2h_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < experts.size(); ++index) {
            const auto& expert = experts[index];
            auto& events = members[index].events;
            if (!launch_weight(expert.gate, device_input, device_gate,
                               *events[0], *events[1]) ||
                !launch_weight(expert.up, device_input, device_up,
                               *events[2], *events[3])) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert gate or up failed");
            }
            if (cudaEventRecord(events[4]->get(), stream_) != cudaSuccess ||
                cuda::launch_situ_glu(
                    device_gate, device_up, device_activation,
                    expert.gate.rows, situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
                cudaEventRecord(events[5]->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert SiTU failed");
            }
            if (!launch_weight(expert.down, device_activation, device_output,
                               *events[6], *events[7], fuse_outputs,
                               fuse_outputs ? contributions[index] : 1.0F,
                               fuse_outputs && index != 0)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert down failed");
            }
            if (!fuse_outputs) {
                outputs[index].resize(expert.down.rows);
                const auto bytes = expert.down.rows * sizeof(float);
                if (cudaMemcpyAsync(outputs[index].data(), device_output, bytes,
                                    cudaMemcpyDeviceToHost, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert output copy failed");
                }
                total_output_bytes += bytes;
            }
        }
        if (fuse_outputs) {
            outputs.front().resize(prepared_mxfp4_->maximum_output);
            total_output_bytes =
                prepared_mxfp4_->maximum_output * sizeof(float);
            if (cudaMemcpyAsync(outputs.front().data(), device_output,
                                total_output_bytes, cudaMemcpyDeviceToHost,
                                stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared fused output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        const auto transfer = async_pipeline_->complete(token);
        if (!transfer) {
            return Result<std::vector<std::vector<float>>>::failure(
                transfer.error(), transfer.message());
        }

        const auto elapsed = [](EventOwner& start, EventOwner& end,
                                std::uint64_t& value) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, start.get(), end.get()) !=
                cudaSuccess) {
                return false;
            }
            value = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
            return true;
        };
        for (std::size_t index = 0; index < experts.size(); ++index) {
            std::array<std::uint64_t, 4> durations{};
            auto& events = members[index].events;
            for (std::size_t pair = 0; pair < durations.size(); ++pair) {
                if (!elapsed(*events[pair * 2], *events[pair * 2 + 1],
                             durations[pair])) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert event timing failed");
                }
            }
            const auto& expert = experts[index];
            const auto logical_bytes = [](const auto& weight) {
                return weight.rows * weight.cols / 2 +
                       weight.rows * weight.cols / 32;
            };
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.gate), 0,
                   durations[0], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.up), 0,
                   durations[1], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.down), 0,
                   durations[3], true);
            record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
                   layer, operation_start, expert.gate.rows * sizeof(float), 0,
                   durations[2], true);
        }
        runtime_stats_.activation_h2d_bytes += input_bytes;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        if (fuse_outputs) {
            ++runtime_stats_.fused_moe_calls;
            runtime_stats_.fused_moe_experts += experts.size();
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, transfer.value().bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        prepared_mxfp4_.reset();
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    BackendMemoryStats memory_stats() const noexcept override { return memory_stats_; }
    std::string_view device_name() const noexcept override { return device_name_; }

private:
    struct PreparedMxfp4Metadata {
        Mxfp4PrefetchToken token;
        std::uint32_t layer{};
        ProfilePhase phase{};
        std::size_t input_cols{};
        std::size_t maximum_intermediate{};
        std::size_t maximum_output{};
        std::size_t expert_count{};
    };

    using DensePlanKey =
        std::tuple<std::size_t, std::size_t, int, int>;

    bool initialize_dense_plan(DensePlan& plan, std::size_t rows,
                               std::size_t cols, cudaDataType_t input_type,
                               cudaDataType_t weight_type) {
        if (cublasLtMatmulDescCreate(plan.operation.out(), CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            !create_row_major_layout(plan.weight_layout, weight_type, rows,
                                     cols, cols) ||
            !create_row_major_layout(plan.input_layout, input_type, cols, 1,
                                     1) ||
            !create_row_major_layout(plan.output_layout, CUDA_R_32F, rows, 1,
                                     1) ||
            cublasLtMatmulPreferenceCreate(plan.preference.out()) !=
                CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        int returned_results = 0;
        return cublasLtMatmulAlgoGetHeuristic(
                   handle_, plan.operation.get(), plan.weight_layout.get(),
                   plan.input_layout.get(), plan.output_layout.get(),
                   plan.output_layout.get(), plan.preference.get(), 1,
                   &plan.heuristic, &returned_results) ==
                   CUBLAS_STATUS_SUCCESS &&
               returned_results == 1 &&
               plan.heuristic.state == CUBLAS_STATUS_SUCCESS;
    }

    static bool valid_dense(std::span<const float> input,
                            DenseWeightView weight) {
        if (input.size() != weight.cols ||
            weight.rows > weight.values.size() ||
            (weight.cols != 0 &&
             weight.rows > weight.values.size() / weight.cols)) {
            return false;
        }
        return weight.values.size() == weight.rows * weight.cols;
    }

    static bool valid_dense_size(std::size_t input_size,
                                 DenseWeightView weight) {
        if (input_size != weight.cols || !weight.rows || !weight.cols ||
            weight.rows > weight.values.size() ||
            weight.rows > weight.values.size() / weight.cols) {
            return false;
        }
        return weight.values.size() == weight.rows * weight.cols;
    }

    static bool valid_dense_mlp(std::span<const float> input,
                                DenseMlpView weights) {
        return valid_dense(input, weights.gate) &&
               valid_dense(input, weights.up) && weights.gate.rows != 0 &&
               weights.gate.rows == weights.up.rows &&
               valid_dense_size(weights.gate.rows, weights.down);
    }

    static bool valid_mxfp4(std::span<const float> input,
                            Mxfp4WeightView weight) {
        return valid_mxfp4_size(input.size(), weight);
    }

    static bool valid_mxfp4_size(std::size_t input_size,
                                 Mxfp4WeightView weight) {
        if (input_size != weight.cols || !weight.rows || !weight.cols ||
            !weight.group_size || weight.cols % weight.group_size ||
            weight.cols % 2 ||
            weight.rows > std::numeric_limits<std::size_t>::max() /
                              weight.cols) {
            return false;
        }
        const auto elements = weight.rows * weight.cols;
        return weight.packed.size() == elements / 2 &&
               weight.scales.size() == elements / weight.group_size &&
               std::none_of(
                   weight.scales.begin(), weight.scales.end(),
                   [](std::byte scale) {
                       return std::to_integer<std::uint8_t>(scale) == 0xFFU;
                   });
    }

    NumericPrecision numeric_precision() const noexcept {
        return options_.dense_precision == DensePrecision::fp32
                   ? NumericPrecision::fp32
                   : NumericPrecision::bf16_rounded;
    }

    static bool create_row_major_layout(MatrixLayoutOwner& owner,
                                        cudaDataType_t type,
                                        std::size_t rows, std::size_t cols,
                                        std::int64_t leading_dimension) {
        if (cublasLtMatrixLayoutCreate(owner.out(), type, rows, cols,
                                       leading_dimension) !=
            CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        constexpr cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        return cublasLtMatrixLayoutSetAttribute(
                   owner.get(), CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                   sizeof(order)) == CUBLAS_STATUS_SUCCESS;
    }

    void record(ProfilePhase phase, ProfileOperation operation,
                NumericPrecision precision, std::uint32_t layer,
                std::chrono::steady_clock::time_point start,
                std::uint64_t logical_bytes, std::uint64_t transfer_bytes,
                std::uint64_t device_nanoseconds, bool success) {
        if (!profiler_) return;
        const auto wall_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        profiler_->record({phase, operation, precision, layer,
                           static_cast<std::uint64_t>(wall_nanoseconds),
                           device_nanoseconds, logical_bytes, transfer_bytes,
                           success});
    }

    BackendOptions options_;
    Profiler* profiler_;
    cudaStream_t stream_{};
    cublasLtHandle_t handle_{};
    std::string device_name_;
    BackendMemoryStats memory_stats_{};
    BackendRuntimeStats runtime_stats_{};
    struct ImmutableWeightIdentity {
        const float* data{};
        std::size_t bytes{};
        std::size_t rows{};
        std::size_t cols{};

        bool operator==(const ImmutableWeightIdentity&) const = default;
    };
    std::unordered_map<std::uint64_t, ImmutableWeightIdentity>
        immutable_weights_;
    cuda::ScratchBuffer dense_input_scratch_;
    cuda::ScratchBuffer dense_weight_scratch_;
    cuda::ScratchBuffer dense_output_scratch_;
    cuda::ScratchBuffer dense_group_output_scratch_;
    cuda::ScratchBuffer ffn_input_scratch_;
    cuda::ScratchBuffer ffn_weight_scratch_;
    cuda::ScratchBuffer ffn_gate_scratch_;
    cuda::ScratchBuffer ffn_up_scratch_;
    cuda::ScratchBuffer ffn_activation_scratch_;
    cuda::ScratchBuffer ffn_output_scratch_;
    cuda::ScratchBuffer mxfp4_input_scratch_;
    cuda::ScratchBuffer mxfp4_packed_scratch_;
    cuda::ScratchBuffer mxfp4_scales_scratch_;
    cuda::ScratchBuffer mxfp4_output_scratch_;
    cuda::ScratchBuffer mxfp4_group_output_scratch_;
    cuda::ScratchBuffer mxfp4_descriptor_scratch_;
    cuda::ScratchBuffer layer_input_scratch_;
    cuda::ScratchBuffer layer_routed_latent_scratch_;
    cuda::ScratchBuffer layer_descriptor_scratch_;
    cuda::ScratchBuffer layer_expert_gate_scratch_;
    cuda::ScratchBuffer layer_expert_up_scratch_;
    cuda::ScratchBuffer layer_expert_activation_scratch_;
    cuda::ScratchBuffer layer_expert_output_scratch_;
    cuda::ScratchBuffer layer_contribution_scratch_;
    cuda::ScratchBuffer layer_mixed_scratch_;
    cuda::ScratchBuffer layer_normalized_scratch_;
    cuda::ScratchBuffer layer_routed_hidden_scratch_;
    cuda::ScratchBuffer layer_shared_gate_scratch_;
    cuda::ScratchBuffer layer_shared_up_scratch_;
    cuda::ScratchBuffer layer_shared_activation_scratch_;
    cuda::ScratchBuffer layer_shared_hidden_scratch_;
    cuda::ScratchBuffer layer_final_hidden_scratch_;
    EventOwner dense_event_start_;
    EventOwner dense_event_end_;
    EventOwner mxfp4_event_start_;
    EventOwner mxfp4_event_end_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_ends_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_ends_;
    std::array<EventOwner, 8> resident_grid_events_;
    std::array<EventOwner, 26> resident_moe_layer_events_;
    std::map<DensePlanKey, std::unique_ptr<DensePlan>> dense_plans_;
    std::unique_ptr<cuda::ResidentWeightTable> resident_weights_;
    std::unique_ptr<cuda::AsyncMxfp4Pipeline> async_pipeline_;
    std::optional<PreparedMxfp4Metadata> prepared_mxfp4_;
};

Result<std::unique_ptr<ComputeBackend>> cuda_failure(
    ErrorCode code, const char* message) {
    return Result<std::unique_ptr<ComputeBackend>>::failure(code, message);
}

}  // namespace

Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions& options, Profiler* profiler) {
    if (options.kind == BackendKind::cpu) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA factory requires a CUDA backend kind");
    }
    if (options.cuda_transfer == CudaTransferMode::prefetch &&
        (options.kind != BackendKind::cuda_custom ||
         options.cuda_boundary != CudaBoundaryMode::ffn_block ||
         options.cuda_allocation != CudaAllocationMode::reused ||
         options.cuda_weights != CudaWeightMode::transient ||
         options.cuda_pinned_bytes == 0)) {
        return cuda_failure(
            ErrorCode::backend_unavailable,
            "CUDA prefetch options do not satisfy the exact capability contract");
    }
    if (options.cuda_transfer == CudaTransferMode::synchronous &&
        options.cuda_pinned_bytes != 0) {
        return cuda_failure(
            ErrorCode::backend_unavailable,
            "synchronous CUDA transfer cannot allocate pinned staging");
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA device selection failed");
    }

    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA capability query failed");
    }
    if (major < 12) {
        return cuda_failure(ErrorCode::unsupported_architecture,
                            "CUDA backend requires compute capability 12.0 or newer");
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA device property query failed");
    }
    int device_overlap = 0;
    if (cudaDeviceGetAttribute(&device_overlap, cudaDevAttrGpuOverlap, device) !=
        cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA overlap capability query failed");
    }

    StreamOwner stream;
    if (cudaStreamCreateWithFlags(stream.out(), cudaStreamNonBlocking) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA stream creation failed");
    }

    CublasLtOwner handle;
    if (cublasLtCreate(handle.out()) != CUBLAS_STATUS_SUCCESS) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "cuBLASLt handle creation failed");
    }

    auto concrete = std::make_unique<CudaBackend>(
        options, profiler, stream.release(), handle.release(), properties.name,
        static_cast<std::uint64_t>(properties.asyncEngineCount),
        device_overlap != 0);
    if (concrete->initialize_async_pipeline() != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA async transfer resource initialization failed");
    }
    std::unique_ptr<ComputeBackend> backend = std::move(concrete);
    return Result<std::unique_ptr<ComputeBackend>>::success(std::move(backend));
}

}  // namespace k3x
