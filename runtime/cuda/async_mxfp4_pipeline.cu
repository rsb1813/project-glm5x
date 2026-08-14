// exact MXFP4 expert payload의 bounded single-flight 비동기 전송을 구현합니다.
#include "async_mxfp4_pipeline.cuh"

#include "device_memory.cuh"
#include "pinned_memory.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace k3x::cuda {
namespace {

std::atomic<std::uint64_t> next_token_id{1};

std::uint64_t nanoseconds(float milliseconds) {
    return static_cast<std::uint64_t>(std::llround(
        static_cast<double>(milliseconds) * 1.0e6));
}

bool checked_add(std::size_t value, std::size_t& total) {
    if (value > std::numeric_limits<std::size_t>::max() - total) return false;
    total += value;
    return true;
}

bool valid_weight(const Mxfp4WeightView& weight) {
    if (!weight.rows || !weight.cols || weight.group_size != 32 ||
        weight.cols % 32 != 0 || weight.cols % 2 != 0 ||
        weight.rows > std::numeric_limits<std::size_t>::max() / weight.cols) {
        return false;
    }
    const auto elements = weight.rows * weight.cols;
    return weight.packed.size() == elements / 2 &&
           weight.scales.size() == elements / 32 &&
           std::none_of(weight.scales.begin(), weight.scales.end(),
                        [](std::byte scale) {
                            return std::to_integer<std::uint8_t>(scale) == 0xffU;
                        });
}

bool valid_expert(const Mxfp4MlpView& expert) {
    return valid_weight(expert.gate) && valid_weight(expert.up) &&
           valid_weight(expert.down) &&
           expert.gate.rows == expert.up.rows &&
           expert.gate.cols == expert.up.cols &&
           expert.down.cols == expert.gate.rows;
}

cudaError_t create_timing_event(cudaEvent_t* event) {
    return cudaEventCreate(event);
}

}  // namespace

struct AsyncMxfp4Pipeline::Impl {
    struct Pending {
        Mxfp4PrefetchToken token;
        std::uint32_t layer{};
        ProfilePhase phase{};
        std::uint64_t bytes{};
        std::uint64_t staging_nanoseconds{};
        bool consumed{};
        bool ready_before_use{};
    };

    Impl(BackendMemoryStats* memory_value, BackendRuntimeStats* runtime_value)
        : memory(memory_value), runtime(runtime_value), pinned(runtime_value),
          device(memory_value, runtime_value) {}

    ~Impl() {
        if (transfer_stream) cudaStreamSynchronize(transfer_stream);
        if (wait_end) cudaEventDestroy(wait_end);
        if (wait_start) cudaEventDestroy(wait_start);
        if (ready) cudaEventDestroy(ready);
        if (transfer_end) cudaEventDestroy(transfer_end);
        if (transfer_start) cudaEventDestroy(transfer_start);
        if (transfer_stream) cudaStreamDestroy(transfer_stream);
    }

    BackendMemoryStats* memory{};
    BackendRuntimeStats* runtime{};
    PinnedBuffer pinned;
    ScratchBuffer device;
    cudaStream_t transfer_stream{};
    cudaEvent_t transfer_start{};
    cudaEvent_t transfer_end{};
    cudaEvent_t ready{};
    cudaEvent_t wait_start{};
    cudaEvent_t wait_end{};
    std::vector<DeviceMxfp4MlpView> views;
    std::optional<Pending> pending;
    bool initialized{};
};

AsyncMxfp4Pipeline::AsyncMxfp4Pipeline(
    BackendMemoryStats* memory, BackendRuntimeStats* runtime)
    : impl_(std::make_unique<Impl>(memory, runtime)) {}

AsyncMxfp4Pipeline::~AsyncMxfp4Pipeline() = default;

cudaError_t AsyncMxfp4Pipeline::initialize(
    std::size_t pinned_capacity) noexcept {
    if (!impl_->memory || !impl_->runtime || impl_->initialized ||
        pinned_capacity == 0) {
        return cudaErrorInvalidValue;
    }
    auto status = cudaStreamCreateWithFlags(
        &impl_->transfer_stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) return status;
    if ((status = create_timing_event(&impl_->transfer_start)) != cudaSuccess ||
        (status = create_timing_event(&impl_->transfer_end)) != cudaSuccess ||
        (status = cudaEventCreateWithFlags(
             &impl_->ready, cudaEventDisableTiming)) != cudaSuccess ||
        (status = create_timing_event(&impl_->wait_start)) != cudaSuccess ||
        (status = create_timing_event(&impl_->wait_end)) != cudaSuccess ||
        (status = impl_->pinned.allocate(pinned_capacity)) != cudaSuccess ||
        (status = impl_->device.reserve(pinned_capacity)) != cudaSuccess) {
        return status;
    }
    impl_->initialized = true;
    return cudaSuccess;
}

Result<Mxfp4PrefetchToken> AsyncMxfp4Pipeline::prepare(
    std::span<const Mxfp4MlpView> experts, std::uint64_t use_sequence,
    std::uint32_t layer, ProfilePhase phase) {
    if (!impl_->initialized) {
        return Result<Mxfp4PrefetchToken>::failure(
            ErrorCode::backend_unavailable, "CUDA async pipeline is not initialized");
    }
    if (impl_->pending || use_sequence == 0) {
        return Result<Mxfp4PrefetchToken>::failure(ErrorCode::invalid_state);
    }
    if (experts.empty()) {
        return Result<Mxfp4PrefetchToken>::failure(ErrorCode::invalid_extent);
    }

    std::size_t total = 0;
    for (const auto& expert : experts) {
        if (!valid_expert(expert)) {
            return Result<Mxfp4PrefetchToken>::failure(ErrorCode::invalid_mxfp4);
        }
        const std::array<std::size_t, 6> sizes{
            expert.gate.packed.size_bytes(), expert.gate.scales.size_bytes(),
            expert.up.packed.size_bytes(), expert.up.scales.size_bytes(),
            expert.down.packed.size_bytes(), expert.down.scales.size_bytes()};
        for (const auto size : sizes) {
            if (!checked_add(size, total)) {
                return Result<Mxfp4PrefetchToken>::failure(
                    ErrorCode::invalid_extent);
            }
        }
    }
    if (total > impl_->pinned.size() || total > impl_->device.capacity()) {
        return Result<Mxfp4PrefetchToken>::failure(ErrorCode::invalid_extent);
    }
    const auto token_value = next_token_id.fetch_add(1, std::memory_order_relaxed);
    if (token_value == 0) {
        return Result<Mxfp4PrefetchToken>::failure(ErrorCode::invalid_state);
    }

    impl_->views.clear();
    impl_->views.reserve(experts.size());
    auto* host = static_cast<std::byte*>(impl_->pinned.get());
    auto* device = static_cast<std::uint8_t*>(impl_->device.get());
    std::size_t offset = 0;
    const auto staging_start = std::chrono::steady_clock::now();
    const auto append = [&](const Mxfp4WeightView& weight) {
        DeviceMxfp4WeightView result{
            weight.tensor_id, device + offset, nullptr,
            weight.rows, weight.cols, weight.group_size};
        std::memcpy(host + offset, weight.packed.data(), weight.packed.size_bytes());
        offset += weight.packed.size_bytes();
        result.scales = device + offset;
        std::memcpy(host + offset, weight.scales.data(), weight.scales.size_bytes());
        offset += weight.scales.size_bytes();
        return result;
    };
    for (const auto& expert : experts) {
        impl_->views.push_back(
            {append(expert.gate), append(expert.up), append(expert.down)});
    }
    const auto measured_staging =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - staging_start)
            .count();
    const auto staging_nanoseconds = static_cast<std::uint64_t>(
        std::max<std::int64_t>(1, measured_staging));

    if (cudaEventRecord(impl_->transfer_start, impl_->transfer_stream) !=
            cudaSuccess ||
        cudaMemcpyAsync(impl_->device.get(), impl_->pinned.get(), total,
                        cudaMemcpyHostToDevice, impl_->transfer_stream) !=
            cudaSuccess ||
        cudaEventRecord(impl_->transfer_end, impl_->transfer_stream) !=
            cudaSuccess ||
        cudaEventRecord(impl_->ready, impl_->transfer_stream) != cudaSuccess) {
        cudaStreamSynchronize(impl_->transfer_stream);
        impl_->views.clear();
        return Result<Mxfp4PrefetchToken>::failure(
            ErrorCode::backend_unavailable, "CUDA async expert upload failed");
    }

    const Mxfp4PrefetchToken token{token_value, use_sequence};
    impl_->pending = Impl::Pending{
        token, layer, phase, static_cast<std::uint64_t>(total),
        staging_nanoseconds, false, false};
    ++impl_->runtime->async_prefetch_calls;
    impl_->runtime->async_prefetch_bytes += total;
    impl_->runtime->weight_h2d_bytes += total;
    impl_->runtime->pinned_staging_nanoseconds += staging_nanoseconds;
    return Result<Mxfp4PrefetchToken>::success(token);
}

Result<std::span<const DeviceMxfp4MlpView>> AsyncMxfp4Pipeline::consume(
    Mxfp4PrefetchToken token, std::uint32_t layer, ProfilePhase phase,
    cudaStream_t compute_stream) {
    if (!impl_->pending || impl_->pending->consumed || token.value == 0 ||
        token.use_sequence == 0 ||
        token.value != impl_->pending->token.value ||
        token.use_sequence != impl_->pending->token.use_sequence ||
        layer != impl_->pending->layer || phase != impl_->pending->phase ||
        !compute_stream) {
        return Result<std::span<const DeviceMxfp4MlpView>>::failure(
            ErrorCode::invalid_state);
    }
    const auto query = cudaEventQuery(impl_->ready);
    if (query != cudaSuccess && query != cudaErrorNotReady) {
        return Result<std::span<const DeviceMxfp4MlpView>>::failure(
            ErrorCode::backend_unavailable, "CUDA readiness query failed");
    }
    if (cudaEventRecord(impl_->wait_start, compute_stream) != cudaSuccess ||
        cudaStreamWaitEvent(compute_stream, impl_->ready, 0) != cudaSuccess ||
        cudaEventRecord(impl_->wait_end, compute_stream) != cudaSuccess) {
        return Result<std::span<const DeviceMxfp4MlpView>>::failure(
            ErrorCode::backend_unavailable, "CUDA transfer dependency failed");
    }
    impl_->pending->ready_before_use = query == cudaSuccess;
    impl_->pending->consumed = true;
    return Result<std::span<const DeviceMxfp4MlpView>>::success(impl_->views);
}

Result<AsyncTransferMetrics> AsyncMxfp4Pipeline::complete(
    Mxfp4PrefetchToken token) {
    if (!impl_->pending || !impl_->pending->consumed || token.value == 0 ||
        token.use_sequence == 0 ||
        token.value != impl_->pending->token.value ||
        token.use_sequence != impl_->pending->token.use_sequence) {
        return Result<AsyncTransferMetrics>::failure(ErrorCode::invalid_state);
    }
    float transfer_ms = 0.0F;
    float stall_ms = 0.0F;
    if (cudaEventElapsedTime(&transfer_ms, impl_->transfer_start,
                             impl_->transfer_end) != cudaSuccess ||
        cudaEventElapsedTime(&stall_ms, impl_->wait_start,
                             impl_->wait_end) != cudaSuccess) {
        return Result<AsyncTransferMetrics>::failure(
            ErrorCode::backend_unavailable, "CUDA async event timing failed");
    }
    const AsyncTransferMetrics metrics{
        impl_->pending->bytes, impl_->pending->staging_nanoseconds,
        nanoseconds(transfer_ms), nanoseconds(stall_ms),
        impl_->pending->ready_before_use};
    ++impl_->runtime->transfer_stream_wait_count;
    if (metrics.ready_before_use) {
        ++impl_->runtime->async_prefetch_ready_before_use;
    } else {
        ++impl_->runtime->async_prefetch_late_at_use;
    }
    impl_->runtime->transfer_device_nanoseconds += metrics.transfer_nanoseconds;
    impl_->runtime->transfer_stall_nanoseconds += metrics.stall_nanoseconds;
    impl_->pending.reset();
    return Result<AsyncTransferMetrics>::success(metrics);
}

}  // namespace k3x::cuda
