// exact MXFP4 expert payload의 단일 요청 비동기 전송 계약을 정의합니다.
#pragma once

#include "k3x/backend.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace k3x::cuda {

struct DeviceMxfp4WeightView {
    std::uint64_t tensor_id{};
    const std::uint8_t* packed{};
    const std::uint8_t* scales{};
    std::size_t rows{};
    std::size_t cols{};
    std::size_t group_size{};
};

struct DeviceMxfp4MlpView {
    DeviceMxfp4WeightView gate;
    DeviceMxfp4WeightView up;
    DeviceMxfp4WeightView down;
};

struct AsyncTransferMetrics {
    std::uint64_t bytes{};
    std::uint64_t staging_nanoseconds{};
    std::uint64_t transfer_nanoseconds{};
    std::uint64_t stall_nanoseconds{};
    bool ready_before_use{};
};

class AsyncMxfp4Pipeline {
public:
    AsyncMxfp4Pipeline(BackendMemoryStats* memory,
                       BackendRuntimeStats* runtime);
    ~AsyncMxfp4Pipeline();
    AsyncMxfp4Pipeline(const AsyncMxfp4Pipeline&) = delete;
    AsyncMxfp4Pipeline& operator=(const AsyncMxfp4Pipeline&) = delete;

    cudaError_t initialize(std::size_t pinned_capacity) noexcept;
    Result<Mxfp4PrefetchToken> prepare(
        std::span<const Mxfp4MlpView> experts, std::uint64_t use_sequence,
        std::uint32_t layer, ProfilePhase phase);
    Result<std::span<const DeviceMxfp4MlpView>> consume(
        Mxfp4PrefetchToken token, std::uint32_t layer, ProfilePhase phase,
        cudaStream_t compute_stream);
    Result<AsyncTransferMetrics> complete(Mxfp4PrefetchToken token);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace k3x::cuda
