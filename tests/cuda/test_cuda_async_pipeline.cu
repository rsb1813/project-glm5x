// exact MXFP4 단일 요청 비동기 전송 pipeline의 검증과 수명 계약을 검증합니다.
#include "async_mxfp4_pipeline.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
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

k3x::Mxfp4MlpView view(ExpertStorage& storage, std::uint64_t base) {
    return {
        {base, storage.gate, storage.gate_scales, 32, 32, 32},
        {base + 1, storage.up, storage.up_scales, 32, 32, 32},
        {base + 2, storage.down, storage.down_scales, 1, 32, 32},
    };
}

bool unchanged_success_counters(const k3x::BackendRuntimeStats& before,
                                const k3x::BackendRuntimeStats& after) {
    return before.async_prefetch_calls == after.async_prefetch_calls &&
           before.async_prefetch_bytes == after.async_prefetch_bytes &&
           before.weight_h2d_bytes == after.weight_h2d_bytes &&
           before.pinned_staging_nanoseconds ==
               after.pinned_staging_nanoseconds;
}

}  // namespace

int main() {
    ExpertStorage first;
    ExpertStorage second;
    first.gate[0] = std::byte{0x11};
    first.up[0] = std::byte{0x12};
    first.down[0] = std::byte{0x13};
    second.gate[0] = std::byte{0x21};
    second.up[0] = std::byte{0x22};
    second.down[0] = std::byte{0x23};
    first.gate_scales.fill(std::byte{127});
    first.up_scales.fill(std::byte{127});
    first.down_scales.fill(std::byte{127});
    second.gate_scales.fill(std::byte{127});
    second.up_scales.fill(std::byte{127});
    second.down_scales.fill(std::byte{127});
    const std::array<k3x::Mxfp4MlpView, 2> experts{
        view(first, 100), view(second, 200)};
    constexpr std::uint64_t expected_bytes = 2210;

    k3x::BackendMemoryStats memory;
    k3x::BackendRuntimeStats runtime;
    k3x::cuda::AsyncMxfp4Pipeline pipeline(&memory, &runtime);
    if (pipeline.initialize(4096) != cudaSuccess) return 1;
    if (runtime.pinned_host_bytes != 4096 || memory.current_device_bytes != 4096)
        return 2;

    const auto prepared = pipeline.prepare(
        experts, 1, 7, k3x::ProfilePhase::decode);
    if (!prepared || prepared.value().value != 1) return 3;
    if (runtime.stream_synchronization_count != 0 ||
        runtime.async_prefetch_calls != 1 ||
        runtime.async_prefetch_bytes != expected_bytes ||
        runtime.weight_h2d_bytes != expected_bytes ||
        runtime.pinned_staging_nanoseconds == 0) {
        return 4;
    }

    const auto after_prepare = runtime;
    const auto duplicate = pipeline.prepare(
        experts, 2, 7, k3x::ProfilePhase::decode);
    if (duplicate || duplicate.error() != k3x::ErrorCode::invalid_state ||
        !unchanged_success_counters(after_prepare, runtime)) {
        return 5;
    }

    cudaStream_t compute{};
    if (cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking) != cudaSuccess)
        return 6;
    k3x::BackendMemoryStats foreign_memory;
    k3x::BackendRuntimeStats foreign_runtime;
    k3x::cuda::AsyncMxfp4Pipeline foreign(&foreign_memory, &foreign_runtime);
    if (foreign.initialize(4096) != cudaSuccess) {
        cudaStreamDestroy(compute);
        return 7;
    }
    const auto foreign_prepared = foreign.prepare(
        experts, 1, 7, k3x::ProfilePhase::decode);
    if (!foreign_prepared ||
        foreign_prepared.value().value == prepared.value().value) {
        cudaStreamDestroy(compute);
        return 8;
    }
    const auto foreign_on_main = pipeline.consume(
        foreign_prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    const auto foreign_consumed = foreign.consume(
        foreign_prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    if (foreign_on_main ||
        foreign_on_main.error() != k3x::ErrorCode::invalid_state ||
        !foreign_consumed || cudaStreamSynchronize(compute) != cudaSuccess ||
        !foreign.complete(foreign_prepared.value())) {
        cudaStreamDestroy(compute);
        return 9;
    }
    const auto invalid_zero = pipeline.consume(
        {0}, 7, k3x::ProfilePhase::decode, compute);
    const auto invalid_stale = pipeline.consume(
        {2}, 7, k3x::ProfilePhase::decode, compute);
    const auto invalid_layer = pipeline.consume(
        {1}, 8, k3x::ProfilePhase::decode, compute);
    const auto invalid_phase = pipeline.consume(
        {1}, 7, k3x::ProfilePhase::prefill, compute);
    const auto invalid_sequence = pipeline.consume(
        {prepared.value().value, prepared.value().use_sequence + 1}, 7,
        k3x::ProfilePhase::decode, compute);
    if (invalid_zero || invalid_stale || invalid_layer || invalid_phase ||
        invalid_sequence ||
        invalid_zero.error() != k3x::ErrorCode::invalid_state ||
        invalid_stale.error() != k3x::ErrorCode::invalid_state ||
        invalid_layer.error() != k3x::ErrorCode::invalid_state ||
        invalid_phase.error() != k3x::ErrorCode::invalid_state ||
        invalid_sequence.error() != k3x::ErrorCode::invalid_state ||
        !unchanged_success_counters(after_prepare, runtime)) {
        cudaStreamDestroy(compute);
        return 10;
    }

    const auto consumed = pipeline.consume(
        prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    if (!consumed || consumed.value().size() != 2) {
        cudaStreamDestroy(compute);
        return 11;
    }
    const auto views = consumed.value();
    if (views[0].gate.tensor_id != 100 || views[0].up.tensor_id != 101 ||
        views[0].down.tensor_id != 102 || views[1].gate.tensor_id != 200 ||
        views[1].up.tensor_id != 201 || views[1].down.tensor_id != 202) {
        cudaStreamDestroy(compute);
        return 12;
    }
    std::array<std::uint8_t, 6> first_bytes{};
    const std::array<const std::uint8_t*, 6> pointers{
        views[0].gate.packed, views[0].up.packed, views[0].down.packed,
        views[1].gate.packed, views[1].up.packed, views[1].down.packed};
    for (std::size_t index = 0; index < pointers.size(); ++index) {
        if (cudaMemcpyAsync(&first_bytes[index], pointers[index], 1,
                            cudaMemcpyDeviceToHost, compute) != cudaSuccess) {
            cudaStreamDestroy(compute);
            return 13;
        }
    }
    if (cudaStreamSynchronize(compute) != cudaSuccess) {
        cudaStreamDestroy(compute);
        return 14;
    }
    const auto completed = pipeline.complete(prepared.value());
    if (!completed || completed.value().bytes != expected_bytes ||
        runtime.transfer_stream_wait_count != 1 ||
        runtime.async_prefetch_ready_before_use +
                runtime.async_prefetch_late_at_use != 1 ||
        runtime.transfer_device_nanoseconds == 0 ||
        first_bytes != std::array<std::uint8_t, 6>{
                           0x11, 0x12, 0x13, 0x21, 0x22, 0x23}) {
        cudaStreamDestroy(compute);
        return 15;
    }
    const auto repeated = pipeline.consume(
        prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    if (repeated || repeated.error() != k3x::ErrorCode::invalid_state) {
        cudaStreamDestroy(compute);
        return 16;
    }
    const auto second_prepared = pipeline.prepare(
        experts, 99, 7, k3x::ProfilePhase::decode);
    if (!second_prepared ||
        second_prepared.value().value <= prepared.value().value ||
        second_prepared.value().value == foreign_prepared.value().value) {
        cudaStreamDestroy(compute);
        return 17;
    }
    const auto old_token = pipeline.consume(
        prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    const auto second_consumed = pipeline.consume(
        second_prepared.value(), 7, k3x::ProfilePhase::decode, compute);
    if (old_token || old_token.error() != k3x::ErrorCode::invalid_state ||
        !second_consumed || cudaStreamSynchronize(compute) != cudaSuccess ||
        !pipeline.complete(second_prepared.value())) {
        cudaStreamDestroy(compute);
        return 18;
    }
    if (cudaStreamDestroy(compute) != cudaSuccess) return 19;

    k3x::BackendMemoryStats small_memory;
    k3x::BackendRuntimeStats small_runtime;
    k3x::cuda::AsyncMxfp4Pipeline small(&small_memory, &small_runtime);
    if (small.initialize(expected_bytes - 1) != cudaSuccess) return 20;
    const auto too_large = small.prepare(
        experts, 1, 0, k3x::ProfilePhase::decode);
    if (too_large || too_large.error() != k3x::ErrorCode::invalid_extent ||
        small_runtime.async_prefetch_calls != 0 ||
        small_runtime.weight_h2d_bytes != 0) {
        return 21;
    }

    auto invalid_group = experts;
    invalid_group[0].gate.group_size = 16;
    const auto before_invalid = runtime;
    const auto bad_group = pipeline.prepare(
        invalid_group, 2, 0, k3x::ProfilePhase::decode);
    invalid_group = experts;
    first.up_scales[0] = std::byte{0xff};
    const auto bad_scale = pipeline.prepare(
        invalid_group, 2, 0, k3x::ProfilePhase::decode);
    first.up_scales[0] = std::byte{127};
    invalid_group = experts;
    invalid_group[1].down.packed = {};
    const auto bad_extent = pipeline.prepare(
        invalid_group, 2, 0, k3x::ProfilePhase::decode);
    const auto empty = pipeline.prepare(
        {}, 2, 0, k3x::ProfilePhase::decode);
    if (bad_group || bad_scale || bad_extent || empty ||
        bad_group.error() != k3x::ErrorCode::invalid_mxfp4 ||
        bad_scale.error() != k3x::ErrorCode::invalid_mxfp4 ||
        bad_extent.error() != k3x::ErrorCode::invalid_mxfp4 ||
        empty.error() != k3x::ErrorCode::invalid_extent ||
        !unchanged_success_counters(before_invalid, runtime)) {
        return 22;
    }
    return 0;
}
