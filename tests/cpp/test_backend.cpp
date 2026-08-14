// exact CPU backend의 dense와 native MXFP4 행렬 연산 계약을 검증합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

int main() {
    k3x::Profiler profiler;
    auto backend = k3x::make_cpu_backend(&profiler);

    if (backend->kind() != k3x::BackendKind::cpu) return 1;
    if (backend->device_name() != "CPU") return 2;
    if (backend->memory_stats().current_device_bytes != 0) return 3;
    if (backend->memory_stats().peak_device_bytes != 0) return 4;

    const k3x::BackendOptions defaults;
    if (defaults.cuda_allocation != k3x::CudaAllocationMode::per_operation) return 21;
    if (defaults.cuda_weights != k3x::CudaWeightMode::transient) return 22;
    if (defaults.cuda_batching != k3x::CudaBatchingMode::scalar) return 23;
    if (defaults.cuda_resident_bytes != 0) return 24;
    if (defaults.cuda_boundary != k3x::CudaBoundaryMode::operation) return 58;
    if (defaults.cuda_transfer != k3x::CudaTransferMode::synchronous) return 67;
    if (defaults.cuda_pinned_bytes != 0) return 68;
    if (k3x::error_code_name(k3x::ErrorCode::invalid_state) !=
        std::string_view{"INVALID_STATE"}) {
        return 69;
    }

    const auto& options = backend->options();
    if (options.kind != k3x::BackendKind::cpu) return 25;
    if (options.dense_precision != k3x::DensePrecision::fp32) return 26;
    if (options.cuda_allocation != k3x::CudaAllocationMode::per_operation) return 27;
    if (options.cuda_weights != k3x::CudaWeightMode::transient) return 28;
    if (options.cuda_batching != k3x::CudaBatchingMode::scalar) return 29;
    if (options.cuda_resident_bytes != 0) return 30;

    const auto runtime_stats = backend->runtime_stats();
    if (runtime_stats.device_allocation_count != 0) return 31;
    if (runtime_stats.device_free_count != 0) return 32;
    if (runtime_stats.stream_synchronization_count != 0) return 33;
    if (runtime_stats.weight_cache_hits != 0) return 34;
    if (runtime_stats.weight_cache_misses != 0) return 35;
    if (runtime_stats.weight_cache_bypasses != 0) return 36;
    if (runtime_stats.resident_weight_bytes != 0) return 37;
    if (runtime_stats.peak_resident_weight_bytes != 0) return 38;
    if (runtime_stats.scratch_bytes != 0) return 39;
    if (runtime_stats.peak_scratch_bytes != 0) return 40;
    if (runtime_stats.weight_h2d_bytes != 0) return 41;
    if (runtime_stats.activation_h2d_bytes != 0) return 42;
    if (runtime_stats.grouped_projection_calls != 0) return 43;
    if (runtime_stats.grouped_projection_members != 0) return 44;
    if (runtime_stats.ffn_block_calls != 0) return 59;
    if (runtime_stats.ffn_block_experts != 0) return 60;
    if (runtime_stats.batched_expert_ffn_calls != 0) return 76;
    if (runtime_stats.batched_expert_ffn_tokens != 0) return 77;
    if (runtime_stats.pinned_host_bytes != 0 ||
        runtime_stats.peak_pinned_host_bytes != 0 ||
        runtime_stats.async_prefetch_calls != 0 ||
        runtime_stats.async_prefetch_bytes != 0 ||
        runtime_stats.async_prefetch_ready_before_use != 0 ||
        runtime_stats.async_prefetch_late_at_use != 0 ||
        runtime_stats.transfer_stream_wait_count != 0 ||
        runtime_stats.pinned_staging_nanoseconds != 0 ||
        runtime_stats.transfer_device_nanoseconds != 0 ||
        runtime_stats.transfer_stall_nanoseconds != 0 ||
        runtime_stats.async_engine_count != 0 || runtime_stats.device_overlap) {
        return 70;
    }

    const auto prefetch = backend->prefetch_mxfp4_situ_mlp_group(
        {}, 1, 0, k3x::ProfilePhase::decode);
    if (prefetch || prefetch.error() != k3x::ErrorCode::backend_unavailable) {
        return 71;
    }
    const auto prepared = backend->mxfp4_situ_mlp_group_prepared(
        {}, k3x::Mxfp4PrefetchToken{1}, 2.0F, std::nullopt, 0,
        k3x::ProfilePhase::decode);
    if (prepared || prepared.error() != k3x::ErrorCode::backend_unavailable) {
        return 72;
    }

    const std::array<float, 3> dense_input{2.0F, -1.0F, 0.5F};
    const std::array<float, 6> dense_weight{1.0F, 2.0F, 3.0F,
                                             -2.0F, 0.25F, 4.0F};
    const auto dense = backend->dense_matvec(
        dense_input, k3x::DenseWeightView{100, dense_weight, 2, 3}, 7,
        k3x::ProfilePhase::prefill);
    if (!dense) return 5;
    if (dense.value()[0] != 1.5F) return 6;
    if (dense.value()[1] != -2.25F) return 7;

    std::array<float, 32> mxfp4_input{};
    mxfp4_input[1] = 2.0F;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    const std::array<std::byte, 1> scales{std::byte{127}};
    const auto mxfp4 = backend->mxfp4_matvec(
        mxfp4_input, k3x::Mxfp4WeightView{200, packed, scales, 1, 32, 32}, 8,
        k3x::ProfilePhase::decode);
    if (!mxfp4) return 8;
    if (mxfp4.value()[0] != 1.0F) return 9;

    const auto invalid = backend->dense_matvec(
        dense_input, k3x::DenseWeightView{300, dense_weight, 2, 4}, 9,
        k3x::ProfilePhase::decode);
    if (invalid) return 10;
    if (invalid.error() != k3x::ErrorCode::invalid_extent) return 11;

    const auto& events = profiler.events();
    if (events.size() != 3) return 12;
    if (events[0].operation != k3x::ProfileOperation::dense_matvec) return 13;
    if (events[0].precision != k3x::NumericPrecision::fp32) return 14;
    if (events[0].layer != 7) return 15;
    if (events[1].operation != k3x::ProfileOperation::mxfp4_matvec) return 16;
    if (events[1].precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 17;
    if (events[1].layer != 8) return 18;
    if (events[2].success) return 19;
    if (events[2].layer != 9) return 20;

    const std::array<float, 2> group_input{2.0F, -1.0F};
    const std::array<float, 4> first_dense{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 2> second_dense{3.0F, -2.0F};
    const std::array<k3x::DenseWeightView, 2> dense_group{{
        {101, first_dense, 2, 2},
        {102, second_dense, 1, 2},
    }};
    const auto dense_outputs = backend->dense_matvec_group(
        group_input, dense_group, 10, k3x::ProfilePhase::decode);
    if (!dense_outputs || dense_outputs.value().size() != 2) return 50;
    if (dense_outputs.value()[0] != std::vector<float>{2.0F, -1.0F}) return 51;
    if (dense_outputs.value()[1] != std::vector<float>{8.0F}) return 52;

    std::array<std::byte, 32> second_packed{};
    second_packed[0] = std::byte{0x10};
    second_packed[16] = std::byte{0x10};
    const std::array<std::byte, 2> second_scales{
        std::byte{127}, std::byte{127}};
    const std::array<k3x::Mxfp4WeightView, 2> mxfp4_group{{
        {201, packed, scales, 1, 32, 32},
        {202, second_packed, second_scales, 2, 32, 32},
    }};
    const auto mxfp4_outputs = backend->mxfp4_matvec_group(
        mxfp4_input, mxfp4_group, 11, k3x::ProfilePhase::decode);
    if (!mxfp4_outputs || mxfp4_outputs.value().size() != 2) return 53;
    if (mxfp4_outputs.value()[0] != std::vector<float>{1.0F}) return 54;
    if (mxfp4_outputs.value()[1] != std::vector<float>{1.0F, 1.0F}) return 55;

    const std::array<k3x::Mxfp4WeightView, 2> invalid_group{{
        {203, packed, scales, 1, 32, 32},
        {204, packed, {}, 1, 32, 32},
    }};
    const auto event_count = profiler.events().size();
    const auto rejected = backend->mxfp4_matvec_group(
        mxfp4_input, invalid_group, 12, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4) return 56;
    if (profiler.events().size() != event_count) return 57;

    const std::array<float, 6> dense_gate{
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    const std::array<float, 6> dense_up{
        0.0F, 0.0F, 2.0F, 1.0F, 0.0F, 0.0F};
    const std::array<float, 4> dense_down{1.0F, 2.0F, -1.0F, 0.5F};
    const k3x::DenseMlpView dense_mlp{
        {301, dense_gate, 2, 3},
        {302, dense_up, 2, 3},
        {303, dense_down, 2, 2},
    };
    const auto dense_block = backend->dense_situ_mlp(
        dense_input, dense_mlp, 2.0F, 1.5F, 13,
        k3x::ProfilePhase::decode);
    if (!dense_block || dense_block.value().size() != 2) return 61;
    std::array<float, 2> dense_activation{};
    const std::array<float, 2> dense_gate_output{2.0F, -1.0F};
    const std::array<float, 2> dense_up_output{1.0F, 2.0F};
    k3x::situ_glu(dense_activation, dense_gate_output, dense_up_output, 2.0F,
                  1.5F);
    const std::array<float, 2> dense_expected{
        dense_activation[0] + 2.0F * dense_activation[1],
        -dense_activation[0] + 0.5F * dense_activation[1],
    };
    if (std::abs(dense_block.value()[0] - dense_expected[0]) > 1.0e-6F ||
        std::abs(dense_block.value()[1] - dense_expected[1]) > 1.0e-6F) {
        return 62;
    }

    std::array<std::byte, 512> expert_gate{};
    std::array<std::byte, 512> expert_up{};
    std::array<std::byte, 16> expert_down_one{};
    std::array<std::byte, 16> expert_down_two{};
    expert_gate[0] = std::byte{0x10};
    expert_up[0] = std::byte{0x20};
    expert_down_one[0] = std::byte{0x02};
    expert_down_two[0] = std::byte{0x04};
    std::array<std::byte, 32> expert_intermediate_scales{};
    expert_intermediate_scales.fill(std::byte{127});
    const std::array<std::byte, 1> expert_down_scales{std::byte{127}};
    const std::array<k3x::Mxfp4MlpView, 2> expert_mlps{{
        {
            {401, expert_gate, expert_intermediate_scales, 32, 32, 32},
            {402, expert_up, expert_intermediate_scales, 32, 32, 32},
            {403, expert_down_one, expert_down_scales, 1, 32, 32},
        },
        {
            {404, expert_gate, expert_intermediate_scales, 32, 32, 32},
            {405, expert_up, expert_intermediate_scales, 32, 32, 32},
            {406, expert_down_two, expert_down_scales, 1, 32, 32},
        },
    }};
    const auto expert_blocks = backend->mxfp4_situ_mlp_group(
        mxfp4_input, expert_mlps, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!expert_blocks || expert_blocks.value().size() != 2) return 63;
    std::array<float, 1> expert_activation{};
    const std::array<float, 1> expert_gate_output{1.0F};
    const std::array<float, 1> expert_up_output{2.0F};
    k3x::situ_glu(expert_activation, expert_gate_output, expert_up_output,
                  2.0F, 1.5F);
    if (std::abs(expert_blocks.value()[0][0] - expert_activation[0]) >
            1.0e-6F ||
        std::abs(expert_blocks.value()[1][0] - 2.0F * expert_activation[0]) >
            1.0e-6F) {
        return 64;
    }
    auto second_mxfp4_input = mxfp4_input;
    second_mxfp4_input[1] = -4.0F;
    std::array<float, 64> flat_expert_inputs{};
    std::copy(mxfp4_input.begin(), mxfp4_input.end(),
              flat_expert_inputs.begin());
    std::copy(second_mxfp4_input.begin(), second_mxfp4_input.end(),
              flat_expert_inputs.begin() + 32);
    const std::array<k3x::Mxfp4MlpView, 1> one_expert{expert_mlps[0]};
    const auto scalar_first = backend->mxfp4_situ_mlp_group(
        mxfp4_input, one_expert, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto scalar_second = backend->mxfp4_situ_mlp_group(
        second_mxfp4_input, one_expert, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!scalar_first || !scalar_second) return 73;
    const auto batched = backend->mxfp4_situ_mlp_batch(
        flat_expert_inputs, 2, expert_mlps[0], 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!batched || batched.value().size() != 2 ||
        batched.value()[0].size() != 1 ||
        batched.value()[1].size() != 1 ||
        std::abs(batched.value()[0][0] - scalar_first.value()[0][0]) >
            1.0e-6F ||
        std::abs(batched.value()[1][0] - scalar_second.value()[0][0]) >
            1.0e-6F) {
        return 74;
    }
    const auto batch_event_count = profiler.events().size();
    const auto rejected_empty_batch = backend->mxfp4_situ_mlp_batch(
        {}, 0, expert_mlps[0], 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto rejected_short_batch = backend->mxfp4_situ_mlp_batch(
        std::span<const float>(flat_expert_inputs).first(63), 2,
        expert_mlps[0], 2.0F, 1.5F, 14, k3x::ProfilePhase::decode);
    const auto rejected_overflow_batch = backend->mxfp4_situ_mlp_batch(
        {}, std::numeric_limits<std::size_t>::max(), expert_mlps[0], 2.0F,
        1.5F, 14, k3x::ProfilePhase::decode);
    const auto rejected_situ_batch = backend->mxfp4_situ_mlp_batch(
        flat_expert_inputs, 2, expert_mlps[0], 0.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    auto invalid_batch_expert = expert_mlps[0];
    invalid_batch_expert.down.scales = {};
    const auto rejected_invalid_batch = backend->mxfp4_situ_mlp_batch(
        flat_expert_inputs, 2, invalid_batch_expert, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (rejected_empty_batch || rejected_short_batch ||
        rejected_overflow_batch || rejected_situ_batch ||
        rejected_invalid_batch ||
        rejected_empty_batch.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_short_batch.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_overflow_batch.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_situ_batch.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_invalid_batch.error() != k3x::ErrorCode::invalid_mxfp4 ||
        profiler.events().size() != batch_event_count) {
        return 75;
    }

    const auto grid = backend->mxfp4_situ_mlp_grid(
        flat_expert_inputs, 2, expert_mlps, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const std::array<std::array<float, 2>, 2> expected_grid{{
        {0.8818111F, 0.26973557F},
        {1.7636222F, 0.53947115F},
    }};
    if (!grid || grid.value().size() != expected_grid.size()) return 76;
    for (std::size_t expert = 0; expert < expected_grid.size(); ++expert) {
        if (grid.value()[expert].size() != expected_grid[expert].size()) {
            return 77;
        }
        for (std::size_t token = 0;
             token < expected_grid[expert].size(); ++token) {
            if (std::abs(grid.value()[expert][token] -
                         expected_grid[expert][token]) > 1.0e-6F) {
                return 78;
            }
        }
    }

    auto duplicate_grid_experts = expert_mlps;
    duplicate_grid_experts[1] = duplicate_grid_experts[0];
    auto mismatched_grid_experts = expert_mlps;
    mismatched_grid_experts[1].down.rows = 2;
    auto zero_id_grid_experts = expert_mlps;
    zero_id_grid_experts[0].gate.tensor_id = 0;
    const auto grid_event_count = profiler.events().size();
    const auto rejected_empty_grid = backend->mxfp4_situ_mlp_grid(
        {}, 0, expert_mlps, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto rejected_no_experts_grid = backend->mxfp4_situ_mlp_grid(
        flat_expert_inputs, 2, {}, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto rejected_short_grid = backend->mxfp4_situ_mlp_grid(
        std::span<const float>(flat_expert_inputs).first(63), 2,
        expert_mlps, 2.0F, 1.5F, 14, k3x::ProfilePhase::decode);
    const auto rejected_overflow_grid = backend->mxfp4_situ_mlp_grid(
        {}, std::numeric_limits<std::size_t>::max(), expert_mlps,
        2.0F, 1.5F, 14, k3x::ProfilePhase::decode);
    const auto rejected_duplicate_grid = backend->mxfp4_situ_mlp_grid(
        flat_expert_inputs, 2, duplicate_grid_experts, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto rejected_mismatched_grid = backend->mxfp4_situ_mlp_grid(
        flat_expert_inputs, 2, mismatched_grid_experts, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    const auto rejected_zero_id_grid = backend->mxfp4_situ_mlp_grid(
        flat_expert_inputs, 2, zero_id_grid_experts, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (rejected_empty_grid || rejected_no_experts_grid ||
        rejected_short_grid || rejected_overflow_grid ||
        rejected_duplicate_grid || rejected_mismatched_grid ||
        rejected_zero_id_grid ||
        rejected_empty_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_no_experts_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_short_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_overflow_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_duplicate_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        rejected_mismatched_grid.error() !=
            k3x::ErrorCode::invalid_mxfp4 ||
        rejected_zero_id_grid.error() != k3x::ErrorCode::invalid_mxfp4 ||
        profiler.events().size() != grid_event_count) {
        return 79;
    }
    const std::array<float, 2> contributions{0.25F, -0.5F};
    const auto mixed = backend->mxfp4_situ_moe(
        mxfp4_input, expert_mlps, contributions, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!mixed || mixed.value().size() != 1 ||
        std::abs(mixed.value()[0] -
                 (0.25F * expert_blocks.value()[0][0] -
                  0.5F * expert_blocks.value()[1][0])) > 1.0e-6F) {
        return 67;
    }
    const std::array<float, 1> wrong_contributions{1.0F};
    const auto rejected_mix = backend->mxfp4_situ_moe(
        mxfp4_input, expert_mlps, wrong_contributions, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (rejected_mix ||
        rejected_mix.error() != k3x::ErrorCode::invalid_mxfp4) return 68;

    auto invalid_expert_mlps = expert_mlps;
    invalid_expert_mlps[1].down.scales = {};
    const auto block_event_count = profiler.events().size();
    const auto rejected_block = backend->mxfp4_situ_mlp_group(
        mxfp4_input, invalid_expert_mlps, 2.0F, 1.5F, 15,
        k3x::ProfilePhase::decode);
    if (rejected_block ||
        rejected_block.error() != k3x::ErrorCode::invalid_mxfp4) {
        return 65;
    }
    if (profiler.events().size() != block_event_count) return 66;

    std::array<float, 96> routed_down_values{};
    routed_down_values[3] = 1.0F;
    const std::array<float, 1> routed_norm_values{1.25F};
    const std::array<float, 3> routed_up_values{1.0F, -0.5F, 2.0F};
    const std::array<float, 6> shared_down_values{
        1.0F, 2.0F, -1.0F, 0.5F, 0.25F, -0.75F};
    const k3x::ResidentMoeLayerView layer_weights{
        {501, routed_down_values, 32, 3},
        {502, routed_norm_values},
        {503, routed_up_values, 3, 1},
        {
            {504, dense_gate, 2, 3},
            {505, dense_up, 2, 3},
            {506, shared_down_values, 3, 2},
        },
    };
    const std::array<float, 2> layer_contributions{0.75F, -0.25F};

    const auto expected_latent = backend->dense_matvec(
        dense_input, layer_weights.routed_down, 16,
        k3x::ProfilePhase::decode);
    if (!expected_latent) return 80;
    const auto expected_experts = backend->mxfp4_situ_mlp_group(
        expected_latent.value(), expert_mlps, 2.0F, 1.5F, 16,
        k3x::ProfilePhase::decode);
    if (!expected_experts) return 81;
    std::array<float, 1> expected_mixed{};
    for (std::size_t slot = 0; slot < expert_mlps.size(); ++slot) {
        expected_mixed[0] +=
            layer_contributions[slot] * expected_experts.value()[slot][0];
    }
    std::array<float, 1> expected_normalized{};
    k3x::rms_norm(expected_normalized, expected_mixed, routed_norm_values,
                  1.0e-5F);
    const auto expected_routed = backend->dense_matvec(
        expected_normalized, layer_weights.routed_up, 16,
        k3x::ProfilePhase::decode);
    const auto expected_shared = backend->dense_situ_mlp(
        dense_input, layer_weights.shared, 2.0F, 1.5F, 16,
        k3x::ProfilePhase::decode);
    if (!expected_routed || !expected_shared) return 82;
    std::array<float, 3> expected_layer_output{};
    for (std::size_t row = 0; row < expected_layer_output.size(); ++row) {
        expected_layer_output[row] =
            expected_routed.value()[row] + expected_shared.value()[row];
    }

    const auto layer_result = backend->resident_mxfp4_moe_layer(
        dense_input, layer_weights, expert_mlps, layer_contributions,
        1.0e-5F, 2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    if (!layer_result || !layer_result.value().executed ||
        layer_result.value().output.size() != expected_layer_output.size()) {
        return 83;
    }
    for (std::size_t row = 0; row < expected_layer_output.size(); ++row) {
        if (std::abs(layer_result.value().output[row] -
                     expected_layer_output[row]) > 1.0e-6F) {
            return 84;
        }
    }

    auto malformed_routed = layer_weights;
    malformed_routed.routed_up.cols = 2;
    auto malformed_shared = layer_weights;
    malformed_shared.shared.down.rows = 2;
    auto short_norm = layer_weights;
    short_norm.routed_norm.values = {};
    auto duplicate_id = layer_weights;
    duplicate_id.routed_up.tensor_id = duplicate_id.routed_down.tensor_id;
    auto zero_id = layer_weights;
    zero_id.routed_norm.tensor_id = 0;
    const std::array<float, 1> short_contributions{1.0F};
    const std::array<float, 2> nonfinite_contributions{
        0.75F, std::numeric_limits<float>::infinity()};
    const auto rejected_empty_experts = backend->resident_mxfp4_moe_layer(
        dense_input, layer_weights, {}, {}, 1.0e-5F, 2.0F, 1.5F, 16,
        k3x::ProfilePhase::decode);
    const auto rejected_short_contributions =
        backend->resident_mxfp4_moe_layer(
            dense_input, layer_weights, expert_mlps, short_contributions,
            1.0e-5F, 2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_nonfinite = backend->resident_mxfp4_moe_layer(
        dense_input, layer_weights, expert_mlps, nonfinite_contributions,
        1.0e-5F, 2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_epsilon = backend->resident_mxfp4_moe_layer(
        dense_input, layer_weights, expert_mlps, layer_contributions, 0.0F,
        2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_routed = backend->resident_mxfp4_moe_layer(
        dense_input, malformed_routed, expert_mlps, layer_contributions,
        1.0e-5F, 2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_shared = backend->resident_mxfp4_moe_layer(
        dense_input, malformed_shared, expert_mlps, layer_contributions,
        1.0e-5F, 2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_norm = backend->resident_mxfp4_moe_layer(
        dense_input, short_norm, expert_mlps, layer_contributions, 1.0e-5F,
        2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_duplicate_id = backend->resident_mxfp4_moe_layer(
        dense_input, duplicate_id, expert_mlps, layer_contributions, 1.0e-5F,
        2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const auto rejected_zero_id = backend->resident_mxfp4_moe_layer(
        dense_input, zero_id, expert_mlps, layer_contributions, 1.0e-5F,
        2.0F, 1.5F, 16, k3x::ProfilePhase::decode);
    const std::array rejected_layers{
        &rejected_empty_experts, &rejected_short_contributions,
        &rejected_nonfinite, &rejected_epsilon, &rejected_routed,
        &rejected_shared, &rejected_norm, &rejected_duplicate_id,
        &rejected_zero_id,
    };
    for (const auto* rejected_layer : rejected_layers) {
        if (*rejected_layer ||
            rejected_layer->error() != k3x::ErrorCode::invalid_mxfp4) {
            return 85;
        }
    }
    const auto layer_stats = backend->runtime_stats();
    if (layer_stats.resident_moe_layer_calls != 0 ||
        layer_stats.resident_moe_layer_experts != 0 ||
        layer_stats.resident_moe_layer_kernel_launches != 0 ||
        layer_stats.resident_moe_layer_fallbacks != 0 ||
        layer_stats.resident_moe_layer_contribution_h2d_bytes != 0) {
        return 86;
    }
    return 0;
}
