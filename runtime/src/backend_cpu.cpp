// 기존 double 누산 dense와 native MXFP4 연산을 exact CPU backend로 제공합니다.
#include "k3x/backend.hpp"

#include "k3x/ops.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace k3x {
namespace {

class CpuBackend final : public ComputeBackend {
public:
    explicit CpuBackend(Profiler* profiler) : profiler_(profiler) {}

    BackendKind kind() const noexcept override { return BackendKind::cpu; }
    const BackendOptions& options() const noexcept override { return options_; }
    BackendRuntimeStats runtime_stats() const noexcept override { return {}; }

    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, DenseWeightView weight,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto start = std::chrono::steady_clock::now();
        if (!valid_dense(input, weight)) {
            record(phase, ProfileOperation::dense_matvec, NumericPrecision::fp32,
                   layer, start, weight.values.size_bytes(), false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }

        std::vector<float> output(weight.rows);
        for (std::size_t row = 0; row < weight.rows; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0; column < weight.cols; ++column) {
                sum += static_cast<double>(
                           weight.values[row * weight.cols + column]) *
                       input[column];
            }
            output[row] = static_cast<float>(sum);
        }
        record(phase, ProfileOperation::dense_matvec, NumericPrecision::fp32,
               layer, start, weight.values.size_bytes(), true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, Mxfp4WeightView weight,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto start = std::chrono::steady_clock::now();
        auto result = mxfp4_matmul(input, weight.packed, weight.scales,
                                   weight.rows, weight.cols, weight.group_size);
        record(phase, ProfileOperation::mxfp4_matvec,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, start,
               weight.packed.size_bytes() + weight.scales.size_bytes(),
               static_cast<bool>(result));
        return result;
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

    Result<std::vector<std::vector<float>>> mxfp4_matvec_group(
        std::span<const float> input, std::span<const Mxfp4WeightView> weights,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& weight : weights) {
            if (!valid_mxfp4(input, weight)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
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

    Result<std::vector<float>> dense_situ_mlp(
        std::span<const float> input, DenseMlpView weights,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        if (!valid_dense_mlp(input, weights)) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
        auto gate = dense_matvec(input, weights.gate, layer, phase);
        auto up = dense_matvec(input, weights.up, layer, phase);
        if (!gate || !up) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
        std::vector<float> activated(weights.gate.rows);
        situ_glu(activated, gate.value(), up.value(), situ_beta, situ_linear);
        return dense_matvec(activated, weights.down, layer, phase);
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& expert : experts) {
            if (!valid_mxfp4_mlp(input, expert)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        std::vector<std::vector<float>> outputs;
        outputs.reserve(experts.size());
        for (const auto& expert : experts) {
            auto gate = mxfp4_matvec(input, expert.gate, layer, phase);
            auto up = mxfp4_matvec(input, expert.up, layer, phase);
            if (!gate || !up) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            std::vector<float> activated(expert.gate.rows);
            situ_glu(activated, gate.value(), up.value(), situ_beta, situ_linear);
            auto output = mxfp4_matvec(activated, expert.down, layer, phase);
            if (!output) {
                return Result<std::vector<std::vector<float>>>::failure(
                    output.error(), output.message());
            }
            outputs.push_back(std::move(output.value()));
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_batch(
        std::span<const float> inputs, std::size_t batch_size,
        Mxfp4MlpView expert, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (batch_size == 0 || expert.gate.cols == 0 ||
            batch_size > std::numeric_limits<std::size_t>::max() /
                             expert.gate.cols ||
            inputs.size() != batch_size * expert.gate.cols ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F)) ||
            !valid_mxfp4_mlp(inputs.first(expert.gate.cols), expert)) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::vector<std::vector<float>> outputs;
        outputs.reserve(batch_size);
        const std::span<const Mxfp4MlpView> one_expert(&expert, 1);
        for (std::size_t row = 0; row < batch_size; ++row) {
            const auto input = inputs.subspan(
                row * expert.gate.cols, expert.gate.cols);
            auto output = mxfp4_situ_mlp_group(
                input, one_expert, situ_beta, situ_linear, layer, phase);
            if (!output) {
                return Result<std::vector<std::vector<float>>>::failure(
                    output.error(), output.message());
            }
            outputs.push_back(std::move(output.value().front()));
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_grid(
        std::span<const float> inputs, std::size_t token_count,
        std::span<const Mxfp4MlpView> experts, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (token_count == 0 || experts.empty() ||
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
                !valid_mxfp4_mlp(inputs.first(input_width), expert)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            for (const auto tensor_id : {
                     expert.gate.tensor_id,
                     expert.up.tensor_id,
                     expert.down.tensor_id}) {
                if (tensor_id == 0 || !tensor_ids.insert(tensor_id).second) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::invalid_mxfp4);
                }
            }
        }
        if (!multiply_fits(token_count, output_width)) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::vector<std::vector<float>> outputs;
        outputs.reserve(experts.size());
        for (const auto& expert : experts) {
            std::vector<float> expert_outputs;
            expert_outputs.reserve(token_count * output_width);
            const std::span<const Mxfp4MlpView> one_expert(&expert, 1);
            for (std::size_t token = 0; token < token_count; ++token) {
                const auto input = inputs.subspan(
                    token * input_width, input_width);
                auto output = mxfp4_situ_mlp_group(
                    input, one_expert, situ_beta, situ_linear, layer, phase);
                if (!output) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        output.error(), output.message());
                }
                expert_outputs.insert(
                    expert_outputs.end(), output.value().front().begin(),
                    output.value().front().end());
            }
            outputs.push_back(std::move(expert_outputs));
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<float>> mxfp4_situ_moe(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (experts.empty() || experts.size() != contributions.size()) {
            return Result<std::vector<float>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        for (const auto contribution : contributions) {
            if (!std::isfinite(contribution)) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        const auto output_rows = experts.front().down.rows;
        for (const auto& expert : experts) {
            if (!valid_mxfp4_mlp(input, expert) ||
                expert.down.rows != output_rows) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        auto outputs = mxfp4_situ_mlp_group(
            input, experts, situ_beta, situ_linear, layer, phase);
        if (!outputs) {
            return Result<std::vector<float>>::failure(
                outputs.error(), outputs.message());
        }
        std::vector<float> mixed(output_rows, 0.0F);
        for (std::size_t expert = 0; expert < outputs.value().size(); ++expert) {
            for (std::size_t row = 0; row < output_rows; ++row) {
                mixed[row] += contributions[expert] *
                              outputs.value()[expert][row];
            }
        }
        return Result<std::vector<float>>::success(std::move(mixed));
    }

    Result<ResidentMoeLayerResult> resident_mxfp4_moe_layer(
        std::span<const float> input, ResidentMoeLayerView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (!valid_resident_moe_layer(input, weights, experts, contributions,
                                      epsilon, situ_beta, situ_linear)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }

        auto latent = dense_matvec(input, weights.routed_down, layer, phase);
        auto expert_outputs = mxfp4_situ_mlp_group(
            latent.value(), experts, situ_beta, situ_linear, layer, phase);
        std::vector<float> mixed(weights.routed_norm.values.size(), 0.0F);
        for (std::size_t slot = 0; slot < experts.size(); ++slot) {
            for (std::size_t row = 0; row < mixed.size(); ++row) {
                mixed[row] += contributions[slot] *
                              expert_outputs.value()[slot][row];
            }
        }

        std::vector<float> normalized(mixed.size());
        rms_norm(normalized, mixed, weights.routed_norm.values, epsilon);
        auto routed = dense_matvec(
            normalized, weights.routed_up, layer, phase);
        auto shared = dense_situ_mlp(
            input, weights.shared, situ_beta, situ_linear, layer, phase);
        for (std::size_t row = 0; row < routed.value().size(); ++row) {
            routed.value()[row] += shared.value()[row];
        }
        return Result<ResidentMoeLayerResult>::success(
            {true, std::move(routed.value())});
    }

    Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
        std::span<const Mxfp4MlpView>, std::uint64_t, std::uint32_t,
        ProfilePhase) override {
        return Result<Mxfp4PrefetchToken>::failure(
            ErrorCode::backend_unavailable);
    }

    Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared(
        std::span<const float>, Mxfp4PrefetchToken, float,
        std::optional<float>, std::uint32_t, ProfilePhase) override {
        return Result<std::vector<std::vector<float>>>::failure(
            ErrorCode::backend_unavailable);
    }

    Result<std::vector<float>> mxfp4_situ_moe_prepared(
        std::span<const float>, Mxfp4PrefetchToken, std::span<const float>,
        float, std::optional<float>, std::uint32_t,
        ProfilePhase) override {
        return Result<std::vector<float>>::failure(
            ErrorCode::backend_unavailable);
    }

    BackendMemoryStats memory_stats() const noexcept override { return {}; }
    std::string_view device_name() const noexcept override { return "CPU"; }

private:
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

    static bool valid_mxfp4(std::span<const float> input,
                            Mxfp4WeightView weight) {
        return valid_mxfp4(input.size(), weight);
    }

    static bool valid_mxfp4(std::size_t input_size,
                            Mxfp4WeightView weight) {
        if (input_size != weight.cols || !weight.rows || !weight.cols ||
            !weight.group_size || weight.cols % weight.group_size ||
            weight.cols % 2 ||
            weight.rows > std::numeric_limits<std::size_t>::max() /
                              weight.cols) {
            return false;
        }
        const auto elements = weight.rows * weight.cols;
        if (weight.packed.size() != elements / 2 ||
            weight.scales.size() != elements / weight.group_size) {
            return false;
        }
        for (const auto scale : weight.scales) {
            if (scale == std::byte{0xff}) return false;
        }
        return true;
    }

    static bool valid_dense_mlp(std::span<const float> input,
                                DenseMlpView weights) {
        return valid_dense(input, weights.gate) &&
               valid_dense(input, weights.up) && weights.gate.rows != 0 &&
               weights.gate.rows == weights.up.rows &&
               valid_dense_size(weights.gate.rows, weights.down);
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

    static bool valid_mxfp4_mlp(std::span<const float> input,
                                 Mxfp4MlpView expert) {
        return valid_mxfp4_mlp(input.size(), expert);
    }

    static bool valid_mxfp4_mlp(std::size_t input_size,
                                 Mxfp4MlpView expert) {
        return valid_mxfp4(input_size, expert.gate) &&
               valid_mxfp4(input_size, expert.up) &&
               expert.gate.rows == expert.up.rows &&
               valid_mxfp4(expert.gate.rows, expert.down);
    }

    static bool all_finite(std::span<const float> values) {
        for (const auto value : values) {
            if (!std::isfinite(value)) return false;
        }
        return true;
    }

    static bool valid_resident_moe_layer(
        std::span<const float> input, ResidentMoeLayerView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear) {
        if (input.empty() || experts.empty() ||
            experts.size() != contributions.size() ||
            !std::isfinite(epsilon) || epsilon <= 0.0F ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F)) ||
            !all_finite(input) || !all_finite(contributions) ||
            !all_finite(weights.routed_down.values) ||
            !all_finite(weights.routed_norm.values) ||
            !all_finite(weights.routed_up.values) ||
            !all_finite(weights.shared.gate.values) ||
            !all_finite(weights.shared.up.values) ||
            !all_finite(weights.shared.down.values) ||
            !valid_dense(input, weights.routed_down) ||
            !valid_dense_mlp(input, weights.shared)) {
            return false;
        }

        const auto latent_width = weights.routed_down.rows;
        const auto routed_width = experts.front().down.rows;
        if (latent_width == 0 || routed_width == 0 ||
            weights.routed_norm.values.size() != routed_width ||
            !valid_dense_size(routed_width, weights.routed_up) ||
            weights.routed_up.rows != input.size() ||
            weights.shared.down.rows != input.size()) {
            return false;
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
            return false;
        }
        const auto intermediate_width = experts.front().gate.rows;
        for (const auto& expert : experts) {
            if (!valid_mxfp4_mlp(latent_width, expert) ||
                expert.gate.rows != intermediate_width ||
                expert.down.rows != routed_width ||
                expert.gate.group_size != 32 ||
                expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !insert_id(expert.gate.tensor_id) ||
                !insert_id(expert.up.tensor_id) ||
                !insert_id(expert.down.tensor_id)) {
                return false;
            }
        }
        return true;
    }

    void record(ProfilePhase phase, ProfileOperation operation,
                NumericPrecision precision, std::uint32_t layer,
                std::chrono::steady_clock::time_point start,
                std::uint64_t logical_bytes, bool success) {
        if (!profiler_) return;
        const auto wall_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        profiler_->record({phase, operation, precision, layer,
                           static_cast<std::uint64_t>(wall_nanoseconds), 0,
                           logical_bytes, 0, success});
    }

    BackendOptions options_{};
    Profiler* profiler_;
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend(Profiler* profiler) {
    return std::make_unique<CpuBackend>(profiler);
}

}  // namespace k3x
