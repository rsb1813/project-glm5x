// GLM-5.2 공식 차원에서 resident expert FFN CUDA 기준선을 측정합니다.
#include "k3x/backend.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kHiddenSize = 6144;
constexpr std::size_t kIntermediateSize = 2048;
constexpr std::size_t kGroupSize = 32;
constexpr std::uint64_t kResidentCapacity = 1ULL << 30;

struct Arguments {
    std::size_t experts{1};
    std::size_t tokens{1};
    std::size_t warmup{};
    std::size_t iterations{1};
    std::string mode{"grid"};
};

struct ExpertStorage {
    std::vector<std::byte> gate_packed;
    std::vector<std::byte> gate_scales;
    std::vector<std::byte> up_packed;
    std::vector<std::byte> up_scales;
    std::vector<std::byte> down_packed;
    std::vector<std::byte> down_scales;
};

std::optional<std::size_t> parse_size(std::string_view text) {
    std::size_t value{};
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return std::nullopt;
        }
        const std::string key = argv[index];
        if (key == "--mode") {
            arguments.mode = argv[index + 1];
            continue;
        }
        const auto value = parse_size(argv[index + 1]);
        if (!value) {
            std::cerr << "invalid option value\n";
            return std::nullopt;
        }
        if (key == "--experts") {
            arguments.experts = *value;
        } else if (key == "--tokens") {
            arguments.tokens = *value;
        } else if (key == "--warmup") {
            arguments.warmup = *value;
        } else if (key == "--iterations") {
            arguments.iterations = *value;
        } else {
            std::cerr << "invalid option: " << key << '\n';
            return std::nullopt;
        }
    }
    if (arguments.experts != 1 && arguments.experts != 8 &&
        arguments.experts != 16) {
        std::cerr << "experts must be one of 1, 8, or 16\n";
        return std::nullopt;
    }
    if (arguments.tokens != 1 && arguments.tokens != 2 &&
        arguments.tokens != 4 && arguments.tokens != 8) {
        std::cerr << "tokens must be one of 1, 2, 4, or 8\n";
        return std::nullopt;
    }
    if (arguments.mode != "grid" && arguments.mode != "expert-batch") {
        std::cerr << "mode must be grid or expert-batch\n";
        return std::nullopt;
    }
    if (arguments.iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return std::nullopt;
    }
    return arguments;
}

std::size_t packed_bytes(std::size_t rows, std::size_t cols) {
    return rows * cols / 2;
}

std::size_t scale_bytes(std::size_t rows, std::size_t cols) {
    return rows * ((cols + kGroupSize - 1) / kGroupSize);
}

void initialize(ExpertStorage& storage) {
    storage.gate_packed.assign(
        packed_bytes(kIntermediateSize, kHiddenSize), std::byte{0});
    storage.gate_scales.assign(
        scale_bytes(kIntermediateSize, kHiddenSize), std::byte{127});
    storage.up_packed.assign(
        packed_bytes(kIntermediateSize, kHiddenSize), std::byte{0});
    storage.up_scales.assign(
        scale_bytes(kIntermediateSize, kHiddenSize), std::byte{127});
    storage.down_packed.assign(
        packed_bytes(kHiddenSize, kIntermediateSize), std::byte{0});
    storage.down_scales.assign(
        scale_bytes(kHiddenSize, kIntermediateSize), std::byte{127});
}

std::uint64_t expert_payload_bytes() {
    return static_cast<std::uint64_t>(
        2 * packed_bytes(kIntermediateSize, kHiddenSize) +
        2 * scale_bytes(kIntermediateSize, kHiddenSize) +
        packed_bytes(kHiddenSize, kIntermediateSize) +
        scale_bytes(kHiddenSize, kIntermediateSize));
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 != 0
        ? values[middle]
        : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

float maximum_absolute_value(
    const std::vector<std::vector<float>>& outputs) {
    float maximum = 0.0F;
    for (const auto& output : outputs) {
        for (const auto value : output) {
            maximum = std::max(maximum, std::abs(value));
        }
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;

    std::vector<ExpertStorage> storage(arguments->experts);
    for (auto& expert : storage) initialize(expert);

    std::vector<k3x::Mxfp4MlpView> experts;
    experts.reserve(arguments->experts);
    for (std::size_t index = 0; index < arguments->experts; ++index) {
        auto& item = storage[index];
        const auto base = 1000ULL + index * 3ULL;
        experts.push_back({
            {base, item.gate_packed, item.gate_scales,
             kIntermediateSize, kHiddenSize, kGroupSize},
            {base + 1, item.up_packed, item.up_scales,
             kIntermediateSize, kHiddenSize, kGroupSize},
            {base + 2, item.down_packed, item.down_scales,
             kHiddenSize, kIntermediateSize, kGroupSize},
        });
    }

    std::vector<float> input(arguments->tokens * kHiddenSize, 0.0F);
    for (std::size_t token = 0; token < arguments->tokens; ++token) {
        for (std::size_t index = 0; index < kHiddenSize; ++index) {
            input[token * kHiddenSize + index] =
                static_cast<float>(static_cast<int>((index + token) % 17) -
                                   8) *
                0.001F;
        }
    }

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    options.cuda_resident_bytes = kResidentCapacity;

    k3x::Profiler profiler;
    auto created = k3x::make_cuda_backend(options, &profiler);
    if (!created) {
        std::cerr << k3x::error_code_name(created.error()) << ": "
                  << created.message() << '\n';
        return 4;
    }
    auto& backend = *created.value();
    const auto execute = [&]() {
        if (arguments->mode == "grid") {
            return backend.mxfp4_situ_mlp_grid(
                input, arguments->tokens, experts, 1.0F, std::nullopt, 1,
                k3x::ProfilePhase::decode);
        }
        std::vector<std::vector<float>> outputs;
        outputs.reserve(arguments->experts * arguments->tokens);
        for (const auto& expert : experts) {
            auto result = backend.mxfp4_situ_mlp_batch(
                input, arguments->tokens, expert, 1.0F, std::nullopt, 1,
                k3x::ProfilePhase::decode);
            if (!result) {
                return k3x::Result<std::vector<std::vector<float>>>::failure(
                    result.error(), result.message());
            }
            for (auto& output : result.value()) {
                outputs.push_back(std::move(output));
            }
        }
        return k3x::Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    };

    const auto cold_start = std::chrono::steady_clock::now();
    auto cold = execute();
    const auto cold_elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - cold_start)
            .count());
    if (!cold) {
        std::cerr << k3x::error_code_name(cold.error()) << ": "
                  << cold.message() << '\n';
        return 4;
    }
    if (maximum_absolute_value(cold.value()) > 1.0e-5F) {
        std::cerr << "synthetic zero-weight validation failed\n";
        return 4;
    }
    const auto runtime_after_cold = backend.runtime_stats();

    for (std::size_t index = 0; index < arguments->warmup; ++index) {
        if (!execute()) return 4;
    }
    const auto runtime_before = backend.runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    std::vector<std::vector<float>> actual;
    samples.reserve(arguments->iterations);
    for (std::size_t index = 0; index < arguments->iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto result = execute();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        if (!result) return 4;
        actual = std::move(result.value());
        samples.push_back(elapsed);
    }
    const auto runtime_after = backend.runtime_stats();
    const auto profile_after = profiler.summary();

    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"glm5.2_shaped_expert_ffn\""
              << ",\"model_family\":\"glm5\""
              << ",\"routing_semantics\":false"
              << ",\"mode\":\"" << arguments->mode << "\""
              << ",\"hidden_size\":" << kHiddenSize
              << ",\"expert_intermediate_size\":" << kIntermediateSize
              << ",\"group_size\":" << kGroupSize
              << ",\"experts\":" << arguments->experts
              << ",\"tokens\":" << arguments->tokens
              << ",\"expert_payload_bytes\":" << expert_payload_bytes()
              << ",\"per_expert_payload_bytes\":" << expert_payload_bytes()
              << ",\"total_resident_payload_bytes\":"
              << expert_payload_bytes() * arguments->experts
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations
              << ",\"cold_latency_nanoseconds\":" << cold_elapsed
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"maximum_absolute_error\":"
              << maximum_absolute_value(actual)
              << ",\"cold_weight_h2d_bytes\":"
              << runtime_after_cold.weight_h2d_bytes
              << ",\"resident_weight_bytes\":"
              << runtime_after_cold.resident_weight_bytes
              << ",\"kernel_nanoseconds\":"
              << profile_after.device_nanoseconds - profile_before.device_nanoseconds
              << ",\"weight_h2d_bytes\":"
              << runtime_after.weight_h2d_bytes - runtime_before.weight_h2d_bytes
              << ",\"activation_h2d_bytes\":"
              << runtime_after.activation_h2d_bytes - runtime_before.activation_h2d_bytes
              << ",\"device_to_host_bytes\":"
              << runtime_after.device_to_host_bytes - runtime_before.device_to_host_bytes
              << ",\"resident_grid_calls\":"
              << runtime_after.resident_grid_calls - runtime_before.resident_grid_calls
              << ",\"resident_grid_kernel_launches\":"
              << runtime_after.resident_grid_kernel_launches -
                     runtime_before.resident_grid_kernel_launches
              << ",\"resident_grid_fallbacks\":"
              << runtime_after.resident_grid_fallbacks -
                     runtime_before.resident_grid_fallbacks
              << ",\"batched_expert_ffn_calls\":"
              << runtime_after.batched_expert_ffn_calls -
                     runtime_before.batched_expert_ffn_calls
              << ",\"batched_expert_ffn_tokens\":"
              << runtime_after.batched_expert_ffn_tokens -
                     runtime_before.batched_expert_ffn_tokens
              << ",\"peak_vram_bytes\":"
              << backend.memory_stats().peak_device_bytes << "}\n";
    return 0;
}
