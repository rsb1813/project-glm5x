// released-dimension expert를 반복 사용해 CUDA routed accumulation 경계를 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/reader.hpp"
#include "k3x/storage_slice.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

std::vector<float> mix_outputs(
    const std::vector<std::vector<float>>& outputs,
    std::span<const float> contributions) {
    std::vector<float> mixed(outputs.front().size(), 0.0F);
    for (std::size_t expert = 0; expert < outputs.size(); ++expert) {
        for (std::size_t row = 0; row < mixed.size(); ++row) {
            mixed[row] += contributions[expert] * outputs[expert][row];
        }
    }
    return mixed;
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path;
    std::string fusion_name = "none";
    std::uint64_t slots = 16;
    std::uint64_t warmup = 0;
    std::uint64_t iterations = 1;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return 2;
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        const auto number = parse_u64(value);
        if (key == "--model") model_path = value;
        else if (key == "--fusion") fusion_name = value;
        else if (key == "--slots" && number) slots = *number;
        else if (key == "--warmup" && number) warmup = *number;
        else if (key == "--iterations" && number) iterations = *number;
        else {
            std::cerr << "invalid option: " << key << '\n';
            return 2;
        }
    }
    if (slots == 0 || slots > 16) {
        std::cerr << "slots must be between 1 and 16\n";
        return 2;
    }
    if (model_path.empty()) {
        std::cerr << "model path is required\n";
        return 2;
    }
    if (iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return 2;
    }
    const bool fused = fusion_name == "routed-accumulate";
    if (!fused && fusion_name != "none") {
        std::cerr << "unknown fusion mode: " << fusion_name << '\n';
        return 2;
    }

    k3x::ReaderOptions reader_options;
    reader_options.verify = k3x::VerifyMode::metadata_only;
    auto reader = k3x::Reader::open(model_path, reader_options);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    auto loaded = k3x::load_storage_expert(reader.value(), 1, 0);
    if (!loaded) {
        write_error(loaded.error(), loaded.message());
        return 4;
    }
    const auto& extents = loaded.value().extents;
    const k3x::Mxfp4MlpView expert{
        {101, extents[0], extents[1], 3072, 3584, 32},
        {102, extents[2], extents[3], 3072, 3584, 32},
        {103, extents[4], extents[5], 3584, 3072, 32},
    };
    std::vector<k3x::Mxfp4MlpView> experts(
        static_cast<std::size_t>(slots), expert);
    std::vector<float> contributions(
        static_cast<std::size_t>(slots),
        1.0F / static_cast<float>(slots));
    std::vector<float> input(3584);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) *
                       0.01F;
    }

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_moe_fusion = fused
        ? k3x::CudaMoeFusionMode::routed_accumulate
        : k3x::CudaMoeFusionMode::none;
    options.cuda_resident_bytes = loaded.value().logical_bytes;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) {
        write_error(backend.error(), backend.message());
        return 4;
    }

    const auto reference_outputs = backend.value()->mxfp4_situ_mlp_group(
        input, experts, 4.0F, 25.0F, 1, k3x::ProfilePhase::decode);
    if (!reference_outputs) {
        write_error(reference_outputs.error(), reference_outputs.message());
        return 4;
    }
    const auto reference = mix_outputs(reference_outputs.value(), contributions);
    const auto execute = [&]() -> k3x::Result<std::vector<float>> {
        if (fused) {
            return backend.value()->mxfp4_situ_moe(
                input, experts, contributions, 4.0F, 25.0F, 1,
                k3x::ProfilePhase::decode);
        }
        auto outputs = backend.value()->mxfp4_situ_mlp_group(
            input, experts, 4.0F, 25.0F, 1,
            k3x::ProfilePhase::decode);
        if (!outputs) {
            return k3x::Result<std::vector<float>>::failure(
                outputs.error(), outputs.message());
        }
        return k3x::Result<std::vector<float>>::success(
            mix_outputs(outputs.value(), contributions));
    };
    for (std::uint64_t index = 0; index < warmup; ++index) {
        const auto result = execute();
        if (!result) {
            write_error(result.error(), result.message());
            return 4;
        }
    }

    const auto runtime_before = backend.value()->runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    std::vector<float> actual;
    for (std::uint64_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto result = execute();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        if (!result) {
            write_error(result.error(), result.message());
            return 4;
        }
        actual = std::move(result.value());
        samples.push_back(elapsed);
    }
    const auto runtime_after = backend.value()->runtime_stats();
    const auto profile_after = profiler.summary();
    std::sort(samples.begin(), samples.end());
    const auto median = samples.size() % 2
        ? samples[samples.size() / 2]
        : samples[samples.size() / 2 - 1] +
              (samples[samples.size() / 2] - samples[samples.size() / 2 - 1]) /
                  2;
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum_error =
            std::max(maximum_error, std::abs(actual[index] - reference[index]));
    }

    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"released_dimension_repeated_view\""
              << ",\"routing_semantics\":false"
              << ",\"fusion\":\"" << fusion_name << "\""
              << ",\"expert_payload_bytes\":" << loaded.value().logical_bytes
              << ",\"slots\":" << slots
              << ",\"warmup\":" << warmup
              << ",\"iterations\":" << iterations
              << ",\"latency_nanoseconds_median\":" << median
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"kernel_nanoseconds\":"
              << profile_after.device_nanoseconds -
                     profile_before.device_nanoseconds
              << ",\"device_to_host_bytes\":"
              << profile_after.device_to_host_bytes -
                     profile_before.device_to_host_bytes
              << ",\"weight_h2d_bytes\":"
              << runtime_after.weight_h2d_bytes - runtime_before.weight_h2d_bytes
              << ",\"activation_h2d_bytes\":"
              << runtime_after.activation_h2d_bytes -
                     runtime_before.activation_h2d_bytes
              << ",\"fused_moe_calls\":"
              << runtime_after.fused_moe_calls - runtime_before.fused_moe_calls
              << ",\"fused_moe_experts\":"
              << runtime_after.fused_moe_experts -
                     runtime_before.fused_moe_experts
              << ",\"peak_vram_bytes\":"
              << backend.value()->memory_stats().peak_device_bytes << "}\n";
    return 0;
}
