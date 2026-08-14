// released 크기의 resident CUDA MoE layer 경계를 직접 측정합니다.
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
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kHidden = 7168;
constexpr std::size_t kLatent = 3584;
constexpr std::size_t kIntermediate = 3072;
constexpr std::uint64_t kResidentCapacity = 1ULL << 30;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kSituBeta = 4.0F;
constexpr float kSituLinear = 25.0F;

enum class Boundary { ffn_block, moe_layer };

struct Arguments {
    std::filesystem::path model;
    std::string boundary_name{"ffn-block"};
    Boundary boundary{Boundary::ffn_block};
    std::size_t experts{1};
    std::size_t warmup{};
    std::size_t iterations{1};
    std::string validation_name{"per-call"};
    k3x::CudaWeightValidationMode validation{
        k3x::CudaWeightValidationMode::per_call};
    bool profiler{true};
};

struct Fixture {
    std::vector<float> input;
    std::vector<float> routed_down;
    std::vector<float> routed_norm;
    std::vector<float> routed_up;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_down;
    std::vector<k3x::Mxfp4MlpView> experts;
    std::vector<float> contributions;
    k3x::ResidentMoeLayerView layer{};
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

bool valid_expert_count(std::size_t value) {
    return value == 1 || value == 4 || value == 16;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return std::nullopt;
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--model") {
            arguments.model = value;
        } else if (key == "--boundary") {
            arguments.boundary_name = value;
        } else if (key == "--experts") {
            const auto parsed = parse_size(value);
            if (!parsed) {
                std::cerr << "invalid option: --experts\n";
                return std::nullopt;
            }
            arguments.experts = *parsed;
        } else if (key == "--warmup") {
            const auto parsed = parse_size(value);
            if (!parsed) {
                std::cerr << "invalid option: --warmup\n";
                return std::nullopt;
            }
            arguments.warmup = *parsed;
        } else if (key == "--iterations") {
            const auto parsed = parse_size(value);
            if (!parsed) {
                std::cerr << "invalid option: --iterations\n";
                return std::nullopt;
            }
            arguments.iterations = *parsed;
        } else if (key == "--validation") {
            arguments.validation_name = value;
        } else if (key == "--profiler") {
            if (value == "on") arguments.profiler = true;
            else if (value == "off") arguments.profiler = false;
            else {
                std::cerr << "unknown profiler mode: " << value << '\n';
                return std::nullopt;
            }
        } else {
            std::cerr << "invalid option: " << key << '\n';
            return std::nullopt;
        }
    }
    if (arguments.boundary_name == "ffn-block") {
        arguments.boundary = Boundary::ffn_block;
    } else if (arguments.boundary_name == "moe-layer") {
        arguments.boundary = Boundary::moe_layer;
    } else {
        std::cerr << "unknown boundary: " << arguments.boundary_name << '\n';
        return std::nullopt;
    }
    if (!valid_expert_count(arguments.experts)) {
        std::cerr << "experts must be one of 1, 4, or 16\n";
        return std::nullopt;
    }
    if (arguments.validation_name == "per-call") {
        arguments.validation = k3x::CudaWeightValidationMode::per_call;
    } else if (arguments.validation_name == "admission") {
        arguments.validation = k3x::CudaWeightValidationMode::admission;
    } else {
        std::cerr << "unknown validation mode: "
                  << arguments.validation_name << '\n';
        return std::nullopt;
    }
    if (arguments.validation == k3x::CudaWeightValidationMode::admission &&
        arguments.boundary != Boundary::moe_layer) {
        std::cerr << "admission validation requires moe-layer boundary\n";
        return std::nullopt;
    }
    if (arguments.iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return std::nullopt;
    }
    if (arguments.model.empty()) {
        std::cerr << "model path is required\n";
        return std::nullopt;
    }
    return arguments;
}

void fill_dense(std::vector<float>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) * 1.0e-4F;
    }
}

Fixture make_fixture(
    const k3x::StorageExpertLoad& loaded, std::size_t expert_count) {
    Fixture fixture;
    fixture.input.resize(kHidden);
    fixture.routed_down.resize(kLatent * kHidden);
    fixture.routed_norm.assign(kLatent, 1.0F);
    fixture.routed_up.resize(kHidden * kLatent);
    fixture.shared_gate.resize(kIntermediate * kHidden);
    fixture.shared_up.resize(kIntermediate * kHidden);
    fixture.shared_down.resize(kHidden * kIntermediate);
    for (std::size_t index = 0; index < fixture.input.size(); ++index) {
        fixture.input[index] =
            static_cast<float>(static_cast<int>(index % 19) - 9) * 0.01F;
    }
    fill_dense(fixture.routed_down);
    fill_dense(fixture.routed_up);
    fill_dense(fixture.shared_gate);
    fill_dense(fixture.shared_up);
    fill_dense(fixture.shared_down);

    fixture.experts.reserve(expert_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        const auto base = 1000 + expert * 3;
        fixture.experts.push_back({
            {base, loaded.extents[0], loaded.extents[1],
             kIntermediate, kLatent, 32},
            {base + 1, loaded.extents[2], loaded.extents[3],
             kIntermediate, kLatent, 32},
            {base + 2, loaded.extents[4], loaded.extents[5],
             kLatent, kIntermediate, 32},
        });
    }
    fixture.contributions.assign(
        expert_count, 1.0F / static_cast<float>(expert_count));
    fixture.layer = {
        {100, fixture.routed_down, kLatent, kHidden},
        {106, fixture.routed_norm},
        {101, fixture.routed_up, kHidden, kLatent},
        {
            {102, fixture.shared_gate, kIntermediate, kHidden},
            {103, fixture.shared_up, kIntermediate, kHidden},
            {104, fixture.shared_down, kHidden, kIntermediate},
        },
    };
    return fixture;
}

std::vector<float> ordered_mix(
    const std::vector<std::vector<float>>& outputs,
    std::span<const float> contributions) {
    std::vector<float> result(outputs.front().size(), 0.0F);
    for (std::size_t expert = 0; expert < outputs.size(); ++expert) {
        for (std::size_t row = 0; row < result.size(); ++row) {
            result[row] += contributions[expert] * outputs[expert][row];
        }
    }
    return result;
}

std::vector<float> strict_rms_norm(
    std::span<const float> input, std::span<const float> weight) {
    double square_sum = 0.0;
    for (const auto value : input) {
        square_sum += static_cast<double>(value) * value;
    }
    const auto inverse = 1.0F /
        std::sqrt(static_cast<float>(square_sum / input.size()) + kEpsilon);
    std::vector<float> result(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        result[index] = input[index] * inverse * weight[index];
    }
    return result;
}

k3x::Result<std::vector<float>> execute_split(
    k3x::ComputeBackend& backend, const Fixture& fixture) {
    auto latent = backend.dense_matvec(
        fixture.input, fixture.layer.routed_down, 1,
        k3x::ProfilePhase::decode);
    if (!latent) {
        return k3x::Result<std::vector<float>>::failure(
            latent.error(), latent.message());
    }
    auto shared = backend.dense_situ_mlp(
        fixture.input, fixture.layer.shared, kSituBeta, kSituLinear, 1,
        k3x::ProfilePhase::decode);
    if (!shared) {
        return k3x::Result<std::vector<float>>::failure(
            shared.error(), shared.message());
    }
    auto expert_outputs = backend.mxfp4_situ_mlp_grid(
        latent.value(), 1, fixture.experts, kSituBeta, kSituLinear, 1,
        k3x::ProfilePhase::decode);
    if (!expert_outputs) {
        return k3x::Result<std::vector<float>>::failure(
            expert_outputs.error(), expert_outputs.message());
    }
    auto normalized = strict_rms_norm(
        ordered_mix(expert_outputs.value(), fixture.contributions),
        fixture.routed_norm);
    auto routed = backend.dense_matvec(
        normalized, fixture.layer.routed_up, 1, k3x::ProfilePhase::decode);
    if (!routed) {
        return k3x::Result<std::vector<float>>::failure(
            routed.error(), routed.message());
    }
    for (std::size_t row = 0; row < routed.value().size(); ++row) {
        routed.value()[row] += shared.value()[row];
    }
    return k3x::Result<std::vector<float>>::success(
        std::move(routed.value()));
}

k3x::Result<std::vector<float>> execute_layer(
    k3x::ComputeBackend& backend, const Fixture& fixture) {
    auto result = backend.resident_mxfp4_moe_layer(
        fixture.input, fixture.layer, fixture.experts, fixture.contributions,
        kEpsilon, kSituBeta, kSituLinear, 1, k3x::ProfilePhase::decode);
    if (!result) {
        return k3x::Result<std::vector<float>>::failure(
            result.error(), result.message());
    }
    if (!result.value().executed) {
        return k3x::Result<std::vector<float>>::failure(
            k3x::ErrorCode::backend_unavailable,
            "released MoE layer capacity bypass");
    }
    return k3x::Result<std::vector<float>>::success(
        std::move(result.value().output));
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

k3x::BackendOptions backend_options(
    Boundary boundary,
    k3x::CudaWeightValidationMode validation =
        k3x::CudaWeightValidationMode::per_call) {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::fp32;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    options.cuda_boundary = boundary == Boundary::moe_layer
        ? k3x::CudaBoundaryMode::moe_layer
        : k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_weight_validation = validation;
    options.cuda_resident_bytes = kResidentCapacity;
    return options;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 != 0
        ? values[middle]
        : values[middle - 1] +
              (values[middle] - values[middle - 1]) / 2;
}

float maximum_error(
    std::span<const float> actual, std::span<const float> expected) {
    float result = 0.0F;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        result = std::max(result, std::abs(actual[index] - expected[index]));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;

    k3x::ReaderOptions reader_options;
    reader_options.verify = k3x::VerifyMode::metadata_only;
    auto reader = k3x::Reader::open(arguments->model, reader_options);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    auto loaded = k3x::load_storage_expert(reader.value(), 1, 0);
    if (!loaded) {
        write_error(loaded.error(), loaded.message());
        return 4;
    }
    auto fixture = make_fixture(loaded.value(), arguments->experts);

    std::vector<float> oracle;
    std::uint64_t oracle_peak_vram_bytes{};
    {
        auto oracle_backend =
            k3x::make_cuda_backend(backend_options(Boundary::ffn_block));
        if (!oracle_backend) {
            write_error(oracle_backend.error(), oracle_backend.message());
            return 4;
        }
        auto oracle_result = execute_split(*oracle_backend.value(), fixture);
        if (!oracle_result) {
            write_error(oracle_result.error(), oracle_result.message());
            return 4;
        }
        oracle = std::move(oracle_result.value());
        oracle_peak_vram_bytes =
            oracle_backend.value()->memory_stats().peak_device_bytes;
    }

    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(
        backend_options(arguments->boundary, arguments->validation),
        arguments->profiler ? &profiler : nullptr);
    if (!backend) {
        write_error(backend.error(), backend.message());
        return 4;
    }
    const auto execute = [&]() {
        return arguments->boundary == Boundary::moe_layer
            ? execute_layer(*backend.value(), fixture)
            : execute_split(*backend.value(), fixture);
    };

    const auto cold_runtime_before = backend.value()->runtime_stats();
    auto cold = execute();
    if (!cold) {
        write_error(cold.error(), cold.message());
        return 4;
    }
    const auto cold_runtime_after = backend.value()->runtime_stats();
    auto observed_maximum_error =
        maximum_error(cold.value(), oracle);

    for (std::size_t index = 0; index < arguments->warmup; ++index) {
        const auto result = execute();
        if (!result) {
            write_error(result.error(), result.message());
            return 4;
        }
    }

    const auto runtime_before = backend.value()->runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    samples.reserve(arguments->iterations);
    std::vector<float> actual;
    for (std::size_t index = 0; index < arguments->iterations; ++index) {
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
    observed_maximum_error = std::max(
        observed_maximum_error, maximum_error(actual, oracle));
    const auto runtime_after = backend.value()->runtime_stats();
    const auto profile_after = profiler.summary();
    const auto memory = backend.value()->memory_stats();

    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"released_dimension_moe_layer\""
              << ",\"routing_semantics\":false"
              << ",\"boundary\":\"" << arguments->boundary_name << "\""
              << ",\"experts\":" << arguments->experts
              << ",\"hidden_width\":" << kHidden
              << ",\"latent_width\":" << kLatent
              << ",\"expert_intermediate_width\":" << kIntermediate
              << ",\"expert_payload_bytes\":" << loaded.value().logical_bytes
              << ",\"resident_capacity_bytes\":" << kResidentCapacity
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations
              << ",\"validation\":\"" << arguments->validation_name << "\""
              << ",\"profiler\":"
              << (arguments->profiler ? "true" : "false")
              << ",\"maximum_absolute_error\":"
              << observed_maximum_error
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"kernel_nanoseconds\":";
    if (arguments->profiler) {
        std::cout << profile_after.device_nanoseconds -
                         profile_before.device_nanoseconds;
    } else {
        std::cout << "null";
    }
    std::cout
              << ",\"activation_h2d_bytes\":"
              << runtime_after.activation_h2d_bytes -
                     runtime_before.activation_h2d_bytes
              << ",\"device_to_host_bytes\":"
              << runtime_after.device_to_host_bytes -
                     runtime_before.device_to_host_bytes
              << ",\"weight_h2d_bytes\":"
              << runtime_after.weight_h2d_bytes -
                     runtime_before.weight_h2d_bytes
              << ",\"stream_synchronization_count\":"
              << runtime_after.stream_synchronization_count -
                     runtime_before.stream_synchronization_count
              << ",\"cold_weight_h2d_bytes\":"
              << cold_runtime_after.weight_h2d_bytes -
                     cold_runtime_before.weight_h2d_bytes
              << ",\"cold_immutable_validation_scans\":"
              << cold_runtime_after.immutable_validation_scans -
                     cold_runtime_before.immutable_validation_scans
              << ",\"cold_immutable_validation_bytes\":"
              << cold_runtime_after.immutable_validation_bytes -
                     cold_runtime_before.immutable_validation_bytes
              << ",\"immutable_validation_scans\":"
              << runtime_after.immutable_validation_scans -
                     runtime_before.immutable_validation_scans
              << ",\"immutable_validation_hits\":"
              << runtime_after.immutable_validation_hits -
                     runtime_before.immutable_validation_hits
              << ",\"immutable_validation_bytes\":"
              << runtime_after.immutable_validation_bytes -
                     runtime_before.immutable_validation_bytes
              << ",\"immutable_validation_nanoseconds\":"
              << runtime_after.immutable_validation_nanoseconds -
                     runtime_before.immutable_validation_nanoseconds
              << ",\"resident_weight_bytes\":"
              << runtime_after.resident_weight_bytes
              << ",\"peak_resident_weight_bytes\":"
              << runtime_after.peak_resident_weight_bytes
              << ",\"oracle_peak_vram_bytes\":"
              << oracle_peak_vram_bytes
              << ",\"peak_vram_bytes\":"
              << std::max(oracle_peak_vram_bytes, memory.peak_device_bytes)
              << ",\"weight_cache_bypasses\":"
              << runtime_after.weight_cache_bypasses -
                     runtime_before.weight_cache_bypasses
              << ",\"resident_grid_calls\":"
              << runtime_after.resident_grid_calls -
                     runtime_before.resident_grid_calls
              << ",\"resident_grid_kernel_launches\":"
              << runtime_after.resident_grid_kernel_launches -
                     runtime_before.resident_grid_kernel_launches
              << ",\"resident_grid_fallbacks\":"
              << runtime_after.resident_grid_fallbacks -
                     runtime_before.resident_grid_fallbacks
              << ",\"resident_moe_layer_calls\":"
              << runtime_after.resident_moe_layer_calls -
                     runtime_before.resident_moe_layer_calls
              << ",\"resident_moe_layer_experts\":"
              << runtime_after.resident_moe_layer_experts -
                     runtime_before.resident_moe_layer_experts
              << ",\"resident_moe_layer_kernel_launches\":"
              << runtime_after.resident_moe_layer_kernel_launches -
                     runtime_before.resident_moe_layer_kernel_launches
              << ",\"resident_moe_layer_fallbacks\":"
              << runtime_after.resident_moe_layer_fallbacks -
                     runtime_before.resident_moe_layer_fallbacks
              << ",\"resident_moe_layer_contribution_h2d_bytes\":"
              << runtime_after.resident_moe_layer_contribution_h2d_bytes -
                     runtime_before.resident_moe_layer_contribution_h2d_bytes
              << "}\n";
    return 0;
}
