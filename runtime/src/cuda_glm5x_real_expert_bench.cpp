// 실제 GLM5X raw BF16 expert를 resident CUDA dense FFN으로 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/checksums.hpp"
#include "k3x/format.hpp"
#include "k3x/glm5x_bundle.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kHiddenSize = 6144;
constexpr std::size_t kIntermediateSize = 2048;

struct Arguments {
    std::filesystem::path artifact_dir;
    std::uint32_t layer{};
    std::uint32_t expert{};
    std::size_t experts{1};
    std::size_t tokens{1};
    std::size_t warmup{2};
    std::size_t iterations{10};
    std::string precision{"fp32"};
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

std::optional<std::uint32_t> parse_u32(std::string_view text) {
    const auto value = parse_size(text);
    if (!value || *value > 0xffffffffULL) return std::nullopt;
    return static_cast<std::uint32_t>(*value);
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) return std::nullopt;
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--artifact-dir") {
            result.artifact_dir = value;
        } else if (key == "--layer") {
            const auto parsed = parse_u32(value);
            if (!parsed) return std::nullopt;
            result.layer = *parsed;
        } else if (key == "--expert") {
            const auto parsed = parse_u32(value);
            if (!parsed) return std::nullopt;
            result.expert = *parsed;
        } else if (key == "--experts") {
            const auto parsed = parse_size(value);
            if (!parsed || *parsed == 0) return std::nullopt;
            result.experts = *parsed;
        } else if (key == "--tokens") {
            const auto parsed = parse_size(value);
            if (!parsed || *parsed == 0 || *parsed > 65535) {
                return std::nullopt;
            }
            result.tokens = *parsed;
        } else if (key == "--warmup") {
            const auto parsed = parse_size(value);
            if (!parsed) return std::nullopt;
            result.warmup = *parsed;
        } else if (key == "--iterations") {
            const auto parsed = parse_size(value);
            if (!parsed || *parsed == 0) return std::nullopt;
            result.iterations = *parsed;
        } else if (key == "--precision") {
            result.precision = value;
        } else {
            return std::nullopt;
        }
    }
    if (result.artifact_dir.empty() || result.precision == "") return std::nullopt;
    if (result.precision != "fp32" && result.precision != "bf16-rounded") {
        return std::nullopt;
    }
    return result;
}

struct RealExpertFixture {
    std::uint32_t expert_id{};
    k3x::GlmBf16ExpertLoad payload;
    std::array<std::vector<float>, 3> weights;
    k3x::DenseMlpView dense{};
};

float bf16_to_float(const std::byte* bytes) {
    const auto low = std::to_integer<std::uint8_t>(bytes[0]);
    const auto high = std::to_integer<std::uint8_t>(bytes[1]);
    const std::uint32_t bits =
        (static_cast<std::uint32_t>(high) << 24U) |
        (static_cast<std::uint32_t>(low) << 16U);
    float value{};
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float round_to_bf16(float value) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    bits &= 0xffff0000U;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> decode_bf16(std::span<const std::byte> bytes) {
    if (bytes.size() % 2 != 0) return {};
    std::vector<float> values(bytes.size() / 2);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] = bf16_to_float(bytes.data() + index * 2);
    }
    return values;
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2
        ? values[middle]
        : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

float maximum_absolute_difference(
    std::span<const float> left, std::span<const float> right) {
    if (left.size() != right.size()) return INFINITY;
    float result = 0.0F;
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

float maximum_absolute_value(std::span<const float> values) {
    float result = 0.0F;
    for (const auto value : values) result = std::max(result, std::abs(value));
    return result;
}

std::vector<std::filesystem::path> artifact_paths(
    const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> result;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".k3x") {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments || !std::filesystem::is_directory(arguments->artifact_dir)) {
        std::cerr << "usage: --artifact-dir DIR --layer N --expert N "
                     "[--experts N] [--tokens N] [--warmup N] [--iterations N] "
                     "[--precision fp32|bf16-rounded]\n";
        return 2;
    }
    const auto paths = artifact_paths(arguments->artifact_dir);
    if (paths.empty()) return 3;
    k3x::ReaderOptions reader_options;
    reader_options.verify = k3x::VerifyMode::metadata_only;
    std::vector<k3x::Reader> readers;
    readers.reserve(paths.size());
    for (const auto& path : paths) {
        auto reader = k3x::Reader::open(path, reader_options);
        if (!reader) {
            std::cerr << k3x::error_code_name(reader.error()) << ": "
                      << reader.message() << '\n';
            return 4;
        }
        readers.push_back(std::move(reader.value()));
    }
    std::vector<const k3x::Reader*> shard_views;
    shard_views.reserve(readers.size());
    for (const auto& reader : readers) shard_views.push_back(&reader);
    std::set<std::uint32_t> available_experts;
    for (const auto& reader : readers) {
        for (const auto& record : reader.tensors()) {
            if (record.layer_id == static_cast<std::int32_t>(arguments->layer) &&
                record.expert_id >= 0) {
                available_experts.insert(static_cast<std::uint32_t>(record.expert_id));
            }
        }
    }
    std::vector<std::uint32_t> expert_ids;
    if (arguments->experts == 1) {
        if (!available_experts.contains(arguments->expert)) return 5;
        expert_ids.push_back(arguments->expert);
    } else {
        if (available_experts.size() < arguments->experts) return 5;
        expert_ids.assign(available_experts.begin(), available_experts.end());
        expert_ids.resize(arguments->experts);
    }
    const auto host_start = std::chrono::steady_clock::now();
    std::vector<RealExpertFixture> fixtures;
    fixtures.reserve(expert_ids.size());
    std::uint64_t host_payload_bytes = 0;
    for (const auto expert_id : expert_ids) {
        auto loaded = k3x::load_glm5x_bf16_expert(
            shard_views, arguments->layer, expert_id,
            kHiddenSize, kIntermediateSize);
        if (!loaded) {
            std::cerr << k3x::error_code_name(loaded.error()) << ": "
                      << loaded.message() << '\n';
            return 6;
        }
        RealExpertFixture fixture;
        fixture.expert_id = expert_id;
        fixture.payload = std::move(loaded.value());
        host_payload_bytes += fixture.payload.payload_bytes;
        for (std::size_t index = 0; index < fixture.weights.size(); ++index) {
            fixture.weights[index] = decode_bf16(fixture.payload.roles[index]);
            if (fixture.weights[index].empty()) return 7;
            if (arguments->precision == "bf16-rounded") {
                for (auto& value : fixture.weights[index]) value = round_to_bf16(value);
            }
        }
        const auto prefix = "model.layers." + std::to_string(arguments->layer) +
            ".mlp.experts." + std::to_string(expert_id) + ".";
        fixture.dense = {
            {k3x::fnv1a64((prefix + "gate_proj.weight").c_str()), fixture.weights[0],
             kIntermediateSize, kHiddenSize},
            {k3x::fnv1a64((prefix + "up_proj.weight").c_str()), fixture.weights[1],
             kIntermediateSize, kHiddenSize},
            {k3x::fnv1a64((prefix + "down_proj.weight").c_str()), fixture.weights[2],
             kHiddenSize, kIntermediateSize},
        };
        fixtures.push_back(std::move(fixture));
    }
    const auto host_load_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - host_start).count());
    const auto input_value_count = arguments->tokens * kHiddenSize;
    std::vector<float> input(input_value_count);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.01F;
        if (arguments->precision == "bf16-rounded") input[index] = round_to_bf16(input[index]);
    }
    std::vector<k3x::DenseMlpView> dense_views;
    dense_views.reserve(fixtures.size());
    for (const auto& fixture : fixtures) dense_views.push_back(fixture.dense);
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = arguments->precision == "bf16-rounded"
        ? k3x::DensePrecision::bf16_rounded : k3x::DensePrecision::fp32;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    const auto use_grid = arguments->precision == "bf16-rounded" &&
        (fixtures.size() > 1 || arguments->tokens > 1);
    options.cuda_batching = use_grid
        ? k3x::CudaBatchingMode::resident_grid : k3x::CudaBatchingMode::scalar;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    options.cuda_resident_bytes = 1ULL << 30;
    auto cuda = k3x::make_cuda_backend(options);
    if (!cuda) {
        std::cerr << k3x::error_code_name(cuda.error()) << ": "
                  << cuda.message() << '\n';
        return 7;
    }
    auto cpu = k3x::make_cpu_backend();
    std::vector<float> cpu_reference;
    cpu_reference.reserve(arguments->tokens * kHiddenSize);
    for (std::size_t token = 0; token < arguments->tokens; ++token) {
        const auto token_input = std::span<const float>(input).subspan(
            token * kHiddenSize, kHiddenSize);
        const auto reference = cpu->dense_situ_mlp(
            token_input, fixtures.back().dense, 1.0F, std::nullopt,
            arguments->layer, k3x::ProfilePhase::decode);
        if (!reference) return 8;
        cpu_reference.insert(cpu_reference.end(), reference.value().begin(),
                             reference.value().end());
    }
    const auto execute = [&]() {
        if (use_grid) {
            auto outputs = cuda.value()->dense_situ_mlp_grid(
                input, arguments->tokens, dense_views, 1.0F, std::nullopt,
                arguments->layer, k3x::ProfilePhase::decode);
            if (!outputs) {
                return k3x::Result<std::vector<float>>::failure(
                    outputs.error(), outputs.message());
            }
            return k3x::Result<std::vector<float>>::success(
                std::move(outputs.value().back()));
        }
        std::vector<float> last;
        last.reserve(arguments->tokens * kHiddenSize);
        for (std::size_t token = 0; token < arguments->tokens; ++token) {
            const auto token_input = std::span<const float>(input).subspan(
                token * kHiddenSize, kHiddenSize);
            k3x::Result<std::vector<float>> token_output =
                k3x::Result<std::vector<float>>::success({});
            for (const auto& fixture : fixtures) {
                token_output = cuda.value()->dense_situ_mlp(
                    token_input, fixture.dense, 1.0F, std::nullopt,
                    arguments->layer, k3x::ProfilePhase::decode);
                if (!token_output) return token_output;
            }
            last.insert(last.end(), token_output.value().begin(),
                        token_output.value().end());
        }
        return k3x::Result<std::vector<float>>::success(std::move(last));
    };
    const auto cold_start = std::chrono::steady_clock::now();
    const auto cold = execute();
    const auto cold_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - cold_start).count());
    if (!cold) {
        std::cerr << k3x::error_code_name(cold.error()) << ": "
                  << cold.message() << '\n';
        return 9;
    }
    const auto stats_after_cold = cuda.value()->runtime_stats();
    for (std::size_t index = 0; index < arguments->warmup; ++index) {
        if (!execute()) return 10;
    }
    const auto stats_before = cuda.value()->runtime_stats();
    std::vector<std::uint64_t> samples;
    samples.reserve(arguments->iterations);
    std::vector<float> actual;
    for (std::size_t index = 0; index < arguments->iterations; ++index) {
        const auto started = std::chrono::steady_clock::now();
        const auto result = execute();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
        if (!result) return 11;
        actual = result.value();
        samples.push_back(elapsed);
    }
    const auto stats_after = cuda.value()->runtime_stats();
    const auto cpu_abs = maximum_absolute_difference(actual, cpu_reference);
    const auto cpu_scale = std::max(maximum_absolute_value(cpu_reference), 1.0e-6F);
    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"glm5.2_real_bf16_expert\""
              << ",\"layer_id\":" << arguments->layer
              << ",\"expert_id\":" << expert_ids.front()
              << ",\"expert_count\":" << expert_ids.size()
              << ",\"token_count\":" << arguments->tokens
              << ",\"last_expert_id\":" << expert_ids.back()
              << ",\"shard_count\":" << paths.size()
              << ",\"precision\":\"" << arguments->precision << "\""
              << ",\"host_payload_bytes\":" << host_payload_bytes
              << ",\"host_load_nanoseconds\":" << host_load_ns
              << ",\"cold_latency_nanoseconds\":" << cold_ns
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"gpu_cpu_max_absolute_error\":" << cpu_abs
              << ",\"gpu_cpu_max_relative_error\":" << cpu_abs / cpu_scale
              << ",\"cold_weight_h2d_bytes\":"
              << stats_after_cold.weight_h2d_bytes
              << ",\"warm_weight_h2d_bytes\":"
              << (stats_after.weight_h2d_bytes - stats_before.weight_h2d_bytes)
              << ",\"resident_weight_bytes\":" << stats_after.resident_weight_bytes
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations << "}\n";
    return 0;
}
