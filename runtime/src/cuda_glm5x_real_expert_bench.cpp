// 실제 GLM5X raw BF16 expert를 resident CUDA dense FFN으로 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/checksums.hpp"
#include "k3x/format.hpp"
#include "k3x/glm5x_bundle.hpp"
#include "k3x/glm5x_activation.hpp"
#include "k3x/routing_policy.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
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
    std::size_t workspace_bytes{};
    std::size_t resident_bytes{1ULL << 30};
    std::string precision{"fp32"};
    std::string output{"fp32"};
    std::string input_mode{"common"};
    std::filesystem::path input_bf16;
    std::filesystem::path expected_bf16;
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
        } else if (key == "--workspace-bytes") {
            const auto parsed = parse_size(value);
            if (!parsed) return std::nullopt;
            result.workspace_bytes = *parsed;
        } else if (key == "--resident-bytes") {
            const auto parsed = parse_size(value);
            if (!parsed || *parsed == 0) return std::nullopt;
            result.resident_bytes = *parsed;
        } else if (key == "--precision") {
            result.precision = value;
        } else if (key == "--output") {
            result.output = value;
        } else if (key == "--input-mode") {
            result.input_mode = value;
        } else if (key == "--input-bf16") {
            result.input_bf16 = value;
        } else if (key == "--expected-bf16") {
            result.expected_bf16 = value;
        } else {
            return std::nullopt;
        }
    }
    if (result.artifact_dir.empty() || result.precision == "" ||
        result.output == "") return std::nullopt;
    if (result.precision != "fp32" && result.precision != "bf16-rounded") {
        return std::nullopt;
    }
    if (result.output != "fp32" && result.output != "bf16") {
        return std::nullopt;
    }
    if (result.input_mode != "common" &&
        result.input_mode != "sparse-packed" &&
        result.input_mode != "expert-major" &&
        result.input_mode != "learned-expert-major" &&
        result.input_mode != "learned-moe-layer") {
        return std::nullopt;
    }
    if (result.precision != "bf16-rounded" && result.output != "fp32") {
        return std::nullopt;
    }
    if (!result.expected_bf16.empty() && result.input_bf16.empty()) {
        return std::nullopt;
    }
    return result;
}

struct RealExpertFixture {
    std::uint32_t expert_id{};
    k3x::GlmBf16ExpertLoad payload;
    std::array<std::vector<float>, 3> weights;
    k3x::DenseMlpView dense{};
    k3x::RawBf16MlpView raw{};
};

struct NamedTensorPayload {
    std::uint16_t dtype{};
    std::uint8_t rank{};
    std::array<std::uint64_t, 4> dimensions{};
    std::vector<std::byte> bytes;
};

struct SharedExpertFixture {
    std::array<NamedTensorPayload, 3> payload;
    std::array<std::vector<float>, 3> weights;
    k3x::RawBf16MlpView raw{};
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

std::vector<float> decode_fp32(std::span<const std::byte> bytes) {
    if (bytes.size() % sizeof(float) != 0) return {};
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

k3x::Result<NamedTensorPayload> load_named_tensor(
    std::span<const k3x::Reader* const> shards, std::string_view name) {
    const std::string owned_name(name);
    const auto tensor_id = k3x::fnv1a64(owned_name.c_str());
    const k3x::Reader* owner = nullptr;
    const k3x::TensorRecord* record = nullptr;
    for (const auto* shard : shards) {
        if (shard == nullptr) continue;
        const auto match = std::find_if(
            shard->tensors().begin(), shard->tensors().end(),
            [tensor_id](const k3x::TensorRecord& candidate) {
                return candidate.tensor_id == tensor_id;
            });
        if (match == shard->tensors().end()) continue;
        if (owner != nullptr) {
            return k3x::Result<NamedTensorPayload>::failure(
                k3x::ErrorCode::invalid_directory,
                "duplicate named GLM tensor");
        }
        owner = shard;
        record = &*match;
    }
    if (owner == nullptr || record == nullptr) {
        return k3x::Result<NamedTensorPayload>::failure(
            k3x::ErrorCode::tensor_not_found, owned_name);
    }
    if (record->quantization != 0 || record->logical_length != record->data_length ||
        record->auxiliary_length != 0 || record->auxiliary_offset != 0 ||
        record->auxiliary_crc32c != 0) {
        return k3x::Result<NamedTensorPayload>::failure(
            k3x::ErrorCode::invalid_directory, "invalid named GLM tensor");
    }
    const auto payload = owner->read_tensor(tensor_id);
    if (!payload) {
        return k3x::Result<NamedTensorPayload>::failure(
            payload.error(), payload.message());
    }
    if (payload.value().size() != record->data_length ||
        k3x::crc32c(payload.value()) != record->data_crc32c) {
        return k3x::Result<NamedTensorPayload>::failure(
            k3x::ErrorCode::data_crc_mismatch, owned_name);
    }
    NamedTensorPayload result;
    result.dtype = record->dtype;
    result.rank = record->rank;
    result.dimensions = record->dimensions;
    result.bytes = payload.value();
    return k3x::Result<NamedTensorPayload>::success(std::move(result));
}

bool has_shape(
    const NamedTensorPayload& tensor,
    std::initializer_list<std::uint64_t> expected) {
    if (tensor.rank != expected.size()) return false;
    std::size_t index = 0;
    for (const auto value : expected) {
        if (tensor.dimensions[index++] != value) return false;
    }
    return true;
}

std::vector<k3x::ExpertMajorTokenRoute> build_learned_routes(
    std::span<const float> input, std::size_t token_count,
    std::span<const float> router_weight,
    std::span<const float> correction_bias) {
    constexpr std::size_t kExpertCount = 256;
    constexpr std::size_t kTopK = 8;
    constexpr float kRoutedScale = 2.5F;
    if (input.size() != token_count * kHiddenSize ||
        router_weight.size() != kExpertCount * kHiddenSize ||
        correction_bias.size() != kExpertCount) {
        return {};
    }
    std::vector<k3x::ExpertMajorTokenRoute> routes(token_count);
    for (std::size_t token = 0; token < token_count; ++token) {
        std::vector<float> scores(kExpertCount);
        const auto token_input = input.subspan(token * kHiddenSize, kHiddenSize);
        for (std::size_t expert = 0; expert < kExpertCount; ++expert) {
            const auto weights = router_weight.subspan(expert * kHiddenSize,
                                                       kHiddenSize);
            float logit = 0.0F;
            for (std::size_t index = 0; index < kHiddenSize; ++index) {
                logit += token_input[index] * weights[index];
            }
            scores[expert] = 1.0F / (1.0F + std::exp(-logit));
        }
        const auto decision = k3x::select_routing(
            scores, correction_bias, kTopK,
            k3x::RoutingPolicyConfig{.mode = k3x::RoutingMode::natural});
        if (!decision) return {};
        auto& route = routes[token];
        route.expert_ids.reserve(kTopK);
        route.contributions.reserve(kTopK);
        for (std::size_t slot = 0; slot < decision.value().selected_k; ++slot) {
            route.expert_ids.push_back(static_cast<std::uint32_t>(
                decision.value().expert_ids[slot]));
            route.contributions.push_back(
                decision.value().normalized_weights[slot] * kRoutedScale);
        }
    }
    return routes;
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
                     "[--workspace-bytes N] [--resident-bytes N] [--precision fp32|bf16-rounded] "
                     "[--output fp32|bf16] [--input-mode common|sparse-packed|expert-major|learned-expert-major|learned-moe-layer] "
                     "[--input-bf16 FILE] [--expected-bf16 FILE]\n";
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
    const bool sparse_packed = arguments->input_mode == "sparse-packed";
    const bool expert_major = arguments->input_mode == "expert-major";
    const bool learned_expert_major =
        arguments->input_mode == "learned-expert-major";
    const bool learned_moe_layer =
        arguments->input_mode == "learned-moe-layer";
    const bool learned_route_mode = learned_expert_major || learned_moe_layer;
    if (learned_route_mode &&
        (arguments->precision != "bf16-rounded" || arguments->tokens == 0)) {
        std::cerr << "learned GLM modes require --precision bf16-rounded\n";
        return 7;
    }
    const auto logical_token_count = arguments->tokens;
    const auto grid_token_count = sparse_packed ? 1 : arguments->tokens;
    const auto input_value_count = logical_token_count * kHiddenSize;
    std::vector<float> input(input_value_count);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 29) - 14) * 0.01F;
        if (arguments->precision == "bf16-rounded") input[index] = round_to_bf16(input[index]);
    }
    if (!arguments->input_bf16.empty()) {
        if (arguments->precision != "bf16-rounded") {
            std::cerr << "--input-bf16 requires --precision bf16-rounded\n";
            return 7;
        }
        const auto activation = k3x::load_glm5x_bf16_activation(
            arguments->input_bf16, static_cast<std::uint32_t>(arguments->tokens),
            static_cast<std::uint32_t>(kHiddenSize));
        if (!activation) {
            std::cerr << k3x::error_code_name(activation.error()) << ": "
                      << activation.message() << '\n';
            return 7;
        }
        input = decode_bf16(activation.value().payload);
        if (input.size() != input_value_count) {
            std::cerr << "activation input shape mismatch\n";
            return 7;
        }
    }
    std::optional<std::vector<float>> expected_output;
    if (!arguments->expected_bf16.empty()) {
        const auto expected = k3x::load_glm5x_bf16_activation(
            arguments->expected_bf16,
            static_cast<std::uint32_t>(arguments->tokens),
            static_cast<std::uint32_t>(kHiddenSize));
        if (!expected) {
            std::cerr << k3x::error_code_name(expected.error()) << ": "
                      << expected.message() << '\n';
            return 7;
        }
        expected_output = decode_bf16(expected.value().payload);
        if (expected_output->size() != input_value_count) {
            std::cerr << "activation expected shape mismatch\n";
            return 7;
        }
    }
    std::vector<k3x::ExpertMajorTokenRoute> learned_routes;
    std::vector<std::uint32_t> expert_ids;
    std::uint64_t router_payload_bytes = 0;
    const auto host_start = std::chrono::steady_clock::now();
    if (learned_route_mode) {
        const auto router = load_named_tensor(
            shard_views,
            "model.layers." + std::to_string(arguments->layer) +
                ".mlp.gate.weight");
        const auto bias = load_named_tensor(
            shard_views,
            "model.layers." + std::to_string(arguments->layer) +
                ".mlp.gate.e_score_correction_bias");
        if (!router || !bias || router.value().dtype != 3 ||
            bias.value().dtype != 1 ||
            !has_shape(router.value(), {256, kHiddenSize}) ||
            !has_shape(bias.value(), {256})) {
            std::cerr << "invalid GLM router tensors\n";
            return 5;
        }
        const auto router_weight = decode_bf16(router.value().bytes);
        const auto correction_bias = decode_fp32(bias.value().bytes);
        router_payload_bytes = router.value().bytes.size() + bias.value().bytes.size();
        learned_routes = build_learned_routes(
            input, arguments->tokens, router_weight, correction_bias);
        if (learned_routes.size() != arguments->tokens) {
            std::cerr << "GLM router selection failed\n";
            return 5;
        }
        std::set<std::uint32_t> selected;
        for (const auto& route : learned_routes) {
            selected.insert(route.expert_ids.begin(), route.expert_ids.end());
        }
        expert_ids.assign(selected.begin(), selected.end());
        for (const auto expert_id : expert_ids) {
            if (!available_experts.contains(expert_id)) {
                std::cerr << "selected GLM expert is absent from bounded shards\n";
                return 5;
            }
        }
    } else if (arguments->experts == 1) {
        if (!available_experts.contains(arguments->expert)) return 5;
        expert_ids.push_back(arguments->expert);
    } else {
        if (available_experts.size() < arguments->experts) return 5;
        expert_ids.assign(available_experts.begin(), available_experts.end());
        expert_ids.resize(arguments->experts);
    }
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
        const auto needs_float_view = arguments->precision == "fp32" ||
            expert_major || learned_route_mode ||
            expert_id == expert_ids.back();
        if (needs_float_view) {
            for (std::size_t index = 0; index < fixture.weights.size(); ++index) {
                fixture.weights[index] = decode_bf16(fixture.payload.roles[index]);
                if (fixture.weights[index].empty()) return 7;
                if (arguments->precision == "bf16-rounded") {
                    for (auto& value : fixture.weights[index]) {
                        value = round_to_bf16(value);
                    }
                }
            }
        }
        const auto prefix = "model.layers." + std::to_string(arguments->layer) +
            ".mlp.experts." + std::to_string(expert_id) + ".";
        const auto gate_id = k3x::fnv1a64(
            (prefix + "gate_proj.weight").c_str());
        const auto up_id = k3x::fnv1a64(
            (prefix + "up_proj.weight").c_str());
        const auto down_id = k3x::fnv1a64(
            (prefix + "down_proj.weight").c_str());
        fixture.raw = {
            {gate_id, fixture.payload.roles[0], kIntermediateSize, kHiddenSize},
            {up_id, fixture.payload.roles[1], kIntermediateSize, kHiddenSize},
            {down_id, fixture.payload.roles[2], kHiddenSize, kIntermediateSize},
        };
        if (needs_float_view) {
            fixture.dense = {
                {gate_id, fixture.weights[0], kIntermediateSize, kHiddenSize},
                {up_id, fixture.weights[1], kIntermediateSize, kHiddenSize},
                {down_id, fixture.weights[2], kHiddenSize, kIntermediateSize},
            };
        }
        fixtures.push_back(std::move(fixture));
    }
    SharedExpertFixture shared_fixture;
    std::uint64_t shared_payload_bytes = 0;
    if (learned_moe_layer) {
        const auto layer_prefix =
            "model.layers." + std::to_string(arguments->layer) +
            ".mlp.shared_experts.";
        const std::array names{
            layer_prefix + "gate_proj.weight",
            layer_prefix + "up_proj.weight",
            layer_prefix + "down_proj.weight",
        };
        const std::array<std::array<std::uint64_t, 2>, 3> shapes{{
            {kIntermediateSize, kHiddenSize},
            {kIntermediateSize, kHiddenSize},
            {kHiddenSize, kIntermediateSize},
        }};
        for (std::size_t index = 0; index < names.size(); ++index) {
            auto payload = load_named_tensor(shard_views, names[index]);
            if (!payload || payload.value().dtype != 3 ||
                !has_shape(payload.value(),
                           {shapes[index][0], shapes[index][1]})) {
                std::cerr << "invalid GLM shared expert tensor\n";
                return 6;
            }
            shared_payload_bytes += payload.value().bytes.size();
            shared_fixture.payload[index] = std::move(payload.value());
            shared_fixture.weights[index] =
                decode_bf16(shared_fixture.payload[index].bytes);
            if (shared_fixture.weights[index].empty()) return 7;
            for (auto& value : shared_fixture.weights[index]) {
                value = round_to_bf16(value);
            }
        }
        const auto gate_id = k3x::fnv1a64(names[0].c_str());
        const auto up_id = k3x::fnv1a64(names[1].c_str());
        const auto down_id = k3x::fnv1a64(names[2].c_str());
        shared_fixture.raw = {
            {gate_id, shared_fixture.payload[0].bytes,
             kIntermediateSize, kHiddenSize},
            {up_id, shared_fixture.payload[1].bytes,
             kIntermediateSize, kHiddenSize},
            {down_id, shared_fixture.payload[2].bytes,
             kHiddenSize, kIntermediateSize},
        };
        shared_fixture.dense = {
            {gate_id, shared_fixture.weights[0], kIntermediateSize,
             kHiddenSize},
            {up_id, shared_fixture.weights[1], kIntermediateSize,
             kHiddenSize},
            {down_id, shared_fixture.weights[2], kHiddenSize,
             kIntermediateSize},
        };
    }
    const auto host_load_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - host_start).count());
    if (sparse_packed &&
        (arguments->precision != "bf16-rounded" || arguments->tokens != 2)) {
        std::cerr << "sparse-packed requires --precision bf16-rounded --tokens 2\n";
        return 7;
    }
    if (expert_major &&
        (arguments->precision != "bf16-rounded" || arguments->tokens != 2 ||
         fixtures.size() < 2)) {
        std::cerr << "expert-major requires --precision bf16-rounded --tokens 2 "
                     "and at least two experts\n";
        return 7;
    }
    if (learned_route_mode &&
        (arguments->precision != "bf16-rounded" || arguments->tokens == 0 ||
         fixtures.size() < 2)) {
        std::cerr << "learned GLM modes require BF16 precision and at least "
                     "two selected experts\n";
        return 7;
    }
    std::vector<k3x::RawBf16MlpView> raw_views;
    raw_views.reserve(fixtures.size());
    for (const auto& fixture : fixtures) {
        raw_views.push_back(fixture.raw);
    }
    std::optional<k3x::ExpertMajorPackedPlan> expert_major_plan;
    std::vector<k3x::RawBf16MlpView> expert_major_views;
    if (expert_major || learned_route_mode) {
        std::vector<k3x::ExpertMajorTokenRoute> routes;
        if (learned_route_mode) {
            routes = learned_routes;
        } else {
            routes.resize(arguments->tokens);
            for (std::size_t token = 0; token < arguments->tokens; ++token) {
                auto& route = routes[token];
                for (std::size_t expert_index = 0;
                     expert_index < fixtures.size(); ++expert_index) {
                    const bool shared = expert_index < 2;
                    const bool alternating = expert_index >= 2 &&
                        ((expert_index + token) % arguments->tokens == 0);
                    if (shared || alternating) {
                        route.expert_ids.push_back(expert_ids[expert_index]);
                    }
                }
                if (route.expert_ids.empty()) return 7;
                route.contributions.assign(
                    route.expert_ids.size(),
                    1.0F / static_cast<float>(route.expert_ids.size()));
            }
        }
        auto built = k3x::build_expert_major_packed_plan(
            input, arguments->tokens, kHiddenSize, routes);
        if (!built) {
            std::cerr << k3x::error_code_name(built.error()) << ": "
                      << built.message() << '\n';
            return 8;
        }
        expert_major_plan = std::move(built.value());
        expert_major_views.reserve(expert_major_plan->groups.size());
        for (const auto& group : expert_major_plan->groups) {
            const auto fixture = std::find_if(
                fixtures.begin(), fixtures.end(),
                [&](const RealExpertFixture& item) {
                    return item.expert_id == group.expert_id;
                });
            if (fixture == fixtures.end()) return 8;
            expert_major_views.push_back(fixture->raw);
        }
    }
    std::vector<float> packed_input;
    if (sparse_packed) {
        packed_input.resize(fixtures.size() * kHiddenSize);
        for (std::size_t expert = 0; expert < fixtures.size(); ++expert) {
            const auto token = expert % logical_token_count;
            std::copy_n(input.begin() + token * kHiddenSize, kHiddenSize,
                        packed_input.begin() + expert * kHiddenSize);
        }
    }
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = arguments->precision == "bf16-rounded"
        ? k3x::DensePrecision::bf16_rounded : k3x::DensePrecision::fp32;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    const auto use_grid = arguments->precision == "bf16-rounded";
    options.cuda_batching = use_grid
        ? k3x::CudaBatchingMode::resident_grid : k3x::CudaBatchingMode::scalar;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_cublas_workspace_bytes = arguments->workspace_bytes;
    options.cuda_bf16_output = arguments->output == "bf16"
        ? k3x::CudaBf16OutputMode::bf16 : k3x::CudaBf16OutputMode::fp32;
    options.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    options.cuda_resident_bytes = arguments->resident_bytes;
    auto cuda = k3x::make_cuda_backend(options);
    if (!cuda) {
        std::cerr << k3x::error_code_name(cuda.error()) << ": "
                  << cuda.message() << '\n';
        return 7;
    }
    auto cpu = k3x::make_cpu_backend();
    std::vector<float> cpu_reference;
    cpu_reference.reserve(grid_token_count * kHiddenSize);
    const auto reference_token_count = sparse_packed ? 1 : arguments->tokens;
    if (expert_major || learned_route_mode) {
        cpu_reference.assign(arguments->tokens * kHiddenSize, 0.0F);
        for (const auto& group : expert_major_plan->groups) {
            const auto fixture = std::find_if(
                fixtures.begin(), fixtures.end(),
                [&](const RealExpertFixture& item) {
                    return item.expert_id == group.expert_id;
                });
            if (fixture == fixtures.end()) return 8;
            for (const auto& assignment : group.assignments) {
                const auto token_input = std::span<const float>(input).subspan(
                    assignment.token_index * kHiddenSize, kHiddenSize);
                const auto reference = cpu->dense_situ_mlp(
                    token_input, fixture->dense, 1.0F, std::nullopt,
                    arguments->layer, k3x::ProfilePhase::decode);
                if (!reference) return 8;
                auto output = std::span<float>(cpu_reference).subspan(
                    assignment.token_index * kHiddenSize, kHiddenSize);
                for (std::size_t value = 0; value < output.size(); ++value) {
                    output[value] += assignment.contribution *
                        reference.value()[value];
                }
            }
        }
        if (learned_moe_layer) {
            for (std::size_t token = 0; token < arguments->tokens; ++token) {
                const auto token_input = std::span<const float>(input).subspan(
                    token * kHiddenSize, kHiddenSize);
                const auto shared = cpu->dense_situ_mlp(
                    token_input, shared_fixture.dense, 1.0F, std::nullopt,
                    arguments->layer, k3x::ProfilePhase::decode);
                if (!shared) return 8;
                auto output = std::span<float>(cpu_reference).subspan(
                    token * kHiddenSize, kHiddenSize);
                for (std::size_t value = 0; value < output.size(); ++value) {
                    output[value] += shared.value()[value];
                }
            }
        }
    } else for (std::size_t token = 0; token < reference_token_count; ++token) {
        const auto logical_token = sparse_packed
            ? ((fixtures.size() - 1) % logical_token_count) : token;
        const auto token_input = std::span<const float>(input).subspan(
            logical_token * kHiddenSize, kHiddenSize);
        const auto reference = cpu->dense_situ_mlp(
            token_input, fixtures.back().dense, 1.0F, std::nullopt,
            arguments->layer, k3x::ProfilePhase::decode);
        if (!reference) return 8;
        cpu_reference.insert(cpu_reference.end(), reference.value().begin(),
                             reference.value().end());
    }
    const auto execute = [&]() {
        if (learned_moe_layer) {
            auto routed = cuda.value()->raw_bf16_situ_mlp_expert_major(
                input, arguments->tokens, *expert_major_plan,
                expert_major_views, 1.0F, std::nullopt, arguments->layer,
                k3x::ProfilePhase::decode);
            if (!routed) return routed;
            const std::array<k3x::RawBf16MlpView, 1> shared_views{
                shared_fixture.raw};
            auto shared = cuda.value()->raw_bf16_situ_mlp_grid(
                input, arguments->tokens, shared_views, 1.0F, std::nullopt,
                arguments->layer, k3x::ProfilePhase::decode);
            if (!shared) {
                return k3x::Result<std::vector<float>>::failure(
                    shared.error(), shared.message());
            }
            if (shared.value().size() != 1 ||
                shared.value().front().size() != routed.value().size()) {
                return k3x::Result<std::vector<float>>::failure(
                    k3x::ErrorCode::invalid_extent,
                    "shared GLM expert output shape mismatch");
            }
            for (std::size_t index = 0; index < routed.value().size(); ++index) {
                routed.value()[index] += shared.value().front()[index];
            }
            return routed;
        }
        if (expert_major || learned_expert_major) {
            return cuda.value()->raw_bf16_situ_mlp_expert_major(
                input, arguments->tokens, *expert_major_plan,
                expert_major_views, 1.0F, std::nullopt, arguments->layer,
                k3x::ProfilePhase::decode);
        }
        if (use_grid) {
            auto outputs = sparse_packed
                ? cuda.value()->raw_bf16_situ_mlp_grid_packed(
                      packed_input, grid_token_count, raw_views, 1.0F,
                      std::nullopt, arguments->layer,
                      k3x::ProfilePhase::decode)
                : cuda.value()->raw_bf16_situ_mlp_grid(
                      input, grid_token_count, raw_views, 1.0F, std::nullopt,
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
    std::optional<float> expected_abs;
    std::optional<float> expected_rel;
    if (expected_output) {
        expected_abs = maximum_absolute_difference(actual, *expected_output);
        const auto expected_scale = std::max(
            maximum_absolute_value(*expected_output), 1.0e-6F);
        expected_rel = *expected_abs / expected_scale;
    }
    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\""
              << ((expert_major || learned_route_mode)
                      ? (learned_moe_layer
                             ? "glm5.2_real_bf16_learned_moe_layer"
                             : (learned_expert_major
                                    ? "glm5.2_real_bf16_learned_expert_major"
                                    : "glm5.2_real_bf16_expert_major"))
                      : "glm5.2_real_bf16_expert")
              << "\""
              << ",\"layer_id\":" << arguments->layer
              << ",\"expert_id\":" << expert_ids.front()
              << ",\"expert_count\":" << expert_ids.size()
              << ",\"token_count\":" << arguments->tokens
              << ",\"last_expert_id\":" << expert_ids.back()
              << ",\"shard_count\":" << paths.size()
              << ",\"precision\":\"" << arguments->precision << "\""
              << ",\"output\":\"" << arguments->output << "\""
              << ",\"input_mode\":\"" << arguments->input_mode << "\""
              << ",\"route_group_count\":"
              << ((expert_major || learned_route_mode)
                      ? expert_major_plan->groups.size() : 0)
              << ",\"route_assignment_count\":"
              << ((expert_major || learned_route_mode)
                      ? expert_major_plan->assignment_count : 0)
              << ",\"grid_token_count\":" << grid_token_count
              << ",\"cublas_workspace_bytes\":"
              << arguments->workspace_bytes
              << ",\"resident_budget_bytes\":"
              << arguments->resident_bytes
              << ",\"host_payload_bytes\":" << host_payload_bytes
              << ",\"router_payload_bytes\":" << router_payload_bytes
              << ",\"shared_payload_bytes\":" << shared_payload_bytes
              << ",\"host_load_nanoseconds\":" << host_load_ns
              << ",\"cold_latency_nanoseconds\":" << cold_ns
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"gpu_cpu_max_absolute_error\":" << cpu_abs
              << ",\"gpu_cpu_max_relative_error\":" << cpu_abs / cpu_scale
              << ",\"expected_max_absolute_error\":";
    if (expected_abs) {
        std::cout << *expected_abs;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"expected_max_relative_error\":";
    if (expected_rel) {
        std::cout << *expected_rel;
    } else {
        std::cout << "null";
    }
    std::cout
              << ",\"cold_weight_h2d_bytes\":"
              << stats_after_cold.weight_h2d_bytes
              << ",\"warm_weight_h2d_bytes\":"
              << (stats_after.weight_h2d_bytes - stats_before.weight_h2d_bytes)
              << ",\"resident_weight_bytes\":" << stats_after.resident_weight_bytes
              << ",\"resident_grid_calls\":"
              << stats_after.resident_grid_calls
              << ",\"resident_grid_kernel_launches\":"
              << stats_after.resident_grid_kernel_launches
              << ",\"resident_grid_descriptor_h2d_bytes\":"
              << stats_after.resident_grid_descriptor_h2d_bytes
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations << "}\n";
    return 0;
}
