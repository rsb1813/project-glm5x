// 실제 GLM5X raw BF16 expert를 resident CUDA dense FFN으로 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/checksums.hpp"
#include "k3x/format.hpp"
#include "k3x/glm5x_bundle.hpp"
#include "k3x/glm5x_activation.hpp"
#include "k3x/glm5x_runtime_index.hpp"
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
#include <functional>
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
constexpr std::size_t kAttentionHeads = 64;
constexpr std::size_t kQRank = 2048;
constexpr std::size_t kKvRank = 512;
constexpr std::size_t kQkNopeDim = 192;
constexpr std::size_t kRopeDim = 64;
constexpr std::size_t kValueDim = 256;
constexpr std::size_t kIndexerHeads = 32;
constexpr std::size_t kIndexerHeadDim = 128;
constexpr double kRopeTheta = 8000000.0;

struct Arguments {
    std::filesystem::path artifact_dir;
    std::filesystem::path runtime_index;
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
    bool device_accumulate{false};
    bool fuse_shared{false};
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
        } else if (key == "--runtime-index") {
            result.runtime_index = value;
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
        } else if (key == "--device-accumulate") {
            if (value != "0" && value != "1" && value != "false" &&
                value != "true") {
                return std::nullopt;
            }
            result.device_accumulate = value == "1" || value == "true";
        } else if (key == "--fuse-shared") {
            if (value != "0" && value != "1" && value != "false" &&
                value != "true") {
                return std::nullopt;
            }
            result.fuse_shared = value == "1" || value == "true";
        } else if (key == "--input-bf16") {
            result.input_bf16 = value;
        } else if (key == "--expected-bf16") {
            result.expected_bf16 = value;
        } else {
            return std::nullopt;
        }
    }
    if (result.artifact_dir.empty() == result.runtime_index.empty() ||
        result.precision == "" ||
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
        result.input_mode != "learned-moe-layer" &&
        result.input_mode != "learned-decoder-layer") {
        return std::nullopt;
    }
    if (result.precision != "bf16-rounded" && result.output != "fp32") {
        return std::nullopt;
    }
    if (!result.expected_bf16.empty() && result.input_bf16.empty()) {
        return std::nullopt;
    }
    if (result.fuse_shared &&
        ((result.input_mode != "learned-moe-layer" &&
          result.input_mode != "learned-decoder-layer") ||
         !result.device_accumulate || result.output != "fp32")) {
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
    std::uint64_t tensor_id{};
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
    result.tensor_id = tensor_id;
    result.rank = record->rank;
    result.dimensions = record->dimensions;
    result.bytes = payload.value();
    return k3x::Result<NamedTensorPayload>::success(std::move(result));
}

k3x::Result<NamedTensorPayload> load_named_tensor(
    const k3x::Glm5xRuntimeIndex& runtime_index,
    std::string_view name) {
    const std::string owned_name(name);
    const auto tensor_id = k3x::fnv1a64(owned_name.c_str());
    auto loaded = runtime_index.read_tensor_with_metadata(tensor_id);
    if (!loaded) {
        return k3x::Result<NamedTensorPayload>::failure(
            loaded.error(), loaded.message());
    }
    const auto& record = loaded.value().record;
    if (record.quantization != 0 ||
        record.logical_length != record.data_length ||
        record.auxiliary_length != 0 || record.auxiliary_offset != 0 ||
        record.auxiliary_crc32c != 0) {
        return k3x::Result<NamedTensorPayload>::failure(
            k3x::ErrorCode::invalid_directory, "invalid named GLM tensor");
    }
    NamedTensorPayload result;
    result.tensor_id = tensor_id;
    result.dtype = record.dtype;
    result.rank = record.rank;
    result.dimensions = record.dimensions;
    result.bytes = std::move(loaded.value().payload);
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

struct DecodedBf16Tensor {
    std::uint64_t tensor_id{};
    std::size_t rows{};
    std::size_t cols{};
    std::vector<float> values;

    k3x::DenseWeightView matrix() const {
        return {tensor_id, values, rows, cols};
    }
};

struct DecoderLayerWeights {
    DecodedBf16Tensor input_norm;
    DecodedBf16Tensor post_attention_norm;
    DecodedBf16Tensor q_a;
    DecodedBf16Tensor q_a_norm;
    DecodedBf16Tensor q_b;
    DecodedBf16Tensor kv_a;
    DecodedBf16Tensor kv_a_norm;
    DecodedBf16Tensor kv_b;
    DecodedBf16Tensor o;
    DecodedBf16Tensor index_wq;
    DecodedBf16Tensor index_wk;
    DecodedBf16Tensor index_k_norm_weight;
    DecodedBf16Tensor index_k_norm_bias;
    DecodedBf16Tensor index_weights;
    std::uint64_t payload_bytes{};
};

struct DecoderAttentionForward {
    std::vector<float> post_attention;
    std::vector<float> moe_input;
    std::vector<std::vector<std::uint32_t>> dsa_topk;
};

using NamedTensorLoader = std::function<
    k3x::Result<NamedTensorPayload>(std::string_view)>;

k3x::Result<DecodedBf16Tensor> load_bf16_tensor(
    const NamedTensorLoader& loader, std::string_view name,
    std::initializer_list<std::uint64_t> shape) {
    auto loaded = loader(name);
    if (!loaded) {
        return k3x::Result<DecodedBf16Tensor>::failure(
            loaded.error(), loaded.message());
    }
    if (loaded.value().dtype != 3 || !has_shape(loaded.value(), shape)) {
        return k3x::Result<DecodedBf16Tensor>::failure(
            k3x::ErrorCode::invalid_directory, std::string(name));
    }
    auto values = decode_bf16(loaded.value().bytes);
    if (values.empty()) {
        return k3x::Result<DecodedBf16Tensor>::failure(
            k3x::ErrorCode::invalid_directory, std::string(name));
    }
    const auto rows = shape.size() == 1 ? 1 : *shape.begin();
    const auto cols = shape.size() == 1 ? *shape.begin() : *(shape.begin() + 1);
    return k3x::Result<DecodedBf16Tensor>::success({
        loaded.value().tensor_id,
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols),
        std::move(values),
    });
}

k3x::Result<DecoderLayerWeights> load_decoder_layer_weights(
    const NamedTensorLoader& loader, std::uint32_t layer) {
    const auto prefix = "model.layers." + std::to_string(layer);
    const auto attention = prefix + ".self_attn";
    const auto indexer = attention + ".indexer";
    auto input_norm = load_bf16_tensor(
        loader, prefix + ".input_layernorm.weight", {kHiddenSize});
    auto post_norm = load_bf16_tensor(
        loader, prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    auto q_a = load_bf16_tensor(
        loader, attention + ".q_a_proj.weight", {kQRank, kHiddenSize});
    auto q_a_norm = load_bf16_tensor(
        loader, attention + ".q_a_layernorm.weight", {kQRank});
    auto q_b = load_bf16_tensor(
        loader, attention + ".q_b_proj.weight",
        {kAttentionHeads * (kQkNopeDim + kRopeDim), kQRank});
    auto kv_a = load_bf16_tensor(
        loader, attention + ".kv_a_proj_with_mqa.weight",
        {kKvRank + kRopeDim, kHiddenSize});
    auto kv_a_norm = load_bf16_tensor(
        loader, attention + ".kv_a_layernorm.weight", {kKvRank});
    auto kv_b = load_bf16_tensor(
        loader, attention + ".kv_b_proj.weight",
        {kAttentionHeads * (kQkNopeDim + kValueDim), kKvRank});
    auto o = load_bf16_tensor(
        loader, attention + ".o_proj.weight",
        {kHiddenSize, kAttentionHeads * kValueDim});
    auto index_wq = load_bf16_tensor(
        loader, indexer + ".wq_b.weight",
        {kIndexerHeads * kIndexerHeadDim, kQRank});
    auto index_wk = load_bf16_tensor(
        loader, indexer + ".wk.weight", {kIndexerHeadDim, kHiddenSize});
    auto index_k_norm_weight = load_bf16_tensor(
        loader, indexer + ".k_norm.weight", {kIndexerHeadDim});
    auto index_k_norm_bias = load_bf16_tensor(
        loader, indexer + ".k_norm.bias", {kIndexerHeadDim});
    auto index_weights = load_bf16_tensor(
        loader, indexer + ".weights_proj.weight",
        {kIndexerHeads, kHiddenSize});
    const std::array<bool, 14> valid{
        static_cast<bool>(input_norm), static_cast<bool>(post_norm),
        static_cast<bool>(q_a), static_cast<bool>(q_a_norm),
        static_cast<bool>(q_b), static_cast<bool>(kv_a),
        static_cast<bool>(kv_a_norm), static_cast<bool>(kv_b),
        static_cast<bool>(o), static_cast<bool>(index_wq),
        static_cast<bool>(index_wk), static_cast<bool>(index_k_norm_weight),
        static_cast<bool>(index_k_norm_bias), static_cast<bool>(index_weights),
    };
    if (std::find(valid.begin(), valid.end(), false) != valid.end()) {
        return k3x::Result<DecoderLayerWeights>::failure(
            k3x::ErrorCode::invalid_directory,
            "invalid GLM decoder layer tensor");
    }
    DecoderLayerWeights result{
        std::move(input_norm.value()),
        std::move(post_norm.value()),
        std::move(q_a.value()),
        std::move(q_a_norm.value()),
        std::move(q_b.value()),
        std::move(kv_a.value()),
        std::move(kv_a_norm.value()),
        std::move(kv_b.value()),
        std::move(o.value()),
        std::move(index_wq.value()),
        std::move(index_wk.value()),
        std::move(index_k_norm_weight.value()),
        std::move(index_k_norm_bias.value()),
        std::move(index_weights.value()),
        0,
    };
    for (const auto* tensor : {
             &result.input_norm, &result.post_attention_norm, &result.q_a,
             &result.q_a_norm, &result.q_b, &result.kv_a,
             &result.kv_a_norm, &result.kv_b, &result.o, &result.index_wq,
             &result.index_wk, &result.index_k_norm_weight,
             &result.index_k_norm_bias, &result.index_weights}) {
        result.payload_bytes += tensor->values.size() * 2ULL;
    }
    return k3x::Result<DecoderLayerWeights>::success(std::move(result));
}

void round_bf16_in_place(std::span<float> values) {
    for (auto& value : values) value = round_to_bf16(value);
}

std::vector<float> rms_norm_bf16(
    std::span<const float> input, std::size_t token_count,
    std::size_t width, std::span<const float> weight, float epsilon) {
    std::vector<float> output(input.size());
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto offset = token * width;
        double squares = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const auto value = input[offset + column];
            squares += static_cast<double>(value) * value;
        }
        const auto inverse = static_cast<float>(
            1.0 / std::sqrt(squares / static_cast<double>(width) + epsilon));
        for (std::size_t column = 0; column < width; ++column) {
            output[offset + column] = round_to_bf16(
                input[offset + column] * inverse * weight[column]);
        }
    }
    return output;
}

std::vector<float> layer_norm_bf16(
    std::span<const float> input, std::size_t token_count,
    std::size_t width, std::span<const float> weight,
    std::span<const float> bias, float epsilon) {
    std::vector<float> output(input.size());
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto offset = token * width;
        double sum = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            sum += input[offset + column];
        }
        const auto mean = static_cast<float>(sum / static_cast<double>(width));
        double squares = 0.0;
        for (std::size_t column = 0; column < width; ++column) {
            const auto centered = input[offset + column] - mean;
            squares += static_cast<double>(centered) * centered;
        }
        const auto inverse = static_cast<float>(
            1.0 / std::sqrt(squares / static_cast<double>(width) + epsilon));
        for (std::size_t column = 0; column < width; ++column) {
            output[offset + column] = round_to_bf16(
                (input[offset + column] - mean) * inverse * weight[column] +
                bias[column]);
        }
    }
    return output;
}

k3x::Result<std::vector<float>> linear_tokens_bf16(
    k3x::ComputeBackend& backend, std::span<const float> input,
    std::size_t token_count, const DecodedBf16Tensor& weight,
    std::uint32_t layer) {
    if (input.size() != token_count * weight.cols) {
        return k3x::Result<std::vector<float>>::failure(
            k3x::ErrorCode::invalid_extent);
    }
    std::vector<float> output;
    output.reserve(token_count * weight.rows);
    for (std::size_t token = 0; token < token_count; ++token) {
        auto projected = backend.dense_matvec(
            input.subspan(token * weight.cols, weight.cols), weight.matrix(),
            layer, k3x::ProfilePhase::decode);
        if (!projected) return projected;
        round_bf16_in_place(projected.value());
        output.insert(output.end(), projected.value().begin(),
                      projected.value().end());
    }
    return k3x::Result<std::vector<float>>::success(std::move(output));
}

void apply_interleaved_rope(
    std::span<float> values, std::size_t token_count,
    std::size_t groups, std::size_t width, std::size_t rope_offset) {
    constexpr auto half = kRopeDim / 2;
    std::array<float, kRopeDim> rotated{};
    for (std::size_t token = 0; token < token_count; ++token) {
        for (std::size_t group = 0; group < groups; ++group) {
            const auto base = (token * groups + group) * width + rope_offset;
            for (std::size_t pair = 0; pair < half; ++pair) {
                const auto exponent = static_cast<double>(pair * 2) /
                                      static_cast<double>(kRopeDim);
                const auto angle = static_cast<double>(token) /
                                   std::pow(kRopeTheta, exponent);
                const auto cosine = static_cast<float>(std::cos(angle));
                const auto sine = static_cast<float>(std::sin(angle));
                const auto even = values[base + pair * 2];
                const auto odd = values[base + pair * 2 + 1];
                rotated[pair] = even * cosine - odd * sine;
                rotated[half + pair] = odd * cosine + even * sine;
            }
            std::copy(rotated.begin(), rotated.end(), values.begin() + base);
        }
    }
}

k3x::Result<DecoderAttentionForward> run_decoder_attention(
    k3x::ComputeBackend& backend, const DecoderLayerWeights& weights,
    std::span<const float> hidden, std::size_t token_count,
    std::uint32_t layer) {
    if (token_count == 0 || token_count > 2048 ||
        hidden.size() != token_count * kHiddenSize) {
        return k3x::Result<DecoderAttentionForward>::failure(
            k3x::ErrorCode::invalid_extent);
    }
    auto normalized = rms_norm_bf16(
        hidden, token_count, kHiddenSize, weights.input_norm.values, 1.0e-5F);
    auto q_resid = linear_tokens_bf16(
        backend, normalized, token_count, weights.q_a, layer);
    if (!q_resid) {
        return k3x::Result<DecoderAttentionForward>::failure(
            q_resid.error(), q_resid.message());
    }
    q_resid.value() = rms_norm_bf16(
        q_resid.value(), token_count, kQRank, weights.q_a_norm.values,
        1.0e-5F);

    auto index_q = linear_tokens_bf16(
        backend, q_resid.value(), token_count, weights.index_wq, layer);
    auto index_k = linear_tokens_bf16(
        backend, normalized, token_count, weights.index_wk, layer);
    auto index_weights = linear_tokens_bf16(
        backend, normalized, token_count, weights.index_weights, layer);
    if (!index_q || !index_k || !index_weights) {
        return k3x::Result<DecoderAttentionForward>::failure(
            k3x::ErrorCode::backend_unavailable,
            "GLM DSA projection failed");
    }
    index_k.value() = layer_norm_bf16(
        index_k.value(), token_count, kIndexerHeadDim,
        weights.index_k_norm_weight.values, weights.index_k_norm_bias.values,
        1.0e-6F);
    apply_interleaved_rope(
        index_q.value(), token_count, kIndexerHeads, kIndexerHeadDim, 0);
    apply_interleaved_rope(
        index_k.value(), token_count, 1, kIndexerHeadDim, 0);
    const auto index_scale = 1.0F / std::sqrt(
        static_cast<float>(kIndexerHeadDim));
    const auto head_scale = 1.0F / std::sqrt(
        static_cast<float>(kIndexerHeads));
    std::vector<std::vector<std::uint32_t>> dsa_topk(token_count);
    for (std::size_t query = 0; query < token_count; ++query) {
        std::vector<std::pair<float, std::uint32_t>> scores;
        scores.reserve(token_count);
        for (std::size_t key = 0; key < token_count; ++key) {
            float combined = -INFINITY;
            if (key <= query) {
                combined = 0.0F;
                for (std::size_t head = 0; head < kIndexerHeads; ++head) {
                    float dot = 0.0F;
                    const auto q_offset =
                        (query * kIndexerHeads + head) * kIndexerHeadDim;
                    const auto k_offset = key * kIndexerHeadDim;
                    for (std::size_t column = 0;
                         column < kIndexerHeadDim; ++column) {
                        dot += index_q.value()[q_offset + column] *
                               index_k.value()[k_offset + column];
                    }
                    combined += std::max(dot * index_scale, 0.0F) *
                        index_weights.value()[query * kIndexerHeads + head] *
                        head_scale;
                }
            }
            scores.emplace_back(combined, static_cast<std::uint32_t>(key));
        }
        std::stable_sort(
            scores.begin(), scores.end(),
            [](const auto& left, const auto& right) {
                return left.first > right.first;
            });
        auto& selected = dsa_topk[query];
        selected.reserve(scores.size());
        for (const auto& [score, key] : scores) {
            static_cast<void>(score);
            selected.push_back(key);
        }
    }

    auto q_states = linear_tokens_bf16(
        backend, q_resid.value(), token_count, weights.q_b, layer);
    auto compressed = linear_tokens_bf16(
        backend, normalized, token_count, weights.kv_a, layer);
    if (!q_states || !compressed) {
        return k3x::Result<DecoderAttentionForward>::failure(
            k3x::ErrorCode::backend_unavailable,
            "GLM MLA projection failed");
    }
    std::vector<float> kv_nope(token_count * kKvRank);
    for (std::size_t token = 0; token < token_count; ++token) {
        const auto source = token * (kKvRank + kRopeDim);
        std::copy_n(compressed.value().begin() + source, kKvRank,
                    kv_nope.begin() + token * kKvRank);
    }
    kv_nope = rms_norm_bf16(
        kv_nope, token_count, kKvRank, weights.kv_a_norm.values, 1.0e-5F);
    apply_interleaved_rope(
        q_states.value(), token_count, kAttentionHeads,
        kQkNopeDim + kRopeDim, kQkNopeDim);
    apply_interleaved_rope(
        compressed.value(), token_count, 1, kKvRank + kRopeDim, kKvRank);
    auto expanded = linear_tokens_bf16(
        backend, kv_nope, token_count, weights.kv_b, layer);
    if (!expanded) {
        return k3x::Result<DecoderAttentionForward>::failure(
            expanded.error(), expanded.message());
    }
    constexpr auto qk_width = kQkNopeDim + kRopeDim;
    constexpr auto kv_width = kQkNopeDim + kValueDim;
    const auto attention_scale = 1.0F / std::sqrt(
        static_cast<float>(qk_width));
    std::vector<float> attended(
        token_count * kAttentionHeads * kValueDim, 0.0F);
    std::vector<float> logits(token_count);
    std::vector<float> probabilities(token_count);
    for (std::size_t query = 0; query < token_count; ++query) {
        for (std::size_t head = 0; head < kAttentionHeads; ++head) {
            float maximum = -INFINITY;
            for (std::size_t key = 0; key <= query; ++key) {
                float dot = 0.0F;
                const auto q_offset = (query * kAttentionHeads + head) * qk_width;
                const auto kv_offset =
                    (key * kAttentionHeads + head) * kv_width;
                const auto rot_offset = key * (kKvRank + kRopeDim) + kKvRank;
                for (std::size_t column = 0; column < kQkNopeDim; ++column) {
                    dot += q_states.value()[q_offset + column] *
                           expanded.value()[kv_offset + column];
                }
                for (std::size_t column = 0; column < kRopeDim; ++column) {
                    dot += q_states.value()[q_offset + kQkNopeDim + column] *
                           compressed.value()[rot_offset + column];
                }
                logits[key] = dot * attention_scale;
                maximum = std::max(maximum, logits[key]);
            }
            float denominator = 0.0F;
            for (std::size_t key = 0; key <= query; ++key) {
                probabilities[key] = std::exp(logits[key] - maximum);
                denominator += probabilities[key];
            }
            for (std::size_t key = 0; key <= query; ++key) {
                probabilities[key] = round_to_bf16(
                    probabilities[key] / denominator);
            }
            const auto output_offset =
                (query * kAttentionHeads + head) * kValueDim;
            for (std::size_t column = 0; column < kValueDim; ++column) {
                float sum = 0.0F;
                for (std::size_t key = 0; key <= query; ++key) {
                    const auto value_offset =
                        (key * kAttentionHeads + head) * kv_width +
                        kQkNopeDim + column;
                    sum += probabilities[key] * expanded.value()[value_offset];
                }
                attended[output_offset + column] = round_to_bf16(sum);
            }
        }
    }
    auto attention_output = linear_tokens_bf16(
        backend, attended, token_count, weights.o, layer);
    if (!attention_output) {
        return k3x::Result<DecoderAttentionForward>::failure(
            attention_output.error(), attention_output.message());
    }
    DecoderAttentionForward result;
    result.post_attention.resize(hidden.size());
    for (std::size_t index = 0; index < hidden.size(); ++index) {
        result.post_attention[index] = round_to_bf16(
            hidden[index] + attention_output.value()[index]);
    }
    result.moe_input = rms_norm_bf16(
        result.post_attention, token_count, kHiddenSize,
        weights.post_attention_norm.values, 1.0e-5F);
    result.dsa_topk = std::move(dsa_topk);
    return k3x::Result<DecoderAttentionForward>::success(std::move(result));
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
    if (!arguments ||
        (!arguments->artifact_dir.empty() &&
         !std::filesystem::is_directory(arguments->artifact_dir)) ||
        (!arguments->runtime_index.empty() &&
         !std::filesystem::is_regular_file(arguments->runtime_index))) {
        std::cerr << "usage: (--artifact-dir DIR | --runtime-index FILE) "
                     "--layer N --expert N "
                     "[--experts N] [--tokens N] [--warmup N] [--iterations N] "
                     "[--workspace-bytes N] [--resident-bytes N] [--precision fp32|bf16-rounded] "
                     "[--output fp32|bf16] [--input-mode common|sparse-packed|expert-major|learned-expert-major|learned-moe-layer|learned-decoder-layer] "
                     "[--device-accumulate 0|1] "
                     "[--fuse-shared 0|1] "
                     "[--input-bf16 FILE] [--expected-bf16 FILE]\n";
        return 2;
    }
    k3x::ReaderOptions reader_options;
    reader_options.verify = k3x::VerifyMode::metadata_only;
    std::vector<std::filesystem::path> paths;
    std::vector<k3x::Reader> readers;
    std::vector<const k3x::Reader*> shard_views;
    std::optional<k3x::Glm5xRuntimeIndex> runtime_index;
    if (!arguments->artifact_dir.empty()) {
        paths = artifact_paths(arguments->artifact_dir);
        if (paths.empty()) return 3;
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
        shard_views.reserve(readers.size());
        for (const auto& reader : readers) shard_views.push_back(&reader);
    } else {
        auto opened = k3x::Glm5xRuntimeIndex::open(
            arguments->runtime_index, reader_options);
        if (!opened) {
            std::cerr << k3x::error_code_name(opened.error()) << ": "
                      << opened.message() << '\n';
            return 4;
        }
        runtime_index.emplace(std::move(opened.value()));
    }
    std::set<std::uint32_t> available_experts;
    if (runtime_index) {
        for (std::uint32_t expert_id = 0; expert_id < 256; ++expert_id) {
            const auto prefix =
                "model.layers." + std::to_string(arguments->layer) +
                ".mlp.experts." + std::to_string(expert_id) + ".";
            if (runtime_index->contains_tensor(
                    k3x::fnv1a64((prefix + "gate_proj.weight").c_str())) &&
                runtime_index->contains_tensor(
                    k3x::fnv1a64((prefix + "up_proj.weight").c_str())) &&
                runtime_index->contains_tensor(
                    k3x::fnv1a64((prefix + "down_proj.weight").c_str()))) {
                available_experts.insert(expert_id);
            }
        }
    } else {
        for (const auto& reader : readers) {
            for (const auto& record : reader.tensors()) {
                if (record.layer_id ==
                        static_cast<std::int32_t>(arguments->layer) &&
                    record.expert_id >= 0) {
                    available_experts.insert(
                        static_cast<std::uint32_t>(record.expert_id));
                }
            }
        }
    }
    const auto load_source_tensor = [&](std::string_view name) {
        return runtime_index
            ? load_named_tensor(*runtime_index, name)
            : load_named_tensor(shard_views, name);
    };
    const auto load_source_expert = [&](std::uint32_t expert_id) {
        return runtime_index
            ? runtime_index->read_expert(
                  arguments->layer, expert_id,
                  kHiddenSize, kIntermediateSize)
            : k3x::load_glm5x_bf16_expert(
                  shard_views, arguments->layer, expert_id,
                  kHiddenSize, kIntermediateSize);
    };
    const bool sparse_packed = arguments->input_mode == "sparse-packed";
    const bool expert_major = arguments->input_mode == "expert-major";
    const bool learned_expert_major =
        arguments->input_mode == "learned-expert-major";
    const bool learned_moe_layer =
        arguments->input_mode == "learned-moe-layer";
    const bool learned_decoder_layer =
        arguments->input_mode == "learned-decoder-layer";
    const bool learned_route_mode =
        learned_expert_major || learned_moe_layer || learned_decoder_layer;
    if (learned_decoder_layer &&
        (arguments->runtime_index.empty() || arguments->input_bf16.empty() ||
         arguments->expected_bf16.empty() || arguments->tokens > 2048)) {
        std::cerr << "learned decoder layer requires runtime index, input, "
                     "expected output, and at most 2048 tokens\n";
        return 7;
    }
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
    options.cuda_expert_major_device_accumulate = arguments->device_accumulate;
    options.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    options.cuda_resident_bytes = arguments->resident_bytes;
    auto cuda = k3x::make_cuda_backend(options);
    if (!cuda) {
        std::cerr << k3x::error_code_name(cuda.error()) << ": "
                  << cuda.message() << '\n';
        return 7;
    }
    std::optional<DecoderLayerWeights> decoder_weights;
    std::optional<DecoderAttentionForward> initial_attention;
    std::vector<float> decoder_layer_input;
    std::uint64_t decoder_prepare_nanoseconds = 0;
    if (learned_decoder_layer) {
        const auto decoder_started = std::chrono::steady_clock::now();
        auto loaded = load_decoder_layer_weights(
            NamedTensorLoader(load_source_tensor), arguments->layer);
        if (!loaded) {
            std::cerr << "invalid GLM decoder layer tensor\n";
            return 5;
        }
        decoder_weights.emplace(std::move(loaded.value()));
        decoder_layer_input = input;
        auto forwarded = run_decoder_attention(
            *cuda.value(), *decoder_weights, decoder_layer_input,
            arguments->tokens, arguments->layer);
        if (!forwarded) {
            std::cerr << k3x::error_code_name(forwarded.error()) << ": "
                      << forwarded.message() << '\n';
            return 5;
        }
        initial_attention.emplace(std::move(forwarded.value()));
        input = initial_attention->moe_input;
        decoder_prepare_nanoseconds = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - decoder_started).count());
    }
    std::vector<k3x::ExpertMajorTokenRoute> learned_routes;
    std::vector<std::uint32_t> expert_ids;
    std::uint64_t router_payload_bytes = 0;
    const auto host_start = std::chrono::steady_clock::now();
    if (learned_route_mode) {
        const auto router = load_source_tensor(
            "model.layers." + std::to_string(arguments->layer) +
                ".mlp.gate.weight");
        const auto bias = load_source_tensor(
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
        auto loaded = load_source_expert(expert_id);
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
    if (learned_moe_layer || learned_decoder_layer) {
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
            auto payload = load_source_tensor(names[index]);
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
    k3x::ReadCounters source_reads;
    if (runtime_index) {
        source_reads = runtime_index->counters();
    } else {
        for (const auto& reader : readers) {
            const auto counters = reader.counters();
            source_reads.calls += counters.calls;
            source_reads.completed_bytes += counters.completed_bytes;
        }
    }
    const auto source_artifact_count = runtime_index
        ? runtime_index->artifact_count() : paths.size();
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
    auto cpu = k3x::make_cpu_backend();
    constexpr auto activation = k3x::MlpActivation::silu;
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
                    arguments->layer, k3x::ProfilePhase::decode, activation);
                if (!reference) return 8;
                auto output = std::span<float>(cpu_reference).subspan(
                    assignment.token_index * kHiddenSize, kHiddenSize);
                for (std::size_t value = 0; value < output.size(); ++value) {
                    output[value] += assignment.contribution *
                        reference.value()[value];
                }
            }
        }
        if (learned_moe_layer || learned_decoder_layer) {
            for (std::size_t token = 0; token < arguments->tokens; ++token) {
                const auto token_input = std::span<const float>(input).subspan(
                    token * kHiddenSize, kHiddenSize);
                const auto shared = cpu->dense_situ_mlp(
                    token_input, shared_fixture.dense, 1.0F, std::nullopt,
                    arguments->layer, k3x::ProfilePhase::decode, activation);
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
            arguments->layer, k3x::ProfilePhase::decode, activation);
        if (!reference) return 8;
        cpu_reference.insert(cpu_reference.end(), reference.value().begin(),
                             reference.value().end());
    }
    if (learned_decoder_layer) {
        if (!initial_attention || cpu_reference.size() !=
                                      initial_attention->post_attention.size()) {
            return 8;
        }
        for (std::size_t index = 0; index < cpu_reference.size(); ++index) {
            cpu_reference[index] = round_to_bf16(
                initial_attention->post_attention[index] +
                round_to_bf16(cpu_reference[index]));
        }
    }
    const auto execute_learned_moe = [&]
        (std::span<const float> current_input,
         const k3x::ExpertMajorPackedPlan& current_plan)
        -> k3x::Result<std::vector<float>> {
        if (learned_moe_layer || learned_decoder_layer) {
            if (arguments->fuse_shared) {
                return cuda.value()->raw_bf16_situ_mlp_expert_major_with_shared(
                    current_input, arguments->tokens, current_plan,
                    expert_major_views, shared_fixture.raw, 1.0F, std::nullopt,
                    arguments->layer, k3x::ProfilePhase::decode, activation);
            }
            auto routed = cuda.value()->raw_bf16_situ_mlp_expert_major(
                current_input, arguments->tokens, current_plan,
                expert_major_views, 1.0F, std::nullopt, arguments->layer,
                k3x::ProfilePhase::decode, activation);
            if (!routed) return routed;
            const std::array<k3x::RawBf16MlpView, 1> shared_views{
                shared_fixture.raw};
            auto shared = cuda.value()->raw_bf16_situ_mlp_grid(
                current_input, arguments->tokens, shared_views, 1.0F, std::nullopt,
                arguments->layer, k3x::ProfilePhase::decode, activation);
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
        return cuda.value()->raw_bf16_situ_mlp_expert_major(
            current_input, arguments->tokens, current_plan,
            expert_major_views, 1.0F, std::nullopt, arguments->layer,
            k3x::ProfilePhase::decode, activation);
    };
    const auto execute = [&]() -> k3x::Result<std::vector<float>> {
        if (learned_decoder_layer) {
            auto attention = run_decoder_attention(
                *cuda.value(), *decoder_weights, decoder_layer_input,
                arguments->tokens, arguments->layer);
            if (!attention) {
                return k3x::Result<std::vector<float>>::failure(
                    attention.error(), attention.message());
            }
            auto plan = k3x::build_expert_major_packed_plan(
                attention.value().moe_input, arguments->tokens, kHiddenSize,
                learned_routes);
            if (!plan || plan.value().groups.size() !=
                             expert_major_views.size()) {
                return k3x::Result<std::vector<float>>::failure(
                    k3x::ErrorCode::invalid_state,
                    "decoder layer route plan changed");
            }
            auto moe = execute_learned_moe(
                attention.value().moe_input, plan.value());
            if (!moe) {
                return moe;
            }
            if (moe.value().size() !=
                attention.value().post_attention.size()) {
                return k3x::Result<std::vector<float>>::failure(
                    k3x::ErrorCode::invalid_extent,
                    "decoder layer MoE output shape mismatch");
            }
            for (std::size_t index = 0; index < moe.value().size(); ++index) {
                moe.value()[index] = round_to_bf16(
                    attention.value().post_attention[index] +
                    round_to_bf16(moe.value()[index]));
            }
            return moe;
        }
        if (learned_moe_layer) {
            return execute_learned_moe(input, *expert_major_plan);
        }
        if (expert_major || learned_expert_major) {
            return execute_learned_moe(input, *expert_major_plan);
        }
        if (use_grid) {
            auto outputs = sparse_packed
                ? cuda.value()->raw_bf16_situ_mlp_grid_packed(
                      packed_input, grid_token_count, raw_views, 1.0F,
                      std::nullopt, arguments->layer,
                      k3x::ProfilePhase::decode, activation)
                : cuda.value()->raw_bf16_situ_mlp_grid(
                      input, grid_token_count, raw_views, 1.0F, std::nullopt,
                      arguments->layer, k3x::ProfilePhase::decode, activation);
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
                    arguments->layer, k3x::ProfilePhase::decode, activation);
                if (!token_output) return token_output;
            }
            last.insert(last.end(), token_output.value().begin(),
                        token_output.value().end());
        }
        return k3x::Result<std::vector<float>>::success(std::move(last));
    };
    const auto stats_before_cold = cuda.value()->runtime_stats();
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
    std::optional<float> cpu_expected_abs;
    std::optional<float> cpu_expected_rel;
    std::optional<float> expected_abs;
    std::optional<float> expected_rel;
    if (expected_output) {
        cpu_expected_abs = maximum_absolute_difference(cpu_reference, *expected_output);
        const auto cpu_expected_scale = std::max(
            maximum_absolute_value(*expected_output), 1.0e-6F);
        cpu_expected_rel = *cpu_expected_abs / cpu_expected_scale;
        auto expected_actual = actual;
        if (arguments->precision == "bf16-rounded") {
            for (auto& value : expected_actual) value = round_to_bf16(value);
        }
        expected_abs = maximum_absolute_difference(
            expected_actual, *expected_output);
        const auto expected_scale = std::max(
            maximum_absolute_value(*expected_output), 1.0e-6F);
        expected_rel = *expected_abs / expected_scale;
    }
    const auto emit_route_telemetry = [&]() {
        if (!learned_route_mode) return;
        std::cout << ",\"route_experts\":[";
        for (std::size_t token = 0; token < learned_routes.size(); ++token) {
            if (token != 0) std::cout << ',';
            std::cout << '[';
            for (std::size_t slot = 0;
                 slot < learned_routes[token].expert_ids.size(); ++slot) {
                if (slot != 0) std::cout << ',';
                std::cout << learned_routes[token].expert_ids[slot];
            }
            std::cout << ']';
        }
        std::cout << "],\"route_contributions\":[";
        for (std::size_t token = 0; token < learned_routes.size(); ++token) {
            if (token != 0) std::cout << ',';
            std::cout << '[';
            for (std::size_t slot = 0;
                 slot < learned_routes[token].contributions.size(); ++slot) {
                if (slot != 0) std::cout << ',';
                std::cout << learned_routes[token].contributions[slot];
            }
            std::cout << ']';
        }
        std::cout << ']';
    };
    const auto emit_decoder_telemetry = [&]() {
        if (!initial_attention) return;
        std::cout << ",\"dsa_topk_indices\":[";
        for (std::size_t token = 0;
             token < initial_attention->dsa_topk.size(); ++token) {
            if (token != 0) std::cout << ',';
            std::cout << '[';
            for (std::size_t slot = 0;
                 slot < initial_attention->dsa_topk[token].size(); ++slot) {
                if (slot != 0) std::cout << ',';
                std::cout << initial_attention->dsa_topk[token][slot];
            }
            std::cout << ']';
        }
        std::cout << ']';
    };
    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\""
              << ((expert_major || learned_route_mode)
                      ? (learned_decoder_layer
                             ? "glm5.2_real_bf16_decoder_layer"
                             : (learned_moe_layer
                                    ? "glm5.2_real_bf16_learned_moe_layer"
                                    : (learned_expert_major
                                           ? "glm5.2_real_bf16_learned_expert_major"
                                           : "glm5.2_real_bf16_expert_major")))
                      : "glm5.2_real_bf16_expert")
              << "\""
              << ",\"layer_id\":" << arguments->layer
              << ",\"expert_id\":" << expert_ids.front()
              << ",\"expert_count\":" << expert_ids.size()
              << ",\"token_count\":" << arguments->tokens
              << ",\"last_expert_id\":" << expert_ids.back()
              << ",\"shard_count\":" << source_artifact_count
              << ",\"source_kind\":\""
              << (runtime_index ? "runtime_index" : "artifact_directory")
              << "\""
              << ",\"source_artifact_count\":" << source_artifact_count
              << ",\"source_read_calls\":" << source_reads.calls
              << ",\"source_read_bytes\":"
              << source_reads.completed_bytes
              << ",\"precision\":\"" << arguments->precision << "\""
              << ",\"output\":\"" << arguments->output << "\""
              << ",\"input_mode\":\"" << arguments->input_mode << "\""
              << ",\"device_expert_accumulate\":"
              << (arguments->device_accumulate ? "true" : "false")
              << ",\"fused_shared\":"
              << (arguments->fuse_shared ? "true" : "false");
    emit_route_telemetry();
    emit_decoder_telemetry();
    std::cout
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
              << ",\"decoder_trunk_payload_bytes\":"
              << (decoder_weights ? decoder_weights->payload_bytes : 0)
              << ",\"decoder_prepare_nanoseconds\":"
              << decoder_prepare_nanoseconds
              << ",\"mla_state_length\":"
              << (learned_decoder_layer ? arguments->tokens : 0)
              << ",\"dsa_state_length\":"
              << (learned_decoder_layer ? arguments->tokens : 0)
              << ",\"host_load_nanoseconds\":" << host_load_ns
              << ",\"cold_latency_nanoseconds\":" << cold_ns
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"gpu_cpu_max_absolute_error\":" << cpu_abs
              << ",\"gpu_cpu_max_relative_error\":" << cpu_abs / cpu_scale
              << ",\"cpu_expected_max_absolute_error\":";
    if (cpu_expected_abs) {
        std::cout << *cpu_expected_abs;
    } else {
        std::cout << "null";
    }
    std::cout << ",\"cpu_expected_max_relative_error\":";
    if (cpu_expected_rel) {
        std::cout << *cpu_expected_rel;
    } else {
        std::cout << "null";
    }
    std::cout
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
              << (stats_after_cold.weight_h2d_bytes -
                  stats_before_cold.weight_h2d_bytes)
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
