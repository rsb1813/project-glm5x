// GLM5X cross-shard raw BF16 expert payload를 정확히 읽습니다.
#include "k3x/glm5x_bundle.hpp"

#include "k3x/checksums.hpp"
#include "k3x/format.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace k3x {
namespace {

const TensorRecord* find_tensor(
    const Reader& reader,
    std::uint64_t tensor_id) {
    const auto match = std::find_if(
        reader.tensors().begin(), reader.tensors().end(),
        [tensor_id](const TensorRecord& record) {
            return record.tensor_id == tensor_id;
        });
    return match == reader.tensors().end() ? nullptr : &*match;
}

}  // namespace

Result<GlmBf16ExpertLoad> load_glm5x_bf16_expert(
    std::span<const Reader* const> shards,
    std::uint32_t layer_id,
    std::uint32_t expert_id,
    std::uint64_t hidden_size,
    std::uint64_t intermediate_size) {
    if (shards.empty() || hidden_size == 0 || intermediate_size == 0) {
        return Result<GlmBf16ExpertLoad>::failure(
            ErrorCode::invalid_directory, "GLM expert bundle is empty");
    }
    const std::string prefix =
        "model.layers." + std::to_string(layer_id) +
        ".mlp.experts." + std::to_string(expert_id) + ".";
    const std::array roles{"gate_proj", "up_proj", "down_proj"};
    const std::array<std::array<std::uint64_t, 2>, 3> shapes{{
        {intermediate_size, hidden_size},
        {intermediate_size, hidden_size},
        {hidden_size, intermediate_size},
    }};
    GlmBf16ExpertLoad result;
    result.shapes = shapes;
    for (std::size_t index = 0; index < roles.size(); ++index) {
        const auto name = prefix + roles[index] + ".weight";
        const auto tensor_id = fnv1a64(name.c_str());
        const Reader* owner = nullptr;
        const TensorRecord* record = nullptr;
        for (const auto* shard : shards) {
            if (shard == nullptr) continue;
            const auto candidate = find_tensor(*shard, tensor_id);
            if (candidate == nullptr) continue;
            if (owner != nullptr) {
                return Result<GlmBf16ExpertLoad>::failure(
                    ErrorCode::invalid_directory, "duplicate GLM expert role");
            }
            owner = shard;
            record = candidate;
        }
        if (owner == nullptr || record == nullptr) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::tensor_not_found, name);
        }
        const auto values = shapes[index][0] * shapes[index][1];
        if (record->dtype != 3 || record->quantization != 0 || record->rank != 2 ||
            record->layer_id != static_cast<std::int32_t>(layer_id) ||
            record->expert_id != static_cast<std::int32_t>(expert_id) ||
            record->dimensions[0] != shapes[index][0] ||
            record->dimensions[1] != shapes[index][1] || record->dimensions[2] != 0 ||
            record->dimensions[3] != 0 || record->data_length != values * 2 ||
            record->logical_length != record->data_length || record->auxiliary_length != 0 ||
            record->auxiliary_offset != 0 || record->auxiliary_crc32c != 0) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::invalid_directory, "invalid GLM BF16 expert tensor");
        }
        auto payload = owner->read_tensor(record->tensor_id);
        if (!payload) {
            return Result<GlmBf16ExpertLoad>::failure(payload.error(), payload.message());
        }
        if (payload.value().size() != record->data_length ||
            crc32c(payload.value()) != record->data_crc32c) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::data_crc_mismatch, name);
        }
        result.payload_bytes += payload.value().size();
        result.roles[index] = std::move(payload.value());
    }
    return Result<GlmBf16ExpertLoad>::success(std::move(result));
}

}  // namespace k3x
