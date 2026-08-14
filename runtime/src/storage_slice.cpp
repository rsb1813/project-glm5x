// Storage fixture에서 정확한 native MXFP4 expert payload를 일괄 복원합니다.
#include "k3x/storage_slice.hpp"

#include "k3x/checksums.hpp"
#include "k3x/format.hpp"

#include <algorithm>
#include <array>
#include <string>

namespace k3x {
namespace {

const TensorRecord* find_tensor(const Reader& reader, const std::string& name) {
    const auto tensor_id = fnv1a64(name.c_str());
    const auto match = std::find_if(
        reader.tensors().begin(), reader.tensors().end(),
        [tensor_id](const TensorRecord& record) {
            return record.tensor_id == tensor_id;
        });
    return match == reader.tensors().end() ? nullptr : &*match;
}

bool valid_native_matrix(
    const TensorRecord& record,
    std::uint32_t layer_id,
    std::uint32_t expert_id,
    std::uint64_t rows,
    std::uint64_t columns) {
    const auto values = rows * columns;
    return record.dtype == 2 && record.quantization == 1 && record.rank == 2 &&
        record.layer_id == static_cast<std::int32_t>(layer_id) &&
        record.expert_id == static_cast<std::int32_t>(expert_id) &&
        record.dimensions[0] == rows && record.dimensions[1] == columns &&
        record.dimensions[2] == 0 && record.dimensions[3] == 0 &&
        record.data_length == values / 2 &&
        record.auxiliary_length == values / 32;
}

}  // namespace

Result<StorageExpertLoad> load_storage_expert(
    Reader& reader,
    std::uint32_t layer_id,
    std::uint32_t expert_id) {
    if (!(reader.superblock().optional_features & optional_storage_fixture)) {
        return Result<StorageExpertLoad>::failure(
            ErrorCode::invalid_extent, "artifact is not a storage fixture");
    }
    const auto base = "model.layers." + std::to_string(layer_id) +
        ".feed_forward.experts." + std::to_string(expert_id) + ".";
    const std::array roles{"gate", "up", "down"};
    const std::array<std::array<std::uint64_t, 2>, 3> shapes{{
        {3072, 3584}, {3072, 3584}, {3584, 3072}}};
    std::array<const TensorRecord*, 3> records{};
    std::array<ExtentRequest, 6> requests{};
    for (std::size_t index = 0; index < roles.size(); ++index) {
        records[index] = find_tensor(reader, base + roles[index]);
        if (records[index] == nullptr) {
            return Result<StorageExpertLoad>::failure(
                ErrorCode::tensor_not_found, "storage expert tensor missing");
        }
        if (!valid_native_matrix(
                *records[index], layer_id, expert_id,
                shapes[index][0], shapes[index][1])) {
            return Result<StorageExpertLoad>::failure(
                ErrorCode::invalid_mxfp4, "invalid storage expert matrix");
        }
        requests[index * 2] = {
            records[index]->data_offset, records[index]->data_length};
        requests[index * 2 + 1] = {
            records[index]->auxiliary_offset, records[index]->auxiliary_length};
    }
    auto extents = reader.read_extents(requests);
    if (!extents) {
        return Result<StorageExpertLoad>::failure(
            extents.error(), extents.message());
    }
    StorageExpertLoad result;
    Sha256Hasher digest;
    for (std::size_t index = 0; index < result.extents.size(); ++index) {
        result.logical_bytes += extents.value()[index].size();
        result.extents[index] = std::move(extents.value()[index]);
        digest.update(result.extents[index]);
    }
    if (result.logical_bytes != 17'547'264) {
        return Result<StorageExpertLoad>::failure(
            ErrorCode::invalid_mxfp4, "invalid storage expert payload length");
    }
    result.ordered_sha256 = digest.finish();
    return Result<StorageExpertLoad>::success(std::move(result));
}

}  // namespace k3x
