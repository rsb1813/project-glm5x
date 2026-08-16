// GLM5X 고정 runtime index와 참조 K3X shard의 무결성을 검증합니다.
#include "k3x/glm5x_runtime_index.hpp"

#include "k3x/checksums.hpp"
#include "k3x/format.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <unordered_set>

namespace k3x {
namespace {

constexpr std::size_t header_bytes = 128;
constexpr std::size_t artifact_record_bytes = 64;
constexpr std::size_t runtime_tensor_record_bytes = 24;

template <typename T>
T little(std::span<const std::byte> data, std::size_t offset) {
    T value{};
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(std::to_integer<unsigned char>(data[offset + index]))
                 << (index * 8);
    }
    return value;
}

bool range_valid(std::uint64_t offset, std::uint64_t length, std::uint64_t size) {
    return offset <= size && length <= size - offset;
}

bool all_zero(std::span<const std::byte> value) {
    return std::all_of(value.begin(), value.end(), [](std::byte item) {
        return item == std::byte{};
    });
}

Result<std::vector<std::byte>> read_index(const std::filesystem::path& path) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < header_bytes ||
        size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::truncated_file);
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::io_error);
    }
    std::vector<std::byte> data(static_cast<std::size_t>(size));
    input.read(reinterpret_cast<char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!input || input.gcount() != static_cast<std::streamsize>(data.size())) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::truncated_file);
    }
    return Result<std::vector<std::byte>>::success(std::move(data));
}

bool contained_by(
    const std::filesystem::path& base,
    const std::filesystem::path& candidate) {
    auto base_item = base.begin();
    auto candidate_item = candidate.begin();
    for (; base_item != base.end(); ++base_item, ++candidate_item) {
        if (candidate_item == candidate.end() || *base_item != *candidate_item) {
            return false;
        }
    }
    return true;
}

ReadCounters& add(ReadCounters& target, const ReadCounters& value) {
    target.calls += value.calls;
    target.requested_bytes += value.requested_bytes;
    target.completed_bytes += value.completed_bytes;
    target.batch_submissions += value.batch_submissions;
    target.storage_submitted_bytes += value.storage_submitted_bytes;
    target.storage_completed_bytes += value.storage_completed_bytes;
    target.completions += value.completions;
    target.short_reads += value.short_reads;
    target.failures += value.failures;
    target.storage_nanoseconds += value.storage_nanoseconds;
    return target;
}

}  // namespace

Result<Glm5xRuntimeIndex> Glm5xRuntimeIndex::open(
    const std::filesystem::path& path,
    ReaderOptions options) {
    auto loaded = read_index(path);
    if (!loaded) {
        return Result<Glm5xRuntimeIndex>::failure(
            loaded.error(), loaded.message());
    }
    const auto data = std::span<const std::byte>(loaded.value());
    constexpr std::array<char, 8> magic{'G', 'L', 'M', '5', 'X', 'I', 'D', 'X'};
    if (std::memcmp(data.data(), magic.data(), magic.size()) != 0) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::bad_magic);
    }
    if (little<std::uint16_t>(data, 8) != 1 ||
        little<std::uint16_t>(data, 10) != 0 ||
        little<std::uint32_t>(data, 12) != header_bytes ||
        little<std::uint32_t>(data, 16) != artifact_record_bytes ||
        little<std::uint32_t>(data, 20) != runtime_tensor_record_bytes) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::unsupported_version);
    }
    if (little<std::uint32_t>(data, 28) != 0 ||
        little<std::uint32_t>(data, 120) != 0) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::invalid_directory);
    }
    if (crc32c(data.first(124)) != little<std::uint32_t>(data, 124)) {
        return Result<Glm5xRuntimeIndex>::failure(
            ErrorCode::superblock_crc_mismatch,
            "GLM5X runtime index header CRC mismatch");
    }
    std::array<std::byte, 32> expected_body_sha{};
    std::copy(data.begin() + 88, data.begin() + 120, expected_body_sha.begin());
    if (sha256(data.subspan(header_bytes)) != expected_body_sha) {
        return Result<Glm5xRuntimeIndex>::failure(
            ErrorCode::directory_sha256_mismatch,
            "GLM5X runtime index body SHA mismatch");
    }

    const auto artifact_count = little<std::uint32_t>(data, 24);
    const auto tensor_count = little<std::uint64_t>(data, 32);
    const auto artifact_offset = little<std::uint64_t>(data, 40);
    const auto artifact_length = little<std::uint64_t>(data, 48);
    const auto tensor_offset = little<std::uint64_t>(data, 56);
    const auto tensor_length = little<std::uint64_t>(data, 64);
    const auto string_offset = little<std::uint64_t>(data, 72);
    const auto string_length = little<std::uint64_t>(data, 80);
    const auto size = static_cast<std::uint64_t>(data.size());
    if (artifact_count == 0 || tensor_count == 0 ||
        artifact_count > std::numeric_limits<std::size_t>::max() /
                             artifact_record_bytes ||
        tensor_count > std::numeric_limits<std::size_t>::max() /
                           runtime_tensor_record_bytes ||
        artifact_length != static_cast<std::uint64_t>(artifact_count) *
                               artifact_record_bytes ||
        tensor_length != tensor_count * runtime_tensor_record_bytes ||
        artifact_offset != header_bytes ||
        tensor_offset != artifact_offset + artifact_length ||
        string_offset != tensor_offset + tensor_length ||
        string_offset + string_length < string_offset ||
        string_offset + string_length != size ||
        !range_valid(artifact_offset, artifact_length, size) ||
        !range_valid(tensor_offset, tensor_length, size) ||
        !range_valid(string_offset, string_length, size)) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::invalid_directory);
    }

    struct ArtifactMetadata {
        std::filesystem::path path;
        std::array<std::byte, 32> root_sha256{};
        std::uint32_t tensor_count{};
    };
    std::error_code filesystem_error;
    auto base = std::filesystem::canonical(
        path.parent_path().empty() ? std::filesystem::current_path()
                                   : path.parent_path(),
        filesystem_error);
    if (filesystem_error) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::io_error);
    }
    std::vector<ArtifactMetadata> artifacts;
    artifacts.reserve(artifact_count);
    std::unordered_set<std::string> seen_paths;
    std::uint64_t expected_tensors{};
    for (std::uint32_t index = 0; index < artifact_count; ++index) {
        const auto offset = static_cast<std::size_t>(artifact_offset) +
                            static_cast<std::size_t>(index) * artifact_record_bytes;
        const auto path_offset = little<std::uint64_t>(data, offset);
        const auto path_length = little<std::uint32_t>(data, offset + 8);
        const auto count = little<std::uint32_t>(data, offset + 12);
        if (path_length == 0 || count == 0 ||
            !range_valid(path_offset, path_length, string_length) ||
            !all_zero(data.subspan(offset + 48, 16))) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory);
        }
        const auto path_begin = static_cast<std::size_t>(string_offset + path_offset);
        std::string relative(
            reinterpret_cast<const char*>(data.data() + path_begin), path_length);
        if (relative.find('\0') != std::string::npos ||
            relative.find('\\') != std::string::npos ||
            !seen_paths.insert(relative).second) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory, "invalid runtime-index artifact path");
        }
        const std::filesystem::path relative_path(relative);
        if (relative_path.is_absolute()) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory, "absolute runtime-index artifact path");
        }
        for (const auto& component : relative_path) {
            if (component.empty() || component == "." || component == "..") {
                return Result<Glm5xRuntimeIndex>::failure(
                    ErrorCode::invalid_directory,
                    "runtime-index artifact path traversal");
            }
        }
        auto artifact_path = std::filesystem::canonical(
            base / relative_path, filesystem_error);
        if (filesystem_error || !contained_by(base, artifact_path)) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory,
                "runtime-index artifact escapes bundle directory");
        }
        ArtifactMetadata metadata;
        metadata.path = std::move(artifact_path);
        metadata.tensor_count = count;
        std::copy(
            data.begin() + static_cast<std::ptrdiff_t>(offset + 16),
            data.begin() + static_cast<std::ptrdiff_t>(offset + 48),
            metadata.root_sha256.begin());
        artifacts.push_back(std::move(metadata));
        if (expected_tensors > std::numeric_limits<std::uint64_t>::max() - count) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory);
        }
        expected_tensors += count;
    }
    if (expected_tensors != tensor_count) {
        return Result<Glm5xRuntimeIndex>::failure(ErrorCode::invalid_directory);
    }

    Glm5xRuntimeIndex result;
    result.tensors_.reserve(static_cast<std::size_t>(tensor_count));
    std::vector<std::vector<bool>> seen_records;
    seen_records.reserve(artifacts.size());
    for (const auto& artifact : artifacts) {
        seen_records.emplace_back(artifact.tensor_count, false);
    }
    std::uint64_t previous_id{};
    bool has_previous = false;
    for (std::uint64_t index = 0; index < tensor_count; ++index) {
        const auto offset = static_cast<std::size_t>(tensor_offset +
                            index * runtime_tensor_record_bytes);
        TensorLocator locator;
        locator.tensor_id = little<std::uint64_t>(data, offset);
        locator.artifact_index = little<std::uint32_t>(data, offset + 8);
        locator.record_index = little<std::uint32_t>(data, offset + 12);
        locator.data_crc32c = little<std::uint32_t>(data, offset + 16);
        if (little<std::uint32_t>(data, offset + 20) != 0 ||
            locator.artifact_index >= artifacts.size() ||
            locator.record_index >= artifacts[locator.artifact_index].tensor_count ||
            (has_previous && locator.tensor_id <= previous_id) ||
            seen_records[locator.artifact_index][locator.record_index]) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory, "invalid runtime-index tensor locator");
        }
        seen_records[locator.artifact_index][locator.record_index] = true;
        previous_id = locator.tensor_id;
        has_previous = true;
        result.tensors_.push_back(locator);
    }
    for (const auto& artifact_records : seen_records) {
        if (std::find(artifact_records.begin(), artifact_records.end(), false) !=
            artifact_records.end()) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory, "incomplete runtime-index tensor map");
        }
    }

    result.readers_.reserve(artifacts.size());
    for (const auto& artifact : artifacts) {
        auto reader = Reader::open(artifact.path, options);
        if (!reader) {
            auto message = artifact.path.string();
            if (!reader.message().empty()) {
                message += ": " + reader.message();
            }
            return Result<Glm5xRuntimeIndex>::failure(
                reader.error(), std::move(message));
        }
        if (reader.value().superblock().root_sha256 != artifact.root_sha256 ||
            reader.value().tensors().size() != artifact.tensor_count) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory,
                "runtime-index artifact identity mismatch");
        }
        result.readers_.push_back(std::move(reader.value()));
    }
    for (const auto& locator : result.tensors_) {
        const auto& record = result.readers_[locator.artifact_index]
                                 .tensors()[locator.record_index];
        if (record.tensor_id != locator.tensor_id ||
            record.data_crc32c != locator.data_crc32c) {
            return Result<Glm5xRuntimeIndex>::failure(
                ErrorCode::invalid_directory,
                "runtime-index tensor metadata mismatch");
        }
    }
    return Result<Glm5xRuntimeIndex>::success(std::move(result));
}

const Glm5xRuntimeIndex::TensorLocator* Glm5xRuntimeIndex::find(
    std::uint64_t tensor_id) const {
    const auto item = std::lower_bound(
        tensors_.begin(), tensors_.end(), tensor_id,
        [](const TensorLocator& locator, std::uint64_t value) {
            return locator.tensor_id < value;
        });
    return item == tensors_.end() || item->tensor_id != tensor_id
        ? nullptr
        : &*item;
}

Result<std::vector<std::byte>> Glm5xRuntimeIndex::read_tensor(
    std::uint64_t tensor_id) const {
    const auto* locator = find(tensor_id);
    if (locator == nullptr) {
        return Result<std::vector<std::byte>>::failure(
            ErrorCode::tensor_not_found);
    }
    const auto& reader = readers_[locator->artifact_index];
    const auto& record = reader.tensors()[locator->record_index];
    const std::array request{ExtentRequest{record.data_offset, record.data_length}};
    auto payloads = reader.read_extents(request);
    if (!payloads) {
        return Result<std::vector<std::byte>>::failure(
            payloads.error(), payloads.message());
    }
    auto payload = std::move(payloads.value().front());
    if (payload.size() != record.data_length ||
        crc32c(payload) != locator->data_crc32c) {
        return Result<std::vector<std::byte>>::failure(
            ErrorCode::data_crc_mismatch);
    }
    return Result<std::vector<std::byte>>::success(std::move(payload));
}

Result<Glm5xTensorLoad> Glm5xRuntimeIndex::read_tensor_with_metadata(
    std::uint64_t tensor_id) const {
    const auto* locator = find(tensor_id);
    if (locator == nullptr) {
        return Result<Glm5xTensorLoad>::failure(ErrorCode::tensor_not_found);
    }
    auto payload = read_tensor(tensor_id);
    if (!payload) {
        return Result<Glm5xTensorLoad>::failure(
            payload.error(), payload.message());
    }
    Glm5xTensorLoad result;
    result.record = readers_[locator->artifact_index]
                        .tensors()[locator->record_index];
    result.payload = std::move(payload.value());
    return Result<Glm5xTensorLoad>::success(std::move(result));
}

bool Glm5xRuntimeIndex::contains_tensor(std::uint64_t tensor_id) const {
    return find(tensor_id) != nullptr;
}

Result<GlmBf16ExpertLoad> Glm5xRuntimeIndex::read_expert(
    std::uint32_t layer_id,
    std::uint32_t expert_id,
    std::uint64_t hidden_size,
    std::uint64_t intermediate_size) const {
    if (hidden_size == 0 || intermediate_size == 0) {
        return Result<GlmBf16ExpertLoad>::failure(
            ErrorCode::invalid_directory, "invalid GLM expert shape");
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
        const auto* locator = find(tensor_id);
        if (locator == nullptr) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::tensor_not_found, name);
        }
        const auto& record = readers_[locator->artifact_index]
                                 .tensors()[locator->record_index];
        const auto values = shapes[index][0] * shapes[index][1];
        if (record.dtype != 3 || record.quantization != 0 || record.rank != 2 ||
            record.layer_id != static_cast<std::int32_t>(layer_id) ||
            record.expert_id != static_cast<std::int32_t>(expert_id) ||
            record.dimensions[0] != shapes[index][0] ||
            record.dimensions[1] != shapes[index][1] ||
            record.dimensions[2] != 0 || record.dimensions[3] != 0 ||
            record.data_length != values * 2 ||
            record.logical_length != record.data_length ||
            record.auxiliary_offset != 0 || record.auxiliary_length != 0 ||
            record.auxiliary_crc32c != 0) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::invalid_directory,
                "invalid GLM BF16 expert tensor");
        }
        auto payload = read_tensor(tensor_id);
        if (!payload) {
            return Result<GlmBf16ExpertLoad>::failure(
                payload.error(), payload.message());
        }
        if (payload.value().size() != record.data_length ||
            crc32c(payload.value()) != record.data_crc32c) {
            return Result<GlmBf16ExpertLoad>::failure(
                ErrorCode::data_crc_mismatch, name);
        }
        result.payload_bytes += payload.value().size();
        result.roles[index] = std::move(payload.value());
    }
    return Result<GlmBf16ExpertLoad>::success(std::move(result));
}

ReadCounters Glm5xRuntimeIndex::counters() const {
    ReadCounters result;
    for (const auto& reader : readers_) add(result, reader.counters());
    return result;
}

}  // namespace k3x
