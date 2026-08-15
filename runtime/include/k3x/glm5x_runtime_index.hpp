// GLM5X 공식 shard의 tensor 소유권을 검증하고 exact payload를 읽습니다.
#pragma once

#include "k3x/glm5x_bundle.hpp"
#include "k3x/reader.hpp"
#include "k3x/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace k3x {

struct Glm5xTensorLoad {
    TensorRecord record;
    std::vector<std::byte> payload;
};

class Glm5xRuntimeIndex {
public:
    Glm5xRuntimeIndex(Glm5xRuntimeIndex&&) noexcept = default;
    Glm5xRuntimeIndex& operator=(Glm5xRuntimeIndex&&) noexcept = default;
    Glm5xRuntimeIndex(const Glm5xRuntimeIndex&) = delete;
    Glm5xRuntimeIndex& operator=(const Glm5xRuntimeIndex&) = delete;

    static Result<Glm5xRuntimeIndex> open(
        const std::filesystem::path& path,
        ReaderOptions options);

    Result<std::vector<std::byte>> read_tensor(std::uint64_t tensor_id) const;
    Result<Glm5xTensorLoad> read_tensor_with_metadata(
        std::uint64_t tensor_id) const;
    bool contains_tensor(std::uint64_t tensor_id) const;
    Result<GlmBf16ExpertLoad> read_expert(
        std::uint32_t layer_id,
        std::uint32_t expert_id,
        std::uint64_t hidden_size = 6144,
        std::uint64_t intermediate_size = 2048) const;

    std::size_t artifact_count() const { return readers_.size(); }
    std::size_t tensor_count() const { return tensors_.size(); }
    ReadCounters counters() const;

private:
    Glm5xRuntimeIndex() = default;

    struct TensorLocator {
        std::uint64_t tensor_id{};
        std::uint32_t artifact_index{};
        std::uint32_t record_index{};
        std::uint32_t data_crc32c{};
    };

    const TensorLocator* find(std::uint64_t tensor_id) const;

    std::vector<Reader> readers_;
    std::vector<TensorLocator> tensors_;
};

}  // namespace k3x
