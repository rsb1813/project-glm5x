// GLM5X 레이어 사이의 BF16 hidden-state 배치 파일 경계를 정의합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace k3x {

struct Glm5xBf16ActivationBatch {
    std::uint32_t token_count{};
    std::uint32_t hidden_size{};
    std::vector<std::byte> payload;
};

Result<Glm5xBf16ActivationBatch> load_glm5x_bf16_activation(
    const std::filesystem::path& path,
    std::uint32_t expected_token_count = 0,
    std::uint32_t expected_hidden_size = 0);

}  // namespace k3x
