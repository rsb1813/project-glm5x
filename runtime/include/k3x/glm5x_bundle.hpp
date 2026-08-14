// GLM5X cross-shard raw BF16 expert를 검증하고 읽는 C++ 경계를 정의합니다.
#pragma once

#include "k3x/reader.hpp"
#include "k3x/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace k3x {

struct GlmBf16ExpertLoad {
    std::array<std::vector<std::byte>, 3> roles;
    std::array<std::array<std::uint64_t, 2>, 3> shapes{};
    std::uint64_t payload_bytes{};
};

Result<GlmBf16ExpertLoad> load_glm5x_bf16_expert(
    std::span<const Reader* const> shards,
    std::uint32_t layer_id,
    std::uint32_t expert_id,
    std::uint64_t hidden_size = 6144,
    std::uint64_t intermediate_size = 2048);

}  // namespace k3x
