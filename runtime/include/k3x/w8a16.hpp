// GLM-5.2 W8A16-G128 가중치의 고정 pack과 reference decode 계약입니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace k3x {

struct W8A16PackedWeight {
    std::vector<std::int8_t> values;
    std::vector<std::byte> scales;
    std::size_t rows{};
    std::size_t cols{};
    std::size_t group_size{};
};

Result<W8A16PackedWeight> pack_w8a16_bf16(
    std::span<const std::byte> bf16, std::size_t rows, std::size_t cols,
    std::size_t group_size = 128);

Result<std::vector<float>> decode_w8a16(const W8A16PackedWeight& weight);

}  // namespace k3x
