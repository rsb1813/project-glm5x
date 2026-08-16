// BF16 expert 가중치를 deterministic W8A16-G128 payload로 변환합니다.
#include "k3x/w8a16.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace k3x {
namespace {

constexpr std::size_t kGroupSize = 128;

float bf16_to_float(const std::byte* bytes) {
    const auto encoded = static_cast<std::uint16_t>(
        std::to_integer<std::uint8_t>(bytes[0]) |
        (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(bytes[1]))
         << 8U));
    return std::bit_cast<float>(static_cast<std::uint32_t>(encoded) << 16U);
}

std::uint16_t float_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

bool valid_shape(std::size_t payload_bytes, std::size_t rows,
                 std::size_t cols, std::size_t group_size) {
    return rows != 0 && cols != 0 && group_size == kGroupSize &&
           cols % group_size == 0 &&
           rows <= std::numeric_limits<std::size_t>::max() / cols &&
           rows * cols <= std::numeric_limits<std::size_t>::max() / 2 &&
           payload_bytes == rows * cols * 2;
}

}  // namespace

Result<W8A16PackedWeight> pack_w8a16_bf16(
    std::span<const std::byte> bf16, std::size_t rows, std::size_t cols,
    std::size_t group_size) {
    if (!valid_shape(bf16.size(), rows, cols, group_size)) {
        return Result<W8A16PackedWeight>::failure(ErrorCode::invalid_extent);
    }
    const auto group_count = rows * (cols / group_size);
    W8A16PackedWeight packed;
    packed.values.resize(rows * cols);
    packed.scales.resize(group_count * 2);
    packed.rows = rows;
    packed.cols = cols;
    packed.group_size = group_size;
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t group = 0; group < cols / group_size; ++group) {
            const auto start = row * cols + group * group_size;
            float maximum = 0.0F;
            for (std::size_t column = 0; column < group_size; ++column) {
                const auto value = bf16_to_float(
                    bf16.data() + (start + column) * 2);
                if (!std::isfinite(value)) {
                    return Result<W8A16PackedWeight>::failure(
                        ErrorCode::invalid_extent);
                }
                maximum = std::max(maximum, std::abs(value));
            }
            const auto scale_bits = float_to_bf16(maximum / 127.0F);
            const auto scale = bf16_to_float(
                reinterpret_cast<const std::byte*>(&scale_bits));
            const auto scale_index = row * (cols / group_size) + group;
            packed.scales[scale_index * 2] =
                std::byte(scale_bits & 0xffU);
            packed.scales[scale_index * 2 + 1] =
                std::byte(scale_bits >> 8U);
            for (std::size_t column = 0; column < group_size; ++column) {
                const auto value = bf16_to_float(
                    bf16.data() + (start + column) * 2);
                const auto quantized = scale == 0.0F
                    ? 0L
                    : std::clamp(std::lround(value / scale), -127L, 127L);
                packed.values[start + column] =
                    static_cast<std::int8_t>(quantized);
            }
        }
    }
    return Result<W8A16PackedWeight>::success(std::move(packed));
}

Result<std::vector<float>> decode_w8a16(const W8A16PackedWeight& weight) {
    if (!valid_shape(weight.values.size() * 2, weight.rows, weight.cols,
                     weight.group_size) ||
        weight.scales.size() !=
            weight.rows * (weight.cols / weight.group_size) * 2) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> decoded(weight.values.size());
    const auto groups_per_row = weight.cols / weight.group_size;
    for (std::size_t row = 0; row < weight.rows; ++row) {
        for (std::size_t group = 0; group < groups_per_row; ++group) {
            const auto scale_index = row * groups_per_row + group;
            const auto scale = bf16_to_float(
                weight.scales.data() + scale_index * 2);
            const auto start = row * weight.cols + group * weight.group_size;
            for (std::size_t column = 0; column < weight.group_size; ++column) {
                decoded[start + column] =
                    static_cast<float>(weight.values[start + column]) * scale;
            }
        }
    }
    return Result<std::vector<float>>::success(std::move(decoded));
}

}  // namespace k3x
