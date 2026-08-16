// W8A16-G128 가중치 pack과 reference decode 계약을 검증합니다.
#include "k3x/w8a16.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace {

std::uint16_t float_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

std::vector<std::byte> make_bf16(std::span<const float> values) {
    std::vector<std::byte> bytes(values.size() * 2);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto encoded = float_to_bf16(values[index]);
        bytes[index * 2] = std::byte(encoded & 0xffU);
        bytes[index * 2 + 1] = std::byte(encoded >> 8U);
    }
    return bytes;
}

}  // namespace

int main() {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 128;
    std::vector<float> source(rows * cols);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(static_cast<int>(index % 31) - 15) /
                        static_cast<float>(index < cols ? 15 : 30);
    }
    const auto bf16 = make_bf16(source);
    const auto packed = k3x::pack_w8a16_bf16(bf16, rows, cols, 128);
    const auto repeated = k3x::pack_w8a16_bf16(bf16, rows, cols, 128);
    if (!packed || !repeated || packed.value().values.size() != rows * cols ||
        packed.value().scales.size() != rows * 2 ||
        packed.value().values != repeated.value().values ||
        packed.value().scales != repeated.value().scales ||
        packed.value().rows != rows || packed.value().cols != cols ||
        packed.value().group_size != 128) {
        return 1;
    }
    const auto decoded = k3x::decode_w8a16(packed.value());
    if (!decoded || decoded.value().size() != source.size()) return 2;
    float maximum_error = 0.0F;
    for (std::size_t index = 0; index < source.size(); ++index) {
        maximum_error = std::max(
            maximum_error, std::abs(decoded.value()[index] - source[index]));
    }
    if (!std::isfinite(maximum_error) || maximum_error > 0.01F) return 3;

    const auto short_payload = std::span<const std::byte>(bf16).first(
        bf16.size() - 2);
    if (k3x::pack_w8a16_bf16(short_payload, rows, cols, 128) ||
        k3x::pack_w8a16_bf16(bf16, rows, cols - 1, 128) ||
        k3x::pack_w8a16_bf16(bf16, rows, cols, 64)) {
        return 4;
    }
    auto malformed = packed.value();
    malformed.scales.pop_back();
    if (k3x::decode_w8a16(malformed)) return 5;
    return 0;
}
