// Castagnoli 다항식으로 portable CRC32C를 계산합니다.
#include "k3x/checksums.hpp"

#include <array>

namespace k3x {
namespace {
constexpr auto make_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t value = 0; value < table.size(); ++value) {
        auto crc = value;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1U) ^ (0x82f63b78U & (0U - (crc & 1U)));
        }
        table[value] = crc;
    }
    return table;
}
constexpr auto table = make_table();
}

std::uint32_t crc32c(std::span<const std::byte> data) {
    std::uint32_t crc = 0xffffffffU;
    for (const auto byte : data) {
        crc = table[(crc ^ std::to_integer<std::uint8_t>(byte)) & 0xffU] ^ (crc >> 8U);
    }
    return ~crc;
}
}
