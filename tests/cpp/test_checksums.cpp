// 표준 벡터로 portable CRC32C와 SHA-256을 검증합니다.
#include "k3x/checksums.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <span>
#include <string_view>

int main() {
    constexpr std::string_view literal = "123456789";
    auto bytes = std::as_bytes(std::span(literal.data(), literal.size()));
    if (k3x::crc32c(bytes) != 0xe3069283U) {
        std::cerr << "CRC32C standard vector mismatch\n";
        return 1;
    }
    constexpr std::array<std::uint8_t, 32> expected{
        0x15, 0xe2, 0xb0, 0xd3, 0xc3, 0x38, 0x91, 0xeb,
        0xb0, 0xf1, 0xef, 0x60, 0x9e, 0xc4, 0x19, 0x42,
        0x0c, 0x20, 0xe3, 0x20, 0xce, 0x94, 0xc6, 0x5f,
        0xbc, 0x8c, 0x33, 0x12, 0x44, 0x8e, 0xb2, 0x25,
    };
    const auto actual = k3x::sha256(bytes);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (std::to_integer<std::uint8_t>(actual[index]) != expected[index]) {
            std::cerr << "SHA-256 standard vector mismatch\n";
            return 1;
        }
    }
    k3x::Sha256Hasher incremental;
    incremental.update(bytes.first(4));
    incremental.update(bytes.subspan(4));
    if (incremental.finish() != actual) return 2;
    return 0;
}
