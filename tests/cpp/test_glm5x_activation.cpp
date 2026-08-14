// GLM5X BF16 hidden-state 배치 헤더와 CRC 검증을 확인합니다.
#include "k3x/checksums.hpp"
#include "k3x/glm5x_activation.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void put_u16(std::array<std::byte, 40>& header, std::size_t offset,
             std::uint16_t value) {
    header[offset] = std::byte(value & 0xffU);
    header[offset + 1] = std::byte((value >> 8U) & 0xffU);
}

void put_u32(std::array<std::byte, 40>& header, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t index = 0; index < 4; ++index) {
        header[offset + index] = std::byte((value >> (index * 8U)) & 0xffU);
    }
}

void put_u64(std::array<std::byte, 40>& header, std::size_t offset,
             std::uint64_t value) {
    for (std::size_t index = 0; index < 8; ++index) {
        header[offset + index] = std::byte((value >> (index * 8U)) & 0xffU);
    }
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() /
                      "glm5x_activation_test.bin";
    const std::vector<std::byte> payload{
        std::byte{0x00}, std::byte{0x3f}, std::byte{0x80}, std::byte{0x3f},
        std::byte{0x00}, std::byte{0x40}, std::byte{0x80}, std::byte{0x40}};
    std::array<std::byte, 40> header{};
    const std::array<char, 8> magic{'G', 'L', 'M', '5', 'X', 'A', 'C', 'T'};
    for (std::size_t index = 0; index < magic.size(); ++index) {
        header[index] = std::byte{static_cast<unsigned char>(magic[index])};
    }
    put_u32(header, 8, 1);
    put_u32(header, 12, 40);
    put_u32(header, 16, 1);
    put_u32(header, 20, 4);
    put_u16(header, 24, 3);
    put_u64(header, 28, payload.size());
    put_u32(header, 36, k3x::crc32c(payload));
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        stream.write(reinterpret_cast<const char*>(header.data()),
                     static_cast<std::streamsize>(header.size()));
        stream.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size()));
    }
    const auto loaded = k3x::load_glm5x_bf16_activation(path, 1, 4);
    std::filesystem::remove(path);
    if (!loaded || loaded.value().token_count != 1 ||
        loaded.value().hidden_size != 4 || loaded.value().payload != payload) {
        return 1;
    }
    return 0;
}
