// GLM5X BF16 hidden-state 배치 파일을 검증하고 스트리밍 없이 읽습니다.
#include "k3x/glm5x_activation.hpp"

#include "k3x/checksums.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>

namespace k3x {
namespace {

constexpr std::array<std::byte, 8> kMagic{
    std::byte{'G'}, std::byte{'L'}, std::byte{'M'}, std::byte{'5'},
    std::byte{'X'}, std::byte{'A'}, std::byte{'C'}, std::byte{'T'}};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kHeaderBytes = 40;
constexpr std::uint16_t kBf16Dtype = 3;

std::uint16_t read_u16(const std::array<std::byte, kHeaderBytes>& header,
                       std::size_t offset) {
    return static_cast<std::uint16_t>(
        std::to_integer<std::uint8_t>(header[offset])) |
        static_cast<std::uint16_t>(
            std::to_integer<std::uint8_t>(header[offset + 1]) << 8U);
}

std::uint32_t read_u32(const std::array<std::byte, kHeaderBytes>& header,
                       std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        value |= static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(
                     header[offset + index]))
                 << (index * 8U);
    }
    return value;
}

std::uint64_t read_u64(const std::array<std::byte, kHeaderBytes>& header,
                       std::size_t offset) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(
                     header[offset + index]))
                 << (index * 8U);
    }
    return value;
}

}  // namespace

Result<Glm5xBf16ActivationBatch> load_glm5x_bf16_activation(
    const std::filesystem::path& path, std::uint32_t expected_token_count,
    std::uint32_t expected_hidden_size) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::io_error, path.string());
    }
    const auto end = stream.tellg();
    if (end < 0 || static_cast<std::uint64_t>(end) < kHeaderBytes) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::truncated_file, "activation header");
    }
    std::array<std::byte, kHeaderBytes> header{};
    stream.seekg(0, std::ios::beg);
    if (!stream.read(reinterpret_cast<char*>(header.data()), header.size())) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::io_error, "activation header read");
    }
    if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::bad_magic, "GLM5XACT");
    }
    if (read_u32(header, 8) != kVersion ||
        read_u32(header, 12) != kHeaderBytes ||
        read_u16(header, 24) != kBf16Dtype || read_u16(header, 26) != 0) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::unsupported_version, "activation header");
    }
    const auto token_count = read_u32(header, 16);
    const auto hidden_size = read_u32(header, 20);
    const auto payload_bytes = read_u64(header, 28);
    const auto expected_crc = read_u32(header, 36);
    if (token_count == 0 || hidden_size == 0 ||
        (expected_token_count != 0 && token_count != expected_token_count) ||
        (expected_hidden_size != 0 && hidden_size != expected_hidden_size) ||
        token_count > std::numeric_limits<std::uint64_t>::max() / hidden_size ||
        payload_bytes != static_cast<std::uint64_t>(token_count) * hidden_size * 2U ||
        payload_bytes > std::numeric_limits<std::size_t>::max() ||
        static_cast<std::uint64_t>(end) != kHeaderBytes + payload_bytes) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::invalid_extent, "activation dimensions");
    }
    Glm5xBf16ActivationBatch result{token_count, hidden_size,
                                    std::vector<std::byte>(
                                        static_cast<std::size_t>(payload_bytes))};
    if (!stream.read(reinterpret_cast<char*>(result.payload.data()),
                     static_cast<std::streamsize>(result.payload.size()))) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::io_error, "activation payload read");
    }
    if (crc32c(result.payload) != expected_crc) {
        return Result<Glm5xBf16ActivationBatch>::failure(
            ErrorCode::data_crc_mismatch, "activation payload");
    }
    return Result<Glm5xBf16ActivationBatch>::success(std::move(result));
}

}  // namespace k3x
