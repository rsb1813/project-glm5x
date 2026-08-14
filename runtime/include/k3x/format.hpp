// K3X v1의 host struct와 고정 크기 상수를 정의합니다.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace k3x {
constexpr std::size_t superblock_bytes = 4096;
constexpr std::size_t tensor_record_bytes = 128;
constexpr std::size_t layer_record_bytes = 64;
constexpr std::size_t expert_record_bytes = 64;
constexpr std::size_t model_config_bytes = 256;
inline constexpr std::uint64_t optional_storage_fixture = 1ULL << 0;

struct Superblock {
    std::uint32_t state{};
    std::uint64_t required_features{};
    std::uint64_t optional_features{};
    std::uint64_t tensor_directory_offset{};
    std::uint64_t tensor_directory_length{};
    std::uint64_t layer_directory_offset{};
    std::uint64_t layer_directory_length{};
    std::uint64_t expert_directory_offset{};
    std::uint64_t expert_directory_length{};
    std::uint64_t model_config_offset{};
    std::uint64_t model_config_length{};
    std::uint64_t payload_offset{};
    std::uint64_t file_length{};
    std::array<std::byte, 32> directory_sha256{};
    std::array<std::byte, 32> root_sha256{};
};

struct TensorRecord {
    std::uint64_t tensor_id{};
    std::uint16_t dtype{};
    std::uint16_t quantization{};
    std::uint8_t rank{};
    std::int32_t layer_id{};
    std::int32_t expert_id{};
    std::array<std::uint64_t, 4> dimensions{};
    std::uint64_t data_offset{};
    std::uint64_t data_length{};
    std::uint64_t logical_length{};
    std::uint64_t auxiliary_offset{};
    std::uint64_t auxiliary_length{};
    std::uint32_t data_crc32c{};
    std::uint32_t auxiliary_crc32c{};
};

std::uint64_t fnv1a64(const char* value);
}
