// K3X 파일 검증에 쓰이는 portable checksum 인터페이스를 선언합니다.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace k3x {

std::uint32_t crc32c(std::span<const std::byte> data);

class Sha256Hasher {
public:
    Sha256Hasher();
    void update(std::span<const std::byte> data);
    std::array<std::byte, 32> finish();
private:
    void compress(const std::byte* block);
    std::array<std::uint32_t, 8> state_{};
    std::array<std::byte, 64> buffer_{};
    std::size_t buffered_{};
    std::uint64_t total_bytes_{};
    bool finished_{};
};

std::array<std::byte, 32> sha256(std::span<const std::byte> data);

}
