// 실제 K3 크기 expert의 여섯 storage extent를 한 batch로 읽는 계약을 정의합니다.
#pragma once

#include "k3x/reader.hpp"
#include "k3x/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace k3x {

struct StorageExpertLoad {
    std::array<std::vector<std::byte>, 6> extents;
    std::array<std::byte, 32> ordered_sha256{};
    std::uint64_t logical_bytes{};
};

Result<StorageExpertLoad> load_storage_expert(
    Reader& reader,
    std::uint32_t layer_id,
    std::uint32_t expert_id);

}  // namespace k3x
