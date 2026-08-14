// C++ reference runtime의 독립 수치 primitive를 선언합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace k3x {
void rms_norm(std::span<float> output, std::span<const float> input,
              std::span<const float> weight, float epsilon);
void situ_glu(std::span<float> output, std::span<const float> gate,
              std::span<const float> up, float beta,
              std::optional<float> linear_beta);
Result<std::vector<float>> decode_mxfp4(std::span<const std::byte> packed,
                                       std::span<const std::byte> scales,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t group_size);
Result<std::vector<float>> mxfp4_matmul(std::span<const float> input,
                                       std::span<const std::byte> packed,
                                       std::span<const std::byte> scales,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t group_size);
}
