// PyTorch oracle와 비교할 portable FP32 및 MXFP4 연산을 구현합니다.
#include "k3x/ops.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace k3x {
void rms_norm(std::span<float> output, std::span<const float> input,
              std::span<const float> weight, float epsilon) {
    double squares = 0.0;
    for (const auto value : input) squares += static_cast<double>(value) * value;
    const float inverse = 1.0F / std::sqrt(static_cast<float>(squares / input.size()) + epsilon);
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * inverse * weight[index];
    }
}

void situ_glu(std::span<float> output, std::span<const float> gate,
              std::span<const float> up, float beta,
              std::optional<float> linear_beta) {
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto sigmoid = 1.0F / (1.0F + std::exp(-gate[index]));
        const auto bounded_gate = beta * std::tanh(gate[index] / beta) * sigmoid;
        const auto bounded_up = linear_beta ? *linear_beta * std::tanh(up[index] / *linear_beta)
                                            : up[index];
        output[index] = bounded_gate * bounded_up;
    }
}

Result<std::vector<float>> decode_mxfp4(std::span<const std::byte> packed,
                                       std::span<const std::byte> scales,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t group_size) {
    if (!rows || !cols || !group_size || cols % group_size || cols % 2 ||
        packed.size() != rows * cols / 2 || scales.size() != rows * cols / group_size) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
    }
    constexpr std::array<float, 16> values{
        0.0F,0.5F,1.0F,1.5F,2.0F,3.0F,4.0F,6.0F,
        -0.0F,-0.5F,-1.0F,-1.5F,-2.0F,-3.0F,-4.0F,-6.0F,
    };
    std::vector<float> output(rows * cols);
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto scale_byte = std::to_integer<std::uint8_t>(scales[index / group_size]);
        if (scale_byte == 0xffU) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
        }
        const auto pair = std::to_integer<std::uint8_t>(packed[index / 2]);
        const auto code = index % 2 ? pair >> 4U : pair & 0x0fU;
        output[index] = std::ldexp(values[code], static_cast<int>(scale_byte) - 127);
    }
    return Result<std::vector<float>>::success(std::move(output));
}

Result<std::vector<float>> mxfp4_matmul(std::span<const float> input,
                                       std::span<const std::byte> packed,
                                       std::span<const std::byte> scales,
                                       std::size_t rows, std::size_t cols,
                                       std::size_t group_size) {
    if (input.size() != cols) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
    }
    auto decoded = decode_mxfp4(packed, scales, rows, cols, group_size);
    if (!decoded) return decoded;
    std::vector<float> output(rows);
    for (std::size_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < cols; ++column) {
            sum += static_cast<double>(input[column]) * decoded.value()[row * cols + column];
        }
        output[row] = static_cast<float>(sum);
    }
    return Result<std::vector<float>>::success(std::move(output));
}
}
