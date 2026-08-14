// C++ RMSNorm, SiTU, MXFP4 literal 결과를 독립 검증합니다.
#include "k3x/ops.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main() {
    std::array<float, 2> input{3.0F, 4.0F}, weight{1.0F, 2.0F}, output{};
    k3x::rms_norm(output, input, weight, 0.0F);
    const float scale = std::sqrt(12.5F);
    if (std::abs(output[0] - 3.0F / scale) > 1e-6F ||
        std::abs(output[1] - 8.0F / scale) > 1e-6F) return 1;
    std::array<float, 2> gate{-1.0F, 2.0F}, up{3.0F, 4.0F}, activated{};
    k3x::situ_glu(activated, gate, up, 1.0F, 1.0F);
    const auto expected0 = std::tanh(-1.0F) * (1.0F / (1.0F + std::exp(1.0F))) * std::tanh(3.0F);
    if (std::abs(activated[0] - expected0) > 1e-6F) return 2;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    std::array<std::byte, 1> scales{std::byte{127}};
    const auto decoded = k3x::decode_mxfp4(packed, scales, 1, 32, 32);
    if (!decoded || decoded.value()[0] != 0.0F || decoded.value()[1] != 0.5F) return 3;
    return 0;
}
