// 합성 K3 전문가에서 resident CUDA expert-grid의 정확성과 지연 시간을 측정합니다.
#include "k3x/backend.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::optional<std::size_t> parse_size(std::string_view text) {
    std::size_t value{};
    const auto parsed =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != text.data() + text.size()) return std::nullopt;
    return value;
}

struct ExpertStorage {
    std::array<std::byte, 512> gate_packed{};
    std::array<std::byte, 32> gate_scales{};
    std::array<std::byte, 512> up_packed{};
    std::array<std::byte, 32> up_scales{};
    std::array<std::byte, 512> down_packed{};
    std::array<std::byte, 32> down_scales{};
};

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 != 0
        ? values[middle]
        : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t expert_count = 1;
    std::size_t token_count = 1;
    std::size_t warmup = 0;
    std::size_t iterations = 1;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) return 2;
        const std::string key = argv[index];
        const auto value = parse_size(argv[index + 1]);
        if (!value) return 2;
        if (key == "--experts") expert_count = *value;
        else if (key == "--tokens") token_count = *value;
        else if (key == "--warmup") warmup = *value;
        else if (key == "--iterations") iterations = *value;
        else return 2;
    }
    const auto supported = [](std::size_t value) {
        return value == 1 || value == 2 || value == 4;
    };
    if (!supported(expert_count) || !supported(token_count) ||
        iterations == 0) return 2;

    std::vector<ExpertStorage> storage(expert_count);
    std::vector<k3x::Mxfp4MlpView> experts;
    experts.reserve(expert_count);
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        auto& item = storage[expert];
        item.gate_packed[0] = std::byte{0x10};
        item.up_packed[0] = std::byte{0x20};
        item.down_packed[0] = std::byte{0x01};
        item.gate_scales.fill(std::byte{127});
        item.up_scales.fill(std::byte{127});
        item.down_scales.fill(std::byte{127});
        experts.push_back({
            {100 + expert * 3, item.gate_packed, item.gate_scales, 32, 32, 32},
            {101 + expert * 3, item.up_packed, item.up_scales, 32, 32, 32},
            {102 + expert * 3, item.down_packed, item.down_scales, 32, 32, 32},
        });
    }
    std::vector<float> inputs(token_count * 32);
    for (std::size_t token = 0; token < token_count; ++token) {
        inputs[token * 32 + 1] = static_cast<float>(token + 1);
    }
    auto cpu = k3x::make_cpu_backend();
    const auto reference = cpu->mxfp4_situ_mlp_grid(
        inputs, token_count, experts, 1.0F, std::nullopt, 1,
        k3x::ProfilePhase::decode);
    if (!reference) return 4;

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_resident_bytes = 8U * 1024U * 1024U;
    k3x::Profiler profiler;
    auto created = k3x::make_cuda_backend(options, &profiler);
    if (!created) return 4;
    auto& backend = *created.value();
    const auto execute = [&] {
        return backend.mxfp4_situ_mlp_grid(
            inputs, token_count, experts, 1.0F, std::nullopt, 1,
            k3x::ProfilePhase::decode);
    };
    for (std::size_t index = 0; index < warmup; ++index) {
        if (!execute()) return 4;
    }
    const auto runtime_before = backend.runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    std::vector<std::vector<float>> actual;
    for (std::size_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto result = execute();
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        if (!result) return 4;
        actual = std::move(result.value());
        samples.push_back(elapsed);
    }
    float maximum_error = 0.0F;
    for (std::size_t expert = 0; expert < expert_count; ++expert) {
        for (std::size_t index = 0; index < actual[expert].size(); ++index) {
            maximum_error = std::max(
                maximum_error,
                std::abs(actual[expert][index] - reference.value()[expert][index]));
        }
    }
    const auto runtime_after = backend.runtime_stats();
    const auto profile_after = profiler.summary();
    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"synthetic_k3_grid\""
              << ",\"experts\":" << expert_count
              << ",\"tokens\":" << token_count
              << ",\"warmup\":" << warmup
              << ",\"iterations\":" << iterations
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"kernel_nanoseconds\":"
              << profile_after.device_nanoseconds - profile_before.device_nanoseconds
              << ",\"host_to_device_bytes\":"
              << profile_after.host_to_device_bytes - profile_before.host_to_device_bytes
              << ",\"device_to_host_bytes\":"
              << profile_after.device_to_host_bytes - profile_before.device_to_host_bytes
              << ",\"resident_grid_calls\":"
              << runtime_after.resident_grid_calls - runtime_before.resident_grid_calls
              << ",\"resident_grid_kernel_launches\":"
              << runtime_after.resident_grid_kernel_launches -
                     runtime_before.resident_grid_kernel_launches
              << ",\"resident_grid_fallbacks\":"
              << runtime_after.resident_grid_fallbacks -
                     runtime_before.resident_grid_fallbacks
              << ",\"peak_vram_bytes\":"
              << backend.memory_stats().peak_device_bytes << "}\n";
    return 0;
}
