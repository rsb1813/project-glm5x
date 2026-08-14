// 실제 GLM5X shard에서 cross-shard BF16 expert loader를 검증합니다.
#include "k3x/glm5x_bundle.hpp"

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

std::optional<std::uint32_t> parse_u32(const char* text) {
    std::uint32_t value{};
    const auto result = std::from_chars(text, text + std::char_traits<char>::length(text), value);
    if (result.ec != std::errc{} || *result.ptr != '\0') return std::nullopt;
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    const auto layer = parse_u32(argv[argc - 2]);
    const auto expert = parse_u32(argv[argc - 1]);
    if (!layer || !expert) return 2;
    k3x::ReaderOptions options;
    options.verify = k3x::VerifyMode::metadata_only;
    std::vector<k3x::Reader> readers;
    readers.reserve(static_cast<std::size_t>(argc - 3));
    for (int index = 1; index < argc - 2; ++index) {
        auto reader = k3x::Reader::open(std::filesystem::path(argv[index]), options);
        if (!reader) {
            std::cerr << k3x::error_code_name(reader.error()) << '\n';
            return 3;
        }
        readers.push_back(std::move(reader.value()));
    }
    std::vector<const k3x::Reader*> shards;
    shards.reserve(readers.size());
    for (const auto& reader : readers) shards.push_back(&reader);
    const auto started = std::chrono::steady_clock::now();
    const auto loaded = k3x::load_glm5x_bf16_expert(
        shards, *layer, *expert);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    if (!loaded) {
        std::cerr << k3x::error_code_name(loaded.error()) << ": "
                  << loaded.message() << '\n';
        return 4;
    }
    std::cout << "{\"layer_id\":" << *layer
              << ",\"expert_id\":" << *expert
              << ",\"payload_bytes\":" << loaded.value().payload_bytes
              << ",\"load_nanoseconds\":" << elapsed
              << ",\"roles\":3}\n";
    return 0;
}
