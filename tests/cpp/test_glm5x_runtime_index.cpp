// Python이 만든 GLM5X runtime index의 C++ exact expert 읽기를 검증합니다.
#include "k3x/checksums.hpp"
#include "k3x/glm5x_runtime_index.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

namespace {

std::optional<std::uint64_t> parse_u64(const char* text) {
    std::uint64_t value{};
    const auto end = text + std::char_traits<char>::length(text);
    const auto result = std::from_chars(text, end, value);
    if (result.ec != std::errc{} || result.ptr != end) return std::nullopt;
    return value;
}

std::string hex(const std::array<std::byte, 32>& value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(64);
    for (const auto item : value) {
        const auto byte = std::to_integer<unsigned char>(item);
        output.push_back(digits[byte >> 4]);
        output.push_back(digits[byte & 15]);
    }
    return output;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    const auto layer = parse_u64(argv[2]);
    const auto expert = parse_u64(argv[3]);
    const auto hidden = parse_u64(argv[4]);
    const auto intermediate = parse_u64(argv[5]);
    if (!layer || !expert || !hidden || !intermediate ||
        *layer > UINT32_MAX || *expert > UINT32_MAX) {
        return 2;
    }
    k3x::ReaderOptions options;
    options.verify = k3x::VerifyMode::metadata_only;
    auto index = k3x::Glm5xRuntimeIndex::open(
        std::filesystem::path(argv[1]), options);
    if (!index) {
        std::cerr << k3x::error_code_name(index.error()) << ": "
                  << index.message() << '\n';
        return 3;
    }
    auto loaded = index.value().read_expert(
        static_cast<std::uint32_t>(*layer),
        static_cast<std::uint32_t>(*expert), *hidden, *intermediate);
    if (!loaded) {
        std::cerr << k3x::error_code_name(loaded.error()) << ": "
                  << loaded.message() << '\n';
        return 4;
    }
    const auto prefix =
        "model.layers." + std::to_string(*layer) + ".mlp.experts." +
        std::to_string(*expert) + ".gate_proj.weight";
    const auto gate_id = k3x::fnv1a64(prefix.c_str());
    if (!index.value().contains_tensor(gate_id) ||
        index.value().contains_tensor(0)) {
        return 5;
    }
    auto gate = index.value().read_tensor_with_metadata(gate_id);
    if (!gate) {
        std::cerr << k3x::error_code_name(gate.error()) << ": "
                  << gate.message() << '\n';
        return 5;
    }
    const auto counters = index.value().counters();
    std::cout << "{\"artifact_count\":" << index.value().artifact_count()
              << ",\"tensor_count\":" << index.value().tensor_count()
              << ",\"payload_bytes\":" << loaded.value().payload_bytes
              << ",\"reader_read_calls\":" << counters.calls
              << ",\"reader_completed_bytes\":" << counters.completed_bytes
              << ",\"gate_dtype\":" << gate.value().record.dtype
              << ",\"gate_rank\":" << static_cast<unsigned>(gate.value().record.rank)
              << ",\"gate_dimensions\":["
              << gate.value().record.dimensions[0] << ','
              << gate.value().record.dimensions[1] << ']'
              << ",\"gate_payload_bytes\":" << gate.value().payload.size()
              << ",\"role_sha256\":[";
    for (std::size_t role = 0; role < loaded.value().roles.size(); ++role) {
        if (role) std::cout << ',';
        std::cout << '\"' << hex(k3x::sha256(loaded.value().roles[role])) << '\"';
    }
    std::cout << "]}\n";
    return 0;
}
