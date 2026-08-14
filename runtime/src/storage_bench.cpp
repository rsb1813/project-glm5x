// Reader I/O mode별 full-dimension expert load latency와 byte accounting을 측정합니다.
#include "k3x/reader.hpp"
#include "k3x/status.hpp"
#include "k3x/storage_slice.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct ProcessIo {
    bool available{};
    std::uint64_t rchar{};
    std::uint64_t read_bytes{};
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

ProcessIo process_io_snapshot() {
#ifdef __linux__
    std::ifstream stream("/proc/self/io");
    std::string key;
    std::uint64_t value{};
    ProcessIo result;
    bool have_rchar = false;
    bool have_read_bytes = false;
    while (stream >> key >> value) {
        if (key == "rchar:") {
            result.rchar = value;
            have_rchar = true;
        } else if (key == "read_bytes:") {
            result.read_bytes = value;
            have_read_bytes = true;
        }
    }
    result.available = have_rchar && have_read_bytes;
    return result;
#else
    return {};
#endif
}

ProcessIo process_io_delta(const ProcessIo& before, const ProcessIo& after) {
    if (!before.available || !after.available || after.rchar < before.rchar ||
        after.read_bytes < before.read_bytes) {
        return {};
    }
    return {true, after.rchar - before.rchar, after.read_bytes - before.read_bytes};
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

std::string digest_hex(const std::array<std::byte, 32>& digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (const auto value : digest) {
        const auto byte = std::to_integer<unsigned char>(value);
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 0x0F]);
    }
    return result;
}

std::uint64_t percentile(
    const std::vector<std::uint64_t>& sorted,
    std::uint64_t numerator,
    std::uint64_t denominator) {
    const auto rank = (sorted.size() * numerator + denominator - 1) / denominator;
    const auto index = static_cast<std::size_t>(
        std::clamp<std::uint64_t>(rank, 1, sorted.size()) - 1);
    return sorted[index];
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path;
    std::uint64_t layer_id = 1;
    std::uint64_t expert_id = 0;
    std::uint64_t warmup = 0;
    std::uint64_t iterations = 1;
    std::string io_name = "pread";
    std::string cache_name = "buffered";
    std::uint64_t queue_depth = 8;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return 2;
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        const auto number = parse_u64(value);
        if (key == "--model") model_path = value;
        else if (key == "--layer" && number) layer_id = *number;
        else if (key == "--expert" && number) expert_id = *number;
        else if (key == "--warmup" && number) warmup = *number;
        else if (key == "--iterations" && number) iterations = *number;
        else if (key == "--l2-queue-depth" && number) queue_depth = *number;
        else if (key == "--l2-io") io_name = value;
        else if (key == "--l2-cache") cache_name = value;
        else {
            std::cerr << "invalid option: " << key << '\n';
            return 2;
        }
    }
    if (model_path.empty()) {
        std::cerr << "model path is required\n";
        return 2;
    }
    if (iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return 2;
    }
    if (layer_id > std::numeric_limits<std::uint32_t>::max() ||
        expert_id > std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "layer or expert id is out of range\n";
        return 2;
    }
    if (queue_depth == 0 || queue_depth > k3x::maximum_l2_queue_depth) {
        std::cerr << "invalid L2 queue depth\n";
        return 2;
    }
    k3x::ReaderOptions options;
    options.verify = k3x::VerifyMode::metadata_only;
    options.queue_depth = static_cast<std::size_t>(queue_depth);
    if (io_name == "pread") options.io_engine = k3x::L2IoEngine::pread;
    else if (io_name == "io-uring") options.io_engine = k3x::L2IoEngine::io_uring;
    else {
        std::cerr << "unknown L2 I/O engine: " << io_name << '\n';
        return 2;
    }
    if (cache_name == "buffered") options.cache_mode = k3x::L2CacheMode::buffered;
    else if (cache_name == "direct") options.cache_mode = k3x::L2CacheMode::direct;
    else {
        std::cerr << "unknown L2 cache mode: " << cache_name << '\n';
        return 2;
    }

    auto warm_reader = k3x::Reader::open(model_path, options);
    if (!warm_reader) {
        write_error(warm_reader.error(), warm_reader.message());
        return 4;
    }
    for (std::uint64_t index = 0; index < warmup; ++index) {
        auto loaded = k3x::load_storage_expert(
            warm_reader.value(), static_cast<std::uint32_t>(layer_id),
            static_cast<std::uint32_t>(expert_id));
        if (!loaded) {
            write_error(loaded.error(), loaded.message());
            return 4;
        }
    }

    auto reader = k3x::Reader::open(model_path, options);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    const auto io_before = process_io_snapshot();
    std::vector<std::uint64_t> samples;
    samples.reserve(static_cast<std::size_t>(iterations));
    std::array<std::byte, 32> digest{};
    std::uint64_t payload_bytes = 0;
    std::uint64_t total_nanoseconds = 0;
    for (std::uint64_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto loaded = k3x::load_storage_expert(
            reader.value(), static_cast<std::uint32_t>(layer_id),
            static_cast<std::uint32_t>(expert_id));
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        if (!loaded) {
            write_error(loaded.error(), loaded.message());
            return 4;
        }
        if (index != 0 && loaded.value().ordered_sha256 != digest) {
            std::cerr << "INVALID_MXFP4: expert digest changed between iterations\n";
            return 4;
        }
        digest = loaded.value().ordered_sha256;
        payload_bytes = loaded.value().logical_bytes;
        samples.push_back(elapsed);
        total_nanoseconds += elapsed;
    }
    const auto process_io = process_io_delta(io_before, process_io_snapshot());
    std::sort(samples.begin(), samples.end());
    const auto median = samples.size() % 2
        ? samples[samples.size() / 2]
        : samples[samples.size() / 2 - 1] +
            (samples[samples.size() / 2] - samples[samples.size() / 2 - 1]) / 2;
    const auto& counters = reader.value().counters();
    const auto loads_per_second = static_cast<double>(iterations) * 1.0e9 /
        static_cast<double>(total_nanoseconds);

    std::cout << std::setprecision(12)
        << "{\"artifact_kind\":\"storage_fixture\""
        << ",\"layer_id\":" << layer_id
        << ",\"expert_id\":" << expert_id
        << ",\"l2_io_engine\":\"" << io_name << "\""
        << ",\"l2_cache_mode\":\"" << cache_name << "\""
        << ",\"l2_queue_depth\":" << queue_depth
        << ",\"warmup\":" << warmup
        << ",\"iterations\":" << iterations
        << ",\"expert_payload_bytes\":" << payload_bytes
        << ",\"ordered_sha256\":\"" << digest_hex(digest) << "\""
        << ",\"expert_load_nanoseconds_median\":" << median
        << ",\"expert_load_nanoseconds_p05\":"
        << percentile(samples, 5, 100)
        << ",\"expert_load_nanoseconds_p95\":"
        << percentile(samples, 95, 100)
        << ",\"expert_loads_per_second\":" << loads_per_second
        << ",\"reader_read_calls\":" << counters.calls
        << ",\"reader_requested_bytes\":" << counters.requested_bytes
        << ",\"reader_completed_bytes\":" << counters.completed_bytes
        << ",\"reader_batch_submissions\":" << counters.batch_submissions
        << ",\"reader_storage_submitted_bytes\":"
        << counters.storage_submitted_bytes
        << ",\"reader_storage_completed_bytes\":"
        << counters.storage_completed_bytes
        << ",\"reader_completions\":" << counters.completions
        << ",\"reader_short_reads\":" << counters.short_reads
        << ",\"reader_failures\":" << counters.failures
        << ",\"reader_storage_nanoseconds\":" << counters.storage_nanoseconds
        << ",\"l2_direct_memory_alignment\":"
        << reader.value().direct_memory_alignment()
        << ",\"l2_direct_offset_alignment\":"
        << reader.value().direct_offset_alignment()
        << ",\"process_io_available\":"
        << (process_io.available ? "true" : "false")
        << ",\"process_rchar_bytes\":";
    if (process_io.available) std::cout << process_io.rchar;
    else std::cout << "null";
    std::cout << ",\"process_read_bytes\":";
    if (process_io.available) std::cout << process_io.read_bytes;
    else std::cout << "null";
    std::cout << "}\n";
    return 0;
}
