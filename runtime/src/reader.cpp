// K3X v1을 aliasing 없이 little-endian으로 해석하고 검증합니다.
#include "k3x/reader.hpp"

#include "k3x/checksums.hpp"
#include "direct_io.hpp"
#include "io_completion.hpp"
#ifdef K3X_ENABLE_IO_URING
#include "io_ring_guard.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>

#ifndef _WIN32
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef K3X_ENABLE_IO_URING
#include <liburing.h>
#endif

namespace k3x {
namespace {
template <typename T>
T little(std::span<const std::byte> bytes, std::size_t offset) {
    T value{};
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(std::to_integer<std::uint8_t>(bytes[offset + index])) << (index * 8U);
    }
    return value;
}

bool range_valid(std::uint64_t offset, std::uint64_t length, std::uint64_t limit) {
    return offset <= limit && length <= limit - offset;
}

bool all_zero(std::span<const std::byte> value) {
    return std::all_of(value.begin(), value.end(), [](std::byte item) {
        return item == std::byte{0};
    });
}

std::optional<std::uint64_t> directory_count(std::span<const std::byte> directory,
                                             const char (&tag)[5],
                                             std::uint32_t record_size) {
    if (directory.size() < 16 || std::memcmp(directory.data(), tag, 4) ||
        little<std::uint32_t>(directory, 4) != record_size) {
        return std::nullopt;
    }
    const auto count = little<std::uint64_t>(directory, 8);
    if (count > (directory.size() - 16) / record_size ||
        directory.size() != 16 + count * record_size) {
        return std::nullopt;
    }
    return count;
}

Result<std::vector<std::byte>> read_raw(const std::filesystem::path& path,
                                        std::uint64_t offset, std::uint64_t length) {
    if (length > std::numeric_limits<std::size_t>::max()) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::invalid_extent);
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return Result<std::vector<std::byte>>::failure(ErrorCode::io_error);
    stream.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::byte> data(static_cast<std::size_t>(length));
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (stream.gcount() != static_cast<std::streamsize>(data.size())) {
        return Result<std::vector<std::byte>>::failure(ErrorCode::truncated_file);
    }
    return Result<std::vector<std::byte>>::success(std::move(data));
}

TensorRecord decode_tensor(std::span<const std::byte> value) {
    TensorRecord result{};
    result.tensor_id = little<std::uint64_t>(value, 0);
    result.dtype = little<std::uint16_t>(value, 12);
    result.quantization = little<std::uint16_t>(value, 14);
    result.rank = little<std::uint8_t>(value, 16);
    result.layer_id = static_cast<std::int32_t>(little<std::uint32_t>(value, 20));
    result.expert_id = static_cast<std::int32_t>(little<std::uint32_t>(value, 24));
    for (std::size_t index = 0; index < 4; ++index) result.dimensions[index] = little<std::uint64_t>(value, 32 + index * 8);
    result.data_offset = little<std::uint64_t>(value, 64);
    result.data_length = little<std::uint64_t>(value, 72);
    result.logical_length = little<std::uint64_t>(value, 80);
    result.auxiliary_offset = little<std::uint64_t>(value, 88);
    result.auxiliary_length = little<std::uint64_t>(value, 96);
    result.data_crc32c = little<std::uint32_t>(value, 104);
    result.auxiliary_crc32c = little<std::uint32_t>(value, 108);
    return result;
}
}

struct Reader::DataPlane {
    mutable std::mutex operation_mutex;
    static Result<std::unique_ptr<DataPlane>> open_source(
        const std::filesystem::path& path, const ReaderOptions& options) {
        auto result = std::unique_ptr<DataPlane>(new DataPlane);
        result->engine = options.io_engine;
        result->cache_mode = options.cache_mode;
#ifdef _WIN32
        if (options.cache_mode == L2CacheMode::direct) {
            return Result<std::unique_ptr<DataPlane>>::failure(
                ErrorCode::storage_unavailable);
        }
        result->path = path;
#else
        int flags = O_RDONLY | O_CLOEXEC;
        if (options.cache_mode == L2CacheMode::direct) {
            auto alignment = detail::query_direct_io_alignment(path);
            if (!alignment) {
                return Result<std::unique_ptr<DataPlane>>::failure(
                    alignment.error(), alignment.message());
            }
            result->direct_alignment = alignment.value();
            flags |= O_DIRECT;
        }
        result->descriptor = ::open(path.c_str(), flags);
        if (result->descriptor < 0) {
            return Result<std::unique_ptr<DataPlane>>::failure(
                ErrorCode::io_error);
        }
#ifdef K3X_ENABLE_IO_URING
        if (options.io_engine == L2IoEngine::io_uring) {
            const auto initialized = io_uring_queue_init(
                static_cast<unsigned>(options.queue_depth),
                &result->ring, 0);
            if (initialized < 0) {
                return Result<std::unique_ptr<DataPlane>>::failure(
                    ErrorCode::storage_unavailable,
                    "io_uring queue initialization failed");
            }
            result->ring_initialized = true;
            auto* probe = io_uring_get_probe_ring(&result->ring);
            const bool read_supported = probe != nullptr &&
                io_uring_opcode_supported(probe, IORING_OP_READ);
            io_uring_free_probe(probe);
            if (!read_supported) {
                return Result<std::unique_ptr<DataPlane>>::failure(
                    ErrorCode::storage_unavailable,
                    "io_uring read opcode is unavailable");
            }
        }
#endif
#endif
        return Result<std::unique_ptr<DataPlane>>::success(
            std::move(result));
    }

    ~DataPlane() {
#ifndef _WIN32
#ifdef K3X_ENABLE_IO_URING
        if (ring_initialized) io_uring_queue_exit(&ring);
#endif
        if (descriptor >= 0) ::close(descriptor);
#endif
    }

    Result<std::vector<std::vector<std::byte>>> read_batch(
        std::span<const ExtentRequest> requests, std::size_t queue_depth,
        ReadCounters& counters) {
#ifdef _WIN32
        std::vector<std::vector<std::byte>> values;
        values.reserve(requests.size());
        for (const auto& request : requests) {
            counters.storage_submitted_bytes += request.length;
            auto value = read_raw(path, request.offset, request.length);
            if (!value) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    value.error(), value.message());
            }
            counters.storage_completed_bytes += value.value().size();
            values.push_back(std::move(value.value()));
        }
        return Result<std::vector<std::vector<std::byte>>>::success(
            std::move(values));
#else
        if (engine == L2IoEngine::io_uring) {
#ifdef K3X_ENABLE_IO_URING
            if (!ring_initialized) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    ErrorCode::storage_unavailable,
                    "io_uring reader is unavailable after an I/O failure");
            }
#endif
        }
        if (cache_mode == L2CacheMode::direct) {
            if (engine == L2IoEngine::pread) {
                return detail::read_direct_pread(
                    descriptor, requests, direct_alignment, counters);
            }
#ifdef K3X_ENABLE_IO_URING
            return detail::read_direct_io_uring(
                descriptor, ring, ring_initialized, requests, queue_depth,
                direct_alignment, counters);
#else
            return Result<std::vector<std::vector<std::byte>>>::failure(
                ErrorCode::storage_unavailable);
#endif
        }
        if (engine == L2IoEngine::io_uring) {
#ifdef K3X_ENABLE_IO_URING
            return read_io_uring(requests, queue_depth, counters);
#else
            return Result<std::vector<std::vector<std::byte>>>::failure(
                ErrorCode::storage_unavailable);
#endif
        }
        std::vector<std::vector<std::byte>> values;
        values.reserve(requests.size());
        for (const auto& request : requests) {
            auto value = read_pread(request, counters);
            if (!value) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    value.error(), value.message());
            }
            values.push_back(std::move(value.value()));
        }
        return Result<std::vector<std::vector<std::byte>>>::success(
            std::move(values));
#endif
    }

    L2IoEngine engine{L2IoEngine::pread};
    L2CacheMode cache_mode{L2CacheMode::buffered};
    detail::DirectIoAlignment direct_alignment{};
#ifdef _WIN32
    std::filesystem::path path;
#else
    Result<std::vector<std::byte>> read_pread(
        const ExtentRequest& request, ReadCounters& counters) const {
        if (request.offset > static_cast<std::uint64_t>(
                                 std::numeric_limits<off_t>::max())) {
            return Result<std::vector<std::byte>>::failure(
                ErrorCode::invalid_extent);
        }
        std::vector<std::byte> data(
            static_cast<std::size_t>(request.length));
        std::size_t completed = 0;
        while (completed < data.size()) {
            const auto remaining = data.size() - completed;
            const auto requested = std::min<std::size_t>(
                remaining, static_cast<std::size_t>(
                               std::numeric_limits<ssize_t>::max()));
            counters.storage_submitted_bytes += requested;
            const auto result = ::pread(
                descriptor, data.data() + completed, requested,
                static_cast<off_t>(request.offset + completed));
            if (result < 0) {
                if (errno == EINTR) continue;
                return Result<std::vector<std::byte>>::failure(
                    ErrorCode::io_error);
            }
            if (result == 0) {
                return Result<std::vector<std::byte>>::failure(
                    ErrorCode::truncated_file);
            }
            completed += static_cast<std::size_t>(result);
            counters.storage_completed_bytes +=
                static_cast<std::uint64_t>(result);
        }
        return Result<std::vector<std::byte>>::success(std::move(data));
    }

#ifdef K3X_ENABLE_IO_URING
    Result<std::vector<std::vector<std::byte>>> read_io_uring(
        std::span<const ExtentRequest> requests, std::size_t queue_depth,
        ReadCounters& counters) {
        struct Operation {
            std::size_t request_index;
            std::size_t buffer_offset;
            ExtentRequest extent;
        };
        constexpr std::uint64_t maximum_read_bytes = 0x7ffff000ULL;
        std::vector<std::vector<std::byte>> values;
        values.reserve(requests.size());
        std::vector<Operation> operations;
        std::vector<ExtentRequest> operation_extents;
        for (std::size_t index = 0; index < requests.size(); ++index) {
            const auto& request = requests[index];
            values.emplace_back(static_cast<std::size_t>(request.length));
            std::uint64_t position = 0;
            while (position < request.length) {
                const auto length = std::min(
                    maximum_read_bytes, request.length - position);
                operations.push_back({
                    index, static_cast<std::size_t>(position),
                    {request.offset + position, length}});
                operation_extents.push_back(
                    {request.offset + position, length});
                position += length;
            }
        }
        for (std::size_t base = 0; base < operations.size();
             base += queue_depth) {
            const auto count = std::min(
                queue_depth, operations.size() - base);
            detail::IoRingBatchGuard ring_guard(ring, ring_initialized);
            for (std::size_t local = 0; local < count; ++local) {
                auto* submission = io_uring_get_sqe(&ring);
                if (submission == nullptr) {
                    return Result<std::vector<std::vector<std::byte>>>::failure(
                        ErrorCode::io_error,
                        "io_uring submission queue exhausted");
                }
                const auto index = base + local;
                const auto& operation = operations[index];
                io_uring_prep_read(
                    submission, descriptor,
                    values[operation.request_index].data() +
                        operation.buffer_offset,
                    static_cast<unsigned>(operation.extent.length),
                    operation.extent.offset);
                io_uring_sqe_set_data64(submission, index);
            }
            std::size_t submitted_total = 0;
            bool submission_failed = false;
            while (submitted_total < count) {
                const auto submitted = io_uring_submit(&ring);
                if (submitted == -EINTR) continue;
                if (submitted <= 0 ||
                    static_cast<std::size_t>(submitted) >
                        count - submitted_total) {
                    submission_failed = true;
                    break;
                }
                for (std::size_t local = 0;
                     local < static_cast<std::size_t>(submitted); ++local) {
                    counters.storage_submitted_bytes +=
                        operations[base + submitted_total + local].extent.length;
                }
                submitted_total += static_cast<std::size_t>(submitted);
            }
            if (submitted_total == 0) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    ErrorCode::io_error, "io_uring submission failed");
            }
            ErrorCode batch_error = ErrorCode::ok;
            for (std::size_t completed = 0; completed < submitted_total;
                 ++completed) {
                io_uring_cqe* completion = nullptr;
                int waited = 0;
                do {
                    waited = io_uring_wait_cqe(&ring, &completion);
                } while (waited == -EINTR);
                if (waited < 0 || completion == nullptr) {
                    return Result<std::vector<std::vector<std::byte>>>::failure(
                        ErrorCode::io_error, "io_uring completion failed");
                }
                const auto index = static_cast<std::size_t>(
                    io_uring_cqe_get_data64(completion));
                const auto result = completion->res;
                const auto completion_error = detail::record_io_completion(
                    operation_extents, index, result, counters);
                if (batch_error == ErrorCode::ok &&
                    completion_error != ErrorCode::ok) {
                    batch_error = completion_error;
                }
                io_uring_cqe_seen(&ring, completion);
            }
            if (submission_failed || submitted_total != count) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    ErrorCode::io_error, "io_uring submission failed");
            }
            ring_guard.release();
            if (batch_error != ErrorCode::ok) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    batch_error);
            }
        }
        return Result<std::vector<std::vector<std::byte>>>::success(
            std::move(values));
    }

    io_uring ring{};
    bool ring_initialized{};
#endif
    int descriptor{-1};
#endif
};

Reader::Reader() = default;
Reader::Reader(Reader&&) noexcept = default;
Reader& Reader::operator=(Reader&&) noexcept = default;
Reader::~Reader() = default;

std::uint64_t Reader::direct_memory_alignment() const {
    return data_plane_ ? data_plane_->direct_alignment.memory : 0;
}

std::uint64_t Reader::direct_offset_alignment() const {
    return data_plane_ ? data_plane_->direct_alignment.offset : 0;
}

std::uint64_t fnv1a64(const char* value) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    while (*value) {
        hash ^= static_cast<unsigned char>(*value++);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

Result<Reader> Reader::open(const std::filesystem::path& path, VerifyMode mode) {
    ReaderOptions options;
    options.verify = mode;
    return open(path, options);
}

Result<Reader> Reader::open(const std::filesystem::path& path,
                            ReaderOptions options) {
    if (options.queue_depth == 0 ||
        options.queue_depth > maximum_l2_queue_depth) {
        return Result<Reader>::failure(ErrorCode::invalid_state,
                                       "L2 queue depth is out of range");
    }
#ifdef _WIN32
    if (options.cache_mode != L2CacheMode::buffered) {
        return Result<Reader>::failure(ErrorCode::storage_unavailable,
                                       "requested L2 mode is not built");
    }
#endif
#ifndef K3X_ENABLE_IO_URING
    if (options.io_engine == L2IoEngine::io_uring) {
        return Result<Reader>::failure(ErrorCode::storage_unavailable,
                                       "requested L2 mode is not built");
    }
#endif
    std::error_code filesystem_error;
    const auto file_length = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || file_length < superblock_bytes) {
        return Result<Reader>::failure(ErrorCode::truncated_file);
    }
    auto raw_block = read_raw(path, 0, superblock_bytes);
    if (!raw_block) return Result<Reader>::failure(raw_block.error());
    const auto block = std::span(raw_block.value());
    constexpr std::array<char, 8> magic{'K','3','X','C','H','K','P','T'};
    if (std::memcmp(block.data(), magic.data(), magic.size()) != 0) {
        return Result<Reader>::failure(ErrorCode::bad_magic);
    }
    if (little<std::uint16_t>(block, 8) != 1 || little<std::uint16_t>(block, 10) != 0 ||
        little<std::uint32_t>(block, 12) != superblock_bytes ||
        little<std::uint32_t>(block, 16) != 4096) {
        return Result<Reader>::failure(ErrorCode::unsupported_version);
    }
    if (crc32c(block.first(4092)) != little<std::uint32_t>(block, 4092)) {
        return Result<Reader>::failure(ErrorCode::superblock_crc_mismatch);
    }
    if (little<std::uint64_t>(block, 24) != 0) {
        return Result<Reader>::failure(ErrorCode::unsupported_required_feature);
    }
    if (!all_zero(block.subspan(232, 4092 - 232))) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    Reader reader;
    reader.path_ = path;
    reader.options_ = options;
    reader.superblock_.state = little<std::uint32_t>(block, 20);
    reader.superblock_.required_features = little<std::uint64_t>(block, 24);
    reader.superblock_.optional_features = little<std::uint64_t>(block, 32);
    std::uint64_t* values[] = {&reader.superblock_.tensor_directory_offset,
                      &reader.superblock_.tensor_directory_length,
                      &reader.superblock_.layer_directory_offset,
                      &reader.superblock_.layer_directory_length,
                      &reader.superblock_.expert_directory_offset,
                      &reader.superblock_.expert_directory_length,
                      &reader.superblock_.model_config_offset,
                      &reader.superblock_.model_config_length,
                      &reader.superblock_.payload_offset,
                      &reader.superblock_.file_length};
    for (std::size_t index = 0; index < 10; ++index) *values[index] = little<std::uint64_t>(block, 88 + index * 8);
    std::copy(block.begin() + 168, block.begin() + 200, reader.superblock_.directory_sha256.begin());
    std::copy(block.begin() + 200, block.begin() + 232, reader.superblock_.root_sha256.begin());
    if (reader.superblock_.state != 1 || reader.superblock_.file_length != file_length) {
        return Result<Reader>::failure(ErrorCode::truncated_file);
    }
    for (const auto [offset, length] : {
             std::pair{reader.superblock_.tensor_directory_offset, reader.superblock_.tensor_directory_length},
             std::pair{reader.superblock_.layer_directory_offset, reader.superblock_.layer_directory_length},
             std::pair{reader.superblock_.expert_directory_offset, reader.superblock_.expert_directory_length},
             std::pair{reader.superblock_.model_config_offset, reader.superblock_.model_config_length}}) {
        if (offset % 4096 || !range_valid(offset, length, file_length)) {
            return Result<Reader>::failure(ErrorCode::invalid_extent);
        }
    }
    auto tensor_directory = read_raw(path, reader.superblock_.tensor_directory_offset,
                                     reader.superblock_.tensor_directory_length);
    auto layer_directory = read_raw(path, reader.superblock_.layer_directory_offset,
                                    reader.superblock_.layer_directory_length);
    auto expert_directory = read_raw(path, reader.superblock_.expert_directory_offset,
                                     reader.superblock_.expert_directory_length);
    auto config = read_raw(path, reader.superblock_.model_config_offset,
                           reader.superblock_.model_config_length);
    if (!tensor_directory || !layer_directory || !expert_directory || !config ||
        config.value().size() != model_config_bytes) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    const auto directory = std::span(tensor_directory.value());
    const auto tensor_count = directory_count(directory, "TENS", tensor_record_bytes);
    const auto layer_bytes = std::span(layer_directory.value());
    const auto layer_count = directory_count(layer_bytes, "LAYR", layer_record_bytes);
    const auto expert_bytes = std::span(expert_directory.value());
    const auto expert_count = directory_count(expert_bytes, "EXPT", expert_record_bytes);
    if (!tensor_count || !layer_count || !expert_count ||
        little<std::uint32_t>(config.value(), 8) != *layer_count ||
        !all_zero(std::span(config.value()).subspan(112)) ||
        (little<std::uint32_t>(config.value(), 108) & ~3U)) {
        return Result<Reader>::failure(ErrorCode::invalid_directory);
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges{
        {0, superblock_bytes},
        {reader.superblock_.tensor_directory_offset,
         reader.superblock_.tensor_directory_offset + reader.superblock_.tensor_directory_length},
        {reader.superblock_.layer_directory_offset,
         reader.superblock_.layer_directory_offset + reader.superblock_.layer_directory_length},
        {reader.superblock_.expert_directory_offset,
         reader.superblock_.expert_directory_offset + reader.superblock_.expert_directory_length},
        {reader.superblock_.model_config_offset,
         reader.superblock_.model_config_offset + reader.superblock_.model_config_length},
    };
    std::unordered_set<std::uint64_t> tensor_ids;
    for (std::size_t index = 0; index < *tensor_count; ++index) {
        const auto raw_record = directory.subspan(16 + index * tensor_record_bytes, tensor_record_bytes);
        auto record = decode_tensor(raw_record);
        if (record.rank > 4 || (record.dtype != 1 && record.dtype != 2) ||
            record.quantization > 1 || little<std::uint32_t>(raw_record, 8) != 0 ||
            little<std::uint8_t>(raw_record, 17) != 0 ||
            little<std::uint16_t>(raw_record, 18) != 0 ||
            little<std::uint32_t>(raw_record, 28) != 0 ||
            !all_zero(raw_record.subspan(112)) || !tensor_ids.insert(record.tensor_id).second) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (std::size_t dimension = record.rank; dimension < 4; ++dimension) {
            if (record.dimensions[dimension]) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
        const bool plain = record.dtype == 1 && record.quantization == 0 &&
                           record.auxiliary_length == 0 && record.auxiliary_offset == 0 &&
                           record.auxiliary_crc32c == 0;
        const bool mxfp4 = record.dtype == 2 && record.quantization == 1 &&
                           record.auxiliary_length != 0;
        if (!record.data_length || !record.logical_length || (!plain && !mxfp4)) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (const auto [offset, length] : {std::pair{record.data_offset, record.data_length},
                                           std::pair{record.auxiliary_offset, record.auxiliary_length}}) {
            if (!length) {
                if (offset) return Result<Reader>::failure(ErrorCode::invalid_extent);
            } else {
                if (offset % 4096 || !range_valid(offset, length, file_length)) {
                    return Result<Reader>::failure(ErrorCode::invalid_extent);
                }
                ranges.emplace_back(offset, offset + length);
            }
        }
        reader.tensors_.push_back(record);
    }
    const auto configured_experts = little<std::uint32_t>(config.value(), 48);
    for (const auto& tensor : reader.tensors_) {
        if (tensor.layer_id < -1 || tensor.layer_id >= static_cast<std::int32_t>(*layer_count) ||
            tensor.expert_id < -1 || tensor.expert_id >= static_cast<std::int32_t>(configured_experts) ||
            (tensor.expert_id >= 0 && tensor.layer_id < 0)) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
    }
    for (std::size_t index = 0; index < *layer_count; ++index) {
        const auto record = layer_bytes.subspan(16 + index * layer_record_bytes, layer_record_bytes);
        const auto layer_index = little<std::uint32_t>(record, 0);
        const auto attention_kind = little<std::uint16_t>(record, 4);
        const auto ffn_kind = little<std::uint16_t>(record, 6);
        const auto first_tensor = little<std::uint32_t>(record, 8);
        const auto count = little<std::uint32_t>(record, 12);
        const auto first_expert = little<std::uint32_t>(record, 16);
        const auto experts = little<std::uint32_t>(record, 20);
        if (layer_index != index || (attention_kind != 1 && attention_kind != 2) ||
            (ffn_kind != 1 && ffn_kind != 2) || little<std::uint32_t>(record, 28) != 0 ||
            !all_zero(record.subspan(32)) || first_tensor > *tensor_count ||
            count > *tensor_count - first_tensor || first_expert > *expert_count ||
            experts > *expert_count - first_expert) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (std::size_t tensor = first_tensor; tensor < first_tensor + count; ++tensor) {
            if (reader.tensors_[tensor].layer_id != static_cast<std::int32_t>(index)) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    std::unordered_set<std::uint32_t> physical_orders;
    for (std::size_t index = 0; index < *expert_count; ++index) {
        const auto record = expert_bytes.subspan(16 + index * expert_record_bytes, expert_record_bytes);
        const auto layer = little<std::uint32_t>(record, 0);
        const auto expert = little<std::uint32_t>(record, 4);
        const auto physical_order = little<std::uint32_t>(record, 8);
        if (layer >= *layer_count || expert >= configured_experts ||
            physical_order >= *expert_count || !physical_orders.insert(physical_order).second ||
            little<std::uint32_t>(record, 12) != 0 ||
            !all_zero(record.subspan(48))) {
            return Result<Reader>::failure(ErrorCode::invalid_directory);
        }
        for (const auto offset : {16U, 24U, 32U}) {
            const auto id = little<std::uint64_t>(record, offset);
            const auto tensor = std::find_if(reader.tensors_.begin(), reader.tensors_.end(),
                                             [id](const auto& item) { return item.tensor_id == id; });
            if (tensor == reader.tensors_.end() || tensor->layer_id != static_cast<std::int32_t>(layer) ||
                tensor->expert_id != static_cast<std::int32_t>(expert) || tensor->quantization != 1) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    for (std::size_t layer = 0; layer < *layer_count; ++layer) {
        const auto record = layer_bytes.subspan(16 + layer * layer_record_bytes, layer_record_bytes);
        const auto first = little<std::uint32_t>(record, 16);
        const auto count = little<std::uint32_t>(record, 20);
        for (std::size_t expert = first; expert < first + count; ++expert) {
            if (little<std::uint32_t>(expert_bytes, 16 + expert * expert_record_bytes) != layer) {
                return Result<Reader>::failure(ErrorCode::invalid_directory);
            }
        }
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first < ranges[index - 1].second) {
            return Result<Reader>::failure(ErrorCode::invalid_extent);
        }
    }
    std::copy(config.value().begin(), config.value().end(), reader.model_config_.begin());
    if (options.verify == VerifyMode::checksums) {
        for (const auto& record : reader.tensors_) {
            auto data = read_raw(path, record.data_offset, record.data_length);
            if (!data || crc32c(data.value()) != record.data_crc32c) {
                return Result<Reader>::failure(ErrorCode::data_crc_mismatch);
            }
            if (record.auxiliary_length) {
                auto auxiliary = read_raw(path, record.auxiliary_offset, record.auxiliary_length);
                if (!auxiliary || crc32c(auxiliary.value()) != record.auxiliary_crc32c) {
                    return Result<Reader>::failure(ErrorCode::auxiliary_crc_mismatch);
                }
            }
        }
        std::vector<std::byte> directories;
        for (const auto* item : {&tensor_directory.value(), &layer_directory.value(),
                                 &expert_directory.value(), &config.value()}) {
            directories.insert(directories.end(), item->begin(), item->end());
        }
        if (sha256(directories) != reader.superblock_.directory_sha256) {
            return Result<Reader>::failure(ErrorCode::directory_sha256_mismatch);
        }
        std::ifstream root_stream(path, std::ios::binary);
        if (!root_stream) return Result<Reader>::failure(ErrorCode::io_error);
        Sha256Hasher root_hasher;
        std::vector<std::byte> root_chunk(1024 * 1024);
        std::uint64_t position = 0;
        while (position < file_length) {
            const auto requested = static_cast<std::size_t>(
                std::min<std::uint64_t>(root_chunk.size(), file_length - position));
            root_stream.read(reinterpret_cast<char*>(root_chunk.data()), requested);
            if (root_stream.gcount() != static_cast<std::streamsize>(requested)) {
                return Result<Reader>::failure(ErrorCode::truncated_file);
            }
            for (const auto [start, end] : {
                     std::pair<std::uint64_t, std::uint64_t>{200, 232},
                     std::pair<std::uint64_t, std::uint64_t>{4092, 4096}}) {
                const auto left = std::max(start, position);
                const auto right = std::min(end, position + requested);
                if (left < right) {
                    std::fill(root_chunk.begin() + (left - position),
                              root_chunk.begin() + (right - position), std::byte{0});
                }
            }
            root_hasher.update(std::span(root_chunk).first(requested));
            position += requested;
        }
        if (root_hasher.finish() != reader.superblock_.root_sha256) {
            return Result<Reader>::failure(ErrorCode::root_sha256_mismatch);
        }
    }
    auto data_plane = DataPlane::open_source(path, options);
    if (!data_plane) {
        return Result<Reader>::failure(data_plane.error(),
                                       data_plane.message());
    }
    reader.data_plane_ = std::move(data_plane.value());
    return Result<Reader>::success(std::move(reader));
}

Result<std::vector<std::vector<std::byte>>> Reader::read_extents(
    std::span<const ExtentRequest> requests) const {
    if (requests.empty()) {
        return Result<std::vector<std::vector<std::byte>>>::success({});
    }
    for (const auto& request : requests) {
        if (!request.length ||
            request.length > std::numeric_limits<std::size_t>::max() ||
            !range_valid(request.offset, request.length,
                         superblock_.file_length)) {
            return Result<std::vector<std::vector<std::byte>>>::failure(
                ErrorCode::invalid_extent);
        }
    }
    std::lock_guard lock(data_plane_->operation_mutex);
    ++counters_.batch_submissions;
    for (const auto& request : requests) {
        ++counters_.calls;
        counters_.requested_bytes += request.length;
    }
    const auto storage_start = std::chrono::steady_clock::now();
    auto results = data_plane_->read_batch(
        requests, options_.queue_depth, counters_);
    counters_.storage_nanoseconds += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - storage_start).count());
    if (!results) {
        if (results.error() == ErrorCode::truncated_file) {
            ++counters_.short_reads;
        }
        ++counters_.failures;
        return results;
    }
    for (const auto& result : results.value()) {
        counters_.completed_bytes += result.size();
        ++counters_.completions;
    }
    return results;
}

ReadCounters Reader::counters() const {
    std::lock_guard lock(data_plane_->operation_mutex);
    return counters_;
}

Result<std::vector<std::byte>> Reader::read_tensor(std::uint64_t tensor_id) const {
    const auto item = std::find_if(tensors_.begin(), tensors_.end(),
                                   [tensor_id](const auto& value) { return value.tensor_id == tensor_id; });
    if (item == tensors_.end()) return Result<std::vector<std::byte>>::failure(ErrorCode::tensor_not_found);
    const std::array request{
        ExtentRequest{item->data_offset, item->data_length}};
    auto result = read_extents(request);
    if (!result) {
        return Result<std::vector<std::byte>>::failure(
            result.error(), result.message());
    }
    return Result<std::vector<std::byte>>::success(
        std::move(result.value().front()));
}

Result<std::vector<std::byte>> Reader::read_auxiliary(std::uint64_t tensor_id) const {
    const auto item = std::find_if(tensors_.begin(), tensors_.end(),
                                   [tensor_id](const auto& value) { return value.tensor_id == tensor_id; });
    if (item == tensors_.end()) return Result<std::vector<std::byte>>::failure(ErrorCode::tensor_not_found);
    const std::array request{
        ExtentRequest{item->auxiliary_offset, item->auxiliary_length}};
    auto result = read_extents(request);
    if (!result) {
        return Result<std::vector<std::byte>>::failure(
            result.error(), result.message());
    }
    return Result<std::vector<std::byte>>::success(
        std::move(result.value().front()));
}
}
