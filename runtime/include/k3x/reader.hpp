// K3X 메타데이터와 tensor extent를 경계 검증 후 읽습니다.
#pragma once

#include "k3x/format.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace k3x {
enum class VerifyMode { checksums, metadata_only };
enum class L2IoEngine { pread, io_uring };
enum class L2CacheMode { buffered, direct };

struct ReaderOptions {
    VerifyMode verify{VerifyMode::checksums};
    L2IoEngine io_engine{L2IoEngine::pread};
    L2CacheMode cache_mode{L2CacheMode::buffered};
    std::size_t queue_depth{8};
};

inline constexpr std::size_t maximum_l2_queue_depth = 1024;

struct ExtentRequest {
    std::uint64_t offset{};
    std::uint64_t length{};
};

struct ReadCounters {
    std::uint64_t calls{};
    std::uint64_t requested_bytes{};
    std::uint64_t completed_bytes{};
    std::uint64_t batch_submissions{};
    std::uint64_t storage_submitted_bytes{};
    std::uint64_t storage_completed_bytes{};
    std::uint64_t completions{};
    std::uint64_t short_reads{};
    std::uint64_t failures{};
    std::uint64_t storage_nanoseconds{};
};

class Reader {
public:
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    ~Reader();
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;

    static Result<Reader> open(const std::filesystem::path& path,
                               VerifyMode mode = VerifyMode::checksums);
    static Result<Reader> open(const std::filesystem::path& path,
                               ReaderOptions options);
    Result<std::vector<std::byte>> read_tensor(std::uint64_t tensor_id) const;
    Result<std::vector<std::byte>> read_auxiliary(std::uint64_t tensor_id) const;
    Result<std::vector<std::vector<std::byte>>> read_extents(
        std::span<const ExtentRequest> requests) const;
    const std::vector<TensorRecord>& tensors() const { return tensors_; }
    const Superblock& superblock() const { return superblock_; }
    const std::array<std::byte, model_config_bytes>& model_config() const { return model_config_; }
    ReadCounters counters() const;
    const ReaderOptions& options() const { return options_; }
    std::uint64_t direct_memory_alignment() const;
    std::uint64_t direct_offset_alignment() const;
private:
    Reader();
    struct DataPlane;
    std::filesystem::path path_;
    std::unique_ptr<DataPlane> data_plane_;
    ReaderOptions options_{};
    Superblock superblock_;
    std::vector<TensorRecord> tensors_;
    std::array<std::byte, model_config_bytes> model_config_{};
    mutable ReadCounters counters_{};
};
}
