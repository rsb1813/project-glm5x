// O_DIRECT 요청을 정렬된 bounce buffer로 실행하고 논리 extent를 복원합니다.
#include "direct_io.hpp"

#ifdef K3X_ENABLE_IO_URING
#include "io_ring_guard.hpp"
#endif

#ifndef _WIN32

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace k3x::detail {
namespace {
struct AlignedBuffer {
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data(std::exchange(other.data, nullptr)), size(other.size) {}
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            size = other.size;
        }
        return *this;
    }
    ~AlignedBuffer() { std::free(data); }

    static Result<AlignedBuffer> allocate(std::size_t size,
                                          std::size_t alignment) {
        void* allocation = nullptr;
        if (posix_memalign(&allocation, alignment, size) != 0) {
            return Result<AlignedBuffer>::failure(ErrorCode::io_error);
        }
        AlignedBuffer result;
        result.data = static_cast<std::byte*>(allocation);
        result.size = size;
        return Result<AlignedBuffer>::success(std::move(result));
    }

    std::byte* data{};
    std::size_t size{};
};

struct RequestState {
    std::size_t logical_index;
    std::size_t prefix;
    std::size_t required_bytes;
    AlignedBuffer buffer;
};

struct Operation {
    std::size_t request_index;
    std::size_t buffer_offset;
    std::size_t minimum_bytes;
    ExtentRequest extent;
};

struct Plan {
    std::vector<RequestState> requests;
    std::vector<Operation> operations;
};

Result<Plan> make_plan(std::span<const ExtentRequest> requests,
                       DirectIoAlignment alignment) {
    if (alignment.memory == 0 ||
        (alignment.memory & (alignment.memory - 1)) != 0 ||
        alignment.offset == 0) {
        return Result<Plan>::failure(ErrorCode::storage_unavailable);
    }
    const auto maximum_chunk =
        (0x7ffff000ULL / alignment.offset) * alignment.offset;
    if (maximum_chunk == 0) {
        return Result<Plan>::failure(ErrorCode::storage_unavailable);
    }
    Plan plan;
    plan.requests.reserve(requests.size());
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const auto& request = requests[index];
        const auto aligned_offset =
            request.offset - request.offset % alignment.offset;
        const auto logical_end = request.offset + request.length;
        const auto remainder = logical_end % alignment.offset;
        const auto padding = remainder == 0 ? 0 :
            alignment.offset - remainder;
        if (logical_end > std::numeric_limits<std::uint64_t>::max() -
                              padding) {
            return Result<Plan>::failure(ErrorCode::invalid_extent);
        }
        const auto aligned_length = logical_end + padding - aligned_offset;
        if (aligned_length > std::numeric_limits<std::size_t>::max()) {
            return Result<Plan>::failure(ErrorCode::invalid_extent);
        }
        auto buffer = AlignedBuffer::allocate(
            static_cast<std::size_t>(aligned_length),
            std::max(sizeof(void*),
                     static_cast<std::size_t>(alignment.memory)));
        if (!buffer) {
            return Result<Plan>::failure(buffer.error(), buffer.message());
        }
        const auto direct_index = plan.requests.size();
        const auto prefix = static_cast<std::size_t>(
            request.offset - aligned_offset);
        plan.requests.push_back({
            index, prefix,
            prefix + static_cast<std::size_t>(request.length),
            std::move(buffer.value())});
        std::uint64_t position = 0;
        while (position < aligned_length) {
            const auto length = std::min<std::uint64_t>(
                maximum_chunk, aligned_length - position);
            const auto required = plan.requests.back().required_bytes;
            const auto minimum = position >= required ? 0 :
                std::min<std::uint64_t>(length, required - position);
            plan.operations.push_back({
                direct_index, static_cast<std::size_t>(position),
                static_cast<std::size_t>(minimum),
                {aligned_offset + position, length}});
            position += length;
        }
    }
    return Result<Plan>::success(std::move(plan));
}

Result<std::vector<std::vector<std::byte>>> finish_plan(
    Plan plan, std::span<const ExtentRequest> logical_requests) {
    std::vector<std::vector<std::byte>> values(logical_requests.size());
    for (auto& request : plan.requests) {
        const auto length = logical_requests[request.logical_index].length;
        values[request.logical_index].assign(
            request.buffer.data + request.prefix,
            request.buffer.data + request.prefix + length);
    }
    return Result<std::vector<std::vector<std::byte>>>::success(
        std::move(values));
}

ErrorCode validate_direct_completion(const Operation& operation, int result,
                                     ReadCounters& counters) {
    if (result < 0 || static_cast<std::uint64_t>(result) >
                          operation.extent.length) {
        return ErrorCode::io_error;
    }
    counters.storage_completed_bytes += static_cast<std::uint64_t>(result);
    return static_cast<std::size_t>(result) < operation.minimum_bytes
        ? ErrorCode::truncated_file
        : ErrorCode::ok;
}
}

Result<DirectIoAlignment> query_direct_io_alignment(
    const std::filesystem::path& path) {
    struct statx information {};
    if (::statx(AT_FDCWD, path.c_str(), AT_STATX_SYNC_AS_STAT,
                STATX_DIOALIGN, &information) != 0 ||
        (information.stx_mask & STATX_DIOALIGN) == 0 ||
        information.stx_dio_mem_align == 0 ||
        information.stx_dio_offset_align == 0) {
        return Result<DirectIoAlignment>::failure(
            ErrorCode::storage_unavailable,
            "direct I/O alignment is unavailable");
    }
    DirectIoAlignment result{
        information.stx_dio_mem_align,
        information.stx_dio_offset_align};
    if (result.memory == 0 ||
        (result.memory & (result.memory - 1)) != 0) {
        return Result<DirectIoAlignment>::failure(
            ErrorCode::storage_unavailable,
            "direct I/O memory alignment is invalid");
    }
    return Result<DirectIoAlignment>::success(result);
}

Result<std::vector<std::vector<std::byte>>> read_direct_pread(
    int descriptor, std::span<const ExtentRequest> requests,
    DirectIoAlignment alignment, ReadCounters& counters) {
    auto plan = make_plan(requests, alignment);
    if (!plan) {
        return Result<std::vector<std::vector<std::byte>>>::failure(
            plan.error(), plan.message());
    }
    for (const auto& operation : plan.value().operations) {
        auto& request = plan.value().requests[operation.request_index];
        ssize_t result = -1;
        do {
            counters.storage_submitted_bytes += operation.extent.length;
            result = ::pread(
                descriptor, request.buffer.data + operation.buffer_offset,
                static_cast<std::size_t>(operation.extent.length),
                static_cast<off_t>(operation.extent.offset));
        } while (result < 0 && errno == EINTR);
        const auto error = validate_direct_completion(
            operation, static_cast<int>(result), counters);
        if (error != ErrorCode::ok) {
            return Result<std::vector<std::vector<std::byte>>>::failure(error);
        }
    }
    return finish_plan(std::move(plan.value()), requests);
}

#ifdef K3X_ENABLE_IO_URING
Result<std::vector<std::vector<std::byte>>> read_direct_io_uring(
    int descriptor, io_uring& ring, bool& ring_initialized,
    std::span<const ExtentRequest> requests, std::size_t queue_depth,
    DirectIoAlignment alignment,
    ReadCounters& counters) {
    auto plan = make_plan(requests, alignment);
    if (!plan) {
        return Result<std::vector<std::vector<std::byte>>>::failure(
            plan.error(), plan.message());
    }
    for (std::size_t base = 0; base < plan.value().operations.size();
         base += queue_depth) {
        const auto count = std::min(
            queue_depth, plan.value().operations.size() - base);
        IoRingBatchGuard ring_guard(ring, ring_initialized);
        for (std::size_t local = 0; local < count; ++local) {
            const auto index = base + local;
            const auto& operation = plan.value().operations[index];
            auto& request = plan.value().requests[operation.request_index];
            auto* submission = io_uring_get_sqe(&ring);
            if (submission == nullptr) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    ErrorCode::io_error);
            }
            io_uring_prep_read(
                submission, descriptor,
                request.buffer.data + operation.buffer_offset,
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
                static_cast<std::size_t>(submitted) > count - submitted_total) {
                submission_failed = true;
                break;
            }
            for (std::size_t local = 0;
                 local < static_cast<std::size_t>(submitted); ++local) {
                counters.storage_submitted_bytes += plan.value().operations[
                    base + submitted_total + local].extent.length;
            }
            submitted_total += static_cast<std::size_t>(submitted);
        }
        if (submitted_total == 0) {
            return Result<std::vector<std::vector<std::byte>>>::failure(
                ErrorCode::io_error);
        }
        ErrorCode batch_error = ErrorCode::ok;
        for (std::size_t completed = 0; completed < submitted_total; ++completed) {
            io_uring_cqe* completion = nullptr;
            int waited = 0;
            do {
                waited = io_uring_wait_cqe(&ring, &completion);
            } while (waited == -EINTR);
            if (waited < 0 || completion == nullptr) {
                return Result<std::vector<std::vector<std::byte>>>::failure(
                    ErrorCode::io_error);
            }
            const auto index = static_cast<std::size_t>(
                io_uring_cqe_get_data64(completion));
            ErrorCode error = ErrorCode::io_error;
            if (index < plan.value().operations.size()) {
                error = validate_direct_completion(
                    plan.value().operations[index], completion->res,
                    counters);
            }
            if (batch_error == ErrorCode::ok && error != ErrorCode::ok) {
                batch_error = error;
            }
            io_uring_cqe_seen(&ring, completion);
        }
        if (submission_failed || submitted_total != count) {
            return Result<std::vector<std::vector<std::byte>>>::failure(
                ErrorCode::io_error);
        }
        ring_guard.release();
        if (batch_error != ErrorCode::ok) {
            return Result<std::vector<std::vector<std::byte>>>::failure(
                batch_error);
        }
    }
    return finish_plan(std::move(plan.value()), requests);
}
#endif
}

#else

namespace k3x::detail {
Result<DirectIoAlignment> query_direct_io_alignment(
    const std::filesystem::path&) {
    return Result<DirectIoAlignment>::failure(ErrorCode::storage_unavailable);
}

Result<std::vector<std::vector<std::byte>>> read_direct_pread(
    int, std::span<const ExtentRequest>, DirectIoAlignment,
    ReadCounters&) {
    return Result<std::vector<std::vector<std::byte>>>::failure(
        ErrorCode::storage_unavailable);
}
}

#endif
