// Bounded exact CUDA resident weight의 admission, hit, bypass를 구현합니다.
#include "resident_weights.cuh"

#include <algorithm>
#include <limits>
#include <map>
#include <tuple>
#include <utility>

namespace k3x::cuda {
namespace {

bool is_dense(WeightRepresentation representation) {
    return representation == WeightRepresentation::dense_fp32 ||
           representation == WeightRepresentation::dense_bf16;
}

bool valid_payload(ResidentWeightKey key,
                   std::span<const std::byte> primary,
                   std::span<const std::byte> secondary) {
    if (!key.rows || !key.cols ||
        key.rows > std::numeric_limits<std::uint64_t>::max() / key.cols) {
        return false;
    }
    const auto elements = key.rows * key.cols;
    if (key.representation == WeightRepresentation::dense_fp32) {
        return key.group_size == 0 && secondary.empty() &&
               elements <= std::numeric_limits<std::uint64_t>::max() / 4 &&
               primary.size() == elements * 4;
    }
    if (key.representation == WeightRepresentation::dense_bf16) {
        return key.group_size == 0 && secondary.empty() &&
               elements <= std::numeric_limits<std::uint64_t>::max() / 2 &&
               primary.size() == elements * 2;
    }
    return key.group_size != 0 && key.cols % key.group_size == 0 &&
           key.cols % 2 == 0 && primary.size() == elements / 2 &&
           secondary.size() == elements / key.group_size;
}

struct KeyLess {
    bool operator()(const ResidentWeightKey& left,
                    const ResidentWeightKey& right) const {
        return std::tie(left.tensor_id, left.representation, left.rows,
                        left.cols, left.group_size) <
               std::tie(right.tensor_id, right.representation, right.rows,
                        right.cols, right.group_size);
    }
};

struct TensorMetadata {
    WeightRepresentation representation;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t group_size;
};

}  // namespace

struct ResidentWeightTable::Impl {
    struct Entry {
        Entry(BackendMemoryStats* memory, BackendRuntimeStats* runtime)
            : primary(memory, runtime), secondary(memory, runtime) {}

        DeviceAllocation primary;
        DeviceAllocation secondary;
    };

    Impl(std::uint64_t capacity_value, BackendMemoryStats* memory_value,
         BackendRuntimeStats* runtime_value, cudaStream_t stream_value)
        : capacity(capacity_value), memory(memory_value), runtime(runtime_value),
          stream(stream_value) {}

    ~Impl() {
        entries.clear();
        if (runtime) runtime->resident_weight_bytes = 0;
    }

    bool compatible(ResidentWeightKey key) const {
        const auto found = metadata.find(key.tensor_id);
        if (found == metadata.end()) return true;
        const auto& existing = found->second;
        if (existing.rows != key.rows || existing.cols != key.cols ||
            existing.group_size != key.group_size) {
            return false;
        }
        return existing.representation == key.representation ||
               (is_dense(existing.representation) &&
                is_dense(key.representation));
    }

    std::uint64_t capacity;
    BackendMemoryStats* memory;
    BackendRuntimeStats* runtime;
    cudaStream_t stream;
    std::map<ResidentWeightKey, std::unique_ptr<Entry>, KeyLess> entries;
    std::map<std::uint64_t, TensorMetadata> metadata;
};

ResidentWeightTable::ResidentWeightTable(
    std::uint64_t capacity, BackendMemoryStats* memory,
    BackendRuntimeStats* runtime, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(capacity, memory, runtime, stream)) {}

ResidentWeightTable::~ResidentWeightTable() = default;

Result<ResidentAcquisition> ResidentWeightTable::acquire(
    ResidentWeightKey key, std::span<const std::byte> primary,
    std::span<const std::byte> secondary) {
    if (!impl_->memory || !impl_->runtime ||
        !valid_payload(key, primary, secondary) || !impl_->compatible(key)) {
        return Result<ResidentAcquisition>::failure(ErrorCode::invalid_extent);
    }
    if (const auto found = impl_->entries.find(key);
        found != impl_->entries.end()) {
        ++impl_->runtime->weight_cache_hits;
        return Result<ResidentAcquisition>::success(
            {ResidentDisposition::hit, found->second->primary.get(),
             found->second->secondary.get(), 0});
    }

    ++impl_->runtime->weight_cache_misses;
    if (secondary.size() >
            std::numeric_limits<std::uint64_t>::max() - primary.size()) {
        return Result<ResidentAcquisition>::failure(ErrorCode::invalid_extent);
    }
    const auto required = primary.size() + secondary.size();
    if (required > impl_->capacity ||
        impl_->runtime->resident_weight_bytes > impl_->capacity - required) {
        ++impl_->runtime->weight_cache_bypasses;
        return Result<ResidentAcquisition>::success(
            {ResidentDisposition::bypass, nullptr, nullptr, 0});
    }

    auto entry = std::make_unique<Impl::Entry>(impl_->memory, impl_->runtime);
    if (entry->primary.allocate(primary.size()) != cudaSuccess ||
        (!secondary.empty() &&
         entry->secondary.allocate(secondary.size()) != cudaSuccess)) {
        return Result<ResidentAcquisition>::failure(
            ErrorCode::backend_unavailable, "CUDA resident allocation failed");
    }
    if (cudaMemcpyAsync(entry->primary.get(), primary.data(), primary.size(),
                        cudaMemcpyHostToDevice, impl_->stream) != cudaSuccess ||
        (!secondary.empty() &&
         cudaMemcpyAsync(entry->secondary.get(), secondary.data(),
                         secondary.size(), cudaMemcpyHostToDevice,
                         impl_->stream) != cudaSuccess)) {
        return Result<ResidentAcquisition>::failure(
            ErrorCode::backend_unavailable, "CUDA resident upload failed");
    }

    auto* primary_pointer = entry->primary.get();
    auto* secondary_pointer = entry->secondary.get();
    impl_->runtime->resident_weight_bytes += required;
    impl_->runtime->peak_resident_weight_bytes = std::max(
        impl_->runtime->peak_resident_weight_bytes,
        impl_->runtime->resident_weight_bytes);
    impl_->metadata.try_emplace(
        key.tensor_id,
        TensorMetadata{key.representation, key.rows, key.cols, key.group_size});
    impl_->entries.emplace(key, std::move(entry));
    return Result<ResidentAcquisition>::success(
        {ResidentDisposition::admitted, primary_pointer, secondary_pointer,
         required});
}

}  // namespace k3x::cuda
