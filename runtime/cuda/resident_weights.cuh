// Stable tensor metadata로 exact CUDA resident weight를 관리하는 내부 계약을 정의합니다.
#pragma once

#include "device_memory.cuh"

#include "k3x/status.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace k3x::cuda {

enum class WeightRepresentation { dense_fp32, dense_bf16, mxfp4 };

struct ResidentWeightKey {
    std::uint64_t tensor_id;
    WeightRepresentation representation;
    std::uint64_t rows;
    std::uint64_t cols;
    std::uint64_t group_size;
};

enum class ResidentDisposition { hit, admitted, bypass };

struct ResidentAcquisition {
    ResidentDisposition disposition;
    const void* primary;
    const void* secondary;
    std::uint64_t uploaded_bytes;
};

class ResidentWeightTable {
public:
    ResidentWeightTable(std::uint64_t capacity, BackendMemoryStats* memory,
                        BackendRuntimeStats* runtime,
                        cudaStream_t stream);
    ~ResidentWeightTable();
    ResidentWeightTable(const ResidentWeightTable&) = delete;
    ResidentWeightTable& operator=(const ResidentWeightTable&) = delete;

    Result<ResidentAcquisition> acquire(
        ResidentWeightKey key, std::span<const std::byte> primary,
        std::span<const std::byte> secondary);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace k3x::cuda
