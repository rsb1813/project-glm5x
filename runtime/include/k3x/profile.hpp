// 런타임 연산별 결정적 프로파일 이벤트와 집계 인터페이스를 정의합니다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace k3x {

enum class ProfilePhase { prefill, decode };

enum class ProfileOperation {
    tensor_read,
    dense_matvec,
    mxfp4_matvec,
    situ_glu,
    moe_mix,
    rms_norm,
    residual_add,
    host_to_device,
    weight_host_to_device,
    activation_host_to_device,
    device_to_host,
};

enum class NumericPrecision { none, fp32, bf16_rounded, mxfp4_e2m1_e8m0 };

inline constexpr std::uint32_t profile_global_layer =
    std::numeric_limits<std::uint32_t>::max();

struct ProfileEvent {
    ProfilePhase phase;
    ProfileOperation operation;
    NumericPrecision precision;
    std::uint32_t layer;
    std::uint64_t wall_nanoseconds;
    std::uint64_t device_nanoseconds;
    std::uint64_t logical_bytes;
    std::uint64_t transfer_bytes;
    bool success;
};

struct ProfileSummary {
    std::uint64_t wall_nanoseconds{};
    std::uint64_t device_nanoseconds{};
    std::uint64_t logical_bytes{};
    std::uint64_t host_to_device_bytes{};
    std::uint64_t weight_host_to_device_bytes{};
    std::uint64_t activation_host_to_device_bytes{};
    std::uint64_t device_to_host_bytes{};
    std::size_t failed_operations{};
};

class Profiler {
public:
    void record(ProfileEvent event);
    const std::vector<ProfileEvent>& events() const noexcept;
    ProfileSummary summary() const noexcept;

private:
    std::vector<ProfileEvent> events_;
};

}  // namespace k3x
