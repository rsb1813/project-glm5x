// 런타임 프로파일 이벤트의 결정적 집계를 검증합니다.
#include "k3x/profile.hpp"

int main() {
    k3x::Profiler profiler;
    profiler.record({k3x::ProfilePhase::prefill,
                     k3x::ProfileOperation::dense_matvec,
                     k3x::NumericPrecision::fp32,
                     0,
                     100,
                     80,
                     64,
                     0,
                     true});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::weight_host_to_device,
                     k3x::NumericPrecision::none,
                     k3x::profile_global_layer,
                     30,
                     20,
                     0,
                     160,
                     true});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::activation_host_to_device,
                     k3x::NumericPrecision::none,
                     k3x::profile_global_layer,
                     10,
                     0,
                     0,
                     96,
                     true});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::mxfp4_matvec,
                     k3x::NumericPrecision::mxfp4_e2m1_e8m0,
                     1,
                     999,
                     888,
                     128,
                     0,
                     false});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::activation_host_to_device,
                     k3x::NumericPrecision::none,
                     k3x::profile_global_layer,
                     777,
                     0,
                     0,
                     4096,
                     false});
    profiler.record({k3x::ProfilePhase::decode,
                     k3x::ProfileOperation::situ_glu,
                     k3x::NumericPrecision::fp32,
                     2,
                     25,
                     15,
                     32,
                     0,
                     true});

    const auto& events = profiler.events();
    if (events.size() != 6) return 1;
    if (events.front().precision != k3x::NumericPrecision::fp32) return 2;
    if (events[3].precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 3;

    const auto summary = profiler.summary();
    if (events[5].operation != k3x::ProfileOperation::situ_glu) return 12;
    if (summary.wall_nanoseconds != 165) return 4;
    if (summary.device_nanoseconds != 115) return 5;
    if (summary.logical_bytes != 96) return 6;
    if (summary.host_to_device_bytes != 256) return 7;
    if (summary.weight_host_to_device_bytes != 160) return 8;
    if (summary.activation_host_to_device_bytes != 96) return 9;
    if (summary.device_to_host_bytes != 0) return 10;
    if (summary.failed_operations != 2) return 11;
    return 0;
}
