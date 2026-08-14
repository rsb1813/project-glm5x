// 기록된 프로파일 이벤트를 성공 여부와 전송 방향에 따라 집계합니다.
#include "k3x/profile.hpp"

#include <utility>

namespace k3x {

void Profiler::record(ProfileEvent event) { events_.push_back(std::move(event)); }

const std::vector<ProfileEvent>& Profiler::events() const noexcept { return events_; }

ProfileSummary Profiler::summary() const noexcept {
    ProfileSummary result;
    for (const auto& event : events_) {
        if (!event.success) {
            ++result.failed_operations;
            continue;
        }

        result.wall_nanoseconds += event.wall_nanoseconds;
        result.device_nanoseconds += event.device_nanoseconds;
        result.logical_bytes += event.logical_bytes;
        if (event.operation == ProfileOperation::host_to_device) {
            result.host_to_device_bytes += event.transfer_bytes;
        } else if (event.operation ==
                   ProfileOperation::weight_host_to_device) {
            result.host_to_device_bytes += event.transfer_bytes;
            result.weight_host_to_device_bytes += event.transfer_bytes;
        } else if (event.operation ==
                   ProfileOperation::activation_host_to_device) {
            result.host_to_device_bytes += event.transfer_bytes;
            result.activation_host_to_device_bytes += event.transfer_bytes;
        } else if (event.operation == ProfileOperation::device_to_host) {
            result.device_to_host_bytes += event.transfer_bytes;
        }
    }
    return result;
}

}  // namespace k3x
