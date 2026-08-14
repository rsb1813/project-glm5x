// bounded L1 expert store의 atomic admission과 exact bypass를 구현합니다.
#include "k3x/host_expert_store.hpp"
#include "k3x/runtime_profile.hpp"

#include <algorithm>
#include <limits>
#include <tuple>

namespace k3x {
namespace {
constexpr std::size_t native_mxfp4_group_size = 32;

bool valid_projection(const ExpertProjection& projection) {
    if (projection.rows == 0 || projection.cols == 0 ||
        projection.cols % 2 != 0 ||
        projection.cols % native_mxfp4_group_size != 0 ||
        projection.rows >
            std::numeric_limits<std::size_t>::max() / projection.cols) {
        return false;
    }
    const auto elements = projection.rows * projection.cols;
    return projection.packed.size() == elements / 2 &&
           projection.scales.size() ==
               elements / native_mxfp4_group_size &&
           std::none_of(
               projection.scales.begin(), projection.scales.end(),
               [](std::byte scale) { return scale == std::byte{0xff}; });
}

Result<std::size_t> charged_bytes(const ExpertMlpPayload& payload) {
    if (!valid_projection(payload.gate) || !valid_projection(payload.up) ||
        !valid_projection(payload.down) ||
        payload.gate.rows != payload.up.rows ||
        payload.gate.cols != payload.up.cols ||
        payload.down.cols != payload.gate.rows ||
        payload.down.rows != payload.gate.cols) {
        return Result<std::size_t>::failure(
            ErrorCode::invalid_mxfp4, "invalid expert payload");
    }
    std::size_t total = 0;
    for (const auto* projection : {&payload.gate, &payload.up, &payload.down}) {
        for (const auto size : {projection->packed.size(),
                                projection->scales.size()}) {
            if (size > std::numeric_limits<std::size_t>::max() - total) {
                return Result<std::size_t>::failure(
                    ErrorCode::invalid_mxfp4, "expert payload size overflow");
            }
            total += size;
        }
    }
    return Result<std::size_t>::success(total);
}
}

HostExpertStore::HostExpertStore(L1ExpertCacheMode mode,
                                 std::size_t capacity_bytes,
                                 RuntimeProfile* profile,
                                 std::uint64_t profile_prior_strength)
    : mode_(mode), capacity_bytes_(capacity_bytes), profile_(profile),
      profile_prior_strength_(profile_prior_strength) {}

std::size_t HostExpertStore::KeyHash::operator()(
    const ExpertKey& key) const noexcept {
    const auto layer = std::hash<std::size_t>{}(key.layer);
    const auto expert = std::hash<std::size_t>{}(key.expert);
    return layer ^ (expert + 0x9e3779b9U + (layer << 6U) + (layer >> 2U));
}

Result<ExpertPayloadHandle> HostExpertStore::get_or_load(
    ExpertKey key, const ExpertPayloadLoader& loader) {
    std::lock_guard lock(mutex_);
    if (mode_ != L1ExpertCacheMode::disabled) {
        if (const auto found = entries_.find(key); found != entries_.end()) {
            ++stats_.hits;
            found->second.last_cycle = active_cycle_;
            found->second.last_access = next_access_++;
            ++found->second.frequency;
            return Result<ExpertPayloadHandle>::success(found->second.handle);
        }
    }

    auto loaded = loader();
    if (!loaded) {
        return Result<ExpertPayloadHandle>::failure(
            loaded.error(), loaded.message());
    }
    auto bytes = charged_bytes(loaded.value());
    if (!bytes) {
        return Result<ExpertPayloadHandle>::failure(bytes.error(), bytes.message());
    }
    ExpertPayloadHandle handle =
        std::make_shared<const ExpertMlpPayload>(std::move(loaded.value()));
    if (mode_ == L1ExpertCacheMode::disabled) {
        return Result<ExpertPayloadHandle>::success(std::move(handle));
    }

    ++stats_.misses;
    if (const auto evicted = evicted_cycle_.find(key);
        evicted != evicted_cycle_.end() && evicted->second == active_cycle_) {
        ++stats_.collision_misses;
    }
    if (bytes.value() > capacity_bytes_) {
        ++stats_.bypasses;
        return Result<ExpertPayloadHandle>::success(std::move(handle));
    }
    if (mode_ == L1ExpertCacheMode::static_admission &&
        bytes.value() > capacity_bytes_ - stats_.resident_bytes) {
        ++stats_.bypasses;
        return Result<ExpertPayloadHandle>::success(std::move(handle));
    }
    while (bytes.value() > capacity_bytes_ - stats_.resident_bytes) {
        auto victim = entries_.end();
        for (auto candidate = entries_.begin(); candidate != entries_.end();
             ++candidate) {
            if (protected_.contains(candidate->first)) continue;
            if (victim == entries_.end()) {
                victim = candidate;
                continue;
            }
            const auto& left = candidate->second;
            const auto& right = victim->second;
            bool preferred = false;
            if (mode_ == L1ExpertCacheMode::lru) {
                preferred = std::tie(left.last_access, left.insertion) <
                            std::tie(right.last_access, right.insertion);
            } else if (mode_ == L1ExpertCacheMode::lfu) {
                preferred = std::tie(left.frequency, left.last_access,
                                     left.insertion) <
                            std::tie(right.frequency, right.last_access,
                                     right.insertion);
            } else if (mode_ == L1ExpertCacheMode::least_stale) {
                const auto left_current = left.last_cycle == active_cycle_;
                const auto right_current = right.last_cycle == active_cycle_;
                const auto spatial_priority = [this](ExpertKey key) {
                    if (key.layer <= active_layer_) {
                        return std::pair{false, key.layer};
                    }
                    return std::pair{
                        true, std::numeric_limits<std::size_t>::max() -
                                  key.layer};
                };
                preferred =
                    std::tuple(left_current,
                               spatial_priority(candidate->first),
                               left.last_access, left.insertion) <
                    std::tuple(right_current,
                               spatial_priority(victim->first),
                               right.last_access, right.insertion);
            } else {
                const auto left_score = profile_ ? profile_->usefulness(
                    candidate->first, profile_prior_strength_) : 0.0;
                const auto right_score = profile_ ? profile_->usefulness(
                    victim->first, profile_prior_strength_) : 0.0;
                preferred = std::tuple(left_score, left.last_access,
                                       left.insertion) <
                            std::tuple(right_score, right.last_access,
                                       right.insertion);
            }
            if (preferred) victim = candidate;
        }
        if (victim == entries_.end()) {
            ++stats_.bypasses;
            return Result<ExpertPayloadHandle>::success(std::move(handle));
        }
        stats_.resident_bytes -= victim->second.bytes;
        evicted_cycle_[victim->first] = active_cycle_;
        entries_.erase(victim);
        ++stats_.evictions;
    }
    entries_.emplace(
        key, Entry{handle, bytes.value(), active_cycle_, next_access_++, 1,
                   next_insertion_++});
    stats_.resident_bytes += bytes.value();
    stats_.peak_resident_bytes =
        std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    return Result<ExpertPayloadHandle>::success(std::move(handle));
}

void HostExpertStore::begin_access_set(
    std::uint64_t forward_cycle, std::size_t layer,
    std::span<const ExpertKey> selected) {
    std::lock_guard lock(mutex_);
    active_cycle_ = forward_cycle;
    active_layer_ = layer;
    if (profile_) profile_->observe(forward_cycle, layer, selected);
    protected_.clear();
    for (const auto key : selected) {
        protected_.insert(key);
        if (const auto found = entries_.find(key); found != entries_.end()) {
            found->second.last_cycle = active_cycle_;
        }
    }
}

bool HostExpertStore::contains(ExpertKey key) const {
    std::lock_guard lock(mutex_);
    return entries_.contains(key);
}

L1ExpertCacheStats HostExpertStore::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}
}
