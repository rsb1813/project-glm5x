// native MXFP4 expert payload를 bounded immutable L1 store에 보관합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace k3x {
class RuntimeProfile;

enum class L1ExpertCacheMode {
    disabled,
    static_admission,
    lru,
    lfu,
    least_stale,
    profiled,
};

struct L1ExpertCacheStats {
    std::uint64_t hits{};
    std::uint64_t misses{};
    std::uint64_t bypasses{};
    std::uint64_t evictions{};
    std::uint64_t collision_misses{};
    std::size_t resident_bytes{};
    std::size_t peak_resident_bytes{};
};

struct ExpertKey {
    std::size_t layer{};
    std::size_t expert{};

    bool operator==(const ExpertKey&) const = default;
};

struct ExpertProjection {
    std::uint64_t id{};
    std::vector<std::byte> packed;
    std::vector<std::byte> scales;
    std::size_t rows{};
    std::size_t cols{};

    Mxfp4WeightView view(std::size_t group_size) const {
        return {id, packed, scales, rows, cols, group_size};
    }
};

struct ExpertMlpPayload {
    ExpertProjection gate;
    ExpertProjection up;
    ExpertProjection down;

    Mxfp4MlpView view(std::size_t group_size) const {
        return {gate.view(group_size), up.view(group_size),
                down.view(group_size)};
    }
};

using ExpertPayloadHandle = std::shared_ptr<const ExpertMlpPayload>;
using ExpertPayloadLoader = std::function<Result<ExpertMlpPayload>()>;

class HostExpertStore {
public:
    HostExpertStore(L1ExpertCacheMode mode, std::size_t capacity_bytes,
                    RuntimeProfile* profile = nullptr,
                    std::uint64_t profile_prior_strength = 0);

    Result<ExpertPayloadHandle> get_or_load(
        ExpertKey key, const ExpertPayloadLoader& loader);
    void begin_access_set(std::uint64_t forward_cycle,
                          std::size_t layer,
                          std::span<const ExpertKey> selected);
    bool contains(ExpertKey key) const;
    L1ExpertCacheStats stats() const;

private:
    struct KeyHash {
        std::size_t operator()(const ExpertKey& key) const noexcept;
    };

    struct Entry {
        ExpertPayloadHandle handle;
        std::size_t bytes{};
        std::uint64_t last_cycle{};
        std::uint64_t last_access{};
        std::uint64_t frequency{};
        std::uint64_t insertion{};
    };

    L1ExpertCacheMode mode_;
    std::size_t capacity_bytes_{};
    RuntimeProfile* profile_{};
    std::uint64_t profile_prior_strength_{};
    L1ExpertCacheStats stats_;
    std::unordered_map<ExpertKey, Entry, KeyHash> entries_;
    std::unordered_set<ExpertKey, KeyHash> protected_;
    std::unordered_map<ExpertKey, std::uint64_t, KeyHash> evicted_cycle_;
    std::uint64_t active_cycle_{};
    std::size_t active_layer_{};
    std::uint64_t next_access_{};
    std::uint64_t next_insertion_{};
    mutable std::mutex mutex_;
};
}
