// 런타임 전용 task/session expert profile을 선언합니다.
#pragma once

#include "k3x/host_expert_store.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace k3x {

class RuntimeProfile {
public:
    static constexpr std::size_t maximum_metadata_records = 32;
    static constexpr std::size_t maximum_metadata_key_bytes = 32;
    static constexpr std::size_t maximum_metadata_value_bytes = 256;
    static constexpr std::size_t maximum_frequency_records = 1U << 20U;
    static constexpr std::size_t maximum_transition_records = 1U << 22U;
    static constexpr std::size_t persisted_hot_bank_size = 64;
    static constexpr std::size_t maximum_profile_bytes = 64U << 20U;

    Result<bool> set_metadata(std::string key, std::string value);
    const std::map<std::string, std::string>& metadata() const noexcept {
        return metadata_;
    }

    void observe(std::uint64_t forward_cycle, std::size_t layer,
                 std::span<const ExpertKey> selected);
    std::uint64_t prior_frequency(ExpertKey key) const;
    std::uint64_t live_frequency(ExpertKey key) const;
    std::uint64_t prior_transition(ExpertKey from, ExpertKey to) const;
    std::uint64_t live_transition(ExpertKey from, ExpertKey to) const;
    std::uint64_t live_route_observations() const noexcept {
        return live_route_observations_;
    }
    double prior_weight(std::uint64_t prior_strength) const noexcept;
    double usefulness(ExpertKey key, std::uint64_t prior_strength) const;
    std::vector<ExpertKey> hot_bank(std::size_t count) const;

    Result<bool> save(const std::filesystem::path& path) const;
    static Result<RuntimeProfile> load(const std::filesystem::path& path);

private:
    using FrequencyKey = std::pair<std::size_t, std::size_t>;
    using TransitionKey =
        std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;

    static FrequencyKey frequency_key(ExpertKey key) {
        return {key.layer, key.expert};
    }
    static TransitionKey transition_key(ExpertKey from, ExpertKey to) {
        return {from.layer, from.expert, to.layer, to.expert};
    }
    static std::uint64_t count_of(
        const std::map<FrequencyKey, std::uint64_t>& counts, ExpertKey key);
    static std::uint64_t count_of(
        const std::map<TransitionKey, std::uint64_t>& counts,
        ExpertKey from, ExpertKey to);

    std::map<std::string, std::string> metadata_;
    std::map<FrequencyKey, std::uint64_t> prior_frequency_;
    std::map<FrequencyKey, std::uint64_t> live_frequency_;
    std::map<TransitionKey, std::uint64_t> prior_transitions_;
    std::map<TransitionKey, std::uint64_t> live_transitions_;
    std::uint64_t prior_route_observations_{};
    std::uint64_t live_route_observations_{};
    std::uint64_t previous_cycle_{};
    std::size_t previous_layer_{};
    bool has_previous_{};
    std::vector<ExpertKey> previous_selected_;
};

}
