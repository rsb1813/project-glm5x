// K3X synthetic runtime을 실행하고 bounded JSON metrics를 기록합니다.
#include "k3x/aurora.hpp"
#include "k3x/model.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct ProcessIoSnapshot {
    bool available{};
    std::uint64_t rchar{};
    std::uint64_t read_bytes{};
};

ProcessIoSnapshot process_io_snapshot() {
#ifdef __linux__
    std::ifstream input("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    ProcessIoSnapshot snapshot;
    bool found_rchar = false;
    bool found_read_bytes = false;
    while (input >> key >> value) {
        if (key == "rchar:") {
            snapshot.rchar = value;
            found_rchar = true;
        } else if (key == "read_bytes:") {
            snapshot.read_bytes = value;
            found_read_bytes = true;
        }
    }
    snapshot.available = found_rchar && found_read_bytes;
    return snapshot;
#else
    return {};
#endif
}

ProcessIoSnapshot process_io_delta(const ProcessIoSnapshot& before,
                                   const ProcessIoSnapshot& after) {
    if (!before.available || !after.available ||
        after.rchar < before.rchar || after.read_bytes < before.read_bytes) {
        return {};
    }
    return ProcessIoSnapshot{
        true, after.rchar - before.rchar, after.read_bytes - before.read_bytes};
}

void write_json_string(std::ostream& output, std::string_view value) {
    output << '"';
    for (const auto character : value) {
        if (character == '"' || character == '\\') output << '\\';
        output << character;
    }
    output << '"';
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

std::uint32_t model_natural_top_k(
    const std::array<std::byte, k3x::model_config_bytes>& config) {
    constexpr std::size_t offset = 13 * sizeof(std::uint32_t);
    return std::to_integer<std::uint32_t>(config[offset]) |
        (std::to_integer<std::uint32_t>(config[offset + 1]) << 8U) |
        (std::to_integer<std::uint32_t>(config[offset + 2]) << 16U) |
        (std::to_integer<std::uint32_t>(config[offset + 3]) << 24U);
}

class ScriptedDraftProvider final : public k3x::DraftProvider {
public:
    explicit ScriptedDraftProvider(std::deque<k3x::DraftProposal> proposals)
        : proposals_(std::move(proposals)) {}

    k3x::Result<k3x::DraftProposal> propose(
        const k3x::DraftRequest&) override {
        if (proposals_.empty()) {
            return k3x::Result<k3x::DraftProposal>::failure(
                k3x::ErrorCode::invalid_state,
                "scripted draft proposals exhausted");
        }
        auto proposal = std::move(proposals_.front());
        proposals_.pop_front();
        return k3x::Result<k3x::DraftProposal>::success(std::move(proposal));
    }

    void update(const k3x::DraftVerification&) override {}

    bool empty() const { return proposals_.empty(); }

private:
    std::deque<k3x::DraftProposal> proposals_;
};

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path, output_path;
    std::filesystem::path runtime_profile_in, runtime_profile_out;
    std::string prompt_text, mode = "incremental";
    std::string runtime_metadata_text;
    std::string profile_prior_strength_text = "64";
    std::string backend_name = "cpu", dense_precision_name = "fp32";
    std::string cuda_allocation_name = "per-operation";
    std::string cuda_weights_name = "transient";
    std::string cuda_batching_name = "scalar";
    std::string cuda_boundary_name = "operation";
    std::string cuda_transfer_name = "synchronous";
    std::string cuda_moe_fusion_name = "none";
    std::string cuda_weight_validation_name = "per-call";
    std::string cuda_resident_bytes_text = "0";
    std::string cuda_pinned_bytes_text = "0";
    std::string l1_expert_cache_name = "disabled";
    std::string l1_expert_cache_bytes_text = "0";
    std::string l2_io_name = "pread";
    std::string l2_cache_name = "buffered";
    std::string l2_queue_depth_text = "8";
    std::string l2_schedule_name = "blocking";
    std::string routing_mode_name = "natural";
    std::string routing_fixed_k_text = "0";
    std::string routing_mass_target_text = "0.9";
    std::string routing_boundary_gap_text = "0";
    std::string routing_agent_failures_text = "0";
    std::string routing_critical_text = "false";
    std::string speculative_mode_name = "none";
    std::string speculative_verification_name = "token-major";
    std::string speculative_block_size_text = "0";
    std::string speculative_script_text;
    std::string aurora_draft_k_text = "0";
    std::string aurora_block_policy_name = "fixed";
    std::string aurora_draft_backend_name = "cpu";
    std::string aurora_draft_resident_bytes_text = "0";
    std::string aurora_draft_batching_name = "grouped";
    std::string aurora_draft_boundary_name = "ffn-block";
    bool aurora_draft_k_supplied = false;
    bool aurora_block_policy_supplied = false;
    bool aurora_draft_backend_supplied = false;
    bool aurora_draft_resident_bytes_supplied = false;
    bool aurora_draft_batching_supplied = false;
    bool aurora_draft_boundary_supplied = false;
    bool diagnostics = false;
    std::size_t count = 0;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--model") model_path = value;
        else if (key == "--prompt-ids") prompt_text = value;
        else if (key == "--generate") count = std::stoull(value);
        else if (key == "--mode") mode = value;
        else if (key == "--diagnostics") diagnostics = value == "true";
        else if (key == "--json") output_path = value;
        else if (key == "--backend") backend_name = value;
        else if (key == "--dense-precision") dense_precision_name = value;
        else if (key == "--cuda-allocation") cuda_allocation_name = value;
        else if (key == "--cuda-weights") cuda_weights_name = value;
        else if (key == "--cuda-batching") cuda_batching_name = value;
        else if (key == "--cuda-boundary") cuda_boundary_name = value;
        else if (key == "--cuda-transfer") cuda_transfer_name = value;
        else if (key == "--cuda-moe-fusion") cuda_moe_fusion_name = value;
        else if (key == "--cuda-weight-validation") cuda_weight_validation_name = value;
        else if (key == "--cuda-resident-bytes") cuda_resident_bytes_text = value;
        else if (key == "--cuda-pinned-bytes") cuda_pinned_bytes_text = value;
        else if (key == "--l1-expert-cache") l1_expert_cache_name = value;
        else if (key == "--l1-expert-cache-bytes") l1_expert_cache_bytes_text = value;
        else if (key == "--profile-prior-strength") profile_prior_strength_text = value;
        else if (key == "--runtime-metadata") runtime_metadata_text = value;
        else if (key == "--runtime-profile-in") runtime_profile_in = value;
        else if (key == "--runtime-profile-out") runtime_profile_out = value;
        else if (key == "--l2-io") l2_io_name = value;
        else if (key == "--l2-cache") l2_cache_name = value;
        else if (key == "--l2-queue-depth") l2_queue_depth_text = value;
        else if (key == "--l2-schedule") l2_schedule_name = value;
        else if (key == "--routing-mode") routing_mode_name = value;
        else if (key == "--routing-fixed-k") routing_fixed_k_text = value;
        else if (key == "--routing-mass-target") routing_mass_target_text = value;
        else if (key == "--routing-min-boundary-gap") routing_boundary_gap_text = value;
        else if (key == "--routing-agent-failures") routing_agent_failures_text = value;
        else if (key == "--routing-critical") routing_critical_text = value;
        else if (key == "--speculative-mode") speculative_mode_name = value;
        else if (key == "--speculative-verification") speculative_verification_name = value;
        else if (key == "--speculative-block-size") speculative_block_size_text = value;
        else if (key == "--speculative-script") speculative_script_text = value;
        else if (key == "--aurora-draft-k") {
            aurora_draft_k_text = value;
            aurora_draft_k_supplied = true;
        }
        else if (key == "--aurora-block-policy") {
            aurora_block_policy_name = value;
            aurora_block_policy_supplied = true;
        }
        else if (key == "--aurora-draft-backend") {
            aurora_draft_backend_name = value;
            aurora_draft_backend_supplied = true;
        }
        else if (key == "--aurora-draft-resident-bytes") {
            aurora_draft_resident_bytes_text = value;
            aurora_draft_resident_bytes_supplied = true;
        }
        else if (key == "--aurora-draft-batching") {
            aurora_draft_batching_name = value;
            aurora_draft_batching_supplied = true;
        }
        else if (key == "--aurora-draft-boundary") {
            aurora_draft_boundary_name = value;
            aurora_draft_boundary_supplied = true;
        }
        else { std::cerr << "unknown argument: " << key << '\n'; return 2; }
    }

    k3x::BackendOptions backend_options;
    k3x::RuntimeOptions runtime_options;
    k3x::ReaderOptions reader_options;
    runtime_options.incremental = mode == "incremental";
    runtime_options.diagnostics = diagnostics;
    if (routing_mode_name == "natural") {
        runtime_options.routing_policy.mode = k3x::RoutingMode::natural;
    } else if (routing_mode_name == "fixed") {
        runtime_options.routing_policy.mode = k3x::RoutingMode::fixed;
    } else if (routing_mode_name == "adaptive") {
        runtime_options.routing_policy.mode = k3x::RoutingMode::adaptive;
    } else {
        std::cerr << "unknown routing mode: " << routing_mode_name << '\n';
        return 2;
    }
    const auto parse_size = [](const std::string& text, std::size_t& value) {
        const auto* begin = text.data();
        const auto* end = begin + text.size();
        const auto parsed = std::from_chars(begin, end, value);
        return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == end;
    };
    const auto parse_float = [](const std::string& text, float& value) {
        const auto* begin = text.data();
        const auto* end = begin + text.size();
        const auto parsed = std::from_chars(
            begin, end, value, std::chars_format::general);
        return !text.empty() && parsed.ec == std::errc{} && parsed.ptr == end &&
               std::isfinite(value);
    };
    std::size_t speculative_block_size = 0;
    if (!parse_size(speculative_block_size_text, speculative_block_size)) {
        std::cerr << "invalid speculative block size: "
                  << speculative_block_size_text << '\n';
        return 2;
    }
    std::size_t aurora_draft_k = 0;
    if (!parse_size(aurora_draft_k_text, aurora_draft_k)) {
        std::cerr << "invalid AURORA draft K: " << aurora_draft_k_text
                  << '\n';
        return 2;
    }
    std::uint64_t aurora_draft_resident_bytes = 0;
    const auto* draft_resident_begin =
        aurora_draft_resident_bytes_text.data();
    const auto* draft_resident_end = draft_resident_begin +
        aurora_draft_resident_bytes_text.size();
    const auto draft_resident_parse = std::from_chars(
        draft_resident_begin, draft_resident_end,
        aurora_draft_resident_bytes);
    if (aurora_draft_resident_bytes_text.empty() ||
        draft_resident_parse.ec != std::errc{} ||
        draft_resident_parse.ptr != draft_resident_end) {
        std::cerr << "invalid AURORA draft resident byte capacity: "
                  << aurora_draft_resident_bytes_text << '\n';
        return 2;
    }
    if (speculative_mode_name != "none" &&
        speculative_mode_name != "scripted-reference" &&
        speculative_mode_name != "aurora-replay" &&
        speculative_mode_name != "aurora-persistent") {
        std::cerr << "unknown speculative mode: " << speculative_mode_name
                  << '\n';
        return 2;
    }
    const bool aurora_mode = speculative_mode_name == "aurora-replay" ||
        speculative_mode_name == "aurora-persistent";
    if (aurora_draft_backend_name != "cpu" &&
        aurora_draft_backend_name != "cuda-custom") {
        std::cerr << "unknown AURORA draft backend: "
                  << aurora_draft_backend_name << '\n';
        return 2;
    }
    if (aurora_draft_batching_name != "grouped" &&
        aurora_draft_batching_name != "resident-grid") {
        std::cerr << "unknown AURORA draft batching mode: "
                  << aurora_draft_batching_name << '\n';
        return 2;
    }
    if (aurora_draft_boundary_name != "ffn-block" &&
        aurora_draft_boundary_name != "moe-layer") {
        std::cerr << "unknown AURORA draft boundary: "
                  << aurora_draft_boundary_name << '\n';
        return 2;
    }
    if (speculative_verification_name == "token-major") {
        runtime_options.speculative_verification =
            k3x::SpeculativeVerificationMode::token_major;
    } else if (speculative_verification_name == "expert-major") {
        runtime_options.speculative_verification =
            k3x::SpeculativeVerificationMode::expert_major;
    } else {
        std::cerr << "unknown speculative verification mode: "
                  << speculative_verification_name << '\n';
        return 2;
    }
    if (runtime_options.speculative_verification ==
            k3x::SpeculativeVerificationMode::expert_major &&
        speculative_mode_name == "none") {
        std::cerr << "expert-major verification requires scripted-reference or AURORA replay speculation\n";
        return 2;
    }
    if (speculative_mode_name == "none" &&
        (speculative_block_size != 0 || !speculative_script_text.empty())) {
        std::cerr << "speculative mode none requires block size 0 and an empty script\n";
        return 2;
    }
    if (speculative_mode_name == "none" &&
        (aurora_draft_k_supplied || aurora_block_policy_supplied ||
         aurora_draft_backend_supplied ||
         aurora_draft_resident_bytes_supplied ||
         aurora_draft_batching_supplied ||
         aurora_draft_boundary_supplied)) {
        std::cerr << "speculative mode none does not accept speculative options\n";
        return 2;
    }
    if (speculative_mode_name == "scripted-reference" &&
        (aurora_draft_k_supplied || aurora_block_policy_supplied ||
         aurora_draft_backend_supplied ||
         aurora_draft_resident_bytes_supplied ||
         aurora_draft_batching_supplied ||
         aurora_draft_boundary_supplied)) {
        std::cerr << "scripted-reference does not accept AURORA options\n";
        return 2;
    }
    if (aurora_draft_resident_bytes_supplied &&
        (speculative_mode_name != "aurora-persistent" ||
         aurora_draft_backend_name != "cuda-custom")) {
        std::cerr << "AURORA draft residency requires persistent cuda-custom draft backend\n";
        return 2;
    }
    if (aurora_draft_batching_supplied &&
        (speculative_mode_name != "aurora-persistent" ||
         aurora_draft_backend_name != "cuda-custom")) {
        std::cerr << "AURORA draft batching requires persistent cuda-custom draft backend\n";
        return 2;
    }
    if (aurora_draft_boundary_supplied &&
        (speculative_mode_name != "aurora-persistent" ||
         aurora_draft_backend_name != "cuda-custom")) {
        std::cerr << "AURORA draft boundary requires persistent cuda-custom draft backend\n";
        return 2;
    }
    if (aurora_draft_boundary_name == "moe-layer" &&
        (aurora_draft_batching_name != "resident-grid" ||
         aurora_draft_resident_bytes == 0)) {
        std::cerr << "AURORA moe-layer draft boundary requires resident-grid batching and positive residency\n";
        return 2;
    }
    if (aurora_draft_batching_name == "resident-grid" &&
        aurora_draft_resident_bytes == 0) {
        std::cerr << "AURORA resident-grid draft batching requires positive residency\n";
        return 2;
    }
    if (speculative_mode_name == "aurora-replay" &&
        aurora_draft_backend_name != "cpu") {
        std::cerr << "AURORA replay requires CPU draft backend\n";
        return 2;
    }
    if (speculative_mode_name == "scripted-reference" &&
        speculative_block_size == 0) {
        std::cerr << "scripted-reference speculation requires a positive block size\n";
        return 2;
    }
    if (speculative_mode_name == "scripted-reference" &&
        !runtime_options.incremental) {
        std::cerr << "scripted-reference speculation requires incremental mode\n";
        return 2;
    }
    const auto allowed_aurora_k = [](std::size_t value) {
        return value == 4 || value == 6 || value == 8 || value == 12;
    };
    if (aurora_mode &&
        !allowed_aurora_k(aurora_draft_k)) {
        std::cerr << "AURORA requires draft K4, K6, K8, or K12\n";
        return 2;
    }
    if (aurora_mode &&
        aurora_block_policy_name != "fixed" &&
        aurora_block_policy_name != "adaptive") {
        std::cerr << "unknown AURORA block policy: "
                  << aurora_block_policy_name << '\n';
        return 2;
    }
    if (aurora_mode &&
        speculative_block_size != 1 && speculative_block_size != 2 &&
        speculative_block_size != 4) {
        std::cerr << "AURORA requires block size 1, 2, or 4\n";
        return 2;
    }
    if (aurora_mode &&
        (!runtime_options.incremental || !speculative_script_text.empty() ||
         runtime_options.routing_policy.mode != k3x::RoutingMode::natural)) {
        std::cerr << "AURORA requires incremental natural routing and an empty script\n";
        return 2;
    }
    std::deque<k3x::DraftProposal> scripted_proposals;
    if (speculative_mode_name == "scripted-reference" &&
        !speculative_script_text.empty()) {
        std::stringstream script_parser(speculative_script_text);
        std::string record;
        while (std::getline(script_parser, record, ';')) {
            const auto separator = record.find(':');
            if (separator == std::string::npos || separator == 0 ||
                record.find(':', separator + 1) != std::string::npos) {
                std::cerr << "invalid speculative script record: " << record
                          << '\n';
                return 2;
            }
            std::uint32_t anchor = 0;
            const auto anchor_text = record.substr(0, separator);
            const auto anchor_parsed = std::from_chars(
                anchor_text.data(), anchor_text.data() + anchor_text.size(),
                anchor);
            if (anchor_parsed.ec != std::errc{} ||
                anchor_parsed.ptr != anchor_text.data() + anchor_text.size()) {
                std::cerr << "invalid speculative script record: " << record
                          << '\n';
                return 2;
            }
            k3x::DraftProposal proposal;
            proposal.anchor_token = anchor;
            const auto candidates_text = record.substr(separator + 1);
            if (!candidates_text.empty()) {
                std::stringstream candidate_parser(candidates_text);
                std::string candidate_text;
                while (std::getline(candidate_parser, candidate_text, ',')) {
                    std::uint32_t candidate = 0;
                    const auto candidate_parsed = std::from_chars(
                        candidate_text.data(),
                        candidate_text.data() + candidate_text.size(),
                        candidate);
                    if (candidate_text.empty() ||
                        candidate_parsed.ec != std::errc{} ||
                        candidate_parsed.ptr !=
                            candidate_text.data() + candidate_text.size()) {
                        std::cerr << "invalid speculative script record: "
                                  << record << '\n';
                        return 2;
                    }
                    proposal.candidate_tokens.push_back(candidate);
                }
                if (candidates_text.back() == ',') {
                    std::cerr << "invalid speculative script record: "
                              << record << '\n';
                    return 2;
                }
            }
            scripted_proposals.push_back(std::move(proposal));
        }
    }
    if (!parse_size(routing_fixed_k_text,
                    runtime_options.routing_policy.fixed_k)) {
        std::cerr << "invalid routing fixed K: " << routing_fixed_k_text << '\n';
        return 2;
    }
    const auto allowed_routing_k = [](std::size_t value) {
        return value == 4 || value == 6 || value == 8 || value == 12 ||
               value == 16;
    };
    if (runtime_options.routing_policy.mode == k3x::RoutingMode::fixed &&
        !allowed_routing_k(runtime_options.routing_policy.fixed_k)) {
        std::cerr << "fixed routing requires K4, K6, K8, K12, or K16\n";
        return 2;
    }
    if (runtime_options.routing_policy.mode == k3x::RoutingMode::natural &&
        runtime_options.routing_policy.fixed_k != 0) {
        std::cerr << "natural routing requires --routing-fixed-k 0\n";
        return 2;
    }
    if (runtime_options.routing_policy.mode == k3x::RoutingMode::adaptive &&
        runtime_options.routing_policy.fixed_k != 0) {
        std::cerr << "adaptive routing requires --routing-fixed-k 0\n";
        return 2;
    }
    if (!parse_float(routing_mass_target_text,
                     runtime_options.routing_policy.mass_target) ||
        runtime_options.routing_policy.mass_target <= 0.0F ||
        runtime_options.routing_policy.mass_target > 1.0F) {
        std::cerr << "invalid routing mass target: "
                  << routing_mass_target_text << '\n';
        return 2;
    }
    if (!parse_float(routing_boundary_gap_text,
                     runtime_options.routing_policy.minimum_boundary_gap) ||
        runtime_options.routing_policy.minimum_boundary_gap < 0.0F ||
        runtime_options.routing_policy.minimum_boundary_gap > 1.0F) {
        std::cerr << "invalid routing boundary gap: "
                  << routing_boundary_gap_text << '\n';
        return 2;
    }
    std::size_t routing_agent_failures = 0;
    if (!parse_size(routing_agent_failures_text, routing_agent_failures)) {
        std::cerr << "invalid routing agent failure count: "
                  << routing_agent_failures_text << '\n';
        return 2;
    }
    if (routing_critical_text != "true" && routing_critical_text != "false") {
        std::cerr << "invalid routing critical flag: "
                  << routing_critical_text << '\n';
        return 2;
    }
    if (routing_critical_text == "true" || routing_agent_failures >= 3) {
        runtime_options.routing_policy.quality_floor_k = 16;
    } else if (routing_agent_failures == 2) {
        runtime_options.routing_policy.quality_floor_k = 12;
    } else if (routing_agent_failures == 1) {
        runtime_options.routing_policy.quality_floor_k = 8;
    }
    if (l1_expert_cache_name == "disabled") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::disabled;
    } else if (l1_expert_cache_name == "static") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::static_admission;
    } else if (l1_expert_cache_name == "lru") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::lru;
    } else if (l1_expert_cache_name == "lfu") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::lfu;
    } else if (l1_expert_cache_name == "least-stale") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::least_stale;
    } else if (l1_expert_cache_name == "profiled") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::profiled;
    } else {
        std::cerr << "unknown L1 expert cache mode: " << l1_expert_cache_name << '\n';
        return 2;
    }
    const auto* l1_begin = l1_expert_cache_bytes_text.data();
    const auto* l1_end = l1_begin + l1_expert_cache_bytes_text.size();
    const auto l1_parse = std::from_chars(
        l1_begin, l1_end, runtime_options.l1_expert_cache_bytes);
    if (l1_expert_cache_bytes_text.empty() || l1_parse.ec != std::errc{} ||
        l1_parse.ptr != l1_end) {
        std::cerr << "invalid L1 expert cache byte capacity: "
                  << l1_expert_cache_bytes_text << '\n';
        return 2;
    }
    if (runtime_options.l1_expert_cache != k3x::L1ExpertCacheMode::disabled &&
        runtime_options.l1_expert_cache_bytes == 0) {
        std::cerr << l1_expert_cache_name
                  << " L1 expert cache requires a positive byte capacity\n";
        return 2;
    }
    if (runtime_options.l1_expert_cache == k3x::L1ExpertCacheMode::disabled &&
        runtime_options.l1_expert_cache_bytes != 0) {
        std::cerr << "disabled L1 expert cache requires a zero byte capacity\n";
        return 2;
    }
    const auto* strength_begin = profile_prior_strength_text.data();
    const auto* strength_end =
        strength_begin + profile_prior_strength_text.size();
    const auto strength_parse = std::from_chars(
        strength_begin, strength_end, runtime_options.profile_prior_strength);
    if (profile_prior_strength_text.empty() ||
        strength_parse.ec != std::errc{} ||
        strength_parse.ptr != strength_end ||
        runtime_options.profile_prior_strength == 0) {
        std::cerr << "invalid profile prior strength: "
                  << profile_prior_strength_text << '\n';
        return 2;
    }
    runtime_options.profile_observation =
        runtime_options.l1_expert_cache == k3x::L1ExpertCacheMode::profiled ||
        !runtime_metadata_text.empty() || !runtime_profile_in.empty() ||
        !runtime_profile_out.empty();
    if (l2_schedule_name == "blocking") {
        runtime_options.l2_expert_schedule =
            k3x::L2ExpertScheduleMode::blocking;
    } else if (l2_schedule_name == "deadline") {
        runtime_options.l2_expert_schedule =
            k3x::L2ExpertScheduleMode::deadline;
    } else {
        std::cerr << "unknown L2 expert schedule mode: "
                  << l2_schedule_name << '\n';
        return 2;
    }
    if (l2_io_name == "pread") {
        reader_options.io_engine = k3x::L2IoEngine::pread;
    } else if (l2_io_name == "io-uring") {
        reader_options.io_engine = k3x::L2IoEngine::io_uring;
    } else {
        std::cerr << "unknown L2 I/O engine: " << l2_io_name << '\n';
        return 2;
    }
    if (l2_cache_name == "buffered") {
        reader_options.cache_mode = k3x::L2CacheMode::buffered;
    } else if (l2_cache_name == "direct") {
        reader_options.cache_mode = k3x::L2CacheMode::direct;
    } else {
        std::cerr << "unknown L2 cache mode: " << l2_cache_name << '\n';
        return 2;
    }
    const auto* l2_queue_begin = l2_queue_depth_text.data();
    const auto* l2_queue_end = l2_queue_begin + l2_queue_depth_text.size();
    const auto l2_queue_parse = std::from_chars(
        l2_queue_begin, l2_queue_end, reader_options.queue_depth);
    if (l2_queue_depth_text.empty() || l2_queue_parse.ec != std::errc{} ||
        l2_queue_parse.ptr != l2_queue_end) {
        std::cerr << "invalid L2 queue depth: " << l2_queue_depth_text << '\n';
        return 2;
    }
    if (reader_options.queue_depth == 0) {
        std::cerr << "L2 queue depth must be positive\n";
        return 2;
    }
    if (reader_options.queue_depth > k3x::maximum_l2_queue_depth) {
        std::cerr << "L2 queue depth exceeds maximum: "
                  << reader_options.queue_depth << '\n';
        return 2;
    }
    if (backend_name == "cpu") {
        backend_options.kind = k3x::BackendKind::cpu;
    } else if (backend_name == "cuda-dense") {
        backend_options.kind = k3x::BackendKind::cuda_dense;
    } else if (backend_name == "cuda-custom") {
        backend_options.kind = k3x::BackendKind::cuda_custom;
    } else {
        std::cerr << "unknown backend: " << backend_name << '\n';
        return 2;
    }
    if (dense_precision_name == "fp32") {
        backend_options.dense_precision = k3x::DensePrecision::fp32;
    } else if (dense_precision_name == "bf16") {
        backend_options.dense_precision = k3x::DensePrecision::bf16_rounded;
    } else {
        std::cerr << "unknown dense precision: " << dense_precision_name << '\n';
        return 2;
    }
    if (cuda_allocation_name == "per-operation") {
        backend_options.cuda_allocation = k3x::CudaAllocationMode::per_operation;
    } else if (cuda_allocation_name == "reused") {
        backend_options.cuda_allocation = k3x::CudaAllocationMode::reused;
    } else {
        std::cerr << "unknown CUDA allocation mode: " << cuda_allocation_name << '\n';
        return 2;
    }
    if (cuda_weights_name == "transient") {
        backend_options.cuda_weights = k3x::CudaWeightMode::transient;
    } else if (cuda_weights_name == "resident") {
        backend_options.cuda_weights = k3x::CudaWeightMode::resident;
    } else {
        std::cerr << "unknown CUDA weight mode: " << cuda_weights_name << '\n';
        return 2;
    }
    if (cuda_batching_name == "scalar") {
        backend_options.cuda_batching = k3x::CudaBatchingMode::scalar;
    } else if (cuda_batching_name == "grouped") {
        backend_options.cuda_batching = k3x::CudaBatchingMode::grouped;
    } else if (cuda_batching_name == "resident-grid") {
        backend_options.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    } else {
        std::cerr << "unknown CUDA batching mode: " << cuda_batching_name << '\n';
        return 2;
    }
    if (cuda_boundary_name == "operation") {
        backend_options.cuda_boundary = k3x::CudaBoundaryMode::operation;
    } else if (cuda_boundary_name == "ffn-block") {
        backend_options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    } else if (cuda_boundary_name == "moe-layer") {
        backend_options.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    } else {
        std::cerr << "unknown CUDA boundary mode: " << cuda_boundary_name << '\n';
        return 2;
    }
    if (cuda_transfer_name == "synchronous") {
        backend_options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    } else if (cuda_transfer_name == "prefetch") {
        backend_options.cuda_transfer = k3x::CudaTransferMode::prefetch;
    } else {
        std::cerr << "unknown CUDA transfer mode: " << cuda_transfer_name << '\n';
        return 2;
    }
    if (cuda_moe_fusion_name == "none") {
        backend_options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    } else if (cuda_moe_fusion_name == "routed-accumulate") {
        backend_options.cuda_moe_fusion =
            k3x::CudaMoeFusionMode::routed_accumulate;
    } else {
        std::cerr << "unknown CUDA MoE fusion mode: "
                  << cuda_moe_fusion_name << '\n';
        return 2;
    }
    if (cuda_weight_validation_name == "per-call") {
        backend_options.cuda_weight_validation =
            k3x::CudaWeightValidationMode::per_call;
    } else if (cuda_weight_validation_name == "admission") {
        backend_options.cuda_weight_validation =
            k3x::CudaWeightValidationMode::admission;
    } else {
        std::cerr << "unknown CUDA weight validation mode: "
                  << cuda_weight_validation_name << '\n';
        return 2;
    }
    const auto* resident_begin = cuda_resident_bytes_text.data();
    const auto* resident_end = resident_begin + cuda_resident_bytes_text.size();
    const auto resident_parse = std::from_chars(
        resident_begin, resident_end, backend_options.cuda_resident_bytes);
    if (cuda_resident_bytes_text.empty() || resident_parse.ec != std::errc{} ||
        resident_parse.ptr != resident_end) {
        std::cerr << "invalid CUDA resident byte capacity: "
                  << cuda_resident_bytes_text << '\n';
        return 2;
    }
    if (backend_options.cuda_batching ==
            k3x::CudaBatchingMode::resident_grid &&
        (backend_options.kind != k3x::BackendKind::cuda_custom ||
         backend_options.cuda_allocation != k3x::CudaAllocationMode::reused ||
         backend_options.cuda_weights != k3x::CudaWeightMode::resident ||
         backend_options.cuda_resident_bytes == 0 ||
         (backend_options.cuda_boundary != k3x::CudaBoundaryMode::ffn_block &&
          backend_options.cuda_boundary != k3x::CudaBoundaryMode::moe_layer) ||
         backend_options.cuda_transfer !=
             k3x::CudaTransferMode::synchronous ||
         backend_options.cuda_moe_fusion != k3x::CudaMoeFusionMode::none)) {
        std::cerr << "resident-grid batching requires resident cuda-custom ffn-block or moe-layer reused synchronous execution with fusion none\n";
        return 2;
    }
    const auto* pinned_begin = cuda_pinned_bytes_text.data();
    const auto* pinned_end = pinned_begin + cuda_pinned_bytes_text.size();
    const auto pinned_parse = std::from_chars(
        pinned_begin, pinned_end, backend_options.cuda_pinned_bytes);
    if (cuda_pinned_bytes_text.empty() || pinned_parse.ec != std::errc{} ||
        pinned_parse.ptr != pinned_end) {
        std::cerr << "invalid CUDA pinned byte capacity: "
                  << cuda_pinned_bytes_text << '\n';
        return 2;
    }
    if (backend_options.cuda_weight_validation ==
            k3x::CudaWeightValidationMode::admission &&
        (backend_options.kind != k3x::BackendKind::cuda_custom ||
         backend_options.dense_precision != k3x::DensePrecision::fp32 ||
         backend_options.cuda_allocation != k3x::CudaAllocationMode::reused ||
         backend_options.cuda_weights != k3x::CudaWeightMode::resident ||
         backend_options.cuda_batching !=
             k3x::CudaBatchingMode::resident_grid ||
         backend_options.cuda_boundary != k3x::CudaBoundaryMode::moe_layer ||
         backend_options.cuda_transfer !=
             k3x::CudaTransferMode::synchronous ||
         backend_options.cuda_moe_fusion != k3x::CudaMoeFusionMode::none ||
         backend_options.cuda_resident_bytes == 0)) {
        std::cerr << "admission CUDA weight validation requires resident "
                     "cuda-custom moe-layer execution\n";
        return 2;
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        backend_options.dense_precision != k3x::DensePrecision::fp32) {
        std::cerr << "bf16 dense precision requires a CUDA backend\n";
        return 2;
    }
    if (backend_options.cuda_boundary == k3x::CudaBoundaryMode::ffn_block &&
        backend_options.kind != k3x::BackendKind::cuda_custom) {
        std::cerr << "ffn-block boundary requires cuda-custom\n";
        return 2;
    }
    if (backend_options.cuda_boundary == k3x::CudaBoundaryMode::moe_layer &&
        (backend_options.kind != k3x::BackendKind::cuda_custom ||
         backend_options.dense_precision != k3x::DensePrecision::fp32 ||
         backend_options.cuda_allocation != k3x::CudaAllocationMode::reused ||
         backend_options.cuda_weights != k3x::CudaWeightMode::resident ||
         backend_options.cuda_batching !=
             k3x::CudaBatchingMode::resident_grid ||
         backend_options.cuda_resident_bytes == 0 ||
         backend_options.cuda_transfer !=
             k3x::CudaTransferMode::synchronous ||
         backend_options.cuda_moe_fusion != k3x::CudaMoeFusionMode::none)) {
        std::cerr << "moe-layer boundary requires FP32 resident cuda-custom resident-grid reused synchronous execution with fusion none\n";
        return 2;
    }
    if (backend_options.cuda_moe_fusion ==
            k3x::CudaMoeFusionMode::routed_accumulate) {
        if (backend_options.kind != k3x::BackendKind::cuda_custom) {
            std::cerr << "routed-accumulate fusion requires cuda-custom\n";
            return 2;
        }
        if (backend_options.cuda_boundary !=
            k3x::CudaBoundaryMode::ffn_block) {
            std::cerr <<
                "routed-accumulate fusion requires ffn-block boundary\n";
            return 2;
        }
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        (backend_options.cuda_allocation != k3x::CudaAllocationMode::per_operation ||
         backend_options.cuda_weights != k3x::CudaWeightMode::transient ||
         backend_options.cuda_batching != k3x::CudaBatchingMode::scalar ||
         backend_options.cuda_transfer != k3x::CudaTransferMode::synchronous ||
         backend_options.cuda_resident_bytes != 0 ||
         backend_options.cuda_pinned_bytes != 0)) {
        std::cerr << "CUDA execution options require a CUDA backend\n";
        return 2;
    }
    if (backend_options.cuda_transfer == k3x::CudaTransferMode::synchronous &&
        backend_options.cuda_pinned_bytes != 0) {
        std::cerr << "synchronous CUDA transfer requires a zero pinned byte capacity\n";
        return 2;
    }
    if (backend_options.cuda_transfer == k3x::CudaTransferMode::prefetch) {
        if (backend_options.cuda_pinned_bytes == 0) {
            std::cerr << "prefetch CUDA transfer requires a positive pinned byte capacity\n";
            return 2;
        }
        if (backend_options.kind != k3x::BackendKind::cuda_custom) {
            std::cerr << "prefetch CUDA transfer requires cuda-custom\n";
            return 2;
        }
        if (backend_options.cuda_boundary != k3x::CudaBoundaryMode::ffn_block) {
            std::cerr << "prefetch CUDA transfer requires ffn-block boundary\n";
            return 2;
        }
        if (backend_options.cuda_allocation != k3x::CudaAllocationMode::reused) {
            std::cerr << "prefetch CUDA transfer requires reused allocation\n";
            return 2;
        }
        if (backend_options.cuda_weights != k3x::CudaWeightMode::transient) {
            std::cerr << "prefetch CUDA transfer requires transient weights\n";
            return 2;
        }
    }
    if (backend_options.kind != k3x::BackendKind::cpu &&
        backend_options.cuda_weights == k3x::CudaWeightMode::resident &&
        backend_options.cuda_resident_bytes == 0) {
        std::cerr << "resident CUDA weights require a positive resident byte capacity\n";
        return 2;
    }
    if (backend_options.kind != k3x::BackendKind::cpu &&
        backend_options.cuda_weights == k3x::CudaWeightMode::transient &&
        backend_options.cuda_resident_bytes != 0) {
        std::cerr << "transient CUDA weights require a zero resident byte capacity\n";
        return 2;
    }
    if (runtime_options.speculative_verification ==
        k3x::SpeculativeVerificationMode::expert_major) {
        if (backend_options.kind != k3x::BackendKind::cpu &&
            backend_options.kind != k3x::BackendKind::cuda_custom) {
            std::cerr << "expert-major verification requires CPU or cuda-custom backend\n";
            return 2;
        }
        if (backend_options.kind == k3x::BackendKind::cuda_custom) {
            if (backend_options.cuda_boundary !=
                k3x::CudaBoundaryMode::ffn_block) {
                std::cerr << "CUDA expert-major verification requires ffn-block boundary\n";
                return 2;
            }
            if (backend_options.cuda_allocation !=
                k3x::CudaAllocationMode::reused) {
                std::cerr << "CUDA expert-major verification requires reused allocation\n";
                return 2;
            }
            if (backend_options.cuda_weights !=
                k3x::CudaWeightMode::transient) {
                std::cerr << "CUDA expert-major verification requires transient weights\n";
                return 2;
            }
            if (backend_options.cuda_transfer !=
                k3x::CudaTransferMode::synchronous) {
                std::cerr << "CUDA expert-major verification requires synchronous transfer\n";
                return 2;
            }
            if (backend_options.cuda_moe_fusion !=
                k3x::CudaMoeFusionMode::none) {
                std::cerr << "CUDA expert-major verification requires CUDA MoE fusion none\n";
                return 2;
            }
        }
        if (runtime_options.l1_expert_cache !=
            k3x::L1ExpertCacheMode::disabled) {
            std::cerr << "expert-major verification requires disabled L1 expert cache\n";
            return 2;
        }
        if (runtime_options.l2_expert_schedule !=
            k3x::L2ExpertScheduleMode::blocking) {
            std::cerr << "expert-major verification requires blocking L2 scheduling\n";
            return 2;
        }
        if (runtime_options.routing_policy.mode != k3x::RoutingMode::natural) {
            std::cerr << "expert-major verification requires natural routing\n";
            return 2;
        }
        if (runtime_options.profile_observation) {
            std::cerr << "expert-major verification does not support runtime profiles\n";
            return 2;
        }
    }

    k3x::Profiler profiler;
    std::unique_ptr<k3x::ComputeBackend> backend;
    if (backend_options.kind == k3x::BackendKind::cpu) {
        backend = k3x::make_cpu_backend(&profiler);
    } else {
        auto cuda_backend = k3x::make_cuda_backend(backend_options, &profiler);
        if (!cuda_backend) {
            write_error(cuda_backend.error(), cuda_backend.message());
            return 4;
        }
        backend = std::move(cuda_backend.value());
    }

    std::vector<std::uint32_t> prompt;
    std::stringstream parser(prompt_text);
    std::string item;
    while (std::getline(parser, item, ',')) prompt.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    k3x::RuntimeProfile runtime_profile;
    std::uint64_t runtime_profile_load_bytes = 0;
    std::uint64_t runtime_profile_load_nanoseconds = 0;
    if (!runtime_profile_in.empty()) {
        const auto load_start = std::chrono::steady_clock::now();
        auto loaded_profile = k3x::RuntimeProfile::load(runtime_profile_in);
        runtime_profile_load_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - load_start).count();
        if (!loaded_profile) {
            write_error(loaded_profile.error(), loaded_profile.message());
            return 4;
        }
        std::error_code size_error;
        runtime_profile_load_bytes =
            std::filesystem::file_size(runtime_profile_in, size_error);
        if (size_error) {
            write_error(k3x::ErrorCode::io_error,
                        "cannot stat loaded runtime profile");
            return 4;
        }
        runtime_profile = std::move(loaded_profile.value());
    }
    if (!runtime_metadata_text.empty()) {
        std::stringstream metadata_parser(runtime_metadata_text);
        std::string record;
        std::set<std::string> metadata_keys;
        while (std::getline(metadata_parser, record, ',')) {
            const auto separator = record.find('=');
            if (separator == std::string::npos || separator == 0 ||
                separator + 1 == record.size() ||
                record.find('=', separator + 1) != std::string::npos) {
                std::cerr << "invalid runtime metadata: " << record << '\n';
                return 2;
            }
            const auto key = record.substr(0, separator);
            if (!metadata_keys.insert(key).second) {
                std::cerr << "invalid runtime metadata: duplicate key "
                          << key << '\n';
                return 2;
            }
            auto inserted = runtime_profile.set_metadata(
                key, record.substr(separator + 1));
            if (!inserted) {
                write_error(inserted.error(), inserted.message());
                return 2;
            }
        }
    }
    auto reader = k3x::Reader::open(model_path, reader_options);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 3;
    }
    if (aurora_mode &&
        aurora_draft_k >= model_natural_top_k(reader.value().model_config())) {
        std::cerr << "AURORA draft K must be below checkpoint natural Top-K\n";
        return 2;
    }
    k3x::RuntimeSession session(runtime_options, std::move(runtime_profile));
    std::unique_ptr<ScriptedDraftProvider> scripted_provider;
    std::optional<k3x::Reader> aurora_reader;
    k3x::Profiler aurora_profiler;
    std::unique_ptr<k3x::ComputeBackend> aurora_backend;
    std::unique_ptr<k3x::DraftProvider> aurora_provider;
    k3x::DraftProvider* draft_provider = nullptr;
    if (speculative_mode_name == "scripted-reference") {
        scripted_provider = std::make_unique<ScriptedDraftProvider>(
            std::move(scripted_proposals));
        draft_provider = scripted_provider.get();
    } else if (aurora_mode) {
        auto opened_draft_reader = k3x::Reader::open(model_path, reader_options);
        if (!opened_draft_reader) {
            write_error(opened_draft_reader.error(),
                        opened_draft_reader.message());
            return 3;
        }
        aurora_reader.emplace(std::move(opened_draft_reader.value()));
        if (aurora_draft_backend_name == "cpu") {
            aurora_backend = k3x::make_cpu_backend(&aurora_profiler);
        } else {
            k3x::BackendOptions draft_backend_options;
            draft_backend_options.kind = k3x::BackendKind::cuda_custom;
            draft_backend_options.dense_precision =
                k3x::DensePrecision::fp32;
            draft_backend_options.cuda_allocation =
                k3x::CudaAllocationMode::reused;
            draft_backend_options.cuda_weights =
                aurora_draft_resident_bytes == 0
                    ? k3x::CudaWeightMode::transient
                    : k3x::CudaWeightMode::resident;
            draft_backend_options.cuda_batching =
                aurora_draft_batching_name == "resident-grid"
                    ? k3x::CudaBatchingMode::resident_grid
                    : k3x::CudaBatchingMode::grouped;
            draft_backend_options.cuda_boundary =
                aurora_draft_boundary_name == "moe-layer"
                    ? k3x::CudaBoundaryMode::moe_layer
                    : k3x::CudaBoundaryMode::ffn_block;
            draft_backend_options.cuda_transfer =
                k3x::CudaTransferMode::synchronous;
            draft_backend_options.cuda_moe_fusion =
                k3x::CudaMoeFusionMode::none;
            if (draft_backend_options.cuda_boundary ==
                k3x::CudaBoundaryMode::moe_layer) {
                draft_backend_options.cuda_weight_validation =
                    backend_options.cuda_weight_validation;
            }
            draft_backend_options.cuda_resident_bytes =
                aurora_draft_resident_bytes;
            auto created_backend = k3x::make_cuda_backend(
                draft_backend_options, &aurora_profiler);
            if (!created_backend) {
                write_error(created_backend.error(),
                            created_backend.message());
                return 4;
            }
            aurora_backend = std::move(created_backend.value());
        }
        k3x::RuntimeOptions draft_options;
        draft_options.incremental = true;
        draft_options.routing_policy.mode = k3x::RoutingMode::fixed;
        draft_options.routing_policy.fixed_k = aurora_draft_k;
        const auto block_policy = aurora_block_policy_name == "adaptive"
            ? k3x::AuroraBlockPolicy::adaptive
            : k3x::AuroraBlockPolicy::fixed;
        if (speculative_mode_name == "aurora-persistent") {
            auto created_provider =
                k3x::AuroraPersistentDraftProvider::create(
                    aurora_reader.value(), *aurora_backend, prompt,
                    draft_options,
                    {.scheduler = {
                        .policy = block_policy,
                        .maximum_length = speculative_block_size}});
            if (!created_provider) {
                write_error(created_provider.error(),
                            created_provider.message());
                return 4;
            }
            aurora_provider = std::move(created_provider.value());
        } else {
            auto created_provider = k3x::AuroraReplayDraftProvider::create(
                aurora_reader.value(), *aurora_backend, prompt,
                draft_options,
                {.scheduler = {
                    .policy = block_policy,
                    .maximum_length = speculative_block_size}});
            if (!created_provider) {
                write_error(created_provider.error(),
                            created_provider.message());
                return 4;
            }
            aurora_provider = std::move(created_provider.value());
        }
        draft_provider = aurora_provider.get();
    }
    const auto process_io_before = process_io_snapshot();
    auto result = draft_provider
        ? k3x::generate_speculative(
              reader.value(), *backend, prompt, count, session,
              *draft_provider, speculative_block_size)
        : k3x::generate_greedy(
              reader.value(), *backend, prompt, count, session);
    const auto process_io = process_io_delta(
        process_io_before, process_io_snapshot());
    if (!result) {
        write_error(result.error(), result.message());
        return 4;
    }
    if (scripted_provider && !scripted_provider->empty()) {
        write_error(k3x::ErrorCode::invalid_state,
                    "unused scripted draft proposals");
        return 4;
    }
    std::uint64_t runtime_profile_save_bytes = 0;
    std::uint64_t runtime_profile_save_nanoseconds = 0;
    if (!runtime_profile_out.empty()) {
        const auto save_start = std::chrono::steady_clock::now();
        auto saved_profile = session.profile().save(runtime_profile_out);
        runtime_profile_save_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - save_start).count();
        if (!saved_profile) {
            write_error(saved_profile.error(), saved_profile.message());
            return 4;
        }
        std::error_code size_error;
        runtime_profile_save_bytes =
            std::filesystem::file_size(runtime_profile_out, size_error);
        if (size_error) {
            write_error(k3x::ErrorCode::io_error,
                        "cannot stat saved runtime profile");
            return 4;
        }
    }
    std::ofstream output(output_path);
    if (!output) return 5;
    const auto profile = profiler.summary();
    const auto memory = backend->memory_stats();
    const auto runtime = backend->runtime_stats();
    const auto& effective_options = backend->options();
    const auto draft_profile = aurora_profiler.summary();
    const auto draft_memory = aurora_backend
        ? aurora_backend->memory_stats()
        : k3x::BackendMemoryStats{};
    const auto draft_runtime = aurora_backend
        ? aurora_backend->runtime_stats()
        : k3x::BackendRuntimeStats{};
    const k3x::BackendOptions default_draft_options;
    const auto& effective_draft_options = aurora_backend
        ? aurora_backend->options()
        : default_draft_options;
    const auto draft_allocation_name =
        effective_draft_options.cuda_allocation ==
                k3x::CudaAllocationMode::reused
            ? "reused"
            : "per-operation";
    const auto draft_weights_name =
        effective_draft_options.cuda_weights == k3x::CudaWeightMode::resident
            ? "resident"
            : "transient";
    const auto draft_batching_name =
        effective_draft_options.cuda_batching ==
                k3x::CudaBatchingMode::resident_grid
            ? "resident-grid"
            : effective_draft_options.cuda_batching ==
                    k3x::CudaBatchingMode::grouped
                ? "grouped"
                : "scalar";
    const auto draft_boundary_name =
        effective_draft_options.cuda_boundary ==
                k3x::CudaBoundaryMode::moe_layer
            ? "moe-layer"
            : effective_draft_options.cuda_boundary ==
                      k3x::CudaBoundaryMode::ffn_block
            ? "ffn-block"
            : "operation";
    const auto draft_transfer_name =
        effective_draft_options.cuda_transfer ==
                k3x::CudaTransferMode::prefetch
            ? "prefetch"
            : "synchronous";
    const auto draft_moe_fusion_name =
        effective_draft_options.cuda_moe_fusion ==
                k3x::CudaMoeFusionMode::routed_accumulate
            ? "routed-accumulate"
            : "none";
    const auto draft_weight_validation_name =
        effective_draft_options.cuda_weight_validation ==
                k3x::CudaWeightValidationMode::admission
            ? "admission"
            : "per-call";
    output << std::setprecision(9);
    output << "{\"backend\":";
    write_json_string(output, backend_name);
    output << ",\"device\":";
    write_json_string(output, backend->device_name());
    output << ",\"dense_precision\":";
    write_json_string(output, dense_precision_name);
    output << ",\"cuda_allocation\":";
    write_json_string(output, cuda_allocation_name);
    output << ",\"cuda_weights\":";
    write_json_string(output, cuda_weights_name);
    output << ",\"cuda_batching\":";
    write_json_string(output, cuda_batching_name);
    output << ",\"cuda_boundary\":";
    write_json_string(output, cuda_boundary_name);
    output << ",\"cuda_transfer\":";
    write_json_string(output, cuda_transfer_name);
    output << ",\"cuda_moe_fusion\":";
    write_json_string(output, cuda_moe_fusion_name);
    output << ",\"cuda_weight_validation\":";
    write_json_string(output, cuda_weight_validation_name);
    output << ",\"cuda_resident_bytes\":"
           << effective_options.cuda_resident_bytes;
    output << ",\"cuda_pinned_bytes\":"
           << effective_options.cuda_pinned_bytes;
    output << ",\"l1_expert_cache_mode\":";
    write_json_string(output, l1_expert_cache_name);
    output << ",\"l1_expert_cache_bytes\":"
           << runtime_options.l1_expert_cache_bytes;
    output << ",\"l1_expert_cache_hits\":"
           << result.value().l1_expert_cache.hits
           << ",\"l1_expert_cache_misses\":"
           << result.value().l1_expert_cache.misses
           << ",\"l1_expert_cache_bypasses\":"
           << result.value().l1_expert_cache.bypasses
           << ",\"l1_expert_cache_evictions\":"
           << result.value().l1_expert_cache.evictions
           << ",\"l1_expert_cache_collision_misses\":"
           << result.value().l1_expert_cache.collision_misses
           << ",\"l1_expert_cache_resident_bytes\":"
           << result.value().l1_expert_cache.resident_bytes
           << ",\"peak_l1_expert_cache_resident_bytes\":"
           << result.value().l1_expert_cache.peak_resident_bytes;
    output << ",\"routing_mode\":";
    write_json_string(output, routing_mode_name);
    output << ",\"speculative_mode\":";
    write_json_string(output, speculative_mode_name);
    output << ",\"speculative_verification\":";
    write_json_string(output, speculative_verification_name);
    output << ",\"speculative_block_size\":" << speculative_block_size
           << ",\"aurora_draft_k\":"
           << (aurora_mode
                   ? aurora_draft_k
                   : 0);
    output << ",\"aurora_block_policy\":";
    write_json_string(
        output, aurora_mode
                    ? aurora_block_policy_name
                    : "none");
    output << ",\"aurora_draft_backend\":";
    write_json_string(
        output, aurora_mode ? aurora_draft_backend_name : "none");
    output << ",\"draft_device\":";
    write_json_string(
        output, aurora_backend ? aurora_backend->device_name() : "CPU");
    output << ",\"draft_cuda_allocation\":";
    write_json_string(output, draft_allocation_name);
    output << ",\"draft_cuda_weights\":";
    write_json_string(output, draft_weights_name);
    output << ",\"draft_cuda_batching\":";
    write_json_string(output, draft_batching_name);
    output << ",\"draft_cuda_boundary\":";
    write_json_string(output, draft_boundary_name);
    output << ",\"draft_cuda_transfer\":";
    write_json_string(output, draft_transfer_name);
    output << ",\"draft_cuda_moe_fusion\":";
    write_json_string(output, draft_moe_fusion_name);
    output << ",\"draft_cuda_weight_validation\":";
    write_json_string(output, draft_weight_validation_name);
    output << ",\"draft_kernel_nanoseconds\":"
           << draft_profile.device_nanoseconds
           << ",\"draft_host_to_device_bytes\":"
           << draft_profile.host_to_device_bytes
           << ",\"draft_weight_h2d_bytes\":"
           << draft_profile.weight_host_to_device_bytes
           << ",\"draft_activation_h2d_bytes\":"
           << draft_profile.activation_host_to_device_bytes
           << ",\"draft_device_to_host_bytes\":"
           << draft_profile.device_to_host_bytes
           << ",\"draft_peak_vram_bytes\":"
           << draft_memory.peak_device_bytes
           << ",\"draft_device_allocation_count\":"
           << draft_runtime.device_allocation_count
           << ",\"draft_stream_synchronization_count\":"
           << draft_runtime.stream_synchronization_count
           << ",\"draft_weight_cache_hits\":"
           << draft_runtime.weight_cache_hits
           << ",\"draft_weight_cache_misses\":"
           << draft_runtime.weight_cache_misses
           << ",\"draft_weight_cache_bypasses\":"
           << draft_runtime.weight_cache_bypasses
           << ",\"draft_cuda_resident_bytes\":"
           << effective_draft_options.cuda_resident_bytes
           << ",\"draft_resident_weight_bytes\":"
           << draft_runtime.resident_weight_bytes
           << ",\"draft_peak_resident_weight_bytes\":"
           << draft_runtime.peak_resident_weight_bytes
           << ",\"draft_immutable_validation_scans\":"
           << draft_runtime.immutable_validation_scans
           << ",\"draft_immutable_validation_hits\":"
           << draft_runtime.immutable_validation_hits
           << ",\"draft_immutable_validation_bytes\":"
           << draft_runtime.immutable_validation_bytes
           << ",\"draft_immutable_validation_nanoseconds\":"
           << draft_runtime.immutable_validation_nanoseconds
           << ",\"draft_resident_grid_calls\":"
           << draft_runtime.resident_grid_calls
           << ",\"draft_resident_grid_experts\":"
           << draft_runtime.resident_grid_experts
           << ",\"draft_resident_grid_tokens\":"
           << draft_runtime.resident_grid_tokens
           << ",\"draft_resident_grid_expert_tokens\":"
           << draft_runtime.resident_grid_expert_tokens
           << ",\"draft_resident_grid_kernel_launches\":"
           << draft_runtime.resident_grid_kernel_launches
           << ",\"draft_resident_grid_fallbacks\":"
           << draft_runtime.resident_grid_fallbacks
           << ",\"draft_resident_grid_descriptor_h2d_bytes\":"
           << draft_runtime.resident_grid_descriptor_h2d_bytes
           << ",\"draft_resident_moe_layer_calls\":"
           << draft_runtime.resident_moe_layer_calls
           << ",\"draft_resident_moe_layer_experts\":"
           << draft_runtime.resident_moe_layer_experts
           << ",\"draft_resident_moe_layer_kernel_launches\":"
           << draft_runtime.resident_moe_layer_kernel_launches
           << ",\"draft_resident_moe_layer_fallbacks\":"
           << draft_runtime.resident_moe_layer_fallbacks
           << ",\"draft_resident_moe_layer_contribution_h2d_bytes\":"
           << draft_runtime.resident_moe_layer_contribution_h2d_bytes
           << ",\"draft_ffn_block_calls\":"
           << draft_runtime.ffn_block_calls
           << ",\"draft_ffn_block_experts\":"
           << draft_runtime.ffn_block_experts;
    output << ",\"draft_proposal_calls\":"
           << result.value().draft_proposal_calls
           << ",\"draft_candidate_tokens\":"
           << result.value().draft_candidate_tokens
           << ",\"draft_replayed_context_tokens\":"
           << result.value().draft_replayed_context_tokens
           << ",\"draft_generation_nanoseconds\":"
           << result.value().draft_generation_nanoseconds
           << ",\"draft_reader_read_calls\":"
           << result.value().draft_reader_calls
           << ",\"draft_reader_completed_bytes\":"
           << result.value().draft_reader_bytes
           << ",\"draft_routing_decisions\":"
           << result.value().draft_routing_decisions
           << ",\"draft_routing_selected_experts\":"
           << result.value().draft_routing_selected_experts
           << ",\"draft_selected_length_1\":"
           << result.value().draft_selected_length_1
           << ",\"draft_selected_length_2\":"
           << result.value().draft_selected_length_2
           << ",\"draft_selected_length_4\":"
           << result.value().draft_selected_length_4
           << ",\"draft_scheduler_growths\":"
           << result.value().draft_scheduler_growths
           << ",\"draft_scheduler_backoffs\":"
           << result.value().draft_scheduler_backoffs
           << ",\"draft_context_prefill_tokens\":"
           << result.value().draft_context_prefill_tokens
           << ",\"draft_incremental_forward_calls\":"
           << result.value().draft_incremental_forward_calls
           << ",\"draft_rollback_events\":"
           << result.value().draft_rollback_events
           << ",\"draft_mla_positions_cropped\":"
           << result.value().draft_mla_positions_cropped
           << ",\"draft_kda_checkpoint_bytes\":"
           << result.value().draft_kda_checkpoint_bytes
           << ",\"speculative_verification_blocks\":"
           << result.value().speculative_verification_blocks
           << ",\"speculative_proposed_draft_tokens\":"
           << result.value().speculative_proposed_draft_tokens
           << ",\"speculative_accepted_draft_tokens\":"
           << result.value().speculative_accepted_draft_tokens
           << ",\"speculative_committed_tokens\":"
           << result.value().speculative_committed_tokens
           << ",\"speculative_max_proposal_tokens\":"
           << result.value().speculative_max_proposal_tokens
           << ",\"target_decode_forward_calls\":"
           << result.value().target_decode_forward_calls
           << ",\"target_block_forward_calls\":"
           << result.value().target_block_forward_calls
           << ",\"target_positions_evaluated\":"
           << result.value().target_positions_evaluated
           << ",\"target_positions_discarded\":"
           << result.value().target_positions_discarded
           << ",\"expert_major_unique_experts_sum\":"
           << result.value().expert_major_unique_experts_sum
           << ",\"expert_major_unique_experts_max\":"
           << result.value().expert_major_unique_experts_max
           << ",\"expert_major_assignments\":"
           << result.value().expert_major_assignments
           << ",\"expert_major_reused_assignments\":"
           << result.value().expert_major_reused_assignments
           << ",\"expert_major_payload_loads\":"
           << result.value().expert_major_payload_loads
           << ",\"speculative_acceptance_rate\":";
    if (result.value().speculative_proposed_draft_tokens == 0) {
        output << "null";
    } else {
        output << static_cast<double>(
                      result.value().speculative_accepted_draft_tokens) /
                      static_cast<double>(
                          result.value().speculative_proposed_draft_tokens);
    }
    const auto routing_decisions = result.value().routing_decisions;
    const auto routing_average = [&](double sum) {
        return routing_decisions ? sum / static_cast<double>(routing_decisions)
                                 : 0.0;
    };
    output << ",\"routing_natural_top_k\":"
           << result.value().routing_natural_top_k
           << ",\"routing_fixed_k\":"
           << runtime_options.routing_policy.fixed_k
           << ",\"routing_mass_target\":"
           << runtime_options.routing_policy.mass_target
           << ",\"routing_min_boundary_gap\":"
           << runtime_options.routing_policy.minimum_boundary_gap
           << ",\"routing_quality_floor_k\":"
           << runtime_options.routing_policy.quality_floor_k
           << ",\"routing_agent_failures\":" << routing_agent_failures
           << ",\"routing_critical\":"
           << (routing_critical_text == "true" ? "true" : "false")
           << ",\"routing_decisions\":" << routing_decisions
           << ",\"routing_selected_experts\":"
           << result.value().routing_selected_experts
           << ",\"routing_average_top_k\":"
           << routing_average(result.value().routing_selected_experts)
           << ",\"routing_average_normalized_entropy\":"
           << routing_average(result.value().routing_normalized_entropy_sum)
           << ",\"routing_average_selected_mass\":"
           << routing_average(result.value().routing_selected_mass_sum)
           << ",\"routing_average_boundary_confidence\":"
           << routing_average(result.value().routing_boundary_confidence_sum)
           << ",\"routing_quality_escalated_decisions\":"
           << result.value().routing_quality_escalated_decisions
           << ",\"cold_rescue_count\":"
           << result.value().cold_rescue_count;
    output << ",\"runtime_profile_metadata_count\":"
           << session.profile().metadata().size()
           << ",\"runtime_profile_prior_weight\":"
           << session.profile().prior_weight(
                  runtime_options.profile_prior_strength)
           << ",\"runtime_profile_live_observations\":"
           << session.profile().live_route_observations()
           << ",\"runtime_profile_load_bytes\":"
           << runtime_profile_load_bytes
           << ",\"runtime_profile_save_bytes\":"
           << runtime_profile_save_bytes
           << ",\"runtime_profile_load_nanoseconds\":"
           << runtime_profile_load_nanoseconds
           << ",\"runtime_profile_save_nanoseconds\":"
           << runtime_profile_save_nanoseconds;
    output << ",\"l2_io_engine\":";
    write_json_string(output, l2_io_name);
    output << ",\"l2_cache_mode\":";
    write_json_string(output, l2_cache_name);
    output << ",\"l2_queue_depth\":" << reader_options.queue_depth
           << ",\"l2_direct_memory_alignment\":"
           << reader.value().direct_memory_alignment()
           << ",\"l2_direct_offset_alignment\":"
           << reader.value().direct_offset_alignment();
    output << ",\"l2_expert_schedule\":";
    write_json_string(output, l2_schedule_name);
    const auto& expert_load = result.value().expert_load_scheduler;
    output << ",\"expert_load_submissions\":"
           << expert_load.submissions
           << ",\"expert_load_inline_resident_hits\":"
           << expert_load.inline_resident_hits
           << ",\"expert_load_completions\":"
           << expert_load.completions
           << ",\"expert_load_ready_before_use\":"
           << expert_load.ready_before_use
           << ",\"expert_load_late_at_use\":"
           << expert_load.late_at_use
           << ",\"expert_load_estimated_deadline_misses\":"
           << expert_load.estimated_deadline_misses
           << ",\"expert_load_requested_bytes\":"
           << expert_load.requested_bytes
           << ",\"expert_load_queue_high_water\":"
           << expert_load.queue_high_water
           << ",\"expert_load_worker_nanoseconds\":"
           << expert_load.worker_nanoseconds
           << ",\"expert_load_exposed_wait_nanoseconds\":"
           << expert_load.exposed_wait_nanoseconds;
    output << ",\"kernel_nanoseconds\":" << profile.device_nanoseconds
           << ",\"host_to_device_bytes\":" << profile.host_to_device_bytes
           << ",\"weight_h2d_bytes\":"
           << profile.weight_host_to_device_bytes
           << ",\"activation_h2d_bytes\":"
           << profile.activation_host_to_device_bytes
           << ",\"device_to_host_bytes\":" << profile.device_to_host_bytes
           << ",\"peak_vram_bytes\":" << memory.peak_device_bytes
           << ",\"device_allocation_count\":" << runtime.device_allocation_count
           << ",\"device_free_count\":" << runtime.device_free_count
           << ",\"stream_synchronization_count\":"
           << runtime.stream_synchronization_count
           << ",\"weight_cache_hits\":" << runtime.weight_cache_hits
           << ",\"weight_cache_misses\":" << runtime.weight_cache_misses
           << ",\"weight_cache_bypasses\":" << runtime.weight_cache_bypasses
           << ",\"resident_weight_bytes\":" << runtime.resident_weight_bytes
           << ",\"peak_resident_weight_bytes\":"
           << runtime.peak_resident_weight_bytes
           << ",\"immutable_validation_scans\":"
           << runtime.immutable_validation_scans
           << ",\"immutable_validation_hits\":"
           << runtime.immutable_validation_hits
           << ",\"immutable_validation_bytes\":"
           << runtime.immutable_validation_bytes
           << ",\"immutable_validation_nanoseconds\":"
           << runtime.immutable_validation_nanoseconds
           << ",\"scratch_bytes\":" << runtime.scratch_bytes
           << ",\"peak_scratch_bytes\":" << runtime.peak_scratch_bytes
           << ",\"grouped_projection_calls\":"
           << runtime.grouped_projection_calls
           << ",\"grouped_projection_members\":"
           << runtime.grouped_projection_members
           << ",\"resident_grid_calls\":" << runtime.resident_grid_calls
           << ",\"resident_grid_experts\":" << runtime.resident_grid_experts
           << ",\"resident_grid_tokens\":" << runtime.resident_grid_tokens
           << ",\"resident_grid_expert_tokens\":"
           << runtime.resident_grid_expert_tokens
           << ",\"resident_grid_kernel_launches\":"
           << runtime.resident_grid_kernel_launches
           << ",\"resident_grid_fallbacks\":"
           << runtime.resident_grid_fallbacks
           << ",\"resident_grid_descriptor_h2d_bytes\":"
           << runtime.resident_grid_descriptor_h2d_bytes
           << ",\"resident_moe_layer_calls\":"
           << runtime.resident_moe_layer_calls
           << ",\"resident_moe_layer_experts\":"
           << runtime.resident_moe_layer_experts
           << ",\"resident_moe_layer_kernel_launches\":"
           << runtime.resident_moe_layer_kernel_launches
           << ",\"resident_moe_layer_fallbacks\":"
           << runtime.resident_moe_layer_fallbacks
           << ",\"resident_moe_layer_contribution_h2d_bytes\":"
           << runtime.resident_moe_layer_contribution_h2d_bytes
           << ",\"ffn_block_calls\":" << runtime.ffn_block_calls
           << ",\"ffn_block_experts\":" << runtime.ffn_block_experts
           << ",\"batched_expert_ffn_calls\":"
           << runtime.batched_expert_ffn_calls
           << ",\"batched_expert_ffn_tokens\":"
           << runtime.batched_expert_ffn_tokens
           << ",\"fused_moe_calls\":" << runtime.fused_moe_calls
           << ",\"fused_moe_experts\":" << runtime.fused_moe_experts
           << ",\"pinned_host_bytes\":" << runtime.pinned_host_bytes
           << ",\"peak_pinned_host_bytes\":"
           << runtime.peak_pinned_host_bytes
           << ",\"async_prefetch_calls\":" << runtime.async_prefetch_calls
           << ",\"async_prefetch_bytes\":" << runtime.async_prefetch_bytes
           << ",\"async_prefetch_ready_before_use\":"
           << runtime.async_prefetch_ready_before_use
           << ",\"async_prefetch_late_at_use\":"
           << runtime.async_prefetch_late_at_use
           << ",\"transfer_stream_wait_count\":"
           << runtime.transfer_stream_wait_count
           << ",\"pinned_staging_nanoseconds\":"
           << runtime.pinned_staging_nanoseconds
           << ",\"transfer_device_nanoseconds\":"
           << runtime.transfer_device_nanoseconds
           << ",\"transfer_stall_nanoseconds\":"
           << runtime.transfer_stall_nanoseconds
           << ",\"async_engine_count\":" << runtime.async_engine_count
           << ",\"device_overlap\":"
           << (runtime.device_overlap ? "true" : "false")
           << ",\"profile_wall_nanoseconds\":" << profile.wall_nanoseconds
           << ",\"profile_logical_bytes\":" << profile.logical_bytes
           << ",\"failed_operations\":" << profile.failed_operations
           << ",\"decode_nanoseconds\":" << result.value().decode_nanoseconds
           << ",\"prefill_nanoseconds\":" << result.value().prefill_nanoseconds
           << ",\"read_bytes\":" << reader.value().counters().completed_bytes
           << ",\"read_calls\":" << reader.value().counters().calls
           << ",\"reader_read_calls\":" << reader.value().counters().calls
           << ",\"reader_requested_bytes\":"
           << reader.value().counters().requested_bytes
           << ",\"reader_completed_bytes\":"
           << reader.value().counters().completed_bytes
           << ",\"reader_batch_submissions\":"
           << reader.value().counters().batch_submissions
           << ",\"reader_storage_submitted_bytes\":"
           << reader.value().counters().storage_submitted_bytes
           << ",\"reader_storage_completed_bytes\":"
           << reader.value().counters().storage_completed_bytes
           << ",\"reader_completions\":"
           << reader.value().counters().completions
           << ",\"reader_short_reads\":"
           << reader.value().counters().short_reads
           << ",\"reader_failures\":"
           << reader.value().counters().failures
           << ",\"reader_storage_nanoseconds\":"
           << reader.value().counters().storage_nanoseconds
           << ",\"process_io_available\":"
           << (process_io.available ? "true" : "false")
           << ",\"process_rchar_bytes\":";
    if (process_io.available) output << process_io.rchar;
    else output << "null";
    output << ",\"process_read_bytes\":";
    if (process_io.available) output << process_io.read_bytes;
    else output << "null";
    output
           << ",\"per_layer_nanoseconds\":[";
    for (std::size_t index = 0; index < result.value().per_layer_nanoseconds.size(); ++index) {
        if (index) output << ',';
        output << result.value().per_layer_nanoseconds[index];
    }
    output << "],\"prefill_layer_outputs\":[";
    for (std::size_t layer = 0; layer < result.value().prefill_layer_outputs.size(); ++layer) {
        if (layer) output << ',';
        output << '[';
        for (std::size_t index = 0; index < result.value().prefill_layer_outputs[layer].size(); ++index) {
            if (index) output << ',';
            output << result.value().prefill_layer_outputs[layer][index];
        }
        output << ']';
    }
    output << "],\"prefill_logits\":[";
    for (std::size_t index = 0; index < result.value().prefill_logits.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_logits[index];
    }
    output << "],\"prefill_state\":[";
    for (std::size_t index = 0; index < result.value().prefill_state.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_state[index];
    }
    output << "],\"prefill_routed_experts\":[";
    for (std::size_t index = 0;
         index < result.value().prefill_routed_experts.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_routed_experts[index];
    }
    output << "],\"prefill_routed_k\":[";
    for (std::size_t index = 0;
         index < result.value().prefill_routed_k.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_routed_k[index];
    }
    output << "],\"final_state\":[";
    for (std::size_t index = 0; index < result.value().final_state.size();
         ++index) {
        if (index) output << ',';
        output << result.value().final_state[index];
    }
    output << "],\"routed_experts\":[";
    for (std::size_t index = 0;
         index < result.value().routed_experts.size(); ++index) {
        if (index) output << ',';
        output << result.value().routed_experts[index];
    }
    output << "],\"routed_k\":[";
    for (std::size_t index = 0; index < result.value().routed_k.size();
         ++index) {
        if (index) output << ',';
        output << result.value().routed_k[index];
    }
    output << "],\"evaluated_routed_experts\":[";
    for (std::size_t index = 0;
         index < result.value().evaluated_routed_experts.size(); ++index) {
        if (index) output << ',';
        output << result.value().evaluated_routed_experts[index];
    }
    output << "],\"evaluated_routed_k\":[";
    for (std::size_t index = 0;
         index < result.value().evaluated_routed_k.size(); ++index) {
        if (index) output << ',';
        output << result.value().evaluated_routed_k[index];
    }
    output << "],\"token_ids\":[";
    for (std::size_t index = 0; index < result.value().token_ids.size(); ++index) {
        if (index) output << ',';
        output << result.value().token_ids[index];
    }
    output << "]}\n";
    return 0;
}
