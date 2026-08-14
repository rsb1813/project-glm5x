// CPU와 CUDA persistent AURORA draft proposal 및 commit lifecycle의 동등성을 검증합니다.
#include "k3x/aurora.hpp"

#include <filesystem>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <vector>

namespace {
void require(
    bool condition,
    const std::source_location location = std::source_location::current()) {
    if (!condition) {
        std::cerr << "CUDA AURORA draft requirement failed at line "
                  << location.line() << '\n';
        throw std::runtime_error("CUDA AURORA draft requirement failed");
    }
}

k3x::RuntimeOptions draft_options() {
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.routing_policy.mode = k3x::RoutingMode::fixed;
    options.routing_policy.fixed_k = 4;
    return options;
}

k3x::BackendOptions cuda_options(
    std::uint64_t resident_bytes = 0,
    k3x::CudaBatchingMode batching = k3x::CudaBatchingMode::grouped) {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::fp32;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = resident_bytes == 0
        ? k3x::CudaWeightMode::transient
        : k3x::CudaWeightMode::resident;
    options.cuda_batching = batching;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_resident_bytes = resident_bytes;
    return options;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto artifact = std::filesystem::path(argv[1]);
    auto cpu_reader = k3x::Reader::open(artifact);
    auto cuda_reader = k3x::Reader::open(artifact);
    auto resident_reader = k3x::Reader::open(artifact);
    auto bypass_reader = k3x::Reader::open(artifact);
    auto grid_reader = k3x::Reader::open(artifact);
    auto grid_bypass_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(cpu_reader));
    require(static_cast<bool>(cuda_reader));
    require(static_cast<bool>(resident_reader));
    require(static_cast<bool>(bypass_reader));
    require(static_cast<bool>(grid_reader));
    require(static_cast<bool>(grid_bypass_reader));
    auto cpu_backend = k3x::make_cpu_backend();
    auto created_cuda_backend = k3x::make_cuda_backend(cuda_options());
    auto created_resident_backend =
        k3x::make_cuda_backend(cuda_options(8ULL * 1024ULL * 1024ULL));
    auto created_bypass_backend = k3x::make_cuda_backend(cuda_options(1));
    auto created_grid_backend = k3x::make_cuda_backend(cuda_options(
        8ULL * 1024ULL * 1024ULL, k3x::CudaBatchingMode::resident_grid));
    auto created_grid_bypass_backend = k3x::make_cuda_backend(cuda_options(
        1, k3x::CudaBatchingMode::resident_grid));
    require(static_cast<bool>(created_cuda_backend));
    require(static_cast<bool>(created_resident_backend));
    require(static_cast<bool>(created_bypass_backend));
    require(static_cast<bool>(created_grid_backend));
    require(static_cast<bool>(created_grid_bypass_backend));
    auto cuda_backend = std::move(created_cuda_backend.value());
    auto resident_backend = std::move(created_resident_backend.value());
    auto bypass_backend = std::move(created_bypass_backend.value());
    auto grid_backend = std::move(created_grid_backend.value());
    auto grid_bypass_backend = std::move(created_grid_bypass_backend.value());
    const std::vector<std::uint32_t> prompt{1, 7, 3, 9};
    const k3x::AuroraPersistentConfig config{
        .scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                      .maximum_length = 4}};

    auto cpu_provider = k3x::AuroraPersistentDraftProvider::create(
        cpu_reader.value(), *cpu_backend, prompt, draft_options(), config);
    auto cuda_provider = k3x::AuroraPersistentDraftProvider::create(
        cuda_reader.value(), *cuda_backend, prompt, draft_options(), config);
    auto resident_provider = k3x::AuroraPersistentDraftProvider::create(
        resident_reader.value(), *resident_backend, prompt, draft_options(),
        config);
    auto bypass_provider = k3x::AuroraPersistentDraftProvider::create(
        bypass_reader.value(), *bypass_backend, prompt, draft_options(),
        config);
    auto grid_provider = k3x::AuroraPersistentDraftProvider::create(
        grid_reader.value(), *grid_backend, prompt, draft_options(), config);
    auto grid_bypass_provider = k3x::AuroraPersistentDraftProvider::create(
        grid_bypass_reader.value(), *grid_bypass_backend, prompt,
        draft_options(), config);
    require(static_cast<bool>(cpu_provider));
    require(static_cast<bool>(cuda_provider));
    require(static_cast<bool>(resident_provider));
    require(static_cast<bool>(bypass_provider));
    require(static_cast<bool>(grid_provider));
    require(static_cast<bool>(grid_bypass_provider));

    std::vector<std::uint32_t> history{43};
    auto request = [&](std::size_t count) {
        return k3x::DraftRequest{
            .anchor_token = history.back(),
            .max_draft_tokens = count,
            .generated_position = history.size(),
            .generated_tokens = history,
        };
    };

    auto cpu_first = cpu_provider.value()->propose(request(2));
    auto cuda_first = cuda_provider.value()->propose(request(2));
    auto resident_first = resident_provider.value()->propose(request(2));
    auto bypass_first = bypass_provider.value()->propose(request(2));
    auto grid_first = grid_provider.value()->propose(request(2));
    auto grid_bypass_first = grid_bypass_provider.value()->propose(request(2));
    require(static_cast<bool>(cpu_first));
    require(static_cast<bool>(cuda_first));
    require(static_cast<bool>(resident_first));
    require(static_cast<bool>(bypass_first));
    require(static_cast<bool>(grid_first));
    require(static_cast<bool>(grid_bypass_first));
    require(cuda_first.value().candidate_tokens ==
            cpu_first.value().candidate_tokens);
    require(resident_first.value().candidate_tokens ==
            cpu_first.value().candidate_tokens);
    require(bypass_first.value().candidate_tokens ==
            cpu_first.value().candidate_tokens);
    require(grid_first.value().candidate_tokens ==
            cpu_first.value().candidate_tokens);
    require(grid_bypass_first.value().candidate_tokens ==
            cpu_first.value().candidate_tokens);
    const auto first = cpu_first.value().candidate_tokens;
    const std::vector<std::uint32_t> full_commit{first[0], first[1], 17};
    const k3x::DraftVerification full_verification{
        .anchor_token = history.back(),
        .proposed_draft_tokens = first.size(),
        .accepted_draft_tokens = first.size(),
        .committed_tokens = full_commit,
        .all_draft_tokens_accepted = true,
    };
    cpu_provider.value()->update(full_verification);
    cuda_provider.value()->update(full_verification);
    resident_provider.value()->update(full_verification);
    bypass_provider.value()->update(full_verification);
    grid_provider.value()->update(full_verification);
    grid_bypass_provider.value()->update(full_verification);
    history.insert(history.end(), full_commit.begin(), full_commit.end());

    auto cpu_second = cpu_provider.value()->propose(request(4));
    auto cuda_second = cuda_provider.value()->propose(request(4));
    auto resident_second = resident_provider.value()->propose(request(4));
    auto bypass_second = bypass_provider.value()->propose(request(4));
    auto grid_second = grid_provider.value()->propose(request(4));
    auto grid_bypass_second = grid_bypass_provider.value()->propose(request(4));
    require(static_cast<bool>(cpu_second));
    require(static_cast<bool>(cuda_second));
    require(static_cast<bool>(resident_second));
    require(static_cast<bool>(bypass_second));
    require(static_cast<bool>(grid_second));
    require(static_cast<bool>(grid_bypass_second));
    require(cuda_second.value().candidate_tokens ==
            cpu_second.value().candidate_tokens);
    require(resident_second.value().candidate_tokens ==
            cpu_second.value().candidate_tokens);
    require(bypass_second.value().candidate_tokens ==
            cpu_second.value().candidate_tokens);
    require(grid_second.value().candidate_tokens ==
            cpu_second.value().candidate_tokens);
    require(grid_bypass_second.value().candidate_tokens ==
            cpu_second.value().candidate_tokens);
    const auto second = cpu_second.value().candidate_tokens;
    const std::vector<std::uint32_t> partial_commit{second[0], 23};
    const k3x::DraftVerification partial_verification{
        .anchor_token = history.back(),
        .proposed_draft_tokens = second.size(),
        .accepted_draft_tokens = 1,
        .committed_tokens = partial_commit,
        .all_draft_tokens_accepted = false,
    };
    cpu_provider.value()->update(partial_verification);
    cuda_provider.value()->update(partial_verification);
    resident_provider.value()->update(partial_verification);
    bypass_provider.value()->update(partial_verification);
    grid_provider.value()->update(partial_verification);
    grid_bypass_provider.value()->update(partial_verification);

    const auto cpu_stats = cpu_provider.value()->stats();
    const auto cuda_stats = cuda_provider.value()->stats();
    require(cuda_stats.proposal_calls == cpu_stats.proposal_calls);
    require(cuda_stats.candidate_tokens == cpu_stats.candidate_tokens);
    require(cuda_stats.context_prefill_tokens ==
            cpu_stats.context_prefill_tokens);
    require(cuda_stats.incremental_forward_calls ==
            cpu_stats.incremental_forward_calls);
    require(cuda_stats.rollback_events == cpu_stats.rollback_events);
    require(cuda_stats.mla_positions_cropped ==
            cpu_stats.mla_positions_cropped);
    require(cuda_stats.rollback_events == 1);
    require(cuda_stats.mla_positions_cropped == 2);

    const auto resident_stats = resident_backend->runtime_stats();
    require(resident_stats.weight_cache_hits > 0);
    require(resident_stats.weight_cache_misses > 0);
    require(resident_stats.weight_cache_bypasses == 0);
    require(resident_stats.resident_weight_bytes > 0);
    require(resident_stats.resident_weight_bytes <= 8ULL * 1024ULL * 1024ULL);
    require(resident_stats.peak_resident_weight_bytes >=
            resident_stats.resident_weight_bytes);
    require(resident_stats.peak_resident_weight_bytes <=
            8ULL * 1024ULL * 1024ULL);

    const auto bypass_stats = bypass_backend->runtime_stats();
    require(bypass_stats.weight_cache_misses > 0);
    require(bypass_stats.weight_cache_bypasses > 0);
    require(bypass_stats.resident_weight_bytes == 0);
    require(bypass_stats.peak_resident_weight_bytes == 0);

    const auto grid_stats = grid_backend->runtime_stats();
    require(grid_stats.resident_grid_calls > 0);
    require(grid_stats.resident_grid_kernel_launches ==
            grid_stats.resident_grid_calls * 4);
    require(grid_stats.resident_grid_fallbacks == 0);
    const auto grid_bypass_stats = grid_bypass_backend->runtime_stats();
    require(grid_bypass_stats.resident_grid_calls == 0);
    require(grid_bypass_stats.resident_grid_fallbacks > 0);

    auto replay_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(replay_reader));
    require(!k3x::AuroraReplayDraftProvider::create(
        replay_reader.value(), *cuda_backend, prompt, draft_options(),
        {.scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                       .maximum_length = 2}}));

    auto invalid = cuda_options();
    invalid.cuda_weights = k3x::CudaWeightMode::resident;
    invalid.cuda_resident_bytes = 0;
    auto invalid_backend = k3x::make_cuda_backend(invalid);
    require(static_cast<bool>(invalid_backend));
    auto invalid_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(invalid_reader));
    require(!k3x::AuroraPersistentDraftProvider::create(
        invalid_reader.value(), *invalid_backend.value(), prompt,
        draft_options(), config));
    return 0;
}
