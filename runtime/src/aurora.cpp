// AURORA draft 후보를 committed prefix 전체 replay로 생성합니다.
#include "k3x/aurora.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace k3x {
namespace {
bool allowed_draft_k(std::size_t value) {
    return value == 4 || value == 6 || value == 8 || value == 12;
}

bool same_tokens(std::span<const std::uint32_t> left,
                 std::span<const std::uint32_t> right) {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin());
}

bool supported_options(const RuntimeOptions& options,
                       std::span<const std::uint32_t> prompt) {
    return options.incremental &&
        options.routing_policy.mode == RoutingMode::fixed &&
        allowed_draft_k(options.routing_policy.fixed_k) &&
        options.l1_expert_cache == L1ExpertCacheMode::disabled &&
        options.l1_expert_cache_bytes == 0 &&
        options.l2_expert_schedule == L2ExpertScheduleMode::blocking &&
        !options.profile_observation && !prompt.empty();
}

bool supported_persistent_backend(const ComputeBackend& backend) {
    if (backend.kind() == BackendKind::cpu) return true;
    if (backend.kind() != BackendKind::cuda_custom) return false;
    const auto& options = backend.options();
    const bool supported_weights =
        (options.cuda_weights == CudaWeightMode::transient &&
         options.cuda_resident_bytes == 0) ||
        (options.cuda_weights == CudaWeightMode::resident &&
         options.cuda_resident_bytes > 0);
    const bool supported_batching =
        options.cuda_batching == CudaBatchingMode::grouped ||
        (options.cuda_batching == CudaBatchingMode::resident_grid &&
         options.cuda_weights == CudaWeightMode::resident &&
         options.cuda_resident_bytes > 0);
    const bool supported_boundary =
        options.cuda_boundary == CudaBoundaryMode::ffn_block ||
        (options.cuda_boundary == CudaBoundaryMode::moe_layer &&
         options.cuda_batching == CudaBatchingMode::resident_grid &&
         options.cuda_weights == CudaWeightMode::resident &&
         options.cuda_resident_bytes > 0);
    return options.kind == BackendKind::cuda_custom &&
        options.dense_precision == DensePrecision::fp32 &&
        options.cuda_allocation == CudaAllocationMode::reused &&
        supported_weights &&
        supported_batching &&
        supported_boundary &&
        options.cuda_transfer == CudaTransferMode::synchronous &&
        options.cuda_moe_fusion == CudaMoeFusionMode::none &&
        options.cuda_pinned_bytes == 0;
}

bool valid_update(const DraftVerification& verification,
                  bool pending, std::uint32_t pending_anchor,
                  std::span<const std::uint32_t> pending_candidates) {
    const auto proposed = pending_candidates.size();
    const auto accepted = verification.accepted_draft_tokens;
    bool valid = pending && verification.anchor_token == pending_anchor &&
        verification.proposed_draft_tokens == proposed &&
        accepted <= proposed &&
        verification.committed_tokens.size() == accepted + 1 &&
        verification.all_draft_tokens_accepted == (accepted == proposed);
    for (std::size_t index = 0; valid && index < accepted; ++index) {
        valid = verification.committed_tokens[index] ==
            pending_candidates[index];
    }
    return valid;
}
}

AuroraReplayDraftProvider::AuroraReplayDraftProvider(
    Reader& reader, ComputeBackend& backend,
    std::vector<std::uint32_t> prompt,
    RuntimeOptions draft_options,
    AdaptiveDraftScheduler scheduler)
    : reader_(reader), backend_(backend), session_(draft_options),
      prompt_(std::move(prompt)), scheduler_(std::move(scheduler)) {}

Result<std::unique_ptr<AuroraReplayDraftProvider>>
AuroraReplayDraftProvider::create(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt,
    RuntimeOptions draft_options, AuroraReplayConfig config) {
    if (backend.kind() != BackendKind::cpu ||
        !supported_options(draft_options, prompt)) {
        return Result<std::unique_ptr<AuroraReplayDraftProvider>>::failure(
            ErrorCode::invalid_state,
            "unsupported AURORA replay runtime combination");
    }
    auto scheduler = AdaptiveDraftScheduler::create(config.scheduler);
    if (!scheduler) {
        return Result<std::unique_ptr<AuroraReplayDraftProvider>>::failure(
            scheduler.error(), scheduler.message());
    }
    return Result<std::unique_ptr<AuroraReplayDraftProvider>>::success(
        std::unique_ptr<AuroraReplayDraftProvider>(
            new AuroraReplayDraftProvider(
                reader, backend,
                std::vector<std::uint32_t>(prompt.begin(), prompt.end()),
                draft_options, std::move(scheduler.value()))));
}

Result<DraftProposal> AuroraReplayDraftProvider::propose(
    const DraftRequest& request) {
    if (lifecycle_error_ || pending_ || request.generated_tokens.empty() ||
        request.generated_position != request.generated_tokens.size() ||
        request.anchor_token != request.generated_tokens.back() ||
        (initialized_ &&
         !same_tokens(request.generated_tokens, expected_generated_))) {
        return Result<DraftProposal>::failure(
            ErrorCode::invalid_state, "invalid AURORA draft lifecycle");
    }

    auto selected = scheduler_.select(request.max_draft_tokens);
    if (!selected) {
        return Result<DraftProposal>::failure(
            selected.error(), selected.message());
    }

    DraftProposal proposal{.anchor_token = request.anchor_token};
    if (selected.value() != 0) {
        std::vector<std::uint32_t> sequence;
        sequence.reserve(prompt_.size() + request.generated_tokens.size());
        sequence.insert(sequence.end(), prompt_.begin(), prompt_.end());
        sequence.insert(sequence.end(), request.generated_tokens.begin(),
                        request.generated_tokens.end());
        const auto reads_before = reader_.counters();
        auto generated = generate_greedy(
            reader_, backend_, sequence, selected.value(), session_);
        const auto reads_after = reader_.counters();
        if (!generated) {
            return Result<DraftProposal>::failure(
                generated.error(), generated.message());
        }
        if (generated.value().token_ids.size() != selected.value() ||
            reads_after.calls < reads_before.calls ||
            reads_after.completed_bytes < reads_before.completed_bytes) {
            return Result<DraftProposal>::failure(
                ErrorCode::invalid_state, "invalid AURORA replay result");
        }
        proposal.candidate_tokens = std::move(generated.value().token_ids);
        stats_.replayed_context_tokens += sequence.size();
        stats_.generation_nanoseconds +=
            generated.value().prefill_nanoseconds +
            generated.value().decode_nanoseconds;
        stats_.reader_calls += reads_after.calls - reads_before.calls;
        stats_.reader_bytes +=
            reads_after.completed_bytes - reads_before.completed_bytes;
        stats_.routing_decisions += generated.value().routing_decisions;
        stats_.routing_selected_experts +=
            generated.value().routing_selected_experts;
    }

    if (!initialized_) {
        expected_generated_.assign(request.generated_tokens.begin(),
                                   request.generated_tokens.end());
        initialized_ = true;
    }
    pending_anchor_ = request.anchor_token;
    pending_candidate_tokens_ = proposal.candidate_tokens;
    pending_ = true;
    ++stats_.proposal_calls;
    stats_.candidate_tokens += proposal.candidate_tokens.size();
    return Result<DraftProposal>::success(std::move(proposal));
}

void AuroraReplayDraftProvider::update(
    const DraftVerification& verification) {
    const auto valid = valid_update(
        verification, pending_, pending_anchor_, pending_candidate_tokens_);
    auto observed = valid
        ? scheduler_.observe(verification)
        : Result<bool>::failure(ErrorCode::invalid_state);
    if (!valid || !observed) {
        lifecycle_error_ = true;
        pending_ = false;
        pending_candidate_tokens_.clear();
        return;
    }
    expected_generated_.insert(expected_generated_.end(),
                               verification.committed_tokens.begin(),
                               verification.committed_tokens.end());
    pending_ = false;
    pending_candidate_tokens_.clear();
}

DraftProviderStats AuroraReplayDraftProvider::stats() const noexcept {
    auto result = stats_;
    const auto scheduler_stats = scheduler_.stats();
    result.selected_length_1 = scheduler_stats.selected_length_1;
    result.selected_length_2 = scheduler_stats.selected_length_2;
    result.selected_length_4 = scheduler_stats.selected_length_4;
    result.scheduler_growths = scheduler_stats.scheduler_growths;
    result.scheduler_backoffs = scheduler_stats.scheduler_backoffs;
    return result;
}

AuroraPersistentDraftProvider::AuroraPersistentDraftProvider(
    Reader& reader, ComputeBackend& backend,
    std::vector<std::uint32_t> prompt,
    RuntimeOptions draft_options,
    AdaptiveDraftScheduler scheduler)
    : reader_(reader), backend_(backend), session_(draft_options),
      prompt_(std::move(prompt)), scheduler_(std::move(scheduler)) {}

Result<std::unique_ptr<AuroraPersistentDraftProvider>>
AuroraPersistentDraftProvider::create(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt,
    RuntimeOptions draft_options, AuroraPersistentConfig config) {
    if (!supported_persistent_backend(backend) ||
        !supported_options(draft_options, prompt)) {
        return Result<std::unique_ptr<AuroraPersistentDraftProvider>>::failure(
            ErrorCode::invalid_state,
            "unsupported AURORA persistent runtime combination");
    }
    auto scheduler = AdaptiveDraftScheduler::create(config.scheduler);
    if (!scheduler) {
        return Result<std::unique_ptr<AuroraPersistentDraftProvider>>::failure(
            scheduler.error(), scheduler.message());
    }
    return Result<std::unique_ptr<AuroraPersistentDraftProvider>>::success(
        std::unique_ptr<AuroraPersistentDraftProvider>(
            new AuroraPersistentDraftProvider(
                reader, backend,
                std::vector<std::uint32_t>(prompt.begin(), prompt.end()),
                draft_options, std::move(scheduler.value()))));
}

Result<DraftProposal> AuroraPersistentDraftProvider::propose(
    const DraftRequest& request) {
    if (lifecycle_error_ || pending_ || request.generated_tokens.empty() ||
        request.generated_position != request.generated_tokens.size() ||
        request.anchor_token != request.generated_tokens.back() ||
        (initialized_ &&
         !same_tokens(request.generated_tokens, expected_generated_))) {
        return Result<DraftProposal>::failure(
            ErrorCode::invalid_state, "invalid AURORA draft lifecycle");
    }

    auto selected = scheduler_.select(request.max_draft_tokens);
    if (!selected) {
        return Result<DraftProposal>::failure(
            selected.error(), selected.message());
    }

    DraftProposal proposal{.anchor_token = request.anchor_token};
    const auto reads_before = reader_.counters();
    const auto started = std::chrono::steady_clock::now();
    if (!cursor_ && selected.value() != 0) {
        std::vector<std::uint32_t> context;
        context.reserve(prompt_.size() + request.generated_tokens.size());
        context.insert(context.end(), prompt_.begin(), prompt_.end());
        context.insert(context.end(), request.generated_tokens.begin(),
                       request.generated_tokens.end());
        auto created = IncrementalDraftCursor::create(
            reader_, backend_, context, session_);
        if (!created) {
            return Result<DraftProposal>::failure(
                created.error(), created.message());
        }
        cursor_ = std::move(created.value());
    }
    if (cursor_) {
        auto candidates = cursor_->propose(selected.value());
        if (!candidates) {
            lifecycle_error_ = true;
            return Result<DraftProposal>::failure(
                candidates.error(), candidates.message());
        }
        proposal.candidate_tokens = std::move(candidates.value());
        cursor_pending_ = true;
    }
    const auto finished = std::chrono::steady_clock::now();
    const auto reads_after = reader_.counters();
    if (reads_after.calls < reads_before.calls ||
        reads_after.completed_bytes < reads_before.completed_bytes) {
        lifecycle_error_ = true;
        return Result<DraftProposal>::failure(
            ErrorCode::invalid_state, "invalid AURORA persistent counters");
    }
    stats_.generation_nanoseconds +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            finished - started).count();
    stats_.reader_calls += reads_after.calls - reads_before.calls;
    stats_.reader_bytes +=
        reads_after.completed_bytes - reads_before.completed_bytes;

    if (!initialized_) {
        expected_generated_.assign(request.generated_tokens.begin(),
                                   request.generated_tokens.end());
        initialized_ = true;
    }
    pending_anchor_ = request.anchor_token;
    pending_candidate_tokens_ = proposal.candidate_tokens;
    pending_ = true;
    ++stats_.proposal_calls;
    stats_.candidate_tokens += proposal.candidate_tokens.size();
    return Result<DraftProposal>::success(std::move(proposal));
}

void AuroraPersistentDraftProvider::update(
    const DraftVerification& verification) {
    const auto valid = valid_update(
        verification, pending_, pending_anchor_, pending_candidate_tokens_);
    if (!valid) {
        lifecycle_error_ = true;
        pending_ = false;
        pending_candidate_tokens_.clear();
        return;
    }

    if (cursor_pending_) {
        const auto reads_before = reader_.counters();
        const auto started = std::chrono::steady_clock::now();
        auto committed = cursor_->commit(
            verification.accepted_draft_tokens,
            verification.committed_tokens);
        const auto finished = std::chrono::steady_clock::now();
        const auto reads_after = reader_.counters();
        if (!committed || reads_after.calls < reads_before.calls ||
            reads_after.completed_bytes < reads_before.completed_bytes) {
            lifecycle_error_ = true;
            pending_ = false;
            cursor_pending_ = false;
            pending_candidate_tokens_.clear();
            return;
        }
        stats_.generation_nanoseconds +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                finished - started).count();
        stats_.reader_calls += reads_after.calls - reads_before.calls;
        stats_.reader_bytes +=
            reads_after.completed_bytes - reads_before.completed_bytes;
    }
    auto observed = scheduler_.observe(verification);
    if (!observed) {
        lifecycle_error_ = true;
        pending_ = false;
        cursor_pending_ = false;
        pending_candidate_tokens_.clear();
        return;
    }
    expected_generated_.insert(expected_generated_.end(),
                               verification.committed_tokens.begin(),
                               verification.committed_tokens.end());
    pending_ = false;
    cursor_pending_ = false;
    pending_candidate_tokens_.clear();
}

DraftProviderStats AuroraPersistentDraftProvider::stats() const noexcept {
    auto result = stats_;
    const auto scheduler_stats = scheduler_.stats();
    result.selected_length_1 = scheduler_stats.selected_length_1;
    result.selected_length_2 = scheduler_stats.selected_length_2;
    result.selected_length_4 = scheduler_stats.selected_length_4;
    result.scheduler_growths = scheduler_stats.scheduler_growths;
    result.scheduler_backoffs = scheduler_stats.scheduler_backoffs;
    if (cursor_) {
        const auto cursor_stats = cursor_->stats();
        result.context_prefill_tokens = cursor_stats.context_prefill_tokens;
        result.incremental_forward_calls =
            cursor_stats.incremental_forward_calls;
        result.rollback_events = cursor_stats.rollback_events;
        result.mla_positions_cropped = cursor_stats.mla_positions_cropped;
        result.kda_checkpoint_bytes = cursor_stats.kda_checkpoint_bytes;
        result.routing_decisions = cursor_stats.routing_decisions;
        result.routing_selected_experts =
            cursor_stats.routing_selected_experts;
    }
    return result;
}
}
