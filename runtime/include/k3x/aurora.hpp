// AURORA reduced-Top-K replay draft provider의 lifecycle을 선언합니다.
#pragma once

#include "k3x/aurora_scheduler.hpp"
#include "k3x/incremental_cursor.hpp"
#include "k3x/model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace k3x {
struct AuroraReplayConfig {
    AuroraSchedulerConfig scheduler{};
};

class AuroraReplayDraftProvider final : public DraftProvider {
public:
    static Result<std::unique_ptr<AuroraReplayDraftProvider>> create(
        Reader& reader, ComputeBackend& backend,
        std::span<const std::uint32_t> prompt,
        RuntimeOptions draft_options, AuroraReplayConfig config);

    Result<DraftProposal> propose(const DraftRequest& request) override;
    void update(const DraftVerification& verification) override;
    DraftProviderStats stats() const noexcept override;

private:
    AuroraReplayDraftProvider(
        Reader& reader, ComputeBackend& backend,
        std::vector<std::uint32_t> prompt,
        RuntimeOptions draft_options,
        AdaptiveDraftScheduler scheduler);

    Reader& reader_;
    ComputeBackend& backend_;
    RuntimeSession session_;
    std::vector<std::uint32_t> prompt_;
    std::vector<std::uint32_t> expected_generated_;
    AdaptiveDraftScheduler scheduler_;
    DraftProviderStats stats_{};
    std::vector<std::uint32_t> pending_candidate_tokens_;
    std::uint32_t pending_anchor_{};
    bool initialized_{};
    bool pending_{};
    bool lifecycle_error_{};
};

struct AuroraPersistentConfig {
    AuroraSchedulerConfig scheduler{};
};

class AuroraPersistentDraftProvider final : public DraftProvider {
public:
    static Result<std::unique_ptr<AuroraPersistentDraftProvider>> create(
        Reader& reader, ComputeBackend& backend,
        std::span<const std::uint32_t> prompt,
        RuntimeOptions draft_options, AuroraPersistentConfig config);

    Result<DraftProposal> propose(const DraftRequest& request) override;
    void update(const DraftVerification& verification) override;
    DraftProviderStats stats() const noexcept override;

private:
    AuroraPersistentDraftProvider(
        Reader& reader, ComputeBackend& backend,
        std::vector<std::uint32_t> prompt,
        RuntimeOptions draft_options,
        AdaptiveDraftScheduler scheduler);

    Reader& reader_;
    ComputeBackend& backend_;
    RuntimeSession session_;
    std::vector<std::uint32_t> prompt_;
    std::vector<std::uint32_t> expected_generated_;
    AdaptiveDraftScheduler scheduler_;
    std::unique_ptr<IncrementalDraftCursor> cursor_;
    DraftProviderStats stats_{};
    std::vector<std::uint32_t> pending_candidate_tokens_;
    std::uint32_t pending_anchor_{};
    bool initialized_{};
    bool cursor_pending_{};
    bool pending_{};
    bool lifecycle_error_{};
};
}
