// 증분 AURORA draft 상태와 proposal transaction 경계를 선언합니다.
#pragma once

#include "k3x/model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace k3x {
struct IncrementalDraftCursorStats {
    std::uint64_t context_prefill_tokens{};
    std::uint64_t incremental_forward_calls{};
    std::uint64_t rollback_events{};
    std::uint64_t mla_positions_cropped{};
    std::uint64_t kda_checkpoint_bytes{};
    std::uint64_t routing_decisions{};
    std::uint64_t routing_selected_experts{};
};

struct IncrementalDraftCursorDiagnostics {
    std::vector<float> flattened_state;
    std::size_t mla_length{};
    std::size_t mla_key_elements{};
    std::size_t mla_value_elements{};
    std::size_t mla_shared_key_elements{};
};

class IncrementalDraftCursor {
public:
    ~IncrementalDraftCursor();

    IncrementalDraftCursor(const IncrementalDraftCursor&) = delete;
    IncrementalDraftCursor& operator=(const IncrementalDraftCursor&) = delete;

    static Result<std::unique_ptr<IncrementalDraftCursor>> create(
        Reader& reader, ComputeBackend& backend,
        std::span<const std::uint32_t> context,
        RuntimeSession& session);

    Result<std::vector<std::uint32_t>> propose(std::size_t count);
    Result<bool> commit(
        std::size_t accepted_prefix,
        std::span<const std::uint32_t> committed_tokens);
    IncrementalDraftCursorStats stats() const noexcept;
    IncrementalDraftCursorDiagnostics diagnostics() const;

private:
    struct Impl;
    explicit IncrementalDraftCursor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};
}
