// 증분 draft cursor의 초기 prefill과 실제 reduced-Top-K proposal을 검증합니다.
#include "k3x/incremental_cursor.hpp"
#include "k3x/model.hpp"

#include <array>
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
        std::cerr << "incremental cursor requirement failed at line "
                  << location.line() << '\n';
        throw std::runtime_error("incremental cursor requirement failed");
    }
}

k3x::RuntimeOptions draft_options() {
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.diagnostics = true;
    options.routing_policy.mode = k3x::RoutingMode::fixed;
    options.routing_policy.fixed_k = 4;
    return options;
}

std::vector<std::uint32_t> oracle_tokens(
    const std::filesystem::path& artifact,
    const std::vector<std::uint32_t>& context,
    std::size_t count) {
    auto reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(reader));
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeSession session(draft_options());
    auto generated = k3x::generate_greedy(
        reader.value(), *backend, context, count, session);
    require(static_cast<bool>(generated));
    return generated.value().token_ids;
}

std::vector<float> oracle_state(
    const std::filesystem::path& artifact,
    const std::vector<std::uint32_t>& context) {
    auto reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(reader));
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeSession session(draft_options());
    auto generated = k3x::generate_greedy(
        reader.value(), *backend, context, 0, session);
    require(static_cast<bool>(generated));
    return generated.value().final_state;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto artifact = std::filesystem::path(argv[1]);
    auto reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(reader));
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeSession session(draft_options());
    const std::vector<std::uint32_t> context{1, 7, 3, 9, 43};

    auto cursor = k3x::IncrementalDraftCursor::create(
        reader.value(), *backend, context, session);
    require(static_cast<bool>(cursor));
    require(cursor.value()->diagnostics().mla_length == context.size());

    auto proposal = cursor.value()->propose(2);
    require(static_cast<bool>(proposal));
    require(proposal.value().size() == 2);
    require(cursor.value()->stats().context_prefill_tokens == context.size());
    require(cursor.value()->stats().incremental_forward_calls == 1);
    require(cursor.value()->diagnostics().mla_length == context.size() + 1);

    require(proposal.value() == oracle_tokens(artifact, context, 2));

    const auto first_candidates = proposal.value();
    const std::array<std::uint32_t, 3> full_commit{
        first_candidates[0], first_candidates[1], 17};
    require(static_cast<bool>(cursor.value()->commit(2, full_commit)));
    std::vector<std::uint32_t> full_context = context;
    full_context.insert(
        full_context.end(), full_commit.begin(), full_commit.end());
    const auto full_diagnostics = cursor.value()->diagnostics();
    require(full_diagnostics.mla_length == full_context.size());
    require(full_diagnostics.flattened_state ==
            oracle_state(artifact, full_context));
    require(cursor.value()->stats().incremental_forward_calls == 3);
    auto full_next = cursor.value()->propose(2);
    require(static_cast<bool>(full_next));
    require(full_next.value() == oracle_tokens(artifact, full_context, 2));

    auto rejected_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(rejected_reader));
    auto rejected_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession rejected_session(draft_options());
    auto rejected_cursor = k3x::IncrementalDraftCursor::create(
        rejected_reader.value(), *rejected_backend, context,
        rejected_session);
    require(static_cast<bool>(rejected_cursor));
    auto rejected_proposal = rejected_cursor.value()->propose(4);
    require(static_cast<bool>(rejected_proposal));
    const std::array<std::uint32_t, 2> corrected_commit{
        rejected_proposal.value()[0], 17};
    require(static_cast<bool>(
        rejected_cursor.value()->commit(1, corrected_commit)));
    std::vector<std::uint32_t> corrected_context = context;
    corrected_context.insert(
        corrected_context.end(), corrected_commit.begin(),
        corrected_commit.end());
    const auto corrected_diagnostics =
        rejected_cursor.value()->diagnostics();
    require(corrected_diagnostics.mla_length == corrected_context.size());
    require(corrected_diagnostics.flattened_state ==
            oracle_state(artifact, corrected_context));
    require(rejected_cursor.value()->stats().rollback_events == 1);
    require(rejected_cursor.value()->stats().mla_positions_cropped == 2);
    auto corrected_next = rejected_cursor.value()->propose(2);
    require(static_cast<bool>(corrected_next));
    require(corrected_next.value() ==
            oracle_tokens(artifact, corrected_context, 2));

    auto zero_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(zero_reader));
    auto zero_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession zero_session(draft_options());
    auto zero_cursor = k3x::IncrementalDraftCursor::create(
        zero_reader.value(), *zero_backend, context, zero_session);
    require(static_cast<bool>(zero_cursor));
    auto zero = zero_cursor.value()->propose(0);
    require(static_cast<bool>(zero));
    require(zero.value().empty());
    const std::array<std::uint32_t, 1> zero_commit{17};
    require(static_cast<bool>(zero_cursor.value()->commit(0, zero_commit)));
    auto zero_context = context;
    zero_context.push_back(17);
    require(zero_cursor.value()->diagnostics().flattened_state ==
            oracle_state(artifact, zero_context));
    require(zero_cursor.value()->stats().incremental_forward_calls == 1);

    auto invalid_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(invalid_reader));
    auto invalid_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession invalid_session(draft_options());
    auto invalid_cursor = k3x::IncrementalDraftCursor::create(
        invalid_reader.value(), *invalid_backend, context,
        invalid_session);
    require(static_cast<bool>(invalid_cursor));
    auto invalid_proposal = invalid_cursor.value()->propose(2);
    require(static_cast<bool>(invalid_proposal));
    const auto reads_before_invalid = invalid_reader.value().counters();
    const std::array<std::uint32_t, 2> wrong_prefix{99, 17};
    require(!invalid_cursor.value()->commit(1, wrong_prefix));
    const auto reads_after_invalid = invalid_reader.value().counters();
    require(reads_after_invalid.calls == reads_before_invalid.calls);
    require(reads_after_invalid.completed_bytes ==
            reads_before_invalid.completed_bytes);
    require(!invalid_cursor.value()->propose(1));
    const auto reads_after_reuse = invalid_reader.value().counters();
    require(reads_after_reuse.calls == reads_before_invalid.calls);
    require(reads_after_reuse.completed_bytes ==
            reads_before_invalid.completed_bytes);

    auto verify_invalid_commit = [&](std::size_t accepted,
                                     std::vector<std::uint32_t> commit) {
        auto case_reader = k3x::Reader::open(artifact);
        require(static_cast<bool>(case_reader));
        auto case_backend = k3x::make_cpu_backend();
        k3x::RuntimeSession case_session(draft_options());
        auto case_cursor = k3x::IncrementalDraftCursor::create(
            case_reader.value(), *case_backend, context, case_session);
        require(static_cast<bool>(case_cursor));
        auto case_proposal = case_cursor.value()->propose(1);
        require(static_cast<bool>(case_proposal));
        if (!commit.empty() && commit[0] == 0) {
            commit[0] = case_proposal.value()[0];
        }
        const auto before = case_reader.value().counters();
        require(!case_cursor.value()->commit(accepted, commit));
        const auto after = case_reader.value().counters();
        require(after.calls == before.calls);
        require(after.completed_bytes == before.completed_bytes);
        require(!case_cursor.value()->propose(1));
    };
    verify_invalid_commit(2, {0, 17, 18});
    verify_invalid_commit(1, {0});
    verify_invalid_commit(1, {0, 17, 18});

    auto outstanding_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(outstanding_reader));
    auto outstanding_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession outstanding_session(draft_options());
    auto outstanding_cursor = k3x::IncrementalDraftCursor::create(
        outstanding_reader.value(), *outstanding_backend, context,
        outstanding_session);
    require(static_cast<bool>(outstanding_cursor));
    auto outstanding = outstanding_cursor.value()->propose(1);
    require(static_cast<bool>(outstanding));
    const auto before_outstanding = outstanding_reader.value().counters();
    require(!outstanding_cursor.value()->propose(1));
    const auto after_outstanding = outstanding_reader.value().counters();
    require(after_outstanding.calls == before_outstanding.calls);
    const std::array<std::uint32_t, 2> outstanding_commit{
        outstanding.value()[0], 17};
    require(static_cast<bool>(
        outstanding_cursor.value()->commit(1, outstanding_commit)));
    require(!outstanding_cursor.value()->commit(1, outstanding_commit));
    return 0;
}
