// 합성 K3 KDA/MLA/MoE/AttnRes graph를 외부 ML library 없이 실행합니다.
#include "k3x/model.hpp"

#include "k3x/expert_major.hpp"
#include "k3x/format.hpp"
#include "k3x/incremental_cursor.hpp"
#include "k3x/ops.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace k3x {
namespace {
using Vector = std::vector<float>;

template <typename T>
T little(const std::array<std::byte, model_config_bytes>& bytes, std::size_t offset) {
    T value{};
    for (std::size_t index = 0; index < sizeof(T); ++index) {
        value |= static_cast<T>(std::to_integer<std::uint8_t>(bytes[offset + index])) << (index * 8U);
    }
    return value;
}

float config_float(const std::array<std::byte, model_config_bytes>& bytes, std::size_t index) {
    const auto raw = little<std::uint32_t>(bytes, 80 + index * 4);
    return std::bit_cast<float>(raw);
}

struct Config {
    std::uint32_t vocab, hidden, layers, kda_heads, kda_dim, conv_kernel;
    std::uint32_t mla_heads, q_rank, kv_rank, q_main, q_extra, value_dim;
    std::uint32_t experts, top_k, shared_experts, latent, expert_intermediate;
    std::uint32_t dense_intermediate, residual_block, group_size;
    float epsilon, kda_lower, routed_scale, situ_beta, situ_linear;
};

Config decode_config(const std::array<std::byte, model_config_bytes>& bytes) {
    std::array<std::uint32_t, 20> integers{};
    for (std::size_t index = 0; index < integers.size(); ++index) {
        integers[index] = little<std::uint32_t>(bytes, index * 4);
    }
    return Config{integers[0],integers[1],integers[2],integers[3],integers[4],integers[5],
                  integers[6],integers[7],integers[8],integers[9],integers[10],integers[11],
                  integers[12],integers[13],integers[14],integers[15],integers[16],
                  integers[17],integers[18],integers[19],config_float(bytes,0),
                  config_float(bytes,1),config_float(bytes,2),config_float(bytes,3),
                  config_float(bytes,4)};
}

float sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }
float silu(float value) { return value * sigmoid(value); }

Vector normalized(std::span<const float> input, std::span<const float> weight, float epsilon) {
    Vector output(input.size());
    rms_norm(output, input, weight, epsilon);
    return output;
}

struct KdaState { Vector conv_q, conv_k, conv_v, recurrent; };
struct MlaState { Vector keys, values, shared_keys; std::size_t length{}; };
struct ModelState { std::vector<KdaState> kda; MlaState mla; };

class Engine {
public:
    Engine(Reader& reader, ComputeBackend& backend, RuntimeSession& session)
        : reader_(reader), backend_(backend),
          config_(decode_config(reader.model_config())),
          trace_routing_(session.options().diagnostics),
          session_(session),
          expert_store_(session.expert_store()),
          expert_loader_(session.expert_loader()) {
        state_template_.kda.resize(3);
        for (auto& state : state_template_.kda) {
            const auto history = (config_.conv_kernel - 1) * config_.hidden;
            state.conv_q.assign(history, 0.0F);
            state.conv_k.assign(history, 0.0F);
            state.conv_v.assign(history, 0.0F);
            state.recurrent.assign(config_.kda_heads * config_.kda_dim * config_.kda_dim, 0.0F);
        }
    }

    ModelState empty_state() const { return state_template_; }
    std::size_t vocabulary_size() const noexcept { return config_.vocab; }
    const std::vector<std::uint64_t>& layer_nanoseconds() const { return layer_nanoseconds_; }
    const std::vector<std::uint32_t>& routed_experts() const {
        return routed_experts_;
    }
    const std::vector<std::uint32_t>& routed_k() const { return routed_k_; }
    L1ExpertCacheStats expert_cache_stats() const {
        return expert_store_.stats();
    }
    void export_routing_stats(GenerationResult& result) const {
        result.routing_natural_top_k = config_.top_k;
        result.routing_decisions = routing_decisions_;
        result.routing_selected_experts = routing_selected_experts_;
        result.routing_quality_escalated_decisions =
            routing_quality_escalated_decisions_;
        result.cold_rescue_count = cold_rescue_count_;
        result.routing_normalized_entropy_sum = routing_normalized_entropy_sum_;
        result.routing_selected_mass_sum = routing_selected_mass_sum_;
        result.routing_boundary_confidence_sum =
            routing_boundary_confidence_sum_;
    }
    std::uint64_t routing_decisions() const noexcept {
        return routing_decisions_;
    }
    std::uint64_t routing_selected_experts() const noexcept {
        return routing_selected_experts_;
    }

    struct ExpertMajorBlockResult {
        std::vector<Vector> logits;
        std::vector<ModelState> states_after_positions;
        std::vector<std::vector<std::uint32_t>>
            routed_experts_by_position;
        std::vector<std::vector<std::uint32_t>> routed_k_by_position;
        std::uint64_t unique_experts_sum{};
        std::uint64_t unique_experts_max{};
        std::uint64_t assignments{};
        std::uint64_t payload_loads{};
    };

    ExpertMajorBlockResult forward_expert_major_block(
        std::span<const std::uint32_t> tokens,
        const ModelState& committed_state, ProfilePhase phase) {
        if (tokens.empty() || config_.layers != 4 ||
            committed_state.kda.size() != 3) {
            throw std::runtime_error("unsupported expert-major graph");
        }
        active_forward_cycle_ = session_.acquire_forward_cycle();

        struct Position {
            Vector hidden;
            std::vector<Vector> sources;
        };
        std::vector<Position> positions;
        positions.reserve(tokens.size());
        const auto& embedding = tensor("model.embeddings");
        for (const auto token : tokens) {
            if (token >= config_.vocab) {
                throw std::runtime_error("token out of range");
            }
            positions.push_back({
                .hidden = Vector(
                    embedding.begin() + token * config_.hidden,
                    embedding.begin() + (token + 1) * config_.hidden),
            });
        }

        ExpertMajorBlockResult result;
        result.routed_experts_by_position.resize(tokens.size());
        result.routed_k_by_position.resize(tokens.size());
        std::vector<std::vector<RoutingDecision>> decisions_by_position(
            tokens.size());
        std::vector<std::vector<KdaState>> kda_snapshots(
            3, std::vector<KdaState>(tokens.size()));
        std::vector<MlaState> mla_snapshots(tokens.size());
        auto working_state = committed_state;

        for (std::size_t layer = 0; layer < config_.layers; ++layer) {
            const auto layer_start = std::chrono::steady_clock::now();
            std::vector<Vector> prefix_sums(tokens.size());
            std::vector<Vector> ffn_inputs(tokens.size());
            for (std::size_t position = 0; position < tokens.size();
                 ++position) {
                auto& item = positions[position];
                const auto prefix = item.hidden;
                Vector attention_input = item.sources.empty()
                    ? prefix
                    : attention_residual(
                          prefix, item.sources,
                          layer_name(layer, "self_attention_residual"));
                const bool pushed = layer % config_.residual_block == 0;
                if (pushed) item.sources.push_back(prefix);
                const auto normed = normalized(
                    attention_input, tensor(layer_name(layer, "input_norm")),
                    config_.epsilon);
                Vector attention;
                if (layer < 3) {
                    attention = kda(normed, layer, working_state.kda[layer],
                                    phase);
                    kda_snapshots[layer][position] =
                        working_state.kda[layer];
                } else {
                    attention = mla(normed, layer, working_state.mla, phase);
                    mla_snapshots[position] = working_state.mla;
                }
                auto& prefix_sum = prefix_sums[position];
                prefix_sum.resize(config_.hidden);
                for (std::size_t index = 0; index < config_.hidden;
                     ++index) {
                    prefix_sum[index] = pushed
                        ? attention[index]
                        : prefix[index] + attention[index];
                }
                const auto ffn_input = attention_residual(
                    prefix_sum, item.sources,
                    layer_name(layer, "mlp_residual"));
                ffn_inputs[position] = normalized(
                    ffn_input,
                    tensor(layer_name(layer, "post_attention_norm")),
                    config_.epsilon);
            }

            if (layer == 0) {
                for (std::size_t position = 0; position < tokens.size();
                     ++position) {
                    const auto ffn = dense(ffn_inputs[position], layer, phase);
                    positions[position].hidden.resize(config_.hidden);
                    for (std::size_t index = 0; index < config_.hidden;
                         ++index) {
                        positions[position].hidden[index] =
                            prefix_sums[position][index] + ffn[index];
                    }
                }
            } else {
                const auto base = layer_name(layer, "feed_forward.");
                std::vector<ExpertMajorTokenRoute> token_routes;
                token_routes.reserve(tokens.size());
                std::vector<Vector> latents;
                std::vector<Vector> shared_outputs;
                latents.reserve(tokens.size());
                shared_outputs.reserve(tokens.size());
                for (std::size_t position = 0; position < tokens.size();
                     ++position) {
                    const auto scores_raw = matvec(
                        base + "router_weight", config_.experts,
                        config_.hidden, ffn_inputs[position], layer, phase);
                    const auto& bias = tensor(base + "correction_bias");
                    Vector scores(config_.experts);
                    for (std::size_t index = 0; index < scores.size();
                         ++index) {
                        scores[index] = sigmoid(scores_raw[index]);
                    }
                    auto routing = select_routing(
                        scores, bias, config_.top_k,
                        session_.options().routing_policy);
                    if (!routing) {
                        throw std::runtime_error(routing.message());
                    }
                    auto decision = std::move(routing.value());
                    ExpertMajorTokenRoute route;
                    route.expert_ids.reserve(decision.selected_k);
                    route.contributions.reserve(decision.selected_k);
                    result.routed_k_by_position[position].push_back(
                        static_cast<std::uint32_t>(decision.selected_k));
                    for (std::size_t slot = 0;
                         slot < decision.selected_k; ++slot) {
                        const auto expert_id = static_cast<std::uint32_t>(
                            decision.expert_ids[slot]);
                        route.expert_ids.push_back(expert_id);
                        route.contributions.push_back(
                            decision.normalized_weights[slot] *
                            config_.routed_scale);
                        result.routed_experts_by_position[position].push_back(
                            expert_id);
                    }
                    token_routes.push_back(std::move(route));
                    decisions_by_position[position].push_back(
                        std::move(decision));
                    latents.push_back(matvec(
                        base + "routed_down_proj", config_.latent,
                        config_.hidden, ffn_inputs[position], layer, phase));
                    shared_outputs.push_back(
                        shared_expert(ffn_inputs[position], layer, phase));
                }

                auto plan = build_expert_major_plan(token_routes);
                if (!plan) throw std::runtime_error(plan.message());
                result.unique_experts_sum += plan.value().groups.size();
                result.unique_experts_max = std::max<std::uint64_t>(
                    result.unique_experts_max, plan.value().groups.size());
                result.assignments += plan.value().assignment_count;

                std::vector<ExpertKey> selected;
                selected.reserve(plan.value().groups.size());
                for (const auto& group : plan.value().groups) {
                    selected.push_back({layer, group.expert_id});
                }
                expert_store_.begin_access_set(
                    active_forward_cycle_, layer, selected);

                std::vector<std::vector<Vector>> routed_outputs(
                    tokens.size());
                for (std::size_t position = 0; position < tokens.size();
                     ++position) {
                    routed_outputs[position].resize(
                        token_routes[position].expert_ids.size());
                }
                for (const auto& group : plan.value().groups) {
                    const auto payload = load_expert(layer, group.expert_id);
                    ++result.payload_loads;
                    if (backend_.kind() == BackendKind::cuda_custom) {
                        std::vector<float> flat_inputs;
                        flat_inputs.reserve(
                            group.assignments.size() * config_.latent);
                        for (const auto& assignment : group.assignments) {
                            const auto& latent =
                                latents[assignment.token_index];
                            flat_inputs.insert(flat_inputs.end(),
                                               latent.begin(), latent.end());
                        }
                        auto batch = backend_.mxfp4_situ_mlp_batch(
                            flat_inputs, group.assignments.size(),
                            payload->view(config_.group_size),
                            config_.situ_beta, config_.situ_linear,
                            static_cast<std::uint32_t>(layer), phase);
                        if (!batch ||
                            batch.value().size() != group.assignments.size()) {
                            throw std::runtime_error(
                                "batched expert FFN backend failure");
                        }
                        for (std::size_t index = 0;
                             index < group.assignments.size(); ++index) {
                            if (batch.value()[index].size() != config_.latent) {
                                throw std::runtime_error(
                                    "batched expert FFN output shape changed");
                            }
                            const auto& assignment = group.assignments[index];
                            routed_outputs[assignment.token_index]
                                          [assignment.router_slot] =
                                std::move(batch.value()[index]);
                        }
                    } else {
                        for (const auto& assignment : group.assignments) {
                            routed_outputs[assignment.token_index]
                                          [assignment.router_slot] =
                                expert_payload(
                                    latents[assignment.token_index], layer,
                                    payload, phase);
                        }
                    }
                }

                for (std::size_t position = 0; position < tokens.size();
                     ++position) {
                    Vector mixed(config_.latent, 0.0F);
                    for (std::size_t slot = 0;
                         slot < routed_outputs[position].size(); ++slot) {
                        for (std::size_t index = 0; index < mixed.size();
                             ++index) {
                            mixed[index] +=
                                token_routes[position].contributions[slot] *
                                routed_outputs[position][slot][index];
                        }
                    }
                    const auto routed_norm = normalized(
                        mixed, tensor(base + "routed_norm"),
                        config_.epsilon);
                    auto output = matvec(
                        base + "routed_up_proj", config_.hidden,
                        config_.latent, routed_norm, layer, phase);
                    for (std::size_t index = 0; index < output.size();
                         ++index) {
                        output[index] += shared_outputs[position][index];
                        positions[position].hidden[index] =
                            prefix_sums[position][index] + output[index];
                    }
                }
            }
            layer_nanoseconds_[layer] +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - layer_start).count();
        }

        result.logits.reserve(tokens.size());
        result.states_after_positions.reserve(tokens.size());
        for (std::size_t position = 0; position < tokens.size(); ++position) {
            auto hidden = attention_residual(
                positions[position].hidden, positions[position].sources,
                "model.output_residual");
            hidden = normalized(
                hidden, tensor("model.final_norm"), config_.epsilon);
            result.logits.push_back(matvec(
                "model.lm_head", config_.vocab, config_.hidden, hidden,
                profile_global_layer, phase));

            auto snapshot = committed_state;
            for (std::size_t layer = 0; layer < 3; ++layer) {
                snapshot.kda[layer] = kda_snapshots[layer][position];
            }
            snapshot.mla = mla_snapshots[position];
            result.states_after_positions.push_back(std::move(snapshot));
        }
        for (const auto& position_decisions : decisions_by_position) {
            for (const auto& decision : position_decisions) {
                record_routing_decision(decision);
            }
        }
        return result;
    }

    void commit_expert_major_routes(
        const ExpertMajorBlockResult& block,
        std::size_t committed_positions) {
        if (committed_positions > block.routed_k_by_position.size()) {
            throw std::runtime_error("invalid committed route prefix");
        }
        if (!trace_routing_) return;
        for (std::size_t position = 0; position < committed_positions;
             ++position) {
            routed_k_.insert(
                routed_k_.end(), block.routed_k_by_position[position].begin(),
                block.routed_k_by_position[position].end());
            routed_experts_.insert(
                routed_experts_.end(),
                block.routed_experts_by_position[position].begin(),
                block.routed_experts_by_position[position].end());
        }
    }

    Vector forward(std::uint32_t token, ModelState& state, ProfilePhase phase,
                   std::vector<Vector>* layer_outputs = nullptr) {
        active_forward_cycle_ = session_.acquire_forward_cycle();
        const auto& embedding = tensor("model.embeddings");
        if (token >= config_.vocab) throw std::runtime_error("token out of range");
        Vector hidden(embedding.begin() + token * config_.hidden,
                      embedding.begin() + (token + 1) * config_.hidden);
        std::vector<Vector> sources;
        for (std::size_t layer = 0; layer < config_.layers; ++layer) {
            const auto layer_start = std::chrono::steady_clock::now();
            const auto prefix = hidden;
            Vector attention_input = sources.empty()
                ? prefix : attention_residual(prefix, sources, layer_name(layer, "self_attention_residual"));
            const bool pushed = layer % config_.residual_block == 0;
            if (pushed) sources.push_back(prefix);
            const auto normed = normalized(attention_input, tensor(layer_name(layer, "input_norm")), config_.epsilon);
            Vector attention = layer < 3 ? kda(normed, layer, state.kda[layer], phase)
                                         : mla(normed, layer, state.mla, phase);
            Vector prefix_sum(config_.hidden);
            for (std::size_t index = 0; index < config_.hidden; ++index) {
                prefix_sum[index] = pushed ? attention[index] : prefix[index] + attention[index];
            }
            const auto ffn_input = attention_residual(prefix_sum, sources,
                                                       layer_name(layer, "mlp_residual"));
            const auto ffn_normed = normalized(ffn_input,
                                               tensor(layer_name(layer, "post_attention_norm")),
                                               config_.epsilon);
            const auto ffn = layer == 0 ? dense(ffn_normed, layer, phase)
                                        : moe(ffn_normed, layer, phase);
            for (std::size_t index = 0; index < config_.hidden; ++index) hidden[index] = prefix_sum[index] + ffn[index];
            if (layer_outputs) layer_outputs->push_back(hidden);
            layer_nanoseconds_[layer] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - layer_start).count();
        }
        hidden = attention_residual(hidden, sources, "model.output_residual");
        hidden = normalized(hidden, tensor("model.final_norm"), config_.epsilon);
        return matvec("model.lm_head", config_.vocab, config_.hidden,
                      hidden, profile_global_layer, phase);
    }

    Vector flatten_state(const ModelState& state) const {
        Vector values;
        for (const auto& item : state.kda) {
            for (const auto* tensor : {&item.conv_q, &item.conv_k, &item.conv_v, &item.recurrent}) {
                values.insert(values.end(), tensor->begin(), tensor->end());
            }
        }
        for (const auto [tensor, width] : {
                 std::pair{&state.mla.keys, config_.q_main},
                 std::pair{&state.mla.values, config_.value_dim}}) {
            for (std::size_t head = 0; head < config_.mla_heads; ++head) {
                for (std::size_t position = 0; position < state.mla.length; ++position) {
                    const auto begin = tensor->begin() +
                        (position * config_.mla_heads + head) * width;
                    values.insert(values.end(), begin, begin + width);
                }
            }
        }
        values.insert(values.end(), state.mla.shared_keys.begin(), state.mla.shared_keys.end());
        return values;
    }

private:
    struct DenseProjection {
        std::string name;
        std::size_t rows;
        std::size_t cols;
    };

    static std::string layer_name(std::size_t layer, const std::string& suffix) {
        return "model.layers." + std::to_string(layer) + "." + suffix;
    }

    const Vector& tensor(const std::string& name) {
        const auto id = fnv1a64(name.c_str());
        if (const auto found = tensors_.find(id); found != tensors_.end()) return found->second;
        auto bytes = reader_.read_tensor(id);
        if (!bytes || bytes.value().size() % sizeof(float)) throw std::runtime_error("missing FP32 tensor: " + name);
        Vector values(bytes.value().size() / sizeof(float));
        std::memcpy(values.data(), bytes.value().data(), bytes.value().size());
        return tensors_.emplace(id, std::move(values)).first->second;
    }

    DenseWeightView dense_weight(const std::string& name, std::size_t rows,
                                 std::size_t cols) {
        return {fnv1a64(name.c_str()), tensor(name), rows, cols};
    }

    static Result<ExpertPayloadHandle> load_expert_result_from(
        Reader& reader, HostExpertStore& expert_store,
        std::size_t layer, std::size_t expert_id) {
        auto result = expert_store.get_or_load(
            {layer, expert_id}, [&, layer, expert_id]() {
                const auto base = layer_name(
                    layer, "feed_forward.experts." +
                               std::to_string(expert_id));
                constexpr std::array suffixes{"gate", "up", "down"};
                std::array<std::uint64_t, suffixes.size()> ids{};
                std::array<const TensorRecord*, suffixes.size()> records{};
                std::array<ExtentRequest, suffixes.size() * 2> requests{};
                for (std::size_t index = 0; index < suffixes.size(); ++index) {
                    ids[index] = fnv1a64(
                        (base + "." + suffixes[index]).c_str());
                    const auto record = std::find_if(
                        reader.tensors().begin(), reader.tensors().end(),
                        [id = ids[index]](const auto& item) {
                            return item.tensor_id == id;
                        });
                    if (record == reader.tensors().end()) {
                        return Result<ExpertMlpPayload>::failure(
                            ErrorCode::tensor_not_found, "missing expert");
                    }
                    records[index] = &*record;
                    requests[index * 2] = {
                        record->data_offset, record->data_length};
                    requests[index * 2 + 1] = {
                        record->auxiliary_offset, record->auxiliary_length};
                }
                auto payloads = reader.read_extents(requests);
                if (!payloads) {
                    return Result<ExpertMlpPayload>::failure(
                        payloads.error(), payloads.message());
                }
                const auto projection = [&](std::size_t index) {
                    return ExpertProjection{
                        ids[index],
                        std::move(payloads.value()[index * 2]),
                        std::move(payloads.value()[index * 2 + 1]),
                        records[index]->dimensions[0],
                        records[index]->dimensions[1]};
                };
                return Result<ExpertMlpPayload>::success({
                    projection(0), projection(1), projection(2)});
            });
        return result;
    }

    Result<ExpertPayloadHandle> load_expert_result(std::size_t layer,
                                                   std::size_t expert_id) {
        return load_expert_result_from(
            reader_, expert_store_, layer, expert_id);
    }

    ExpertPayloadHandle load_expert(std::size_t layer,
                                    std::size_t expert_id) {
        auto result = load_expert_result(layer, expert_id);
        if (!result) throw std::runtime_error("missing expert");
        return std::move(result.value());
    }

    std::uint64_t expert_payload_bytes(std::size_t layer,
                                       std::size_t expert_id) const {
        const auto base = layer_name(
            layer, "feed_forward.experts." + std::to_string(expert_id));
        std::uint64_t bytes = 0;
        for (const auto suffix : {"gate", "up", "down"}) {
            const auto id = fnv1a64((base + "." + suffix).c_str());
            const auto record = std::find_if(
                reader_.tensors().begin(), reader_.tensors().end(),
                [id](const auto& item) { return item.tensor_id == id; });
            if (record == reader_.tensors().end() ||
                record->data_length >
                    std::numeric_limits<std::uint64_t>::max() - bytes ||
                record->auxiliary_length >
                    std::numeric_limits<std::uint64_t>::max() - bytes -
                        record->data_length) {
                throw std::runtime_error("missing expert");
            }
            bytes += record->data_length + record->auxiliary_length;
        }
        return bytes;
    }

    ExpertLoadTicket schedule_expert(std::size_t layer,
                                     std::size_t expert_id,
                                     bool resident,
                                     std::chrono::nanoseconds estimate) {
        if (!expert_loader_) throw std::runtime_error("expert scheduler disabled");
        auto ticket = expert_loader_->submit(
            {std::chrono::steady_clock::now() + estimate, estimate,
             expert_payload_bytes(layer, expert_id),
             resident},
            [reader = &reader_, store = &expert_store_, layer, expert_id] {
                return load_expert_result_from(
                    *reader, *store, layer, expert_id);
            });
        if (!ticket) throw std::runtime_error("expert scheduler queue failure");
        return std::move(ticket.value());
    }

    Vector matvec(const std::string& name, std::size_t rows,
                  std::size_t cols, std::span<const float> input,
                  std::uint32_t layer, ProfilePhase phase) {
        auto result = backend_.dense_matvec(
            input, dense_weight(name, rows, cols), layer, phase);
        if (!result) throw std::runtime_error("dense backend failure");
        return std::move(result.value());
    }

    std::vector<Vector> matvec_group(
        std::span<const DenseProjection> projections,
        std::span<const float> input, std::uint32_t layer,
        ProfilePhase phase) {
        if (backend_.options().cuda_batching != CudaBatchingMode::grouped) {
            std::vector<Vector> outputs;
            outputs.reserve(projections.size());
            for (const auto& projection : projections) {
                outputs.push_back(matvec(projection.name, projection.rows,
                                         projection.cols, input, layer, phase));
            }
            return outputs;
        }
        std::vector<DenseWeightView> weights;
        weights.reserve(projections.size());
        for (const auto& projection : projections) {
            weights.push_back(dense_weight(
                projection.name, projection.rows, projection.cols));
        }
        auto result = backend_.dense_matvec_group(input, weights, layer, phase);
        if (!result) throw std::runtime_error("grouped dense backend failure");
        return std::move(result.value());
    }

    Vector attention_residual(const Vector& prefix, const std::vector<Vector>& sources,
                              const std::string& base) {
        std::vector<const Vector*> values;
        for (const auto& source : sources) values.push_back(&source);
        values.push_back(&prefix);
        const auto& norm_weight = tensor(base + ".norm");
        const auto& projection = tensor(base + ".projection");
        Vector scores(values.size());
        for (std::size_t depth = 0; depth < values.size(); ++depth) {
            const auto key = normalized(*values[depth], norm_weight, config_.epsilon);
            scores[depth] = std::inner_product(key.begin(), key.end(), projection.begin(), 0.0F);
        }
        const auto maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0F;
        for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
        Vector output(config_.hidden, 0.0F);
        for (std::size_t depth = 0; depth < values.size(); ++depth) {
            const auto probability = scores[depth] / denominator;
            for (std::size_t index = 0; index < output.size(); ++index) output[index] += probability * (*values[depth])[index];
        }
        return output;
    }

    Vector short_conv(const Vector& projected, Vector& history, const Vector& weight) {
        Vector output(config_.hidden);
        for (std::size_t channel = 0; channel < config_.hidden; ++channel) {
            double sum = 0.0;
            for (std::size_t tap = 0; tap + 1 < config_.conv_kernel; ++tap) {
                sum += static_cast<double>(history[tap * config_.hidden + channel]) *
                       weight[channel * config_.conv_kernel + tap];
            }
            sum += static_cast<double>(projected[channel]) *
                   weight[channel * config_.conv_kernel + config_.conv_kernel - 1];
            output[channel] = silu(static_cast<float>(sum));
        }
        std::move(history.begin() + config_.hidden, history.end(), history.begin());
        std::copy(projected.begin(), projected.end(), history.end() - config_.hidden);
        return output;
    }

    Vector kda(const Vector& input, std::size_t layer, KdaState& state,
               ProfilePhase phase) {
        const auto base = layer_name(layer, "attention.");
        const std::array<DenseProjection, 3> qkv_projections{{
            {base + "q_proj", config_.hidden, config_.hidden},
            {base + "k_proj", config_.hidden, config_.hidden},
            {base + "v_proj", config_.hidden, config_.hidden},
        }};
        auto qkv = matvec_group(qkv_projections, input, layer, phase);
        auto q = short_conv(std::move(qkv[0]),
                            state.conv_q, tensor(base + "q_conv"));
        auto k = short_conv(std::move(qkv[1]),
                            state.conv_k, tensor(base + "k_conv"));
        auto v = short_conv(std::move(qkv[2]),
                            state.conv_v, tensor(base + "v_conv"));
        const auto ones = Vector(config_.kda_dim, 1.0F);
        const auto q_scale = 1.0F / static_cast<float>(config_.kda_dim);
        const auto k_scale = 1.0F / std::sqrt(static_cast<float>(config_.kda_dim));
        for (std::size_t head = 0; head < config_.kda_heads; ++head) {
            auto q_head = normalized(std::span(q).subspan(head * config_.kda_dim, config_.kda_dim), ones, 1e-6F);
            auto k_head = normalized(std::span(k).subspan(head * config_.kda_dim, config_.kda_dim), ones, 1e-6F);
            for (std::size_t channel = 0; channel < config_.kda_dim; ++channel) {
                q[head * config_.kda_dim + channel] = q_head[channel] * q_scale;
                k[head * config_.kda_dim + channel] = k_head[channel] * k_scale;
            }
        }
        const auto f_a = matvec(base + "f_a_proj", config_.kda_dim,
                                  config_.hidden, input, layer, phase);
        const auto forget = matvec(base + "f_b_proj", config_.hidden,
                                   config_.kda_dim, f_a, layer, phase);
        const auto beta_raw = matvec(base + "b_proj", config_.kda_heads,
                                     config_.hidden, input, layer, phase);
        const auto& a_log = tensor(base + "a_log");
        const auto& dt_bias = tensor(base + "dt_bias");
        Vector recurrent_output(config_.hidden);
        for (std::size_t head = 0; head < config_.kda_heads; ++head) {
            const auto beta = sigmoid(beta_raw[head]);
            auto* matrix = state.recurrent.data() + head * config_.kda_dim * config_.kda_dim;
            for (std::size_t row = 0; row < config_.kda_dim; ++row) {
                const auto decay = std::exp(config_.kda_lower * sigmoid(
                    std::exp(a_log[head]) * (forget[head * config_.kda_dim + row] +
                                              dt_bias[head * config_.kda_dim + row])));
                for (std::size_t column = 0; column < config_.kda_dim; ++column) matrix[row * config_.kda_dim + column] *= decay;
            }
            Vector prediction(config_.kda_dim, 0.0F);
            for (std::size_t column = 0; column < config_.kda_dim; ++column)
                for (std::size_t row = 0; row < config_.kda_dim; ++row)
                    prediction[column] += k[head * config_.kda_dim + row] * matrix[row * config_.kda_dim + column];
            for (std::size_t row = 0; row < config_.kda_dim; ++row)
                for (std::size_t column = 0; column < config_.kda_dim; ++column)
                    matrix[row * config_.kda_dim + column] += k[head * config_.kda_dim + row] *
                        (v[head * config_.kda_dim + column] - prediction[column]) * beta;
            for (std::size_t column = 0; column < config_.kda_dim; ++column)
                for (std::size_t row = 0; row < config_.kda_dim; ++row)
                    recurrent_output[head * config_.kda_dim + column] +=
                        q[head * config_.kda_dim + row] * matrix[row * config_.kda_dim + column];
        }
        const auto& o_norm = tensor(base + "o_norm");
        for (std::size_t head = 0; head < config_.kda_heads; ++head) {
            auto head_output = normalized(std::span(recurrent_output).subspan(head * config_.kda_dim, config_.kda_dim),
                                          o_norm, config_.epsilon);
            std::copy(head_output.begin(), head_output.end(), recurrent_output.begin() + head * config_.kda_dim);
        }
        const auto gate = matvec(base + "g_proj", config_.hidden,
                                 config_.hidden, input, layer, phase);
        for (std::size_t index = 0; index < recurrent_output.size(); ++index) recurrent_output[index] *= sigmoid(gate[index]);
        return matvec(base + "o_proj", config_.hidden, config_.hidden,
                      recurrent_output, layer, phase);
    }

    Vector mla(const Vector& input, std::size_t layer, MlaState& state,
               ProfilePhase phase) {
        const auto base = layer_name(layer, "attention.");
        auto q_latent = matvec(base + "q_a_proj", config_.q_rank,
                               config_.hidden, input, layer, phase);
        q_latent = normalized(q_latent, tensor(base + "q_a_norm"), config_.epsilon);
        const auto query_width = config_.q_main + config_.q_extra;
        const auto query = matvec(base + "q_b_proj",
                                  config_.mla_heads * query_width,
                                  config_.q_rank, q_latent, layer, phase);
        const auto compressed = matvec(base + "kv_a_proj",
                                       config_.kv_rank + config_.q_extra,
                                       config_.hidden, input, layer, phase);
        Vector latent(compressed.begin(), compressed.begin() + config_.kv_rank);
        latent = normalized(latent, tensor(base + "kv_a_norm"), config_.epsilon);
        const auto expanded = matvec(base + "kv_b_proj",
                                     config_.mla_heads * (config_.q_main + config_.value_dim),
                                     config_.kv_rank, latent, layer, phase);
        const auto old_length = state.length++;
        state.keys.resize(state.length * config_.mla_heads * config_.q_main);
        state.values.resize(state.length * config_.mla_heads * config_.value_dim);
        state.shared_keys.insert(state.shared_keys.end(), compressed.end() - config_.q_extra, compressed.end());
        for (std::size_t head = 0; head < config_.mla_heads; ++head) {
            const auto expanded_base = head * (config_.q_main + config_.value_dim);
            std::copy_n(expanded.begin() + expanded_base, config_.q_main,
                        state.keys.begin() + (old_length * config_.mla_heads + head) * config_.q_main);
            std::copy_n(expanded.begin() + expanded_base + config_.q_main, config_.value_dim,
                        state.values.begin() + (old_length * config_.mla_heads + head) * config_.value_dim);
        }
        Vector merged(config_.mla_heads * config_.value_dim);
        const auto scale = 1.0F / std::sqrt(static_cast<float>(query_width));
        for (std::size_t head = 0; head < config_.mla_heads; ++head) {
            Vector scores(state.length);
            for (std::size_t position = 0; position < state.length; ++position) {
                double score = 0.0;
                for (std::size_t index = 0; index < config_.q_main; ++index)
                    score += query[head * query_width + index] *
                             state.keys[(position * config_.mla_heads + head) * config_.q_main + index];
                for (std::size_t index = 0; index < config_.q_extra; ++index)
                    score += query[head * query_width + config_.q_main + index] *
                             state.shared_keys[position * config_.q_extra + index];
                scores[position] = static_cast<float>(score) * scale;
            }
            const auto maximum = *std::max_element(scores.begin(), scores.end());
            float denominator = 0.0F;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (std::size_t position = 0; position < state.length; ++position)
                for (std::size_t index = 0; index < config_.value_dim; ++index)
                    merged[head * config_.value_dim + index] += scores[position] / denominator *
                        state.values[(position * config_.mla_heads + head) * config_.value_dim + index];
        }
        const auto gate = matvec(base + "g_proj", merged.size(),
                                 config_.hidden, input, layer, phase);
        for (std::size_t index = 0; index < merged.size(); ++index) merged[index] *= sigmoid(gate[index]);
        return matvec(base + "o_proj", config_.hidden, merged.size(),
                      merged, layer, phase);
    }

    Vector activated_mlp(const Vector& input, const std::string& base,
                         std::size_t intermediate, std::size_t layer,
                         ProfilePhase phase) {
        if (backend_.options().cuda_boundary == CudaBoundaryMode::ffn_block ||
            backend_.options().cuda_boundary == CudaBoundaryMode::moe_layer) {
            const DenseMlpView weights{
                dense_weight(base + ".gate", intermediate, input.size()),
                dense_weight(base + ".up", intermediate, input.size()),
                dense_weight(base + ".down", config_.hidden, intermediate),
            };
            auto output = backend_.dense_situ_mlp(
                input, weights, config_.situ_beta, config_.situ_linear,
                static_cast<std::uint32_t>(layer), phase);
            if (!output) throw std::runtime_error("dense FFN backend failure");
            return std::move(output.value());
        }
        const std::array<DenseProjection, 2> projections{{
            {base + ".gate", intermediate, input.size()},
            {base + ".up", intermediate, input.size()},
        }};
        auto gate_up = matvec_group(projections, input, layer, phase);
        Vector activated(intermediate);
        situ_glu(activated, gate_up[0], gate_up[1], config_.situ_beta,
                 config_.situ_linear);
        return matvec(base + ".down", config_.hidden, intermediate,
                      activated, layer, phase);
    }

    Vector dense(const Vector& input, std::size_t layer, ProfilePhase phase) {
        return activated_mlp(input, layer_name(layer, "feed_forward"),
                             config_.dense_intermediate, layer, phase);
    }

    Vector expert_payload(const Vector& input, std::size_t layer,
                          const ExpertPayloadHandle& payload,
                          ProfilePhase phase) {
        const std::array<Mxfp4WeightView, 2> gate_up_views{{
            payload->gate.view(config_.group_size),
            payload->up.view(config_.group_size),
        }};
        auto gate_up = backend_.mxfp4_matvec_group(
            input, gate_up_views, static_cast<std::uint32_t>(layer), phase);
        if (!gate_up) throw std::runtime_error("invalid expert");
        Vector activated(config_.expert_intermediate);
        situ_glu(activated, gate_up.value()[0], gate_up.value()[1],
                 config_.situ_beta, config_.situ_linear);
        auto output = backend_.mxfp4_matvec(
            activated, payload->down.view(config_.group_size),
            static_cast<std::uint32_t>(layer), phase);
        if (!output) throw std::runtime_error("invalid expert");
        return output.value();
    }

    Vector expert(const Vector& input, std::size_t layer, std::size_t expert_id,
                  ProfilePhase phase) {
        return expert_payload(
            input, layer, load_expert(layer, expert_id), phase);
    }

    Vector shared_expert(const Vector& input, std::size_t layer,
                         ProfilePhase phase) {
        const auto base = layer_name(layer, "feed_forward.");
        if (backend_.options().cuda_boundary == CudaBoundaryMode::ffn_block) {
            const DenseMlpView shared_weights{
                dense_weight(base + "shared_gate", config_.expert_intermediate,
                             config_.hidden),
                dense_weight(base + "shared_up", config_.expert_intermediate,
                             config_.hidden),
                dense_weight(base + "shared_down", config_.hidden,
                             config_.expert_intermediate),
            };
            auto shared_result = backend_.dense_situ_mlp(
                input, shared_weights, config_.situ_beta, config_.situ_linear,
                static_cast<std::uint32_t>(layer), phase);
            if (!shared_result) {
                throw std::runtime_error("shared FFN backend failure");
            }
            return std::move(shared_result.value());
        }
        const std::array<DenseProjection, 2> shared_projections{{
            {base + "shared_gate", config_.expert_intermediate, config_.hidden},
            {base + "shared_up", config_.expert_intermediate, config_.hidden},
        }};
        auto shared_gate_up = matvec_group(
            shared_projections, input, layer, phase);
        Vector shared_activated(config_.expert_intermediate);
        situ_glu(shared_activated, shared_gate_up[0], shared_gate_up[1],
                 config_.situ_beta, config_.situ_linear);
        return matvec(base + "shared_down", config_.hidden,
                      config_.expert_intermediate, shared_activated,
                      layer, phase);
    }

    void record_routing_decision(const RoutingDecision& decision) {
        ++routing_decisions_;
        routing_selected_experts_ += decision.selected_k;
        routing_normalized_entropy_sum_ += decision.normalized_entropy;
        routing_selected_mass_sum_ += decision.selected_cumulative_mass;
        routing_boundary_confidence_sum_ += decision.boundary_confidence;
        if (decision.quality_floor_escalated) {
            ++routing_quality_escalated_decisions_;
        }
    }

    Vector moe(const Vector& input, std::size_t layer, ProfilePhase phase) {
        const auto base = layer_name(layer, "feed_forward.");
        const auto scores_raw = matvec(base + "router_weight",
                                       config_.experts, config_.hidden, input,
                                       layer, phase);
        const auto& bias = tensor(base + "correction_bias");
        Vector scores(config_.experts);
        for (std::size_t index = 0; index < scores.size(); ++index) scores[index] = sigmoid(scores_raw[index]);
        auto routing = select_routing(
            scores, bias, config_.top_k, session_.options().routing_policy);
        if (!routing) throw std::runtime_error(routing.message());
        const auto& decision = routing.value();
        const auto& order = decision.expert_ids;
        const auto selected_k = decision.selected_k;
        record_routing_decision(decision);
        Vector latent;
        Vector mixed(config_.latent, 0.0F);
        std::vector<ExpertKey> selected;
        selected.reserve(selected_k);
        for (std::size_t slot = 0; slot < selected_k; ++slot) {
            selected.push_back({layer, order[slot]});
            if (session_.options().l1_expert_cache !=
                    L1ExpertCacheMode::disabled &&
                !expert_store_.contains(selected.back())) {
                ++cold_rescue_count_;
            }
        }
        expert_store_.begin_access_set(active_forward_cycle_, layer, selected);
        if (trace_routing_) {
            routed_k_.push_back(static_cast<std::uint32_t>(selected_k));
            for (std::size_t slot = 0; slot < selected_k; ++slot) {
                routed_experts_.push_back(
                    static_cast<std::uint32_t>(order[slot]));
            }
        }
        std::vector<ExpertLoadTicket> expert_tickets;
        if (expert_loader_) {
            const auto counters = reader_.counters();
            const auto estimate = counters.batch_submissions
                ? std::chrono::nanoseconds{
                      counters.storage_nanoseconds / counters.batch_submissions}
                : std::chrono::nanoseconds{0};
            expert_tickets.resize(selected_k);
            std::vector<bool> resident(selected_k);
            for (std::size_t slot = 0; slot < selected_k; ++slot) {
                resident[slot] = expert_store_.contains(
                    {layer, order[slot]});
            }
            for (const bool resident_pass : {true, false}) {
                for (std::size_t slot = 0; slot < selected_k; ++slot) {
                    if (resident[slot] == resident_pass) {
                        expert_tickets[slot] = schedule_expert(
                            layer, order[slot], resident[slot], estimate);
                    }
                }
            }
        }
        if (backend_.options().cuda_transfer ==
                CudaTransferMode::synchronous &&
            backend_.options().cuda_boundary != CudaBoundaryMode::moe_layer) {
            latent = matvec(base + "routed_down_proj",
                            config_.latent, config_.hidden, input,
                            layer, phase);
        }
        Vector shared;
        if (expert_loader_ &&
            backend_.options().cuda_boundary != CudaBoundaryMode::moe_layer) {
            shared = shared_expert(input, layer, phase);
        }
        if (backend_.options().cuda_boundary == CudaBoundaryMode::ffn_block ||
            backend_.options().cuda_boundary == CudaBoundaryMode::moe_layer) {
            std::vector<ExpertPayloadHandle> payloads;
            payloads.reserve(selected_k);
            for (std::size_t slot = 0; slot < selected_k; ++slot) {
                if (expert_loader_) {
                    auto payload = expert_tickets[slot].wait();
                    if (!payload) throw std::runtime_error("missing expert");
                    payloads.push_back(std::move(payload.value()));
                } else {
                    payloads.push_back(load_expert(layer, order[slot]));
                }
            }
            std::vector<Mxfp4MlpView> expert_views;
            expert_views.reserve(payloads.size());
            for (const auto& payload : payloads) {
                expert_views.push_back(payload->view(config_.group_size));
            }
            std::vector<float> contributions;
            contributions.reserve(selected_k);
            for (std::size_t slot = 0; slot < selected_k; ++slot) {
                contributions.push_back(
                    decision.normalized_weights[slot] * config_.routed_scale);
            }
            if (backend_.options().cuda_boundary ==
                CudaBoundaryMode::moe_layer) {
                const ResidentMoeLayerView layer_weights{
                    dense_weight(base + "routed_down_proj", config_.latent,
                                 config_.hidden),
                    {fnv1a64((base + "routed_norm").c_str()),
                     tensor(base + "routed_norm")},
                    dense_weight(base + "routed_up_proj", config_.hidden,
                                 config_.latent),
                    {
                        dense_weight(base + "shared_gate",
                                     config_.expert_intermediate,
                                     config_.hidden),
                        dense_weight(base + "shared_up",
                                     config_.expert_intermediate,
                                     config_.hidden),
                        dense_weight(base + "shared_down", config_.hidden,
                                     config_.expert_intermediate),
                    },
                };
                auto layer_result = backend_.resident_mxfp4_moe_layer(
                    input, layer_weights, expert_views, contributions,
                    config_.epsilon, config_.situ_beta, config_.situ_linear,
                    static_cast<std::uint32_t>(layer), phase);
                if (!layer_result) {
                    throw std::runtime_error(
                        "resident MoE layer backend failure");
                }
                if (layer_result.value().executed) {
                    return std::move(layer_result.value().output);
                }
                latent = matvec(base + "routed_down_proj", config_.latent,
                                config_.hidden, input, layer, phase);
                if (expert_loader_) {
                    shared = shared_expert(input, layer, phase);
                }
            }
            if (backend_.options().cuda_moe_fusion ==
                CudaMoeFusionMode::routed_accumulate) {
                Result<std::vector<float>> fused_output =
                    Result<std::vector<float>>::failure(
                        ErrorCode::backend_unavailable);
                if (backend_.options().cuda_transfer ==
                    CudaTransferMode::prefetch) {
                    auto token = backend_.prefetch_mxfp4_situ_mlp_group(
                        expert_views, next_prefetch_sequence_++,
                        static_cast<std::uint32_t>(layer), phase);
                    if (!token) {
                        throw std::runtime_error(
                            "expert prefetch backend failure");
                    }
                    latent = matvec(base + "routed_down_proj",
                                    config_.latent, config_.hidden, input,
                                    layer, phase);
                    fused_output = backend_.mxfp4_situ_moe_prepared(
                        latent, token.value(), contributions,
                        config_.situ_beta, config_.situ_linear,
                        static_cast<std::uint32_t>(layer), phase);
                } else {
                    fused_output = backend_.mxfp4_situ_moe(
                        latent, expert_views, contributions,
                        config_.situ_beta, config_.situ_linear,
                        static_cast<std::uint32_t>(layer), phase);
                }
                if (!fused_output) {
                    throw std::runtime_error(
                        "fused expert FFN backend failure");
                }
                mixed = std::move(fused_output.value());
            } else {
                Result<std::vector<std::vector<float>>> expert_outputs =
                    Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable);
                if (backend_.options().cuda_transfer ==
                    CudaTransferMode::prefetch) {
                    auto token = backend_.prefetch_mxfp4_situ_mlp_group(
                        expert_views, next_prefetch_sequence_++,
                        static_cast<std::uint32_t>(layer), phase);
                    if (!token) {
                        throw std::runtime_error(
                            "expert prefetch backend failure");
                    }
                    latent = matvec(base + "routed_down_proj",
                                    config_.latent, config_.hidden, input,
                                    layer, phase);
                    expert_outputs =
                        backend_.mxfp4_situ_mlp_group_prepared(
                            latent, token.value(), config_.situ_beta,
                            config_.situ_linear,
                            static_cast<std::uint32_t>(layer), phase);
                } else {
                    expert_outputs = backend_.options().cuda_batching ==
                            CudaBatchingMode::resident_grid
                        ? backend_.mxfp4_situ_mlp_grid(
                              latent, 1, expert_views, config_.situ_beta,
                              config_.situ_linear,
                              static_cast<std::uint32_t>(layer), phase)
                        : backend_.mxfp4_situ_mlp_group(
                              latent, expert_views, config_.situ_beta,
                              config_.situ_linear,
                              static_cast<std::uint32_t>(layer), phase);
                }
                if (!expert_outputs) {
                    throw std::runtime_error("expert FFN backend failure");
                }
                for (std::size_t slot = 0; slot < selected_k; ++slot) {
                    for (std::size_t index = 0; index < mixed.size(); ++index) {
                        mixed[index] += contributions[slot] *
                            expert_outputs.value()[slot][index];
                    }
                }
            }
        } else {
            for (std::size_t slot = 0; slot < selected_k; ++slot) {
                Vector expert_output;
                if (expert_loader_) {
                    auto payload = expert_tickets[slot].wait();
                    if (!payload) throw std::runtime_error("missing expert");
                    expert_output = expert_payload(
                        latent, layer, payload.value(), phase);
                } else {
                    expert_output = expert(
                        latent, layer, order[slot], phase);
                }
                const auto weight = decision.normalized_weights[slot] *
                                    config_.routed_scale;
                for (std::size_t index = 0; index < mixed.size(); ++index) {
                    mixed[index] += weight * expert_output[index];
                }
            }
        }
        const auto routed_norm = normalized(mixed, tensor(base + "routed_norm"), config_.epsilon);
        auto output = matvec(base + "routed_up_proj", config_.hidden,
                             config_.latent, routed_norm, layer, phase);
        if (!expert_loader_) {
            shared = shared_expert(input, layer, phase);
        }
        for (std::size_t index = 0; index < output.size(); ++index) output[index] += shared[index];
        return output;
    }

    Reader& reader_;
    ComputeBackend& backend_;
    Config config_;
    bool trace_routing_{};
    RuntimeSession& session_;
    HostExpertStore& expert_store_;
    DeadlineExpertLoader* expert_loader_{};
    ModelState state_template_;
    std::unordered_map<std::uint64_t, Vector> tensors_;
    std::vector<std::uint64_t> layer_nanoseconds_ = std::vector<std::uint64_t>(config_.layers);
    std::vector<std::uint32_t> routed_experts_;
    std::vector<std::uint32_t> routed_k_;
    std::uint64_t routing_decisions_{};
    std::uint64_t routing_selected_experts_{};
    std::uint64_t routing_quality_escalated_decisions_{};
    std::uint64_t cold_rescue_count_{};
    double routing_normalized_entropy_sum_{};
    double routing_selected_mass_sum_{};
    double routing_boundary_confidence_sum_{};
    std::uint64_t active_forward_cycle_{};
    std::uint64_t next_prefetch_sequence_{1};
};

std::uint32_t argmax(const Vector& values) {
    return static_cast<std::uint32_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}
}

struct IncrementalDraftCursor::Impl {
    struct MlaMark {
        std::size_t length{};
        std::size_t keys{};
        std::size_t values{};
        std::size_t shared_keys{};
    };

    struct Checkpoint {
        std::vector<KdaState> kda;
        MlaMark mla;
        Vector logits;
    };

    Impl(Reader& reader, ComputeBackend& backend, RuntimeSession& session)
        : engine(reader, backend, session), state(engine.empty_state()),
          session(session) {}

    std::uint64_t kda_bytes() const {
        std::uint64_t elements = 0;
        for (const auto& item : state.kda) {
            for (const auto* values : {
                     &item.conv_q, &item.conv_k,
                     &item.conv_v, &item.recurrent}) {
                if (values->size() >
                    (std::numeric_limits<std::uint64_t>::max() - elements)) {
                    throw std::overflow_error("KDA checkpoint size overflow");
                }
                elements += values->size();
            }
        }
        if (elements >
            std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
            throw std::overflow_error("KDA checkpoint byte overflow");
        }
        return elements * sizeof(float);
    }
    Checkpoint capture() {
        const auto bytes = kda_bytes();
        if (stats.kda_checkpoint_bytes >
            std::numeric_limits<std::uint64_t>::max() - bytes) {
            throw std::overflow_error("KDA checkpoint counter overflow");
        }
        stats.kda_checkpoint_bytes += bytes;
        return {
            .kda = state.kda,
            .mla = {
                .length = state.mla.length,
                .keys = state.mla.keys.size(),
                .values = state.mla.values.size(),
                .shared_keys = state.mla.shared_keys.size(),
            },
            .logits = logits,
        };
    }

    void restore(const Checkpoint& checkpoint) {
        if (checkpoint.mla.length > state.mla.length ||
            checkpoint.mla.keys > state.mla.keys.size() ||
            checkpoint.mla.values > state.mla.values.size() ||
            checkpoint.mla.shared_keys > state.mla.shared_keys.size()) {
            throw std::runtime_error("invalid incremental draft checkpoint");
        }
        stats.mla_positions_cropped +=
            state.mla.length - checkpoint.mla.length;
        state.kda = checkpoint.kda;
        state.mla.keys.resize(checkpoint.mla.keys);
        state.mla.values.resize(checkpoint.mla.values);
        state.mla.shared_keys.resize(checkpoint.mla.shared_keys);
        state.mla.length = checkpoint.mla.length;
        logits = checkpoint.logits;
        ++stats.rollback_events;
    }

    Engine engine;
    ModelState state;
    Vector logits;
    RuntimeSession& session;
    IncrementalDraftCursorStats stats{};
    std::vector<std::uint32_t> pending_candidates;
    std::unique_ptr<Checkpoint> base_checkpoint;
    std::vector<Checkpoint> processed_checkpoints;
    bool pending{};
    bool failed{};
};

IncrementalDraftCursor::IncrementalDraftCursor(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

IncrementalDraftCursor::~IncrementalDraftCursor() = default;

Result<std::unique_ptr<IncrementalDraftCursor>>
IncrementalDraftCursor::create(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> context,
    RuntimeSession& session) {
    auto generation_guard = session.acquire_generation_guard();
    if (reader.superblock().optional_features & optional_storage_fixture) {
        return Result<std::unique_ptr<IncrementalDraftCursor>>::failure(
            ErrorCode::non_executable_artifact);
    }
    if (!session.options().incremental || context.empty()) {
        return Result<std::unique_ptr<IncrementalDraftCursor>>::failure(
            ErrorCode::invalid_state, "invalid incremental draft context");
    }
    try {
        auto impl = std::make_unique<Impl>(reader, backend, session);
        for (const auto token : context) {
            impl->logits = impl->engine.forward(
                token, impl->state, ProfilePhase::prefill);
        }
        impl->stats.context_prefill_tokens = context.size();
        return Result<std::unique_ptr<IncrementalDraftCursor>>::success(
            std::unique_ptr<IncrementalDraftCursor>(
                new IncrementalDraftCursor(std::move(impl))));
    } catch (const std::exception& error) {
        if (auto* loader = session.expert_loader()) loader->wait_idle();
        return Result<std::unique_ptr<IncrementalDraftCursor>>::failure(
            ErrorCode::invalid_extent, error.what());
    }
}

Result<std::vector<std::uint32_t>> IncrementalDraftCursor::propose(
    std::size_t count) {
    auto generation_guard = impl_->session.acquire_generation_guard();
    if (impl_->failed || impl_->pending || impl_->logits.empty()) {
        return Result<std::vector<std::uint32_t>>::failure(
            ErrorCode::invalid_state, "invalid incremental draft lifecycle");
    }
    try {
        std::vector<std::uint32_t> candidates;
        candidates.reserve(count);
        impl_->processed_checkpoints.clear();
        impl_->base_checkpoint =
            std::make_unique<Impl::Checkpoint>(impl_->capture());
        for (std::size_t index = 0; index < count; ++index) {
            const auto token = argmax(impl_->logits);
            candidates.push_back(token);
            if (index + 1 < count) {
                impl_->logits = impl_->engine.forward(
                    token, impl_->state, ProfilePhase::decode);
                ++impl_->stats.incremental_forward_calls;
                impl_->processed_checkpoints.push_back(impl_->capture());
            }
        }
        impl_->pending_candidates = candidates;
        impl_->pending = true;
        return Result<std::vector<std::uint32_t>>::success(
            std::move(candidates));
    } catch (const std::exception& error) {
        impl_->failed = true;
        if (auto* loader = impl_->session.expert_loader()) {
            loader->wait_idle();
        }
        return Result<std::vector<std::uint32_t>>::failure(
            ErrorCode::invalid_extent, error.what());
    }
}

Result<bool> IncrementalDraftCursor::commit(
    std::size_t accepted_prefix,
    std::span<const std::uint32_t> committed_tokens) {
    auto generation_guard = impl_->session.acquire_generation_guard();
    bool valid = !impl_->failed && impl_->pending &&
        impl_->base_checkpoint &&
        accepted_prefix <= impl_->pending_candidates.size() &&
        committed_tokens.size() == accepted_prefix + 1;
    for (std::size_t index = 0; valid && index < accepted_prefix; ++index) {
        valid = committed_tokens[index] == impl_->pending_candidates[index];
    }
    if (!valid) {
        impl_->failed = true;
        impl_->pending = false;
        return Result<bool>::failure(
            ErrorCode::invalid_state, "invalid incremental draft commit");
    }

    try {
        const auto processed_candidates = impl_->pending_candidates.empty()
            ? std::size_t{0}
            : impl_->pending_candidates.size() - 1;
        if (accepted_prefix < processed_candidates) {
            const auto& checkpoint = accepted_prefix == 0
                ? *impl_->base_checkpoint
                : impl_->processed_checkpoints[accepted_prefix - 1];
            impl_->restore(checkpoint);
        }
        if (accepted_prefix == impl_->pending_candidates.size() &&
            accepted_prefix != 0) {
            impl_->logits = impl_->engine.forward(
                impl_->pending_candidates.back(), impl_->state,
                ProfilePhase::decode);
            ++impl_->stats.incremental_forward_calls;
        }
        impl_->logits = impl_->engine.forward(
            committed_tokens.back(), impl_->state, ProfilePhase::decode);
        ++impl_->stats.incremental_forward_calls;
        impl_->pending = false;
        impl_->pending_candidates.clear();
        impl_->processed_checkpoints.clear();
        impl_->base_checkpoint.reset();
        return Result<bool>::success(true);
    } catch (const std::exception& error) {
        impl_->failed = true;
        impl_->pending = false;
        if (auto* loader = impl_->session.expert_loader()) {
            loader->wait_idle();
        }
        return Result<bool>::failure(ErrorCode::invalid_extent, error.what());
    }
}

IncrementalDraftCursorStats IncrementalDraftCursor::stats() const noexcept {
    auto result = impl_->stats;
    result.routing_decisions = impl_->engine.routing_decisions();
    result.routing_selected_experts =
        impl_->engine.routing_selected_experts();
    return result;
}

IncrementalDraftCursorDiagnostics
IncrementalDraftCursor::diagnostics() const {
    auto generation_guard = impl_->session.acquire_generation_guard();
    return {
        .flattened_state = impl_->engine.flatten_state(impl_->state),
        .mla_length = impl_->state.mla.length,
        .mla_key_elements = impl_->state.mla.keys.size(),
        .mla_value_elements = impl_->state.mla.values.size(),
        .mla_shared_key_elements = impl_->state.mla.shared_keys.size(),
    };
}

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeSession& session) {
    auto generation_guard = session.acquire_generation_guard();
    if (reader.superblock().optional_features & optional_storage_fixture) {
        return Result<GenerationResult>::failure(
            ErrorCode::non_executable_artifact);
    }
    const auto& options = session.options();
    if (prompt.empty()) return Result<GenerationResult>::failure(ErrorCode::invalid_extent, "empty prompt");
    try {
        Engine engine(reader, backend, session);
        GenerationResult result;
        if (options.incremental) {
            auto state = engine.empty_state();
            Vector logits;
            if (options.diagnostics) result.prefill_layer_outputs.resize(engine.layer_nanoseconds().size());
            const auto prefill_start = std::chrono::steady_clock::now();
            for (const auto token : prompt) {
                std::vector<Vector> token_layers;
                logits = engine.forward(token, state, ProfilePhase::prefill,
                                        options.diagnostics ? &token_layers : nullptr);
                if (options.diagnostics) {
                    result.prefill_logits.insert(result.prefill_logits.end(), logits.begin(), logits.end());
                    for (std::size_t layer = 0; layer < token_layers.size(); ++layer) {
                        result.prefill_layer_outputs[layer].insert(
                            result.prefill_layer_outputs[layer].end(),
                            token_layers[layer].begin(), token_layers[layer].end());
                    }
                }
            }
            result.prefill_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - prefill_start).count();
        if (options.diagnostics) {
                result.prefill_state = engine.flatten_state(state);
                result.prefill_routed_experts = engine.routed_experts();
                result.prefill_routed_k = engine.routed_k();
            }
            if (count) {
                auto token = argmax(logits);
                result.token_ids.push_back(token);
                if (count > 1) {
                    const auto decode_start = std::chrono::steady_clock::now();
                    for (std::size_t index = 1; index < count; ++index) {
                        logits = engine.forward(token, state, ProfilePhase::decode);
                        ++result.target_decode_forward_calls;
                        token = argmax(logits);
                        result.token_ids.push_back(token);
                    }
                    result.decode_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - decode_start).count();
                }
            }
            if (options.diagnostics) {
                result.final_state = engine.flatten_state(state);
            }
        } else {
            std::vector<std::uint32_t> sequence(prompt.begin(), prompt.end());
            const auto decode_start = std::chrono::steady_clock::now();
            for (std::size_t generated = 0; generated < count; ++generated) {
                auto state = engine.empty_state();
                Vector logits;
                for (std::size_t index = 0; index < sequence.size(); ++index) {
                    const auto phase = index < prompt.size() ? ProfilePhase::prefill
                                                             : ProfilePhase::decode;
                    logits = engine.forward(sequence[index], state, phase);
                    if (phase == ProfilePhase::decode) {
                        ++result.target_decode_forward_calls;
                    }
                }
                const auto token = argmax(logits);
                result.token_ids.push_back(token);
                sequence.push_back(token);
            }
            result.decode_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - decode_start).count();
        }
        if (auto* loader = session.expert_loader()) loader->wait_idle();
        result.per_layer_nanoseconds = engine.layer_nanoseconds();
        result.l1_expert_cache = engine.expert_cache_stats();
        result.expert_load_scheduler =
            session.expert_load_scheduler_stats();
        if (options.diagnostics) {
            result.routed_experts = engine.routed_experts();
            result.routed_k = engine.routed_k();
        }
        engine.export_routing_stats(result);
        return Result<GenerationResult>::success(std::move(result));
    } catch (const std::exception& error) {
        if (auto* loader = session.expert_loader()) loader->wait_idle();
        return Result<GenerationResult>::failure(ErrorCode::invalid_extent, error.what());
    }
}

Result<GenerationResult> generate_speculative(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt, std::size_t count,
    RuntimeSession& session, DraftProvider& draft_provider,
    std::size_t block_size) {
    auto generation_guard = session.acquire_generation_guard();
    if (reader.superblock().optional_features & optional_storage_fixture) {
        return Result<GenerationResult>::failure(
            ErrorCode::non_executable_artifact);
    }
    const auto& options = session.options();
    if (!options.incremental || block_size == 0) {
        return Result<GenerationResult>::failure(ErrorCode::invalid_state);
    }
    const bool expert_major =
        options.speculative_verification ==
        SpeculativeVerificationMode::expert_major;
    const auto& backend_options = backend.options();
    const bool cuda_expert_major =
        backend.kind() == BackendKind::cuda_custom &&
        backend_options.cuda_boundary == CudaBoundaryMode::ffn_block &&
        backend_options.cuda_allocation == CudaAllocationMode::reused &&
        backend_options.cuda_weights == CudaWeightMode::transient &&
        backend_options.cuda_transfer == CudaTransferMode::synchronous &&
        backend_options.cuda_moe_fusion == CudaMoeFusionMode::none;
    if (expert_major &&
        ((backend.kind() != BackendKind::cpu && !cuda_expert_major) ||
         options.l1_expert_cache != L1ExpertCacheMode::disabled ||
         options.l2_expert_schedule != L2ExpertScheduleMode::blocking ||
         options.routing_policy.mode != RoutingMode::natural ||
         options.profile_observation ||
         decode_config(reader.model_config()).layers != 4)) {
        return Result<GenerationResult>::failure(
            ErrorCode::invalid_state,
            "unsupported expert-major runtime combination");
    }
    if (prompt.empty()) {
        return Result<GenerationResult>::failure(
            ErrorCode::invalid_extent, "empty prompt");
    }
    try {
        Engine engine(reader, backend, session);
        GenerationResult result;
        auto state = engine.empty_state();
        Vector logits;
        if (options.diagnostics) {
            result.prefill_layer_outputs.resize(
                engine.layer_nanoseconds().size());
        }
        const auto prefill_start = std::chrono::steady_clock::now();
        for (const auto token : prompt) {
            std::vector<Vector> token_layers;
            logits = engine.forward(
                token, state, ProfilePhase::prefill,
                options.diagnostics ? &token_layers : nullptr);
            if (options.diagnostics) {
                result.prefill_logits.insert(
                    result.prefill_logits.end(), logits.begin(), logits.end());
                for (std::size_t layer = 0; layer < token_layers.size();
                     ++layer) {
                    result.prefill_layer_outputs[layer].insert(
                        result.prefill_layer_outputs[layer].end(),
                        token_layers[layer].begin(),
                        token_layers[layer].end());
                }
            }
        }
        result.prefill_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - prefill_start).count();
        if (options.diagnostics) {
            result.prefill_state = engine.flatten_state(state);
            result.prefill_routed_experts = engine.routed_experts();
            result.prefill_routed_k = engine.routed_k();
        }

        if (count != 0) {
            result.token_ids.push_back(argmax(logits));
            const auto decode_start = std::chrono::steady_clock::now();
            while (result.token_ids.size() < count) {
                const auto remaining = count - result.token_ids.size();
                const auto max_draft_tokens =
                    std::min(block_size, remaining - 1);
                const DraftRequest request{
                    .anchor_token = result.token_ids.back(),
                    .max_draft_tokens = max_draft_tokens,
                    .generated_position = result.token_ids.size(),
                    .generated_tokens = result.token_ids,
                };
                auto proposal = draft_provider.propose(request);
                if (!proposal) {
                    if (auto* loader = session.expert_loader()) {
                        loader->wait_idle();
                    }
                    return Result<GenerationResult>::failure(
                        proposal.error(), proposal.message());
                }
                if (proposal.value().anchor_token != request.anchor_token) {
                    if (auto* loader = session.expert_loader()) {
                        loader->wait_idle();
                    }
                    return Result<GenerationResult>::failure(
                        ErrorCode::invalid_state,
                        "draft anchor does not match committed token");
                }
                auto verification = Result<DraftVerification>::failure(
                    ErrorCode::invalid_state);
                std::size_t feedback_positions_evaluated = 0;
                std::size_t feedback_positions_discarded = 0;
                std::size_t feedback_payload_loads = 0;
                std::size_t feedback_assignments = 0;
                const auto target_calls_before =
                    result.target_decode_forward_calls;
                if (expert_major) {
                    if (proposal.value().candidate_tokens.size() >
                        max_draft_tokens) {
                        return Result<GenerationResult>::failure(
                            ErrorCode::invalid_extent);
                    }
                    for (const auto candidate :
                         proposal.value().candidate_tokens) {
                        if (candidate >= engine.vocabulary_size()) {
                            return Result<GenerationResult>::failure(
                                ErrorCode::invalid_extent);
                        }
                    }
                    std::vector<std::uint32_t> block_inputs;
                    block_inputs.reserve(
                        proposal.value().candidate_tokens.size() + 1);
                    block_inputs.push_back(proposal.value().anchor_token);
                    block_inputs.insert(
                        block_inputs.end(),
                        proposal.value().candidate_tokens.begin(),
                        proposal.value().candidate_tokens.end());
                    auto block = engine.forward_expert_major_block(
                        block_inputs, state, ProfilePhase::decode);
                    std::vector<std::uint32_t> target_tokens;
                    target_tokens.reserve(block.logits.size());
                    for (const auto& target_logits : block.logits) {
                        target_tokens.push_back(argmax(target_logits));
                    }
                    verification = verify_greedy_target_block(
                        proposal.value(), max_draft_tokens,
                        engine.vocabulary_size(), target_tokens);
                    if (verification) {
                        const auto committed_positions =
                            verification.value().committed_tokens.size();
                        if (committed_positions == 0 ||
                            committed_positions >
                                block.states_after_positions.size()) {
                            return Result<GenerationResult>::failure(
                                ErrorCode::invalid_state);
                        }
                        state = std::move(
                            block.states_after_positions[
                                committed_positions - 1]);
                        engine.commit_expert_major_routes(
                            block, committed_positions);
                        ++result.target_block_forward_calls;
                        result.target_positions_evaluated +=
                            block_inputs.size();
                        result.target_positions_discarded +=
                            block_inputs.size() - committed_positions;
                        result.target_decode_forward_calls +=
                            block_inputs.size();
                        result.expert_major_unique_experts_sum +=
                            block.unique_experts_sum;
                        result.expert_major_unique_experts_max =
                            std::max(
                                result.expert_major_unique_experts_max,
                                block.unique_experts_max);
                        result.expert_major_assignments += block.assignments;
                        result.expert_major_payload_loads +=
                            block.payload_loads;
                        result.expert_major_reused_assignments +=
                            block.assignments - block.payload_loads;
                        feedback_positions_evaluated = block_inputs.size();
                        feedback_positions_discarded =
                            block_inputs.size() - committed_positions;
                        feedback_payload_loads = block.payload_loads;
                        feedback_assignments = block.assignments;
                        if (options.diagnostics) {
                            for (std::size_t position = 0;
                                 position <
                                     block.routed_k_by_position.size();
                                 ++position) {
                                result.evaluated_routed_k.insert(
                                    result.evaluated_routed_k.end(),
                                    block.routed_k_by_position[position]
                                        .begin(),
                                    block.routed_k_by_position[position]
                                        .end());
                                result.evaluated_routed_experts.insert(
                                    result.evaluated_routed_experts.end(),
                                    block.routed_experts_by_position[position]
                                        .begin(),
                                    block.routed_experts_by_position[position]
                                        .end());
                            }
                        }
                    }
                } else {
                    verification = verify_greedy_draft(
                        proposal.value(), max_draft_tokens,
                        engine.vocabulary_size(),
                        [&](std::uint32_t input_token) {
                            auto target_logits = engine.forward(
                                input_token, state, ProfilePhase::decode);
                            ++result.target_decode_forward_calls;
                            return Result<std::uint32_t>::success(
                                argmax(target_logits));
                        });
                }
                if (!verification) {
                    if (auto* loader = session.expert_loader()) {
                        loader->wait_idle();
                    }
                    return Result<GenerationResult>::failure(
                        verification.error(), verification.message());
                }
                if (!expert_major) {
                    feedback_positions_evaluated =
                        result.target_decode_forward_calls -
                        target_calls_before;
                }
                verification.value().target_positions_evaluated =
                    feedback_positions_evaluated;
                verification.value().target_positions_discarded =
                    feedback_positions_discarded;
                verification.value().expert_major_payload_loads =
                    feedback_payload_loads;
                verification.value().expert_major_assignments =
                    feedback_assignments;
                draft_provider.update(verification.value());
                ++result.speculative_verification_blocks;
                result.speculative_proposed_draft_tokens +=
                    verification.value().proposed_draft_tokens;
                result.speculative_accepted_draft_tokens +=
                    verification.value().accepted_draft_tokens;
                result.speculative_committed_tokens +=
                    verification.value().committed_tokens.size();
                result.speculative_max_proposal_tokens = std::max(
                    result.speculative_max_proposal_tokens,
                    static_cast<std::uint64_t>(
                        verification.value().proposed_draft_tokens));
                result.token_ids.insert(
                    result.token_ids.end(),
                    verification.value().committed_tokens.begin(),
                    verification.value().committed_tokens.end());
            }
            result.decode_nanoseconds =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - decode_start).count();
        }
        if (options.diagnostics) {
            result.final_state = engine.flatten_state(state);
            result.routed_experts = engine.routed_experts();
            result.routed_k = engine.routed_k();
        }
        if (auto* loader = session.expert_loader()) loader->wait_idle();
        result.per_layer_nanoseconds = engine.layer_nanoseconds();
        result.l1_expert_cache = engine.expert_cache_stats();
        result.expert_load_scheduler =
            session.expert_load_scheduler_stats();
        engine.export_routing_stats(result);
        const auto draft_stats = draft_provider.stats();
        result.draft_proposal_calls = draft_stats.proposal_calls;
        result.draft_candidate_tokens = draft_stats.candidate_tokens;
        result.draft_replayed_context_tokens =
            draft_stats.replayed_context_tokens;
        result.draft_generation_nanoseconds =
            draft_stats.generation_nanoseconds;
        result.draft_reader_calls = draft_stats.reader_calls;
        result.draft_reader_bytes = draft_stats.reader_bytes;
        result.draft_routing_decisions = draft_stats.routing_decisions;
        result.draft_routing_selected_experts =
            draft_stats.routing_selected_experts;
        result.draft_selected_length_1 = draft_stats.selected_length_1;
        result.draft_selected_length_2 = draft_stats.selected_length_2;
        result.draft_selected_length_4 = draft_stats.selected_length_4;
        result.draft_scheduler_growths = draft_stats.scheduler_growths;
        result.draft_scheduler_backoffs = draft_stats.scheduler_backoffs;
        result.draft_context_prefill_tokens =
            draft_stats.context_prefill_tokens;
        result.draft_incremental_forward_calls =
            draft_stats.incremental_forward_calls;
        result.draft_rollback_events = draft_stats.rollback_events;
        result.draft_mla_positions_cropped =
            draft_stats.mla_positions_cropped;
        result.draft_kda_checkpoint_bytes =
            draft_stats.kda_checkpoint_bytes;
        return Result<GenerationResult>::success(std::move(result));
    } catch (const std::exception& error) {
        if (auto* loader = session.expert_loader()) loader->wait_idle();
        return Result<GenerationResult>::failure(
            ErrorCode::invalid_extent, error.what());
    }
}

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options) {
    RuntimeSession session(options);
    return generate_greedy(reader, backend, prompt, count, session);
}

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics) {
    return generate_greedy(reader, backend, prompt, count,
                           RuntimeOptions{incremental, diagnostics});
}

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics) {
    auto backend = make_cpu_backend();
    return generate_greedy(reader, *backend, prompt, count, incremental,
                           diagnostics);
}

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options) {
    auto backend = make_cpu_backend();
    return generate_greedy(reader, *backend, prompt, count, options);
}
}
