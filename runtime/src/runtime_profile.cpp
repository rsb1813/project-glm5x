// 런타임 task/session profile의 bounded 관측과 canonical 영속화를 구현합니다.
#include "k3x/runtime_profile.hpp"

#include "k3x/checksums.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <system_error>

namespace k3x {
namespace {
constexpr std::string_view profile_magic = "K3X_RUNTIME_PROFILE_V1";

template <typename Integer>
bool parse_integer(std::string_view text, Integer& value) {
    if (text.empty()) return false;
    const auto parsed = std::from_chars(
        text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} &&
           parsed.ptr == text.data() + text.size();
}

bool valid_utf8(std::string_view value) {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto lead = static_cast<unsigned char>(value[index]);
        std::size_t continuation = 0;
        std::uint32_t codepoint = 0;
        if (lead <= 0x7f) codepoint = lead;
        else if ((lead & 0xe0U) == 0xc0U) {
            continuation = 1;
            codepoint = lead & 0x1fU;
        } else if ((lead & 0xf0U) == 0xe0U) {
            continuation = 2;
            codepoint = lead & 0x0fU;
        } else if ((lead & 0xf8U) == 0xf0U) {
            continuation = 3;
            codepoint = lead & 0x07U;
        } else return false;
        if (index + continuation >= value.size()) return false;
        for (std::size_t offset = 1; offset <= continuation; ++offset) {
            const auto byte = static_cast<unsigned char>(value[index + offset]);
            if ((byte & 0xc0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (byte & 0x3fU);
        }
        if ((continuation == 1 && codepoint < 0x80U) ||
            (continuation == 2 && codepoint < 0x800U) ||
            (continuation == 3 && codepoint < 0x10000U) ||
            codepoint > 0x10ffffU ||
            (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint < 0x20U || codepoint == 0x7fU) return false;
        index += continuation + 1;
    }
    return !value.empty();
}

bool valid_key(std::string_view key) {
    return !key.empty() &&
           key.size() <= RuntimeProfile::maximum_metadata_key_bytes &&
           std::all_of(key.begin(), key.end(), [](char character) {
               return (character >= 'A' && character <= 'Z') ||
                      (character >= '0' && character <= '9') ||
                      character == '_';
           });
}

std::string hex_encode(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string encoded(value.size() * 2, '0');
    for (std::size_t index = 0; index < value.size(); ++index) {
        const auto byte = static_cast<unsigned char>(value[index]);
        encoded[index * 2] = digits[byte >> 4U];
        encoded[index * 2 + 1] = digits[byte & 0x0fU];
    }
    return encoded;
}

Result<std::string> hex_decode(std::string_view value) {
    if (value.size() % 2 != 0) {
        return Result<std::string>::failure(
            ErrorCode::invalid_state, "odd metadata hex length");
    }
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        return -1;
    };
    std::string decoded(value.size() / 2, '\0');
    for (std::size_t index = 0; index < decoded.size(); ++index) {
        const auto high = nibble(value[index * 2]);
        const auto low = nibble(value[index * 2 + 1]);
        if (high < 0 || low < 0) {
            return Result<std::string>::failure(
                ErrorCode::invalid_state, "invalid metadata hex");
        }
        decoded[index] = static_cast<char>((high << 4) | low);
    }
    return Result<std::string>::success(std::move(decoded));
}

std::span<const std::byte> bytes_of(std::string_view value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

template <typename Map, typename Key>
bool bounded_increment(Map& counts, const Key& key, std::size_t limit) {
    if (auto found = counts.find(key); found != counts.end()) {
        if (found->second != std::numeric_limits<std::uint64_t>::max()) {
            ++found->second;
            return true;
        }
    } else if (counts.size() < limit) {
        counts.emplace(key, 1);
        return true;
    }
    return false;
}

template <typename Map>
bool checked_merge(Map& destination, const Map& source) {
    for (const auto& [key, value] : source) {
        auto& total = destination[key];
        if (value > std::numeric_limits<std::uint64_t>::max() - total) {
            return false;
        }
        total += value;
    }
    return true;
}
}

Result<bool> RuntimeProfile::set_metadata(std::string key, std::string value) {
    if (!valid_key(key) || value.size() > maximum_metadata_value_bytes ||
        !valid_utf8(value)) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "invalid runtime metadata");
    }
    if (metadata_.size() >= maximum_metadata_records && !metadata_.contains(key)) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "excessive runtime metadata");
    }
    metadata_.insert_or_assign(std::move(key), std::move(value));
    return Result<bool>::success(true);
}

void RuntimeProfile::observe(std::uint64_t forward_cycle, std::size_t layer,
                             std::span<const ExpertKey> selected) {
    if (has_previous_ && previous_cycle_ == forward_cycle &&
        previous_layer_ + 1 == layer) {
        for (const auto from : previous_selected_) {
            for (const auto to : selected) {
                (void)bounded_increment(
                    live_transitions_, transition_key(from, to),
                    maximum_transition_records);
            }
        }
    }
    for (const auto key : selected) {
        if (bounded_increment(live_frequency_, frequency_key(key),
                              maximum_frequency_records) &&
            live_route_observations_ !=
                std::numeric_limits<std::uint64_t>::max()) {
            ++live_route_observations_;
        }
    }
    previous_cycle_ = forward_cycle;
    previous_layer_ = layer;
    has_previous_ = true;
    previous_selected_.assign(selected.begin(), selected.end());
}

std::uint64_t RuntimeProfile::count_of(
    const std::map<FrequencyKey, std::uint64_t>& counts, ExpertKey key) {
    const auto found = counts.find(frequency_key(key));
    return found == counts.end() ? 0 : found->second;
}

std::uint64_t RuntimeProfile::count_of(
    const std::map<TransitionKey, std::uint64_t>& counts,
    ExpertKey from, ExpertKey to) {
    const auto found = counts.find(transition_key(from, to));
    return found == counts.end() ? 0 : found->second;
}

std::uint64_t RuntimeProfile::prior_frequency(ExpertKey key) const {
    return count_of(prior_frequency_, key);
}

std::uint64_t RuntimeProfile::live_frequency(ExpertKey key) const {
    return count_of(live_frequency_, key);
}

std::uint64_t RuntimeProfile::prior_transition(ExpertKey from,
                                               ExpertKey to) const {
    return count_of(prior_transitions_, from, to);
}

std::uint64_t RuntimeProfile::live_transition(ExpertKey from,
                                              ExpertKey to) const {
    return count_of(live_transitions_, from, to);
}

double RuntimeProfile::prior_weight(std::uint64_t prior_strength) const noexcept {
    if (prior_route_observations_ == 0) return 0.0;
    if (live_route_observations_ == 0) return 1.0;
    return static_cast<double>(prior_strength) /
           (static_cast<double>(prior_strength) +
            static_cast<double>(live_route_observations_));
}

double RuntimeProfile::usefulness(ExpertKey key,
                                  std::uint64_t prior_strength) const {
    const auto alpha = prior_weight(prior_strength);
    const auto prior = prior_route_observations_ == 0
                           ? 0.0
                           : static_cast<double>(prior_frequency(key)) /
                                 static_cast<double>(prior_route_observations_);
    const auto live = live_route_observations_ == 0
                          ? 0.0
                          : static_cast<double>(live_frequency(key)) /
                                static_cast<double>(live_route_observations_);
    return alpha * prior + (1.0 - alpha) * live;
}

std::vector<ExpertKey> RuntimeProfile::hot_bank(std::size_t count) const {
    std::map<FrequencyKey, std::uint64_t> merged = prior_frequency_;
    for (const auto& [key, value] : live_frequency_) {
        auto& total = merged[key];
        total = value > std::numeric_limits<std::uint64_t>::max() - total
                    ? std::numeric_limits<std::uint64_t>::max()
                    : total + value;
    }
    std::vector<std::pair<FrequencyKey, std::uint64_t>> ordered(
        merged.begin(), merged.end());
    std::sort(ordered.begin(), ordered.end(), [](const auto& left,
                                                  const auto& right) {
        if (left.second != right.second) return left.second > right.second;
        return left.first < right.first;
    });
    if (ordered.size() > count) ordered.resize(count);
    std::vector<ExpertKey> result;
    result.reserve(ordered.size());
    for (const auto& [key, unused] : ordered) {
        (void)unused;
        result.push_back({key.first, key.second});
    }
    return result;
}

Result<bool> RuntimeProfile::save(const std::filesystem::path& path) const {
    std::map<FrequencyKey, std::uint64_t> frequencies = prior_frequency_;
    if (!checked_merge(frequencies, live_frequency_)) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "runtime profile frequency overflow");
    }
    std::map<TransitionKey, std::uint64_t> transitions = prior_transitions_;
    if (!checked_merge(transitions, live_transitions_)) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "runtime profile transition overflow");
    }
    if (live_route_observations_ >
        std::numeric_limits<std::uint64_t>::max() - prior_route_observations_) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "runtime profile observation overflow");
    }
    const auto observations =
        prior_route_observations_ + live_route_observations_;
    const auto hot = hot_bank(persisted_hot_bank_size);

    std::ostringstream body;
    body << profile_magic << '\n';
    body << "META_COUNT " << metadata_.size() << '\n';
    for (const auto& [key, value] : metadata_) {
        body << "META " << key << ' ' << hex_encode(value) << '\n';
    }
    body << "FREQ_COUNT " << frequencies.size() << '\n';
    for (const auto& [key, value] : frequencies) {
        body << "FREQ " << key.first << ' ' << key.second << ' ' << value << '\n';
    }
    body << "TRANSITION_COUNT " << transitions.size() << '\n';
    for (const auto& [key, value] : transitions) {
        body << "TRANSITION " << std::get<0>(key) << ' ' << std::get<1>(key)
             << ' ' << std::get<2>(key) << ' ' << std::get<3>(key) << ' '
             << value << '\n';
    }
    body << "OBSERVATIONS " << observations << '\n';
    body << "HOT_COUNT " << hot.size() << '\n';
    for (const auto key : hot) {
        body << "HOT " << key.layer << ' ' << key.expert << '\n';
    }
    const auto canonical = body.str();
    std::ostringstream document;
    document << canonical << "CRC32C " << std::hex << std::setfill('0')
             << std::setw(8) << crc32c(bytes_of(canonical)) << '\n';
    const auto bytes = document.str();
    if (bytes.size() > maximum_profile_bytes) {
        return Result<bool>::failure(ErrorCode::invalid_state,
                                     "runtime profile exceeds size limit");
    }
    const auto temporary = std::filesystem::path(path.string() + ".tmp");
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return Result<bool>::failure(
            ErrorCode::io_error, "cannot open profile temporary file");
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output) {
            output.close();
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return Result<bool>::failure(
                ErrorCode::io_error, "cannot write profile temporary file");
        }
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return Result<bool>::failure(ErrorCode::io_error,
                                     "cannot publish runtime profile");
    }
    return Result<bool>::success(true);
}

Result<RuntimeProfile> RuntimeProfile::load(const std::filesystem::path& path) {
    std::error_code size_error;
    const auto size = std::filesystem::file_size(path, size_error);
    if (size_error) return Result<RuntimeProfile>::failure(
        ErrorCode::io_error, "cannot stat runtime profile");
    if (size == 0 || size > maximum_profile_bytes) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid runtime profile size");
    }
    std::ifstream input(path, std::ios::binary);
    std::string document(static_cast<std::size_t>(size), '\0');
    input.read(document.data(), static_cast<std::streamsize>(document.size()));
    if (!input || input.peek() != std::char_traits<char>::eof()) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::io_error, "cannot read runtime profile");
    }
    const auto marker = document.rfind("CRC32C ");
    if (marker == std::string::npos || marker == 0 || document.back() != '\n' ||
        document[marker - 1] != '\n') {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "missing runtime profile checksum");
    }
    const std::string_view body(document.data(), marker);
    const std::string_view checksum_text(document.data() + marker + 7,
                                         document.size() - marker - 8);
    std::uint32_t expected = 0;
    const auto parsed = std::from_chars(
        checksum_text.data(), checksum_text.data() + checksum_text.size(),
        expected, 16);
    if (checksum_text.size() != 8 || parsed.ec != std::errc{} ||
        parsed.ptr != checksum_text.data() + checksum_text.size() ||
        crc32c(bytes_of(body)) != expected) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "runtime profile checksum mismatch");
    }

    std::istringstream lines{std::string(body)};
    std::string line;
    if (!std::getline(lines, line) || line != profile_magic) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "unsupported runtime profile");
    }
    RuntimeProfile result;
    const auto read_count = [&](std::string_view label, std::size_t limit,
                                std::size_t& count) {
        return std::getline(lines, line) && line.starts_with(label) &&
               parse_integer(std::string_view(line).substr(label.size()), count) &&
               count <= limit;
    };
    std::size_t count = 0;
    if (!read_count("META_COUNT ", maximum_metadata_records, count)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid metadata count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::getline(lines, line)) return Result<RuntimeProfile>::failure(ErrorCode::invalid_state);
        std::istringstream fields(line);
        std::string tag, key, encoded, extra;
        if (!(fields >> tag >> key >> encoded) || fields >> extra || tag != "META") {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "invalid metadata record");
        }
        auto decoded = hex_decode(encoded);
        if (!decoded) return Result<RuntimeProfile>::failure(decoded.error(), decoded.message());
        if (result.metadata_.contains(key)) {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "duplicate runtime metadata");
        }
        auto inserted = result.set_metadata(std::move(key), std::move(decoded.value()));
        if (!inserted) return Result<RuntimeProfile>::failure(inserted.error(), inserted.message());
    }
    if (!read_count("FREQ_COUNT ", maximum_frequency_records, count)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid frequency count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::getline(lines, line)) return Result<RuntimeProfile>::failure(ErrorCode::invalid_state);
        std::istringstream fields(line);
        std::string tag, extra;
        std::size_t layer = 0, expert = 0;
        std::uint64_t value = 0;
        if (!(fields >> tag >> layer >> expert >> value) || fields >> extra ||
            tag != "FREQ" || value == 0 ||
            !result.prior_frequency_.emplace(FrequencyKey{layer, expert}, value).second) {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "invalid frequency record");
        }
    }
    if (!read_count("TRANSITION_COUNT ", maximum_transition_records, count)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid transition count");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::getline(lines, line)) return Result<RuntimeProfile>::failure(ErrorCode::invalid_state);
        std::istringstream fields(line);
        std::string tag, extra;
        std::size_t fl = 0, fe = 0, tl = 0, te = 0;
        std::uint64_t value = 0;
        if (!(fields >> tag >> fl >> fe >> tl >> te >> value) ||
            fields >> extra || tag != "TRANSITION" || value == 0 ||
            !result.prior_transitions_
                 .emplace(TransitionKey{fl, fe, tl, te}, value).second) {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "invalid transition record");
        }
    }
    if (!std::getline(lines, line) || !line.starts_with("OBSERVATIONS ") ||
        !parse_integer(std::string_view(line).substr(13),
                       result.prior_route_observations_)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid observation count");
    }
    std::uint64_t total = 0;
    for (const auto& [unused, value] : result.prior_frequency_) {
        (void)unused;
        if (value > std::numeric_limits<std::uint64_t>::max() - total) {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "frequency total overflow");
        }
        total += value;
    }
    if (total != result.prior_route_observations_) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "observation count mismatch");
    }
    if (!read_count("HOT_COUNT ", persisted_hot_bank_size, count)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "invalid hot bank count");
    }
    std::vector<ExpertKey> stored_hot;
    for (std::size_t index = 0; index < count; ++index) {
        if (!std::getline(lines, line)) return Result<RuntimeProfile>::failure(ErrorCode::invalid_state);
        std::istringstream fields(line);
        std::string tag, extra;
        ExpertKey key;
        if (!(fields >> tag >> key.layer >> key.expert) || fields >> extra ||
            tag != "HOT") {
            return Result<RuntimeProfile>::failure(
                ErrorCode::invalid_state, "invalid hot bank record");
        }
        stored_hot.push_back(key);
    }
    if (stored_hot != result.hot_bank(persisted_hot_bank_size) ||
        std::getline(lines, line)) {
        return Result<RuntimeProfile>::failure(
            ErrorCode::invalid_state, "non-canonical runtime profile");
    }
    return Result<RuntimeProfile>::success(std::move(result));
}

}
