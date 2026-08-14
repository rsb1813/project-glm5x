// 런타임 전용 task/session profile의 관측과 영속화 계약을 검증합니다.
#include "k3x/runtime_profile.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                       \
            throw std::runtime_error("profile requirement failed: " #condition); \
        }                                                                        \
    } while (false)

int main() {
    using k3x::ExpertKey;
    using k3x::RuntimeProfile;

    RuntimeProfile profile;
    assert(profile.set_metadata("TASK", "coding"));
    assert(profile.set_metadata("LANG", "cpp"));
    assert(profile.set_metadata("TASK", "agentic-coding"));
    assert(!profile.set_metadata("bad-key", "value"));
    assert(!profile.set_metadata("REPO", "line\nbreak"));
    assert(profile.metadata().at("TASK") == "agentic-coding");

    const std::array first{ExpertKey{0, 3}, ExpertKey{0, 1}};
    const std::array second{ExpertKey{1, 4}, ExpertKey{1, 2}};
    profile.observe(7, 0, first);
    profile.observe(7, 1, second);
    profile.observe(8, 0, std::array{ExpertKey{0, 3}});

    assert(profile.live_route_observations() == 5);
    assert(profile.live_frequency({0, 3}) == 2);
    assert(profile.live_frequency({0, 1}) == 1);
    assert(profile.live_transition({0, 3}, {1, 4}) == 1);
    assert(profile.live_transition({0, 1}, {1, 2}) == 1);
    assert(profile.live_transition({1, 4}, {0, 3}) == 0);

    const auto hot = profile.hot_bank(3);
    assert(hot.size() == 3);
    assert((hot[0] == ExpertKey{0, 3}));
    assert((hot[1] == ExpertKey{0, 1}));
    assert((hot[2] == ExpertKey{1, 2}));

    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const auto root = std::filesystem::temp_directory_path() /
                      ("k3x-profile-test-" + std::to_string(nonce));
    std::filesystem::create_directories(root);
    const auto path = root / "session.k3xp";
    assert(profile.save(path));
    assert(std::filesystem::exists(path));
    assert(!std::filesystem::exists(path.string() + ".tmp"));

    auto loaded = RuntimeProfile::load(path);
    assert(loaded);
    assert(loaded.value().metadata() == profile.metadata());
    assert(loaded.value().prior_frequency({0, 3}) == 2);
    assert(loaded.value().prior_frequency({0, 1}) == 1);
    assert(loaded.value().prior_transition({0, 3}, {1, 4}) == 1);
    assert(loaded.value().live_route_observations() == 0);
    assert(loaded.value().hot_bank(3) == hot);
    assert(loaded.value().prior_weight(64) == 1.0);

    loaded.value().observe(9, 0, std::array{ExpertKey{2, 7}});
    assert(loaded.value().prior_weight(1) == 0.5);
    assert(loaded.value().usefulness({0, 3}, 1) > 0.0);
    assert(loaded.value().usefulness({2, 7}, 1) > 0.0);

    const auto canonical = root / "canonical.k3xp";
    assert(loaded.value().save(canonical));
    auto canonical_loaded = RuntimeProfile::load(canonical);
    assert(canonical_loaded);
    const auto canonical_again = root / "canonical-again.k3xp";
    assert(canonical_loaded.value().save(canonical_again));
    std::ifstream left(canonical, std::ios::binary);
    std::ifstream right(canonical_again, std::ios::binary);
    const std::string left_bytes((std::istreambuf_iterator<char>(left)), {});
    const std::string right_bytes((std::istreambuf_iterator<char>(right)), {});
    assert(left_bytes == right_bytes);

    const auto corrupt = root / "corrupt.k3xp";
    auto corrupt_bytes = left_bytes;
    assert(!corrupt_bytes.empty());
    corrupt_bytes[corrupt_bytes.size() / 2] ^= 1;
    {
        std::ofstream output(corrupt, std::ios::binary | std::ios::trunc);
        output.write(corrupt_bytes.data(),
                     static_cast<std::streamsize>(corrupt_bytes.size()));
    }
    auto rejected = RuntimeProfile::load(corrupt);
    assert(!rejected);
    assert(rejected.error() == k3x::ErrorCode::invalid_state);

    std::filesystem::remove_all(root);
    return 0;
}
