// Python writer가 만든 K3X artifact를 C++ reader로 검증합니다.
#include "k3x/reader.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    const auto path = std::filesystem::path(argv[1]);
    if (argc == 3 &&
        (std::string_view(argv[2]) == "io-uring" ||
         std::string_view(argv[2]) == "direct" ||
         std::string_view(argv[2]) == "io-uring-direct")) {
        k3x::ReaderOptions options;
        const auto mode = std::string_view(argv[2]);
        if (mode != "direct") {
            options.io_engine = k3x::L2IoEngine::io_uring;
        }
        if (mode != "io-uring") {
            options.cache_mode = k3x::L2CacheMode::direct;
        }
        options.queue_depth = 8;
        const auto mode_reader = k3x::Reader::open(path, options);
        if (!mode_reader) {
            std::cerr << k3x::error_code_name(mode_reader.error());
            if (!mode_reader.message().empty()) {
                std::cerr << ": " << mode_reader.message();
            }
            std::cerr << '\n';
            return 12;
        }
        const auto payload = mode_reader.value().read_tensor(
            mode_reader.value().tensors().front().tensor_id);
        const auto quantized = std::find_if(
            mode_reader.value().tensors().begin(),
            mode_reader.value().tensors().end(),
            [](const auto& record) { return record.auxiliary_length > 0; });
        if (quantized == mode_reader.value().tensors().end()) return 13;
        const auto scales = mode_reader.value().read_auxiliary(
            quantized->tensor_id);
        const auto last = std::max_element(
            mode_reader.value().tensors().begin(),
            mode_reader.value().tensors().end(),
            [](const auto& left, const auto& right) {
                const auto left_end = std::max(
                    left.data_offset + left.data_length,
                    left.auxiliary_offset + left.auxiliary_length);
                const auto right_end = std::max(
                    right.data_offset + right.data_length,
                    right.auxiliary_offset + right.auxiliary_length);
                return left_end < right_end;
            });
        const bool last_is_auxiliary = last->auxiliary_length > 0 &&
            last->auxiliary_offset + last->auxiliary_length >
                last->data_offset + last->data_length;
        const auto final_extent = last_is_auxiliary
            ? mode_reader.value().read_auxiliary(last->tensor_id)
            : mode_reader.value().read_tensor(last->tensor_id);
        const auto final_length = last_is_auxiliary
            ? last->auxiliary_length : last->data_length;
        if (!payload || payload.value().empty() || !scales ||
            scales.value().size() != quantized->auxiliary_length ||
            !final_extent || final_extent.value().size() != final_length ||
            mode_reader.value().options().io_engine != options.io_engine ||
            mode_reader.value().options().cache_mode != options.cache_mode ||
            (options.cache_mode == k3x::L2CacheMode::direct &&
             (mode_reader.value().direct_memory_alignment() == 0 ||
              mode_reader.value().direct_offset_alignment() == 0 ||
              mode_reader.value().counters().storage_submitted_bytes <=
                  mode_reader.value().counters().requested_bytes ||
              mode_reader.value().counters().storage_completed_bytes <
                  mode_reader.value().counters().completed_bytes))) {
            return 13;
        }
        return 0;
    }
    const auto reader = k3x::Reader::open(path);
    if (!reader) {
        std::cerr << k3x::error_code_name(reader.error()) << '\n';
        return 1;
    }
    if (reader.value().tensors().empty()) return 3;
    if (argc == 3) {
        if (std::string_view(argv[2]) == "storage-fixture") {
            return reader.value().superblock().optional_features ==
                    k3x::optional_storage_fixture
                ? 0
                : 14;
        }
        if (std::string_view(argv[2]) != "persistent") return 10;
#ifdef __linux__
        const auto moved = path.string() + ".moved";
        std::filesystem::rename(path, moved);
        const auto persistent = reader.value().read_tensor(
            reader.value().tensors().front().tensor_id);
        std::filesystem::rename(moved, path);
        if (!persistent || persistent.value().empty()) return 11;
#endif
        return 0;
    }
    const auto payload = reader.value().read_tensor(reader.value().tensors().front().tensor_id);
    if (!payload || payload.value().empty()) return 4;
    if (reader.value().counters().calls != 1 ||
        reader.value().counters().batch_submissions != 1) {
        return 5;
    }

    const auto& records = reader.value().tensors();
    if (records.size() < 2) return 6;
    const std::array requests{
        k3x::ExtentRequest{records[1].data_offset, records[1].data_length},
        k3x::ExtentRequest{records[0].data_offset, records[0].data_length},
    };
    const auto batch = reader.value().read_extents(requests);
    if (!batch || batch.value().size() != requests.size() ||
        batch.value()[0].size() != requests[0].length ||
        batch.value()[1].size() != requests[1].length ||
        reader.value().counters().calls != 3 ||
        reader.value().counters().batch_submissions != 2 ||
        reader.value().counters().completions != 3) {
        return 7;
    }

    const auto before = reader.value().counters();
    const std::array invalid{
        k3x::ExtentRequest{reader.value().superblock().file_length, 1}};
    const auto rejected = reader.value().read_extents(invalid);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_extent ||
        reader.value().counters().calls != before.calls ||
        reader.value().counters().batch_submissions != before.batch_submissions ||
        reader.value().counters().failures != before.failures) {
        return 8;
    }

    const std::span<const k3x::ExtentRequest> empty;
    const auto empty_result = reader.value().read_extents(empty);
    if (!empty_result || !empty_result.value().empty() ||
        reader.value().counters().batch_submissions != before.batch_submissions) {
        return 9;
    }
    return 0;
}
