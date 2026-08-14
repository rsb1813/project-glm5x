// C++ runtime의 안정적 오류 코드와 결과 값을 정의합니다.
#pragma once

#include <optional>
#include <string>
#include <utility>

namespace k3x {
enum class ErrorCode {
    ok,
    io_error,
    bad_magic,
    unsupported_version,
    unsupported_required_feature,
    non_executable_artifact,
    superblock_crc_mismatch,
    truncated_file,
    invalid_directory,
    invalid_extent,
    data_crc_mismatch,
    auxiliary_crc_mismatch,
    directory_sha256_mismatch,
    root_sha256_mismatch,
    tensor_not_found,
    invalid_mxfp4,
    invalid_state,
    backend_unavailable,
    storage_unavailable,
    unsupported_architecture,
};

inline const char* error_code_name(ErrorCode code) {
    switch (code) {
        case ErrorCode::ok: return "OK";
        case ErrorCode::io_error: return "IO_ERROR";
        case ErrorCode::bad_magic: return "BAD_MAGIC";
        case ErrorCode::unsupported_version: return "UNSUPPORTED_VERSION";
        case ErrorCode::unsupported_required_feature: return "UNSUPPORTED_REQUIRED_FEATURE";
        case ErrorCode::non_executable_artifact: return "NON_EXECUTABLE_ARTIFACT";
        case ErrorCode::superblock_crc_mismatch: return "SUPERBLOCK_CRC_MISMATCH";
        case ErrorCode::truncated_file: return "TRUNCATED_FILE";
        case ErrorCode::invalid_directory: return "INVALID_DIRECTORY";
        case ErrorCode::invalid_extent: return "INVALID_EXTENT";
        case ErrorCode::data_crc_mismatch: return "DATA_CRC_MISMATCH";
        case ErrorCode::auxiliary_crc_mismatch: return "AUXILIARY_CRC_MISMATCH";
        case ErrorCode::directory_sha256_mismatch: return "DIRECTORY_SHA256_MISMATCH";
        case ErrorCode::root_sha256_mismatch: return "ROOT_SHA256_MISMATCH";
        case ErrorCode::tensor_not_found: return "TENSOR_NOT_FOUND";
        case ErrorCode::invalid_mxfp4: return "INVALID_MXFP4";
        case ErrorCode::invalid_state: return "INVALID_STATE";
        case ErrorCode::backend_unavailable: return "BACKEND_UNAVAILABLE";
        case ErrorCode::storage_unavailable: return "STORAGE_UNAVAILABLE";
        case ErrorCode::unsupported_architecture: return "UNSUPPORTED_ARCHITECTURE";
    }
    return "UNKNOWN";
}

template <typename T>
class Result {
public:
    static Result success(T value) { return Result(std::move(value)); }
    static Result failure(ErrorCode code, std::string message = {}) {
        return Result(code, std::move(message));
    }
    explicit operator bool() const { return value_.has_value(); }
    T& value() { return *value_; }
    const T& value() const { return *value_; }
    ErrorCode error() const { return error_; }
    const std::string& message() const { return message_; }
private:
    explicit Result(T value) : value_(std::move(value)), error_(ErrorCode::ok) {}
    Result(ErrorCode code, std::string message) : error_(code), message_(std::move(message)) {}
    std::optional<T> value_;
    ErrorCode error_{ErrorCode::ok};
    std::string message_;
};
}
