#include "qallow/logging.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {
std::mutex g_logger_mutex;
std::shared_ptr<spdlog::logger> g_logger;
std::string g_log_directory = "data/logs";
std::string g_log_filename = "qallow_runtime.log";

void ensure_directory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
}

std::shared_ptr<spdlog::logger> build_logger() {
    ensure_directory(g_log_directory);
    std::vector<spdlog::sink_ptr> sinks;
    sinks.emplace_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
    sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        g_log_directory + "/" + g_log_filename, true));
    auto logger = std::make_shared<spdlog::logger>("qallow", begin(sinks), end(sinks));
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->set_level(spdlog::level::info);
    return logger;
}

std::string format_message(const char* fmt, va_list args) {
    if (!fmt) {
        return {};
    }

    va_list copy;
    va_copy(copy, args);
    int required = vsnprintf(nullptr, 0, fmt, copy);
    va_end(copy);
    if (required <= 0) {
        return {};
    }

    std::string buffer(static_cast<size_t>(required), '\0');
    vsnprintf(buffer.data(), buffer.size() + 1, fmt, args);
    return buffer;
}

void log_internal(spdlog::level::level_enum level, const char* scope, const char* fmt, va_list args) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (!g_logger) {
        g_logger = build_logger();
    }

    std::string message = format_message(fmt, args);
    if (scope && *scope) {
        g_logger->log(level, "[{}] {}", scope, message);
    } else {
        g_logger->log(level, "{}", message);
    }
}
}  // namespace

extern "C" {

void qallow_logging_init(void) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (!g_logger) {
        g_logger = build_logger();
    }
}

void qallow_logging_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (g_logger) {
        g_logger->flush();
        g_logger.reset();
    }
}

void qallow_logging_set_directory(const char* dir) {
    if (!dir) return;
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    g_log_directory = dir;
    g_logger.reset();
}

void qallow_logging_flush(void) {
    std::lock_guard<std::mutex> lock(g_logger_mutex);
    if (g_logger) {
        g_logger->flush();
    }
}

void qallow_log_info(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal(spdlog::level::info, scope, fmt, args);
    va_end(args);
}

void qallow_log_warn(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal(spdlog::level::warn, scope, fmt, args);
    va_end(args);
}

void qallow_log_error(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal(spdlog::level::err, scope, fmt, args);
    va_end(args);
}

}  // extern "C"
