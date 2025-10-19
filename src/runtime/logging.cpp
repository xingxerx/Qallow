#include "qallow/logging.h"

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>
#include <time.h>

namespace {
std::mutex g_log_mutex;
std::string g_log_directory = "data/logs";
std::string g_log_filename = "qallow_runtime.log";
FILE* g_log_file = nullptr;

void ensure_directory(const std::string& path) {
    std::error_code ec;
    #include <cstdarg>
    #include <filesystem>
    #include <mutex>
    #include <string>

    #include <spdlog/sinks/basic_file_sink.h>
    #include <spdlog/sinks/stdout_color_sinks.h>
    #include <spdlog/spdlog.h>

    namespace {
    std::mutex g_log_mutex;
    std::string g_log_directory = "data/logs";
    std::string g_log_filename = "qallow_runtime.log";
    std::shared_ptr<spdlog::logger> g_logger;

    void ensure_directory(const std::string& path) {
        std::error_code ec;
        std::filesystem::create_directories(path, ec);
    }

    void rebuild_logger_locked() {
        ensure_directory(g_log_directory);
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
            g_log_directory + "/" + g_log_filename, true);
        console_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] [%n] %v");
        file_sink->set_pattern("[%Y-%m-%d %H:%M:%S] [%l] [%n] %v");

        std::vector<spdlog::sink_ptr> sinks{console_sink, file_sink};
        g_logger = std::make_shared<spdlog::logger>("qallow", sinks.begin(), sinks.end());
        g_logger->set_level(spdlog::level::info);
        g_logger->flush_on(spdlog::level::info);
    }

    void ensure_logger() {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_logger) {
            rebuild_logger_locked();
        }
    }

    void log_internal(spdlog::level::level_enum lvl, const char* scope, const char* fmt, va_list args) {
        ensure_logger();

        char message_buffer[2048];
        if (fmt) {
            vsnprintf(message_buffer, sizeof(message_buffer), fmt, args);
        } else {
            message_buffer[0] = '\0';
        }

        std::lock_guard<std::mutex> lock(g_log_mutex);
        if (!g_logger) {
            return;
        }
        if (scope && *scope) {
            g_logger->log(lvl, "[{}] {}", scope, message_buffer);
        } else {
            g_logger->log(lvl, "{}", message_buffer);
        }
    }
    }  // namespace

    extern "C" {

    void qallow_logging_init(void) {
        ensure_logger();
    }

    void qallow_logging_shutdown(void) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_logger.reset();
    }

    void qallow_logging_set_directory(const char* dir) {
        if (!dir || !*dir) return;
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log_directory = dir;
        rebuild_logger_locked();
    }

    void qallow_logging_flush(void) {
        std::lock_guard<std::mutex> lock(g_log_mutex);
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
