#include "qallow/logging.h"

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string>
#include <time.h>

namespace {
std::mutex g_log_mutex;
std::string g_log_directory = "data/logs";
std::string g_log_filename = "qallow_runtime.log";
FILE* g_log_file = nullptr;

void ensure_directory(const std::string& path) {
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
}

void open_log_unlocked() {
    if (g_log_file) return;
    ensure_directory(g_log_directory);
    std::string full_path = g_log_directory + "/" + g_log_filename;
    g_log_file = std::fopen(full_path.c_str(), "a");
}

void close_log_unlocked() {
    if (g_log_file) {
        std::fflush(g_log_file);
        std::fclose(g_log_file);
        g_log_file = nullptr;
    }
}

std::string format_timestamp() {
    char buffer[64];
    std::time_t now = std::time(nullptr);
    std::tm tm_now{};
    if (std::localtime_r(&now, &tm_now)) {
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_now);
        return buffer;
    }
    return "0000-00-00 00:00:00";
}

void log_internal(const char* level, const char* scope, const char* fmt, va_list args) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    open_log_unlocked();

    char message_buffer[2048];
    if (fmt) {
        vsnprintf(message_buffer, sizeof(message_buffer), fmt, args);
    } else {
        message_buffer[0] = '\0';
    }

    std::string timestamp = format_timestamp();
    if (scope && *scope) {
        std::fprintf(stdout, "[%s] [%s] [%s] %s\n", timestamp.c_str(), level, scope, message_buffer);
        if (g_log_file) {
            std::fprintf(g_log_file, "[%s] [%s] [%s] %s\n", timestamp.c_str(), level, scope, message_buffer);
        }
    } else {
        std::fprintf(stdout, "[%s] [%s] %s\n", timestamp.c_str(), level, message_buffer);
        if (g_log_file) {
            std::fprintf(g_log_file, "[%s] [%s] %s\n", timestamp.c_str(), level, message_buffer);
        }
    }
}
}  // namespace

extern "C" {

void qallow_logging_init(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    open_log_unlocked();
}

void qallow_logging_shutdown(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    close_log_unlocked();
}

void qallow_logging_set_directory(const char* dir) {
    if (!dir) return;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log_directory = dir;
    close_log_unlocked();
}

void qallow_logging_flush(void) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log_file) {
        std::fflush(g_log_file);
    }
}

void qallow_log_info(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal("INFO", scope, fmt, args);
    va_end(args);
}

void qallow_log_warn(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal("WARN", scope, fmt, args);
    va_end(args);
}

void qallow_log_error(const char* scope, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_internal("ERROR", scope, fmt, args);
    va_end(args);
}

}  // extern "C"
