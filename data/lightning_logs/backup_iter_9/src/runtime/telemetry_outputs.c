#include "qallow/telemetry_outputs.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static int ensure_dir_exists(const char* path) {
    if (!path || !*path) {
        return -1;
    }

    char buffer[PATH_MAX];
    size_t len = strnlen(path, sizeof(buffer));
    if (len == 0 || len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1);

    for (size_t i = 1; i <= len; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\' || buffer[i] == '\0') {
            char prev = buffer[i];
            buffer[i] = '\0';
            if (buffer[0] != '\0') {
#if defined(_WIN32)
                if (_mkdir(buffer) != 0 && errno != EEXIST) {
                    buffer[i] = prev;
                    return -1;
                }
#else
                if (mkdir(buffer, 0755) != 0 && errno != EEXIST) {
                    buffer[i] = prev;
                    return -1;
                }
#endif
            }
            buffer[i] = prev;
        }
    }
    return 0;
}

static int ensure_parent_dir(const char* path) {
    if (!path) {
        return -1;
    }
    char buffer[PATH_MAX];
    size_t len = strnlen(path, sizeof(buffer));
    if (len == 0 || len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1);
    for (size_t i = len; i > 0; --i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            buffer[i] = '\0';
            break;
        }
    }
    if (buffer[0] == '\0') {
        return 0;
    }
    return ensure_dir_exists(buffer);
}

static int ensure_base_dirs(void) {
    if (ensure_dir_exists("data") != 0) {
        return -1;
    }
    if (ensure_dir_exists("data/logs") != 0) {
        return -1;
    }
    return 0;
}

static int ensure_latest_dir(void) {
    if (ensure_base_dirs() != 0) {
        return -1;
    }
    if (ensure_dir_exists("data/logs/latest") != 0) {
        return -1;
    }
    return 0;
}

const char* qallow_audit_tag_fallback(void) {
    const char* tag = getenv("QALLOW_AUDIT_TAG");
    if (tag && *tag) {
        return tag;
    }
    return "default";
}

int qallow_phase_resolve_log_path(const char* phase,
                                  const char* override_path,
                                  char* out_path,
                                  size_t out_len) {
    if (!phase || !out_path || out_len == 0) {
        return -1;
    }

    const char* chosen = override_path;
    char buffer[PATH_MAX];

    if (!chosen || !*chosen) {
        if (ensure_base_dirs() != 0) {
            return -1;
        }
        if ((size_t)snprintf(buffer, sizeof(buffer), "data/logs/%s.csv", phase) >= sizeof(buffer)) {
            return -1;
        }
        chosen = buffer;
    } else {
        char tmp[PATH_MAX];
        if (strnlen(chosen, sizeof(tmp)) >= sizeof(tmp)) {
            return -1;
        }
        memcpy(tmp, chosen, strnlen(chosen, sizeof(tmp)) + 1);
        if (ensure_parent_dir(tmp) != 0) {
            return -1;
        }
    }

    size_t needed = strnlen(chosen, PATH_MAX);
    if (needed + 1 > out_len) {
        return -1;
    }
    memcpy(out_path, chosen, needed + 1);
    return 0;
}

static int absolute_path(const char* input, char* resolved, size_t resolved_len) {
    if (!input || !resolved || resolved_len == 0) {
        return -1;
    }
#if defined(_WIN32)
    if (!_fullpath(resolved, input, resolved_len)) {
        return -1;
    }
#else
    if (!realpath(input, resolved)) {
        return -1;
    }
#endif
    return 0;
}

int qallow_phase_update_latest_symlink(const char* phase, const char* csv_path) {
    if (!phase || !csv_path) {
        return -1;
    }
    if (ensure_latest_dir() != 0) {
        return -1;
    }

    char link_path[PATH_MAX];
    if ((size_t)snprintf(link_path, sizeof(link_path), "data/logs/latest/%s.csv", phase) >= sizeof(link_path)) {
        return -1;
    }

    char abs_target[PATH_MAX];
    if (absolute_path(csv_path, abs_target, sizeof(abs_target)) != 0) {
        return -1;
    }

#if defined(_WIN32)
    DeleteFileA(link_path);
    if (CreateSymbolicLinkA(link_path, abs_target, 0) == 0) {
        if (CopyFileA(abs_target, link_path, FALSE) == 0) {
            return -1;
        }
    }
#else
    unlink(link_path);
    if (symlink(abs_target, link_path) != 0) {
        return -1;
    }
#endif
    return 0;
}

static int format_timestamp(char* buffer, size_t len) {
    time_t now = time(NULL);
    struct tm tm_now;
#if defined(_WIN32)
    if (gmtime_s(&tm_now, &now) != 0) {
        return -1;
    }
#else
    if (!gmtime_r(&now, &tm_now)) {
        return -1;
    }
#endif
    if (strftime(buffer, len, "%Y-%m-%dT%H:%M:%SZ", &tm_now) == 0) {
        return -1;
    }
    return 0;
}

int qallow_phase_write_summary(const char* phase,
                               const char* audit_tag,
                               const char* csv_path,
                               const char* metrics_json) {
    if (!phase || !csv_path) {
        return -1;
    }
    if (ensure_base_dirs() != 0) {
        return -1;
    }

    const char* tag = (audit_tag && *audit_tag) ? audit_tag : qallow_audit_tag_fallback();
    const char* metrics = (metrics_json && *metrics_json) ? metrics_json : "{}";

    char timestamp[32];
    if (format_timestamp(timestamp, sizeof(timestamp)) != 0) {
        strncpy(timestamp, "1970-01-01T00:00:00Z", sizeof(timestamp));
        timestamp[sizeof(timestamp) - 1] = '\0';
    }

    FILE* fp = fopen("data/logs/phase_summary.json", "w");
    if (!fp) {
        return -1;
    }

    fprintf(fp,
            "{\n"
            "  \"phase\": \"%s\",\n"
            "  \"audit_tag\": \"%s\",\n"
            "  \"csv\": \"%s\",\n"
            "  \"generated_at\": \"%s\",\n"
            "  \"metrics\": %s\n"
            "}\n",
            phase,
            tag,
            csv_path,
            timestamp,
            metrics);

    fclose(fp);
    return 0;
}
