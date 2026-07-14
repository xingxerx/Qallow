#include "qallow/telemetry_outputs.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define qallow_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define qallow_mkdir(path) mkdir(path, 0755)
#endif

static int qallow_ensure_log_dir(void) {
    /* Ignore EEXIST-style failures; only care that the dirs end up present. */
    (void)qallow_mkdir("data");
    (void)qallow_mkdir("data/logs");
    FILE* probe = fopen("data/logs/.probe", "w");
    if (!probe) {
        return 1;
    }
    fclose(probe);
    remove("data/logs/.probe");
    return 0;
}

int qallow_phase_resolve_log_path(const char* phase_name,
                                  const char* requested,
                                  char* out,
                                  size_t out_size) {
    if (!out || out_size == 0) {
        return 1;
    }
    if (requested && *requested) {
        if (strlen(requested) >= out_size) {
            return 1;
        }
        strcpy(out, requested);
        return 0;
    }
    if (!phase_name || !*phase_name) {
        return 1;
    }
    if (qallow_ensure_log_dir() != 0) {
        return 1;
    }
    int written = snprintf(out, out_size, "data/logs/%s.csv", phase_name);
    if (written < 0 || (size_t)written >= out_size) {
        return 1;
    }
    return 0;
}

const char* qallow_audit_tag_fallback(void) {
    return "untagged";
}

int qallow_phase_update_latest_symlink(const char* phase_name,
                                       const char* csv_path) {
    if (!phase_name || !csv_path) {
        return 1;
    }
    if (qallow_ensure_log_dir() != 0) {
        return 1;
    }

    char latest_path[PATH_MAX];
    int written = snprintf(latest_path, sizeof(latest_path),
                           "data/logs/%s_latest.csv", phase_name);
    if (written < 0 || (size_t)written >= sizeof(latest_path)) {
        return 1;
    }

    /* Windows symlinks need elevated privileges; copy the file instead. */
    FILE* src = fopen(csv_path, "rb");
    if (!src) {
        return 1;
    }
    FILE* dst = fopen(latest_path, "wb");
    if (!dst) {
        fclose(src);
        return 1;
    }

    char buffer[8192];
    size_t n;
    int rc = 0;
    while ((n = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, n, dst) != n) {
            rc = 1;
            break;
        }
    }
    fclose(src);
    fclose(dst);
    return rc;
}

int qallow_phase_write_summary(const char* phase_name,
                               const char* tag,
                               const char* csv_path,
                               const char* metrics_json) {
    if (!phase_name || !csv_path) {
        return 1;
    }
    if (qallow_ensure_log_dir() != 0) {
        return 1;
    }

    FILE* f = fopen("data/logs/phase_summary.json", "w");
    if (!f) {
        return 1;
    }
    fprintf(f,
            "{\n"
            "  \"phase\": \"%s\",\n"
            "  \"audit_tag\": \"%s\",\n"
            "  \"csv_path\": \"%s\",\n"
            "  \"metrics\": %s\n"
            "}\n",
            phase_name,
            tag ? tag : qallow_audit_tag_fallback(),
            csv_path,
            (metrics_json && *metrics_json) ? metrics_json : "{}");
    fclose(f);
    return 0;
}
