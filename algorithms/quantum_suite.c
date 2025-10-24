#include "quantum_suite.h"

#include "cJSON.h"
#include "qallow/logging.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#define QALLOW_MKDIR(path) _mkdir(path)
#define QALLOW_ACCESS(path, mode) _access(path, mode)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#define QALLOW_MKDIR(path) mkdir(path, 0755)
#define QALLOW_ACCESS(path, mode) access(path, mode)
#endif

#if !defined(PATH_MAX)
#define PATH_MAX 4096
#endif

#define QALLOW_TMP_RESULTS "/tmp/quantum_results.json"
#define QALLOW_RESULTS_PATH "data/logs/quantum_results.json"
#define QALLOW_COMBINED_METRICS "data/logs/combined_metrics.csv"
#define QALLOW_INTROSPECTION_TRACE "data/logs/introspection_trace.csv"

static quantum_suite_metrics_t g_latest_metrics;

static void metrics_reset(quantum_suite_metrics_t* metrics) {
    if (!metrics) {
        return;
    }
    memset(metrics, 0, sizeof(*metrics));
    metrics->grover_probability = -1.0;
    metrics->vqe_best_energy = 0.0;
    metrics->timestamp[0] = '\0';
    metrics->valid = 0;
}

static void iso_timestamp_now(char* buffer, size_t len) {
    if (!buffer || len == 0) {
        return;
    }
    time_t now = time(NULL);
    struct tm tm_now;
#if defined(_WIN32)
    gmtime_s(&tm_now, &now);
#else
    gmtime_r(&now, &tm_now);
#endif
    if (strftime(buffer, len, "%Y-%m-%dT%H:%M:%SZ", &tm_now) == 0) {
        if (len > 0) {
            buffer[0] = '\0';
        }
    }
}

static int ensure_directory_chain(const char* path) {
    if (!path || !*path) {
        return -1;
    }
    char buffer[PATH_MAX];
    size_t len = strnlen(path, sizeof(buffer));
    if (len == 0 || len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1);
    for (size_t i = 1; i < len; ++i) {
        if (buffer[i] == '/' || buffer[i] == '\\') {
            char saved = buffer[i];
            buffer[i] = '\0';
            if (buffer[0] != '\0') {
                if (QALLOW_MKDIR(buffer) != 0 && errno != EEXIST) {
                    buffer[i] = saved;
                    return -1;
                }
            }
            buffer[i] = saved;
        }
    }
    if (QALLOW_MKDIR(buffer) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int ensure_parent_directory(const char* path) {
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
        if (buffer[i - 1] == '/' || buffer[i - 1] == '\\') {
            buffer[i - 1] = '\0';
            break;
        }
    }
    if (buffer[0] == '\0') {
        return 0;
    }
    return ensure_directory_chain(buffer);
}

static int ensure_logs_directory(void) {
    if (ensure_directory_chain("data") != 0) {
        return -1;
    }
    if (ensure_directory_chain("data/logs") != 0) {
        return -1;
    }
    return 0;
}

static const char* detect_python_binary(void) {
    const char* env_python = getenv("QALLOW_PYTHON");
    if (env_python && *env_python && QALLOW_ACCESS(env_python, X_OK) == 0) {
        return env_python;
    }
    if (QALLOW_ACCESS("./qiskit-env/bin/python", X_OK) == 0) {
        return "./qiskit-env/bin/python";
    }
    if (QALLOW_ACCESS("python3", X_OK) == 0) {
        return "python3";
    }
    if (QALLOW_ACCESS("python", X_OK) == 0) {
        return "python";
    }
    return NULL;
}

static int run_python_suite(char* error_buffer, size_t error_buffer_len) {
    const char* python = detect_python_binary();
    if (!python) {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "python interpreter not found");
        }
        return -1;
    }
    char command[512];
    int written = snprintf(command,
                           sizeof(command),
                           "\"%s\" quantum_algorithms/unified_quantum_framework.py",
                           python);
    if (written < 0 || (size_t)written >= sizeof(command)) {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "command overflow for python runner");
        }
        return -1;
    }
    int rc = system(command);
    if (rc == -1) {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "system() invocation failed");
        }
        return -1;
    }
#if defined(_WIN32)
    if (rc != 0) {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "python exited with code %d", rc);
        }
        return -1;
    }
#else
    if (WIFEXITED(rc)) {
        int status = WEXITSTATUS(rc);
        if (status != 0) {
            if (error_buffer && error_buffer_len) {
                snprintf(error_buffer, error_buffer_len, "python exited with code %d", status);
            }
            return -1;
        }
    } else {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "python process terminated unexpectedly");
        }
        return -1;
    }
#endif
    if (QALLOW_ACCESS(QALLOW_TMP_RESULTS, R_OK) != 0) {
        if (error_buffer && error_buffer_len) {
            snprintf(error_buffer, error_buffer_len, "expected results file missing: %s", QALLOW_TMP_RESULTS);
        }
        return -1;
    }
    return 0;
}

static char* read_entire_file(const char* path, size_t* out_size) {
    if (!path) {
        return NULL;
    }
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char* buffer = (char*)malloc((size_t)size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    size_t read = fread(buffer, 1, (size_t)size, fp);
    fclose(fp);
    if (read != (size_t)size) {
        free(buffer);
        return NULL;
    }
    buffer[size] = '\0';
    if (out_size) {
        *out_size = (size_t)size;
    }
    return buffer;
}

static int write_text_file(const char* path, const char* data) {
    if (!path || !data) {
        return -1;
    }
    if (ensure_parent_directory(path) != 0) {
        return -1;
    }
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }
    size_t len = strlen(data);
    size_t written = fwrite(data, 1, len, fp);
    fclose(fp);
    return written == len ? 0 : -1;
}

static const char* find_matching_brace(const char* start) {
    if (!start || *start != '{') {
        return NULL;
    }
    int depth = 0;
    bool in_string = false;
    for (const char* p = start; *p; ++p) {
        char c = *p;
        if (c == '"') {
            bool escaped = (p > start && *(p - 1) == '\\');
            if (!escaped) {
                in_string = !in_string;
            }
            continue;
        }
        if (in_string) {
            continue;
        }
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) {
                return p;
            }
        }
    }
    return NULL;
}

static const char* find_matching_bracket(const char* start) {
    if (!start || *start != '[') {
        return NULL;
    }
    int depth = 0;
    bool in_string = false;
    for (const char* p = start; *p; ++p) {
        char c = *p;
        if (c == '"') {
            bool escaped = (p > start && *(p - 1) == '\\');
            if (!escaped) {
                in_string = !in_string;
            }
            continue;
        }
        if (in_string) {
            continue;
        }
        if (c == '[') {
            depth++;
        } else if (c == ']') {
            depth--;
            if (depth == 0) {
                return p;
            }
        }
    }
    return NULL;
}

static const char* locate_key(const char* start, const char* end, const char* key) {
    size_t key_len = strlen(key);
    const char* cursor = start;
    while (cursor && cursor < end) {
        const char* pos = strstr(cursor, key);
        if (!pos || pos >= end) {
            return NULL;
        }
        const char* colon = strchr(pos + key_len, ':');
        if (!colon || colon >= end) {
            cursor = pos + key_len;
            continue;
        }
        return colon + 1;
    }
    return NULL;
}

static int extract_string_between(const char* start, const char* end, const char* key, char* out, size_t out_len) {
    if (!out || out_len == 0) {
        return 0;
    }
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* value_start = locate_key(start, end, pattern);
    if (!value_start) {
        return 0;
    }
    while (value_start < end && isspace((unsigned char)*value_start)) {
        ++value_start;
    }
    if (value_start >= end || *value_start != '"') {
        return 0;
    }
    ++value_start;
    const char* value_end = value_start;
    while (value_end < end) {
        if (*value_end == '"' && *(value_end - 1) != '\\') {
            break;
        }
        ++value_end;
    }
    if (value_end >= end) {
        return 0;
    }
    size_t copy_len = (size_t)(value_end - value_start);
    if (copy_len >= out_len) {
        copy_len = out_len - 1;
    }
    memcpy(out, value_start, copy_len);
    out[copy_len] = '\0';
    return 1;
}

static int extract_bool_between(const char* start, const char* end, const char* key, bool* out_value) {
    if (!out_value) {
        return 0;
    }
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* value_start = locate_key(start, end, pattern);
    if (!value_start) {
        return 0;
    }
    while (value_start < end && isspace((unsigned char)*value_start)) {
        ++value_start;
    }
    if (value_start >= end) {
        return 0;
    }
    if (strncmp(value_start, "true", 4) == 0) {
        *out_value = true;
        return 1;
    }
    if (strncmp(value_start, "false", 5) == 0) {
        *out_value = false;
        return 1;
    }
    return 0;
}

static int extract_double_between(const char* start,
                                  const char* end,
                                  const char* key,
                                  double* out_value) {
    if (!out_value) {
        return 0;
    }
    char pattern[96];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* value_start = locate_key(start, end, pattern);
    if (!value_start) {
        return 0;
    }
    while (value_start < end && isspace((unsigned char)*value_start)) {
        ++value_start;
    }
    if (value_start >= end) {
        return 0;
    }
    if (strncmp(value_start, "np.float64", 10) == 0) {
        const char* paren = strchr(value_start, '(');
        if (!paren || paren >= end) {
            return 0;
        }
        value_start = paren + 1;
    }
    char* parse_end = NULL;
    double value = strtod(value_start, &parse_end);
    if (parse_end == value_start || parse_end > end) {
        return 0;
    }
    *out_value = value;
    return 1;
}

static void parse_timestamp(const char* payload, quantum_suite_metrics_t* metrics) {
    if (!payload || !metrics) {
        return;
    }
    const char* start = payload;
    const char* end = payload + strlen(payload);
    if (!extract_string_between(start, end, "timestamp", metrics->timestamp, sizeof(metrics->timestamp))) {
        iso_timestamp_now(metrics->timestamp, sizeof(metrics->timestamp));
    }
}

static int parse_algorithms(const char* payload, quantum_suite_metrics_t* metrics) {
    if (!payload || !metrics) {
        return -1;
    }
    const char* array_anchor = strstr(payload, "\"algorithms\"");
    if (!array_anchor) {
        return -1;
    }
    const char* array_start = strchr(array_anchor, '[');
    if (!array_start) {
        return -1;
    }
    const char* array_end = find_matching_bracket(array_start);
    if (!array_end) {
        return -1;
    }
    const char* cursor = array_start;
    int total = 0;
    int passed = 0;
    double grover_probability = -1.0;
    double vqe_best_energy = 0.0;
    while (cursor < array_end) {
        const char* algorithm_key = strstr(cursor, "\"algorithm\"");
        if (!algorithm_key || algorithm_key >= array_end) {
            break;
        }
        const char* item_start = strchr(algorithm_key, '{');
        if (!item_start || item_start >= array_end) {
            break;
        }
        const char* item_end = find_matching_brace(item_start);
        if (!item_end || item_end > array_end) {
            break;
        }
        char name[64];
        if (!extract_string_between(item_start, item_end, "algorithm", name, sizeof(name))) {
            cursor = item_end + 1;
            continue;
        }
        bool success = false;
        if (extract_bool_between(item_start, item_end, "success", &success)) {
            if (success) {
                passed++;
            }
        }
        total++;
        if (strcmp(name, "grover") == 0) {
            double value = -1.0;
            if (extract_double_between(item_start, item_end, "marked_state_probability", &value)) {
                grover_probability = value;
            }
        } else if (strcmp(name, "vqe") == 0) {
            double value = 0.0;
            if (extract_double_between(item_start, item_end, "best_energy", &value)) {
                vqe_best_energy = value;
            }
        }
        cursor = item_end + 1;
    }
    metrics->algorithms_total = total;
    metrics->algorithms_passed = passed;
    metrics->grover_probability = grover_probability;
    metrics->vqe_best_energy = vqe_best_energy;
    metrics->valid = total > 0;
    return metrics->valid ? 0 : -1;
}

static int write_export_json(const quantum_suite_metrics_t* metrics, const char* raw_payload) {
    if (!metrics) {
        return -1;
    }
    if (ensure_logs_directory() != 0) {
        return -1;
    }
    cJSON* root = cJSON_CreateObject();
    if (!root) {
        return -1;
    }
    cJSON_AddStringToObject(root, "generated_at", metrics->timestamp);
    cJSON_AddStringToObject(root, "source", "quantum_algorithms/unified_quantum_framework.py");

    cJSON* summary = cJSON_CreateObject();
    if (!summary) {
        cJSON_Delete(root);
        return -1;
    }
    cJSON_AddNumberToObject(summary, "algorithms_total", (double)metrics->algorithms_total);
    cJSON_AddNumberToObject(summary, "algorithms_passed", (double)metrics->algorithms_passed);
    if (metrics->grover_probability >= 0.0) {
        cJSON_AddNumberToObject(summary, "grover_probability", metrics->grover_probability);
    } else {
        cJSON_AddNullToObject(summary, "grover_probability");
    }
    cJSON_AddNumberToObject(summary, "vqe_best_energy", metrics->vqe_best_energy);
    cJSON_AddItemToObject(root, "summary", summary);

    if (raw_payload) {
        cJSON_AddStringToObject(root, "raw_payload", raw_payload);
    }

    char* json_string = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (!json_string) {
        return -1;
    }
    int rc = write_text_file(QALLOW_RESULTS_PATH, json_string);
    free(json_string);
    return rc;
}

static int update_combined_metrics_csv(const quantum_suite_metrics_t* metrics) {
    if (!metrics || !metrics->valid) {
        return -1;
    }
    if (ensure_logs_directory() != 0) {
        return -1;
    }
    bool write_header = (QALLOW_ACCESS(QALLOW_COMBINED_METRICS, F_OK) != 0);
    FILE* fp = fopen(QALLOW_COMBINED_METRICS, "a");
    if (!fp) {
        return -1;
    }
    if (write_header) {
        fprintf(fp, "timestamp,algorithms_total,algorithms_passed,grover_probability,vqe_best_energy\n");
    }
    fprintf(fp, "%s,%d,%d,", metrics->timestamp, metrics->algorithms_total, metrics->algorithms_passed);
    if (metrics->grover_probability >= 0.0) {
        fprintf(fp, "%.6f,", metrics->grover_probability);
    } else {
        fprintf(fp, "NA,");
    }
    fprintf(fp, "%.6f\n", metrics->vqe_best_energy);
    fclose(fp);
    return 0;
}

static void append_introspection_trace(const quantum_suite_metrics_t* metrics) {
    if (!metrics || !metrics->valid) {
        return;
    }
    if (ensure_logs_directory() != 0) {
        return;
    }
    bool write_header = (QALLOW_ACCESS(QALLOW_INTROSPECTION_TRACE, F_OK) != 0);
    FILE* fp = fopen(QALLOW_INTROSPECTION_TRACE, "a");
    if (!fp) {
        return;
    }
    if (write_header) {
        fprintf(fp, "timestamp,component,event,details\n");
    }
    char grover_buffer[32];
    if (metrics->grover_probability >= 0.0) {
        snprintf(grover_buffer, sizeof(grover_buffer), "%.3f", metrics->grover_probability);
    } else {
        snprintf(grover_buffer, sizeof(grover_buffer), "NA");
    }
    fprintf(fp,
            "\"%s\",phase16_quantum_suite,quantum_benchmark,\"passed=%d/%d;grover=%s;vqe_best=%.6f\"\n",
            metrics->timestamp,
            metrics->algorithms_passed,
            metrics->algorithms_total,
            grover_buffer,
            metrics->vqe_best_energy);
    fclose(fp);
}

int quantum_run_all(quantum_suite_metrics_t* out_metrics) {
    const char* skip_env = getenv("QALLOW_SKIP_QUANTUM_SUITE");
    if (skip_env && (*skip_env == '1' || *skip_env == 'T' || *skip_env == 't')) {
        metrics_reset(&g_latest_metrics);
        if (out_metrics) {
            metrics_reset(out_metrics);
        }
        qallow_log_info("quantum", "quantum suite skipped via environment override");
        return 1;
    }

    metrics_reset(&g_latest_metrics);
    quantum_suite_metrics_t metrics;
    metrics_reset(&metrics);

    char error_buffer[128];
    if (run_python_suite(error_buffer, sizeof(error_buffer)) != 0) {
        qallow_log_warn("quantum", "unified quantum framework execution failed: %s", error_buffer);
        if (out_metrics) {
            metrics_reset(out_metrics);
        }
        return -1;
    }

    size_t payload_size = 0;
    char* payload = read_entire_file(QALLOW_TMP_RESULTS, &payload_size);
    if (!payload) {
        qallow_log_warn("quantum", "unable to read quantum results payload from %s", QALLOW_TMP_RESULTS);
        if (out_metrics) {
            metrics_reset(out_metrics);
        }
        return -1;
    }

    parse_timestamp(payload, &metrics);
    if (parse_algorithms(payload, &metrics) != 0) {
        qallow_log_warn("quantum", "failed to parse algorithm metrics from payload");
        free(payload);
        if (out_metrics) {
            metrics_reset(out_metrics);
        }
        return -1;
    }

    if (write_export_json(&metrics, payload) != 0) {
        qallow_log_warn("quantum", "unable to persist quantum_results.json");
    }

    if (update_combined_metrics_csv(&metrics) != 0) {
        qallow_log_warn("quantum", "unable to update combined_metrics.csv");
    }

    append_introspection_trace(&metrics);

    g_latest_metrics = metrics;
    if (out_metrics) {
        *out_metrics = metrics;
    }

    qallow_log_info("quantum",
                    "suite complete algorithms=%d passed=%d grover=%.3f vqe_best=%.6f",
                    metrics.algorithms_total,
                    metrics.algorithms_passed,
                    metrics.grover_probability,
                    metrics.vqe_best_energy);

    free(payload);
    return 0;
}

const quantum_suite_metrics_t* quantum_suite_get_metrics(void) {
    if (!g_latest_metrics.valid) {
        return NULL;
    }
    return &g_latest_metrics;
}
