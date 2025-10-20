#include "runtime/meta_introspect.h"

#include "meta_introspect.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define META_EVENT_CAP 256
#define META_ROLLUP_CAP 128

static void meta_safe_localtime(time_t now, struct tm* out) {
    if (!out) {
        return;
    }
    struct tm* tm_ptr = localtime(&now);
    if (tm_ptr) {
        *out = *tm_ptr;
    } else {
        memset(out, 0, sizeof(*out));
    }
}

extern int qallow_meta_introspect_gpu(const float* durations,
                                      const float* coherence,
                                      const float* ethics,
                                      float* improvement_scores,
                                      int count);

typedef struct meta_event_record_s {
    char phase[32];
    char module[64];
    char objective_id[64];
    float duration_s;
    float coherence;
    float ethics;
    float score;
} meta_event_record_t;

typedef struct meta_rollup_entry_s {
    char phase[32];
    char module[64];
    char objective_id[64];
    float duration_total;
    float score_total;
    int event_count;
} meta_rollup_entry_t;

typedef struct objective_entry_s {
    char id[64];
    char label[128];
} objective_entry_t;

static meta_event_record_t g_events[META_EVENT_CAP];
static size_t g_event_count = 0;

static meta_rollup_entry_t g_rollups[META_ROLLUP_CAP];
static size_t g_rollup_count = 0;

static objective_entry_t g_objectives[64];
static size_t g_objective_count = 0;

static int g_enabled = 0;
static int g_gpu_enabled = 0;
static int g_configured = 0;
static int g_config_explicit = 0;
static int g_enable_explicit = 0;

static char g_log_dir[PATH_MAX] = "/var/log/qallow";
static char g_csv_path[PATH_MAX] = "";
static char g_json_path[PATH_MAX] = "";
static char g_rollup_path[PATH_MAX] = "";
static char g_hints_path[PATH_MAX] = "";
static char g_objective_map_path[PATH_MAX] = "";

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int ensure_directory(const char* path) {
    if (!path || !*path) return -1;

    char current[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(current)) return -1;
    strcpy(current, path);

    for (size_t i = 1; i < len; ++i) {
        if (current[i] == '/' || current[i] == '\\') {
            char saved = current[i];
            current[i] = '\0';
            if (current[0] && mkdir(current, 0755) != 0) {
                if (errno != EEXIST) {
                    current[i] = saved;
                    return -1;
                }
            }
            current[i] = saved;
        }
    }

    if (mkdir(current, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int ensure_parent_directory(const char* path) {
    if (!path || !*path) return -1;
    char temp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(temp)) return -1;
    strcpy(temp, path);
    for (size_t i = len; i > 0; --i) {
        if (temp[i - 1] == '/' || temp[i - 1] == '\\') {
            temp[i - 1] = '\0';
            break;
        }
    }
    if (!temp[0]) {
        return 0;
    }
    return ensure_directory(temp);
}

static void build_paths(void) {
    const char* csv_suffix = "/self_state.csv";
    const char* json_suffix = "/self_state.json";
    const char* rollup_suffix = "/rollup_daily.json";
    const char* hints_suffix = "/hints.ndjson";
    size_t base_len = strlen(g_log_dir);

    if (base_len + strlen(csv_suffix) < sizeof(g_csv_path)) {
        snprintf(g_csv_path, sizeof(g_csv_path), "%s%s", g_log_dir, csv_suffix);
    } else {
        g_csv_path[0] = '\0';
    }

    if (base_len + strlen(json_suffix) < sizeof(g_json_path)) {
        snprintf(g_json_path, sizeof(g_json_path), "%s%s", g_log_dir, json_suffix);
    } else {
        g_json_path[0] = '\0';
    }

    if (base_len + strlen(rollup_suffix) < sizeof(g_rollup_path)) {
        snprintf(g_rollup_path, sizeof(g_rollup_path), "%s%s", g_log_dir, rollup_suffix);
    } else {
        g_rollup_path[0] = '\0';
    }

    if (base_len + strlen(hints_suffix) < sizeof(g_hints_path)) {
        snprintf(g_hints_path, sizeof(g_hints_path), "%s%s", g_log_dir, hints_suffix);
    } else {
        g_hints_path[0] = '\0';
    }
}

static void reset_objectives(void) {
    g_objective_count = 0;
    g_objectives[0].id[0] = '\0';
}

static void load_objectives(const char* path) {
    reset_objectives();
    if (!path || !*path) {
        return;
    }
    FILE* f = fopen(path, "r");
    if (!f) {
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char key[64];
        char value[128];
        if (sscanf(line, " \"%63[^\"]\"%*[^\"]\"%127[^\"]\"", key, value) == 2) {
            if (g_objective_count < sizeof(g_objectives) / sizeof(g_objectives[0])) {
                snprintf(g_objectives[g_objective_count].id,
                         sizeof(g_objectives[g_objective_count].id),
                         "%s",
                         key);
                snprintf(g_objectives[g_objective_count].label,
                         sizeof(g_objectives[g_objective_count].label),
                         "%s",
                         value);
                g_objective_count++;
            }
        }
    }
    fclose(f);
}

static const char* lookup_objective_label(const char* id) {
    if (!id || !*id) return "";
    for (size_t i = 0; i < g_objective_count; ++i) {
        if (strcmp(g_objectives[i].id, id) == 0) {
            return g_objectives[i].label;
        }
    }
    return "";
}

static meta_rollup_entry_t* get_or_create_rollup(const char* phase, const char* module) {
    for (size_t i = 0; i < g_rollup_count; ++i) {
        if (strcmp(g_rollups[i].phase, phase) == 0 && strcmp(g_rollups[i].module, module) == 0) {
            return &g_rollups[i];
        }
    }
    if (g_rollup_count >= META_ROLLUP_CAP) {
        return NULL;
    }
    meta_rollup_entry_t* entry = &g_rollups[g_rollup_count++];
    memset(entry, 0, sizeof(*entry));
    if (phase) {
        size_t len = strlen(phase);
        if (len >= sizeof(entry->phase)) {
            len = sizeof(entry->phase) - 1;
        }
        memcpy(entry->phase, phase, len);
        entry->phase[len] = '\0';
    } else {
        entry->phase[0] = '\0';
    }

    if (module) {
        size_t len = strlen(module);
        if (len >= sizeof(entry->module)) {
            len = sizeof(entry->module) - 1;
        }
        memcpy(entry->module, module, len);
        entry->module[len] = '\0';
    } else {
        entry->module[0] = '\0';
    }
    return entry;
}

static void write_snapshot(size_t events_written) {
    if (!g_json_path[0]) return;
    if (ensure_parent_directory(g_json_path) != 0) return;

    FILE* f = fopen(g_json_path, "w");
    if (!f) return;

    time_t now = time(NULL);
    fprintf(f, "{ \"updated\": %ld, \"csv\": \"%s\", \"events\": %zu }\n",
            (long)now, g_csv_path, events_written);
    fclose(f);
}

static void write_rollup_json(void) {
    if (!g_rollup_path[0]) return;
    if (ensure_parent_directory(g_rollup_path) != 0) return;

    FILE* f = fopen(g_rollup_path, "w");
    if (!f) return;

    time_t now = time(NULL);
    struct tm tm_now;
    meta_safe_localtime(now, &tm_now);

    fprintf(f, "{\n  \"date\": \"%04d-%02d-%02d\",\n  \"entries\": [\n",
            tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday);

    for (size_t i = 0; i < g_rollup_count; ++i) {
        const meta_rollup_entry_t* entry = &g_rollups[i];
        float avg_score = entry->event_count > 0 ? entry->score_total / (float)entry->event_count : 0.0f;
        const char* label = lookup_objective_label(entry->objective_id);
        fprintf(f,
                "    { \"phase\": \"%s\", \"module\": \"%s\", \"objective\": \"%s\", \"label\": \"%s\", \"events\": %d, \"duration_s\": %.3f, \"avg_score\": %.3f }%s\n",
                entry->phase,
                entry->module,
                entry->objective_id,
                label,
                entry->event_count,
                entry->duration_total,
                avg_score,
                (i + 1) < g_rollup_count ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

static void emit_hint(const meta_rollup_entry_t* entry, float avg_score) {
    if (!entry || !g_hints_path[0]) return;
    if (avg_score >= 0.75f || entry->duration_total < 600.0f) return;

    if (ensure_parent_directory(g_hints_path) != 0) return;

    FILE* hints = fopen(g_hints_path, "a");
    if (!hints) return;

    time_t now = time(NULL);
    fprintf(hints,
            "{\"ts\":%ld,\"action\":\"decrease_lr\",\"target\":\"%s.%s\",\"reason\":\"low_score\",\"score\":%.3f}\n",
            (long)now,
            entry->phase,
            entry->module,
            avg_score);
    fclose(hints);
}

static int configure_internal(const char* base_dir, const char* objective_map_path, int explicit_source) {
    const char* resolved_dir = (base_dir && *base_dir) ? base_dir : "/var/log/qallow";
    if (snprintf(g_log_dir, sizeof(g_log_dir), "%s", resolved_dir) >= (int)sizeof(g_log_dir)) {
        return -1;
    }
    build_paths();
    if (ensure_directory(g_log_dir) != 0) {
        return -1;
    }

    const char* resolved_objectives = (objective_map_path && *objective_map_path) ? objective_map_path : "/etc/qallow/objectives.json";
    if (snprintf(g_objective_map_path, sizeof(g_objective_map_path), "%s", resolved_objectives) >= (int)sizeof(g_objective_map_path)) {
        g_objective_map_path[0] = '\0';
    }
    load_objectives(g_objective_map_path);

    g_configured = 1;
    if (explicit_source) {
        g_config_explicit = 1;
    }
    return 0;
}

int meta_introspect_configure(const char* base_dir, const char* objective_map_path) {
    return configure_internal(base_dir, objective_map_path, 1);
}

void meta_introspect_enable(int enabled) {
    g_enabled = enabled ? 1 : 0;
    g_enable_explicit = 1;
    if (g_enabled && !g_configured) {
        configure_internal(NULL, NULL, 1);
    }
}

int meta_introspect_enabled(void) {
    return g_enabled;
}

void meta_introspect_set_gpu_available(int available) {
    g_gpu_enabled = available ? 1 : 0;
}

void meta_introspect_apply_environment_defaults(void) {
    const char* env_dir = getenv("QALLOW_SELF_AUDIT_PATH");
    const char* env_obj = getenv("QALLOW_OBJECTIVES_PATH");
    if (!g_configured || !g_config_explicit) {
        configure_internal(env_dir, env_obj, 0);
    }

    const char* env_enable = getenv("QALLOW_SELF_AUDIT");
    if (env_enable && !g_enable_explicit) {
        int enable = 0;
        if (*env_enable) {
            if (strcmp(env_enable, "1") == 0 || strcmp(env_enable, "true") == 0 || strcmp(env_enable, "TRUE") == 0) {
                enable = 1;
            }
        }
        meta_introspect_enable(enable);
        g_enable_explicit = 0; // preserve ability for user override later
    }
}

static void ensure_configured_defaults(void) {
    if (!g_configured) {
        configure_internal(NULL, NULL, 0);
    }
}

void meta_introspect_push(const learn_event_t* event) {
    if (!event || !g_enabled) {
        return;
    }
    ensure_configured_defaults();

    if (g_event_count >= META_EVENT_CAP) {
        meta_introspect_flush();
        if (g_event_count >= META_EVENT_CAP) {
            return;
        }
    }

    meta_event_record_t* rec = &g_events[g_event_count++];
    memset(rec, 0, sizeof(*rec));
    if (event->phase) {
        snprintf(rec->phase, sizeof(rec->phase), "%s", event->phase);
    }
    if (event->module) {
        snprintf(rec->module, sizeof(rec->module), "%s", event->module);
    }
    if (event->objective_id) {
        snprintf(rec->objective_id, sizeof(rec->objective_id), "%s", event->objective_id);
    }
    rec->duration_s = event->duration_s;
    rec->coherence = clampf(event->coherence, 0.0f, 1.0f);
    rec->ethics = clampf(event->ethics, 0.0f, 1.0f);
    float duration = fmaxf(rec->duration_s, 0.0f);
    float fallback = 0.4f * rec->coherence + 0.4f * rec->ethics + 0.2f * logf(1.0f + duration);
    rec->score = clampf(fallback, 0.0f, 1.0f);
}

void meta_introspect_flush(void) {
    if (!g_enabled) {
        write_snapshot(0);
        return;
    }
    ensure_configured_defaults();

    if (g_event_count == 0) {
        write_snapshot(0);
        return;
    }

    float durations[META_EVENT_CAP];
    float coherences[META_EVENT_CAP];
    float ethics[META_EVENT_CAP];
    for (size_t i = 0; i < g_event_count; ++i) {
        durations[i] = g_events[i].duration_s;
        coherences[i] = g_events[i].coherence;
        ethics[i] = g_events[i].ethics;
    }

    if (g_gpu_enabled) {
        float scores[META_EVENT_CAP];
        if (qallow_meta_introspect_gpu(durations, coherences, ethics, scores, (int)g_event_count) == 0) {
            for (size_t i = 0; i < g_event_count; ++i) {
                g_events[i].score = clampf(scores[i], 0.0f, 1.0f);
            }
        }
    }

    if (ensure_parent_directory(g_csv_path) != 0) {
        g_event_count = 0;
        return;
    }

    int need_header = 0;
    FILE* existing = fopen(g_csv_path, "r");
    if (!existing) {
        need_header = 1;
    } else {
        fclose(existing);
    }

    FILE* csv = fopen(g_csv_path, "a");
    if (!csv) {
        g_event_count = 0;
        return;
    }
    if (need_header) {
        fprintf(csv, "timestamp,phase,module,duration_s,coherence,ethics,score,objective_id\n");
    }

    time_t now = time(NULL);
    size_t events_written = 0;
    for (size_t i = 0; i < g_event_count; ++i) {
        meta_event_record_t* rec = &g_events[i];
        fprintf(csv, "%ld,%s,%s,%.3f,%.3f,%.3f,%.3f,%s\n",
                (long)now,
                rec->phase,
                rec->module,
                rec->duration_s,
                rec->coherence,
                rec->ethics,
                rec->score,
                rec->objective_id);
        events_written++;

        meta_rollup_entry_t* roll = get_or_create_rollup(rec->phase, rec->module);
        if (roll) {
            roll->duration_total += rec->duration_s;
            roll->score_total += rec->score;
            roll->event_count += 1;
            if (!roll->objective_id[0] && rec->objective_id[0]) {
                snprintf(roll->objective_id, sizeof(roll->objective_id), "%s", rec->objective_id);
            }
        }
    }
    fclose(csv);

    for (size_t i = 0; i < g_rollup_count; ++i) {
        meta_rollup_entry_t* roll = &g_rollups[i];
        float avg_score = roll->event_count > 0 ? roll->score_total / (float)roll->event_count : 0.0f;
        emit_hint(roll, avg_score);
    }

    write_snapshot(events_written);
    write_rollup_json();

    g_event_count = 0;
}

const char* meta_introspect_log_dir(void) {
    return g_log_dir;
}

int meta_introspect_export_pocket_map(const char* output_path) {
    if (!output_path || !*output_path) {
        return -1;
    }
    if (ensure_parent_directory(output_path) != 0) {
        return -1;
    }

    static const char* pockets[][2] = {
        {"core", "\u2705"},
        {"ethics", "\u2705"},
        {"overlay", "\u2705"},
        {"sandbox", "\u2705"},
        {"pocket", "\u2705"},
        {"govern", "\u2705"},
        {"telemetry", "\u2705"},
        {"accelerator", "\u2705"},
        {"quantum-cache", "\u2705"}
    };

    FILE* f = fopen(output_path, "w");
    if (!f) {
        return -1;
    }

    time_t now = time(NULL);
    struct tm tm_now;
    meta_safe_localtime(now, &tm_now);

    fprintf(f, "{\n  \"date\": \"%04d-%02d-%02d\",\n  \"pockets\": [\n",
            tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday);
    for (size_t i = 0; i < sizeof(pockets) / sizeof(pockets[0]); ++i) {
        fprintf(f, "    {\"id\":%zu,\"name\":\"%s\",\"status\":\"%s\"}%s\n",
                i,
                pockets[i][0],
                pockets[i][1],
                (i + 1) < (sizeof(pockets) / sizeof(pockets[0])) ? "," : "");
    }
    fprintf(f, "  ],\n  \"auditor\": \"phase16_meta_introspect\"\n}\n");
    fclose(f);
    return 0;
}
