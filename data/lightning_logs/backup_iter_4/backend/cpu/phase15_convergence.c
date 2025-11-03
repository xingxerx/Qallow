#include "phase15.h"
#include "phase14.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bool active;
    bool no_split;
    bool audit_unified;
    float convergence_signal;
    float audit_score;
    float entropy_index;
    unsigned tick_count;
    bool seeded_from_quantum;
} phase15_state_internal_t;

static phase15_state_internal_t g_phase15_state;

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

static const char* phase15_metrics_default_path(int index) {
    static const char* fallbacks[] = {
        "data/quantum/phase15_metrics.json",
        "data/calibrations/phase15_metrics.json",
        "phase15_metrics.json",
        NULL,
    };
    return fallbacks[index];
}

static const char* phase15_next_candidate_path(const char* env_path, int* cursor) {
    if (*cursor == 0 && env_path && *env_path) {
        (*cursor)++;
        return env_path;
    }
    const char* fallback = phase15_metrics_default_path(*cursor - (env_path && *env_path ? 1 : 0));
    if (fallback) {
        (*cursor)++;
    }
    return fallback;
}

static bool phase15_parse_json_double(const char* json, const char* key, double* out_val) {
    if (!json || !key || !out_val) {
        return false;
    }
    char pattern[64];
    size_t key_len = strlen(key);
    if (key_len >= sizeof(pattern) - 4) {
        return false;
    }
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* cursor = strstr(json, pattern);
    if (!cursor) {
        return false;
    }
    cursor += strlen(pattern);
    while (*cursor && (isspace((unsigned char)*cursor) || *cursor == ':')) {
        cursor++;
    }
    if (!*cursor) {
        return false;
    }
    char* endptr = NULL;
    double value = strtod(cursor, &endptr);
    if (cursor == endptr) {
        return false;
    }
    *out_val = value;
    return true;
}

static bool phase15_load_quantum_seed(float* score_out,
                                      float* stability_out,
                                      float* convergence_tick_out) {
    const char* env_path = getenv("QALLOW_PHASE15_METRICS");
    int cursor = 0;
    const char* candidate = NULL;
    while ((candidate = phase15_next_candidate_path(env_path, &cursor)) != NULL) {
        FILE* f = fopen(candidate, "rb");
        if (!f) {
            continue;
        }
        char buffer[4096];
        size_t n = fread(buffer, 1, sizeof(buffer) - 1, f);
        fclose(f);
        buffer[n] = '\0';

        double score = 0.0;
        double stability = 0.0;
        double convergence_tick = 0.0;

        bool ok = true;
        ok = ok && phase15_parse_json_double(buffer, "score", &score);
        ok = ok && phase15_parse_json_double(buffer, "stability", &stability);
        if (phase15_parse_json_double(buffer, "convergence_tick", &convergence_tick)) {
            // optional field; ignore failure
        }

        if (ok) {
            if (score_out) *score_out = (float)score;
            if (stability_out) *stability_out = (float)stability;
            if (convergence_tick_out) *convergence_tick_out = (float)convergence_tick;
            printf("[PHASE15] Quantum seed loaded from %s (score=%.6f stability=%.6f)\n",
                   candidate,
                   score,
                   stability);
            return true;
        }
    }
    return false;
}

void phase15_initialize(const qallow_state_t* state) {
    memset(&g_phase15_state, 0, sizeof(g_phase15_state));
    g_phase15_state.convergence_signal = state ? clamp01(state->global_coherence) : 0.4f;
    g_phase15_state.audit_score = state ? clamp01((state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f) : 0.5f;
    g_phase15_state.entropy_index = state ? clamp01(1.0f - state->decoherence_level * 8.0f) : 0.5f;

    float score = 0.0f;
    float stability = 0.0f;
    float convergence_tick = 0.0f;
    if (phase15_load_quantum_seed(&score, &stability, &convergence_tick)) {
        g_phase15_state.convergence_signal = clamp01(score);
        g_phase15_state.audit_score = clamp01((score + stability) * 0.5f);
        g_phase15_state.entropy_index = clamp01(stability);
        (void)convergence_tick;
        g_phase15_state.seeded_from_quantum = true;
    }
}

void phase15_configure(const phase15_config_t* cfg) {
    if (!cfg) {
        g_phase15_state.active = false;
        g_phase15_state.no_split = false;
        g_phase15_state.audit_unified = false;
        return;
    }

    g_phase15_state.active = cfg->enable;
    g_phase15_state.no_split = cfg->no_split_mode;
    g_phase15_state.audit_unified = cfg->audit_unified;
}

void phase15_tick(qallow_state_t* state) {
    if (!g_phase15_state.active || !state) {
        return;
    }

    float entanglement = phase14_is_active() ? phase14_get_entanglement_strength() : state->global_coherence;
    float ethics_mean = clamp01((state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f);
    float base_convergence = clamp01(state->global_coherence * 0.5f + entanglement * 0.3f + ethics_mean * 0.2f);

    float no_split_gain = g_phase15_state.no_split ? 1.08f : 1.0f;
    g_phase15_state.convergence_signal = clamp01(0.86f * g_phase15_state.convergence_signal + 0.14f * base_convergence * no_split_gain);

    for (int i = 0; i < NUM_OVERLAYS; ++i) {
        float current = clamp01(state->overlays[i].stability);
        float blend = g_phase15_state.convergence_signal * (g_phase15_state.no_split ? 0.24f : 0.18f);
        state->overlays[i].stability = clamp01(current * (1.0f - blend) + g_phase15_state.convergence_signal * blend);
    }

    state->global_coherence = (state->overlays[OVERLAY_ORBITAL].stability +
                               state->overlays[OVERLAY_RIVER_DELTA].stability +
                               state->overlays[OVERLAY_MYCELIAL].stability) / 3.0f;

    float ethics_target = clamp01((g_phase15_state.convergence_signal + ethics_mean) * 0.5f);
    float ethics_gain = g_phase15_state.no_split ? 0.12f : 0.08f;
    state->ethics_S = clamp01(state->ethics_S * (1.0f - ethics_gain) + ethics_target * ethics_gain);
    state->ethics_C = clamp01(state->ethics_C * (1.0f - ethics_gain) + ethics_target * ethics_gain);
    state->ethics_H = clamp01(state->ethics_H * (1.0f - ethics_gain) + ethics_target * ethics_gain);

    float audit_signal = state->global_coherence * ethics_mean;
    float audit_gain = g_phase15_state.audit_unified ? 0.18f : 0.10f;
    g_phase15_state.audit_score = clamp01(0.88f * g_phase15_state.audit_score + audit_gain * audit_signal);

    float entropy_delta = g_phase15_state.convergence_signal * 0.0006f;
    state->decoherence_level -= entropy_delta;
    if (state->decoherence_level < 0.0f) {
        state->decoherence_level = 0.0f;
    }
    if (state->decoherence_level > 0.1f) {
        state->decoherence_level = 0.1f;
    }

    g_phase15_state.entropy_index = clamp01(0.90f * g_phase15_state.entropy_index + 0.10f * (1.0f - state->decoherence_level * 8.0f));
    g_phase15_state.tick_count++;
}

void phase15_collect_metrics(phase15_metrics_t* out) {
    if (!out) {
        return;
    }

    out->active = g_phase15_state.active;
    out->convergence_signal = g_phase15_state.convergence_signal;
    out->audit_score = g_phase15_state.audit_score;
    out->entropy_index = g_phase15_state.entropy_index;
}

float phase15_get_convergence(void) {
    return g_phase15_state.convergence_signal;
}

bool phase15_is_active(void) {
    return g_phase15_state.active;
}
