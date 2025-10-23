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
    bool share_cuda;
    float entanglement_strength;
    float ethics_alignment;
    float pocket_flux;
    float decoherence_buffer;
    unsigned tick_count;
    bool seeded_from_quantum;
} phase14_state_internal_t;

static phase14_state_internal_t g_phase14_state;

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

static const char* phase14_metrics_default_path(int index) {
    static const char* fallbacks[] = {
        "data/quantum/phase14_metrics.json",
        "data/calibrations/phase14_metrics.json",
        "phase14_metrics.json",
        NULL,
    };
    return fallbacks[index];
}

static const char* phase14_next_candidate_path(const char* env_path, int* cursor) {
    if (*cursor == 0 && env_path && *env_path) {
        (*cursor)++;
        return env_path;
    }
    const char* fallback = phase14_metrics_default_path(*cursor - (env_path && *env_path ? 1 : 0));
    if (fallback) {
        (*cursor)++;
    }
    return fallback;
}

static bool phase14_parse_json_double(const char* json, const char* key, double* out_val) {
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

static bool phase14_load_quantum_seed(float* fidelity_out,
                                      float* target_out,
                                      float* alpha_base_out,
                                      float* alpha_used_out) {
    const char* env_path = getenv("QALLOW_PHASE14_METRICS");
    int cursor = 0;
    const char* candidate = NULL;
    while ((candidate = phase14_next_candidate_path(env_path, &cursor)) != NULL) {
        FILE* f = fopen(candidate, "rb");
        if (!f) {
            continue;
        }
        char buffer[4096];
        size_t n = fread(buffer, 1, sizeof(buffer) - 1, f);
        fclose(f);
        buffer[n] = '\0';

        double fidelity = 0.0;
        double target = 0.0;
        double alpha_base = 0.0;
        double alpha_used = 0.0;

        bool ok = true;
        ok = ok && phase14_parse_json_double(buffer, "fidelity", &fidelity);
        ok = ok && phase14_parse_json_double(buffer, "target", &target);
        ok = ok && phase14_parse_json_double(buffer, "alpha_base", &alpha_base);
        ok = ok && phase14_parse_json_double(buffer, "alpha_used", &alpha_used);

        if (ok) {
            if (fidelity_out) *fidelity_out = (float)fidelity;
            if (target_out) *target_out = (float)target;
            if (alpha_base_out) *alpha_base_out = (float)alpha_base;
            if (alpha_used_out) *alpha_used_out = (float)alpha_used;
            printf("[PHASE14] Quantum seed loaded from %s (fidelity=%.6f target=%.6f)\n",
                   candidate,
                   fidelity,
                   target);
            return true;
        }
    }
    return false;
}

void phase14_initialize(const qallow_state_t* state) {
    memset(&g_phase14_state, 0, sizeof(g_phase14_state));
    g_phase14_state.entanglement_strength = state ? clamp01(state->global_coherence) : 0.5f;
    g_phase14_state.ethics_alignment = 0.5f;
    g_phase14_state.decoherence_buffer = 0.5f;
    g_phase14_state.share_cuda = state && state->cuda_enabled;

    float fidelity = 0.0f;
    float target = 0.0f;
    float alpha_base = 0.0f;
    float alpha_used = 0.0f;
    if (phase14_load_quantum_seed(&fidelity, &target, &alpha_base, &alpha_used)) {
        g_phase14_state.entanglement_strength = clamp01(fidelity);
        g_phase14_state.ethics_alignment = clamp01(target);
        g_phase14_state.decoherence_buffer = clamp01(1.0f - alpha_used * 128.0f);
        g_phase14_state.pocket_flux = clamp01(fabsf(target - fidelity) * 4.0f);
        g_phase14_state.seeded_from_quantum = true;
        (void)alpha_base;
    }
}

void phase14_configure(const phase14_config_t* cfg) {
    if (!cfg) {
        g_phase14_state.active = false;
        g_phase14_state.no_split = false;
        return;
    }

    g_phase14_state.active = cfg->enable;
    g_phase14_state.no_split = cfg->no_split_mode;
    g_phase14_state.share_cuda = cfg->share_cuda_blocks;
}

void phase14_tick(qallow_state_t* state) {
    if (!g_phase14_state.active || !state) {
        return;
    }

    float orbital = clamp01(state->overlays[OVERLAY_ORBITAL].stability);
    float river = clamp01(state->overlays[OVERLAY_RIVER_DELTA].stability);
    float mycelial = clamp01(state->overlays[OVERLAY_MYCELIAL].stability);
    float mean = (orbital + river + mycelial) / 3.0f;

    float span = (fabsf(orbital - mean) + fabsf(river - mean) + fabsf(mycelial - mean)) / 3.0f;
    float ethics_mean = clamp01((state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f);

    float coherence_drive = clamp01(1.0f - span);
    float cuda_bias = g_phase14_state.share_cuda ? 1.08f : 0.97f;
    float mode_bias = g_phase14_state.no_split ? 1.12f : 1.0f;
    float coupling = (0.02f + ethics_mean * 0.03f + coherence_drive * 0.01f) * cuda_bias * mode_bias;
    if (coupling > 0.18f) coupling = 0.18f;
    if (coupling < 0.0f) coupling = 0.0f;

    g_phase14_state.entanglement_strength = clamp01(0.90f * g_phase14_state.entanglement_strength + 0.10f * coherence_drive * cuda_bias);
    g_phase14_state.ethics_alignment = clamp01(0.88f * g_phase14_state.ethics_alignment + 0.12f * ethics_mean);
    g_phase14_state.decoherence_buffer = clamp01(0.88f * g_phase14_state.decoherence_buffer + 0.12f * (1.0f - state->decoherence_level * 8.0f));
    g_phase14_state.pocket_flux = clamp01(0.85f * g_phase14_state.pocket_flux + 0.15f * span);

    for (int i = 0; i < NUM_OVERLAYS; ++i) {
        float current = clamp01(state->overlays[i].stability);
        float adjusted = current * (1.0f - coupling) + mean * coupling;
        state->overlays[i].stability = clamp01(adjusted);
    }

    state->global_coherence = (state->overlays[OVERLAY_ORBITAL].stability +
                               state->overlays[OVERLAY_RIVER_DELTA].stability +
                               state->overlays[OVERLAY_MYCELIAL].stability) / 3.0f;

    float deco_reduction = 1.0f - (0.0035f * g_phase14_state.entanglement_strength);
    if (deco_reduction < 0.90f) deco_reduction = 0.90f;
    state->decoherence_level *= deco_reduction;
    if (state->decoherence_level < 0.0f) {
        state->decoherence_level = 0.0f;
    }

    g_phase14_state.tick_count++;
}

void phase14_collect_metrics(phase14_metrics_t* out) {
    if (!out) {
        return;
    }

    out->active = g_phase14_state.active;
    out->entanglement_strength = g_phase14_state.entanglement_strength;
    out->ethics_alignment = g_phase14_state.ethics_alignment;
    out->pocket_flux = g_phase14_state.pocket_flux;
    out->decoherence_buffer = g_phase14_state.decoherence_buffer;
}

float phase14_get_entanglement_strength(void) {
    return g_phase14_state.entanglement_strength;
}

bool phase14_is_active(void) {
    return g_phase14_state.active;
}

/**
 * phase14_gain_from_csr: Extract alpha_eff from a CSV file containing J-graph data
 *
 * @param csv_path: Path to the CSV file
 * @param N: Number of nodes (used for validation)
 * @param out_alpha_eff: Output parameter for the computed alpha_eff value
 * @param gain_base: Base gain value
 * @param gain_span: Gain span value
 * @return: 0 on success, non-zero on failure
 */
int phase14_gain_from_csr(const char* csv_path, int N, double* out_alpha_eff,
                          double gain_base, double gain_span) {
    if (!csv_path || !out_alpha_eff) {
        return -1;
    }

    FILE* f = fopen(csv_path, "r");
    if (!f) {
        return -1;
    }

    // Initialize output
    *out_alpha_eff = 0.0;

    // Read CSV and compute alpha_eff from J-graph data
    // Expected format: each line contains J values or coupling strengths
    char line[1024];
    int line_count = 0;
    double sum_j = 0.0;
    int valid_entries = 0;

    while (fgets(line, sizeof(line), f)) {
        line_count++;

        // Skip empty lines and comments
        if (line[0] == '\0' || line[0] == '#' || line[0] == '\n') {
            continue;
        }

        // Parse CSV values (simple comma-separated format)
        char* ptr = line;
        double val;
        while (sscanf(ptr, "%lf", &val) == 1) {
            sum_j += val;
            valid_entries++;

            // Move to next comma or end of line
            while (*ptr && *ptr != ',' && *ptr != '\n') {
                ptr++;
            }
            if (*ptr == ',') {
                ptr++;
            } else {
                break;
            }
        }
    }

    fclose(f);

    // Compute alpha_eff from the J-graph data
    if (valid_entries > 0) {
        double mean_j = sum_j / valid_entries;
        // Alpha is derived from gain parameters and J-graph statistics
        // Formula: alpha_eff = gain_base + (mean_j / N) * gain_span
        *out_alpha_eff = gain_base + (mean_j / (double)N) * gain_span;

        // Clamp to reasonable range [0.0001, 0.1]
        if (*out_alpha_eff < 0.0001) {
            *out_alpha_eff = 0.0001;
        }
        if (*out_alpha_eff > 0.1) {
            *out_alpha_eff = 0.1;
        }

        return 0;
    }

    return -1;
}
