#include "phase15.h"
#include "phase14.h"

#include <math.h>
#include <string.h>

typedef struct {
    bool active;
    bool no_split;
    bool audit_unified;
    float convergence_signal;
    float audit_score;
    float entropy_index;
    unsigned tick_count;
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

void phase15_initialize(const qallow_state_t* state) {
    memset(&g_phase15_state, 0, sizeof(g_phase15_state));
    g_phase15_state.convergence_signal = state ? clamp01(state->global_coherence) : 0.4f;
    g_phase15_state.audit_score = state ? clamp01((state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f) : 0.5f;
    g_phase15_state.entropy_index = state ? clamp01(1.0f - state->decoherence_level * 8.0f) : 0.5f;
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
