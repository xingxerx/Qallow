#include "phase14.h"

#include <math.h>
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

void phase14_initialize(const qallow_state_t* state) {
    memset(&g_phase14_state, 0, sizeof(g_phase14_state));
    g_phase14_state.entanglement_strength = state ? clamp01(state->global_coherence) : 0.5f;
    g_phase14_state.ethics_alignment = 0.5f;
    g_phase14_state.decoherence_buffer = 0.5f;
    g_phase14_state.share_cuda = state && state->cuda_enabled;
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
