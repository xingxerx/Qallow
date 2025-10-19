#include "qallow_phase14.h"

#include <math.h>
#include <string.h>

static float clamp01(float value) {
    if (value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static float overlay_mean(const overlay_t* overlay) {
    if (!overlay || overlay->node_count <= 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int i = 0; i < overlay->node_count; ++i) {
        sum += overlay->values[i];
    }
    return sum / (float)overlay->node_count;
}

static float compute_alignment_metric(const float* means, int count) {
    if (!means || count <= 0) {
        return 0.0f;
    }
    float global_mean = 0.0f;
    for (int i = 0; i < count; ++i) {
        global_mean += means[i];
    }
    global_mean /= (float)count;

    float deviation = 0.0f;
    for (int i = 0; i < count; ++i) {
        deviation += fabsf(means[i] - global_mean);
    }
    return deviation / (float)count;
}

void phase14_config_default(phase14_config_t* cfg) {
    if (!cfg) {
        return;
    }
    cfg->iterations = 12;
    cfg->coupling_gain = 0.08f;
    cfg->harmonic_gain = 0.015f;
    cfg->decoherence_damping = 0.01f;
    cfg->ethics_gain = 0.006f;
}

void phase14_entanglement_integrate(qallow_state_t* state,
                                    const phase14_config_t* cfg,
                                    phase14_status_t* status) {
    if (!state || !cfg) {
        return;
    }

    phase14_status_t local_status = {0};
    const float baseline_coherence = state->global_coherence;
    float initial_means[NUM_OVERLAYS];
    for (int i = 0; i < NUM_OVERLAYS; ++i) {
        initial_means[i] = overlay_mean(&state->overlays[i]);
    }
    const float alignment_before = compute_alignment_metric(initial_means, NUM_OVERLAYS);

    const float damping = fmaxf(0.0f, fminf(cfg->decoherence_damping, 0.05f));
    const float ethics_gain = fmaxf(0.0f, cfg->ethics_gain);

    for (int iter = 0; iter < cfg->iterations; ++iter) {
        float overlay_means[NUM_OVERLAYS];
        for (int i = 0; i < NUM_OVERLAYS; ++i) {
            overlay_means[i] = overlay_mean(&state->overlays[i]);
        }

        float global_mean = 0.0f;
        for (int i = 0; i < NUM_OVERLAYS; ++i) {
            global_mean += overlay_means[i];
        }
        global_mean /= (float)NUM_OVERLAYS;

        for (int overlay_idx = 0; overlay_idx < NUM_OVERLAYS; ++overlay_idx) {
            overlay_t* overlay = &state->overlays[overlay_idx];
            const float overlay_mean_val = overlay_means[overlay_idx];
            const int left = (overlay_idx == 0) ? NUM_OVERLAYS - 1 : overlay_idx - 1;
            const int right = (overlay_idx + 1) % NUM_OVERLAYS;
            const float neighbor_mean = 0.5f * (overlay_means[left] + overlay_means[right]);

            for (int node = 0; node < overlay->node_count; ++node) {
                const float prev_value = overlay->values[node];
                const float memory = overlay->history[node];

                const float towards_global = (global_mean - prev_value) * cfg->coupling_gain;
                const float neighbor_bias = (neighbor_mean - overlay_mean_val) * cfg->harmonic_gain;
                const float memory_feedback = (memory - prev_value) * 0.2f;

                float next_value = prev_value + towards_global + neighbor_bias + memory_feedback;
                overlay->history[node] = prev_value;
                overlay->values[node] = clamp01(next_value);
            }

            overlay->stability = qallow_calculate_stability(overlay);
        }

        float stability_sum = 0.0f;
        for (int i = 0; i < NUM_OVERLAYS; ++i) {
            stability_sum += state->overlays[i].stability;
        }
        state->global_coherence = stability_sum / (float)NUM_OVERLAYS;

        state->decoherence_level = fmaxf(0.0f, state->decoherence_level * (1.0f - damping));
        state->ethics_S = clamp01(state->ethics_S + ethics_gain * 0.5f);
        state->ethics_C = clamp01(state->ethics_C + ethics_gain * 0.3f);
        state->ethics_H = clamp01(state->ethics_H + ethics_gain * 0.4f);
    }

    float final_means[NUM_OVERLAYS];
    for (int i = 0; i < NUM_OVERLAYS; ++i) {
        final_means[i] = overlay_mean(&state->overlays[i]);
    }
    const float alignment_after = compute_alignment_metric(final_means, NUM_OVERLAYS);

    local_status.coherence_delta = state->global_coherence - baseline_coherence;
    local_status.cross_alignment_delta = alignment_before - alignment_after;
    local_status.ethics_projection =
        (state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f;
    local_status.entanglement_index = alignment_after;

    if (status) {
        *status = local_status;
    }
}
