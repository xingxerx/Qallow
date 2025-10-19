#include "qallow_phase15.h"

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

void phase15_config_default(phase15_config_t* cfg) {
    if (!cfg) {
        return;
    }
    cfg->review_passes = 3;
    cfg->convergence_gain = 0.12f;
    cfg->audit_gain = 0.65f;
    cfg->bayesian_weight = 0.45f;
    cfg->ethics_floor = 0.72f;
}

void phase15_singularity_converge(qallow_state_t* state,
                                  const phase15_config_t* cfg,
                                  const phase14_status_t* phase14_feedback,
                                  phase15_status_t* status) {
    if (!state || !cfg) {
        return;
    }

    phase15_status_t local_status = {0};
    const float baseline_coherence = state->global_coherence;
    const float base_ethics =
        (state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f;

    float entanglement_hint = 0.0f;
    if (phase14_feedback) {
        entanglement_hint = fmaxf(0.0f, phase14_feedback->cross_alignment_delta);
        entanglement_hint += fmaxf(0.0f, phase14_feedback->coherence_delta);
        entanglement_hint = fminf(entanglement_hint, 0.2f);
    }

    const float convergence = fmaxf(0.0f, fminf(cfg->convergence_gain, 0.35f));
    const float audit_gain = fmaxf(0.0f, fminf(cfg->audit_gain, 1.0f));
    const float bayesian_weight = fmaxf(0.0f, fminf(cfg->bayesian_weight, 1.0f));
    const float ethics_floor = clamp01(cfg->ethics_floor);

    float audit_accumulator = 0.0f;

    for (int pass = 0; pass < cfg->review_passes; ++pass) {
        float overlay_stability_sum = 0.0f;

        for (int overlay_idx = 0; overlay_idx < NUM_OVERLAYS; ++overlay_idx) {
            overlay_t* overlay = &state->overlays[overlay_idx];
            const float overlay_mean_val = overlay_mean(overlay);

            for (int node = 0; node < overlay->node_count; ++node) {
                const float previous = overlay->values[node];
                const float toward_mean = (overlay_mean_val - previous) * convergence;
                const float toward_core = (state->global_coherence - previous) * 0.5f * convergence;
                const float entanglement_drive =
                    (overlay_mean_val - previous) * 0.25f * entanglement_hint;

                float next = previous + toward_mean + toward_core + entanglement_drive;
                overlay->history[node] = previous;
                overlay->values[node] = clamp01(next);
            }

            overlay->stability = qallow_calculate_stability(overlay);
            overlay_stability_sum += overlay->stability;
        }

        const float stability_avg = overlay_stability_sum / (float)NUM_OVERLAYS;
        state->global_coherence =
            (state->global_coherence * 0.6f) + (stability_avg * 0.4f);

        const float ethics_mean =
            (state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f;
        const float inference_trace =
            stability_avg * (1.0f - bayesian_weight) + ethics_mean * bayesian_weight;
        audit_accumulator = audit_accumulator * 0.5f + inference_trace * 0.5f;

        const float target_floor =
            fmaxf(ethics_floor, base_ethics - 0.02f + entanglement_hint * 0.15f);
        state->ethics_S = fmaxf(state->ethics_S, target_floor);
        state->ethics_C = fmaxf(state->ethics_C, target_floor);
        state->ethics_H = fmaxf(state->ethics_H, target_floor);

        const float deco_damp = fminf(convergence * 0.3f + entanglement_hint * 0.05f, 0.06f);
        state->decoherence_level = fmaxf(0.0f, state->decoherence_level * (1.0f - deco_damp));
    }

    const float final_ethics =
        (state->ethics_S + state->ethics_C + state->ethics_H) / 3.0f;

    local_status.coherence_scalar = state->global_coherence;
    local_status.inference_entropy = fabsf(state->global_coherence - baseline_coherence);
    local_status.ethics_floor = final_ethics;
    local_status.audit_score =
        audit_accumulator * audit_gain + final_ethics * (1.0f - audit_gain);

    if (status) {
        *status = local_status;
    }
}
