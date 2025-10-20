#include "qallow_phase13.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "meta_introspect.h"

#define QALLOW_PHASE13_MAX_POCKETS 32

static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float avg_phase(const float* phases, int count) {
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {
        total += phases[i];
    }
    return count > 0 ? total / (float)count : 0.0f;
}

int run_phase13_harmonic(const char* log_path, int pockets, int ticks, float coupling) {
    clock_t start_clock = clock();
    if (ticks <= 0) {
        fprintf(stderr, "[PHASE13] Invalid tick count: %d\n", ticks);
        return 1;
    }

    if (pockets < 2) {
        fprintf(stderr, "[PHASE13] Pocket count too low, promoting to 2\n");
        pockets = 2;
    }

    if (pockets > QALLOW_PHASE13_MAX_POCKETS) {
        fprintf(stderr, "[PHASE13] Pocket count capped to %d\n", QALLOW_PHASE13_MAX_POCKETS);
        pockets = QALLOW_PHASE13_MAX_POCKETS;
    }

    FILE* log = NULL;
    if (log_path) {
        log = fopen(log_path, "w");
        if (!log) {
            fprintf(stderr, "[PHASE13] Failed to open log file: %s\n", log_path);
            return 2;
        }
        fprintf(log, "tick,avg_coherence,phase_drift,phase_energy\n");
        fflush(log);
    }

    float phase_curr[QALLOW_PHASE13_MAX_POCKETS];
    float phase_prev[QALLOW_PHASE13_MAX_POCKETS];
    float coh_curr[QALLOW_PHASE13_MAX_POCKETS];
    float coh_prev[QALLOW_PHASE13_MAX_POCKETS];

    for (int i = 0; i < pockets; ++i) {
        float offset = (float)i / (float)pockets;
        phase_curr[i] = offset * 0.4f;
        coh_curr[i] = 0.78f + 0.02f * (float)(i % 3);
    }

    float start_coherence = 0.0f;
    float start_drift = 0.0f;
    {
        float phase_mean = avg_phase(phase_curr, pockets);
        float drift_acc = 0.0f;
        float coh_acc = 0.0f;
        for (int i = 0; i < pockets; ++i) {
            drift_acc += fabsf(phase_curr[i] - phase_mean);
            coh_acc += coh_curr[i];
        }
        start_coherence = coh_acc / (float)pockets;
        start_drift = drift_acc / (float)pockets;
    }

    float final_coherence = start_coherence;
    float final_drift = start_drift;

    memcpy(phase_prev, phase_curr, sizeof(float) * QALLOW_PHASE13_MAX_POCKETS);
    memcpy(coh_prev, coh_curr, sizeof(float) * QALLOW_PHASE13_MAX_POCKETS);

    float drift_history = start_drift;
    for (int tick = 1; tick <= ticks; ++tick) {
        memcpy(phase_prev, phase_curr, sizeof(float) * pockets);
        memcpy(coh_prev, coh_curr, sizeof(float) * pockets);

        float phase_mean = avg_phase(phase_prev, pockets);
        float drift_acc = 0.0f;
        float coh_acc = 0.0f;
        float energy_acc = 0.0f;

        for (int i = 0; i < pockets; ++i) {
            int left = (i == 0) ? pockets - 1 : i - 1;
            int right = (i + 1) % pockets;

            float neighbor_phase = 0.5f * (phase_prev[left] + phase_prev[right]);
            float phase_delta = neighbor_phase - phase_prev[i];

            float alignment_gain = coupling * 120.0f;
            float damping = coupling * 15.0f;
            float new_phase = phase_prev[i] + alignment_gain * phase_delta - damping * phase_prev[i] * 0.01f;

            phase_curr[i] = new_phase;

            float neighbor_coh = 0.5f * (coh_prev[left] + coh_prev[right]);
            float drift_pressure = clampf(1.0f - drift_history, 0.0f, 1.0f);
            float bias = 0.0004f + coupling * 0.25f + drift_pressure * 0.0003f;
            float new_coh = coh_prev[i] + coupling * 60.0f * (neighbor_coh - coh_prev[i]) + bias;
            coh_curr[i] = clampf(new_coh, 0.0f, 1.0f);

            float diff = phase_curr[i] - phase_mean;
            drift_acc += fabsf(diff);
            energy_acc += 0.5f * diff * diff;
            coh_acc += coh_curr[i];
        }

        drift_history = (drift_history * 0.85f) + (drift_acc / (float)pockets) * 0.15f;

        final_coherence = coh_acc / (float)pockets;
        final_drift = drift_acc / (float)pockets;

        if (log) {
            fprintf(log, "%d,%.6f,%.6f,%.6f\n", tick, final_coherence, final_drift, energy_acc);
            if ((tick % 25) == 0) {
                fflush(log);
            }
        }
    }

    if (log) {
        fflush(log);
        fclose(log);
    }

    printf("[PHASE13] Harmonic propagation complete: pockets=%d ticks=%d k=%.6f\n",
           pockets, ticks, coupling);
    printf("[PHASE13] avg_coherence: %.6f → %.6f\n", start_coherence, final_coherence);
    printf("[PHASE13] phase_drift  : %.6f → %.6f\n", start_drift, final_drift);

    float duration_s = (float)(clock() - start_clock) / (float)CLOCKS_PER_SEC;
    float coherence_metric = clampf(final_coherence, 0.0f, 1.0f);
    float ethics_metric = clampf(1.0f - final_drift, 0.0f, 1.0f);
    learn_event_t ev = {
        .phase = "phase13",
        .module = "harmonic",
        .objective_id = "phase13.harmonic",
        .duration_s = duration_s,
        .coherence = coherence_metric,
        .ethics = ethics_metric
    };
    meta_introspect_push(&ev);
    meta_introspect_flush();

    return 0;
}
