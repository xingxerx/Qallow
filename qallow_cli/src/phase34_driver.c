/* Small C drivers that run the phase 3/4 engines against a locally
 * constructed qallow_state_t, so the Rust CLI only needs a simple
 * int-returning entry point per phase. */

#include <stdbool.h>
#include <stdio.h>

#include "phase3.h"
#include "phase4.h"

/* Initial kernel state shared by both drivers. Values chosen to represent a
 * healthy-but-imperfect system: overlay stabilities in the 0.93-0.96 band,
 * ethics components at 0.85, near-zero decoherence, coherence = mean of the
 * overlay stabilities, CPU-only. */
static qallow_state_t qallow_driver_make_state(void) {
    qallow_state_t state;
    state.overlays[OVERLAY_ORBITAL].stability = 0.96f;
    state.overlays[OVERLAY_RIVER_DELTA].stability = 0.93f;
    state.overlays[OVERLAY_MYCELIAL].stability = 0.94f;
    state.ethics_S = 0.85f;
    state.ethics_C = 0.85f;
    state.ethics_H = 0.85f;
    state.decoherence_level = 0.00001f;
    state.global_coherence = (state.overlays[OVERLAY_ORBITAL].stability +
                              state.overlays[OVERLAY_RIVER_DELTA].stability +
                              state.overlays[OVERLAY_MYCELIAL].stability) / 3.0f;
    state.cuda_enabled = false;
    return state;
}

int qallow_run_phase3(int ticks) {
    if (ticks <= 0) {
        fprintf(stderr, "[PHASE3] Invalid tick count: %d\n", ticks);
        return 1;
    }

    qallow_state_t state = qallow_driver_make_state();

    phase3_initialize(&state);
    phase3_config_t cfg;
    cfg.enable = true;
    cfg.no_split_mode = false;
    cfg.share_cuda_blocks = false;
    phase3_configure(&cfg);

    for (int t = 0; t < ticks; ++t) {
        phase3_tick(&state);
    }

    phase3_metrics_t metrics;
    phase3_collect_metrics(&metrics);
    printf("[PHASE3] Coherence run complete: ticks=%d\n", ticks);
    printf("[PHASE3] entanglement=%.6f ethics_alignment=%.6f pocket_flux=%.6f "
           "deco_buffer=%.6f global_coherence=%.6f\n",
           metrics.entanglement_strength,
           metrics.ethics_alignment,
           metrics.pocket_flux,
           metrics.decoherence_buffer,
           state.global_coherence);
    return 0;
}

int qallow_run_phase4(int ticks, int audit_unified) {
    if (ticks <= 0) {
        fprintf(stderr, "[PHASE4] Invalid tick count: %d\n", ticks);
        return 1;
    }

    qallow_state_t state = qallow_driver_make_state();

    phase4_initialize(&state);
    phase4_config_t cfg;
    cfg.enable = true;
    cfg.no_split_mode = false;
    cfg.audit_unified = audit_unified != 0;
    phase4_configure(&cfg);

    for (int t = 0; t < ticks; ++t) {
        phase4_tick(&state);
    }

    phase4_metrics_t metrics;
    phase4_collect_metrics(&metrics);
    printf("[PHASE4] Convergence run complete: ticks=%d unified=%s\n",
           ticks, audit_unified ? "true" : "false");
    printf("[PHASE4] convergence=%.6f audit_score=%.6f entropy_index=%.6f "
           "global_coherence=%.6f\n",
           metrics.convergence_signal,
           metrics.audit_score,
           metrics.entropy_index,
           state.global_coherence);
    return 0;
}
