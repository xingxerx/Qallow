#include "qallow_integration.h"

#include "phase14.h"
#include "phase15.h"
#include "telemetry.h"
#include "ethics.h"

#include <stdio.h>
#include <string.h>

void qallow_lattice_config_init(qallow_lattice_config_t* cfg) {
    if (!cfg) {
        return;
    }
    cfg->phase_mask = QALLOW_LATTICE_PHASE14 | QALLOW_LATTICE_PHASE15;
    cfg->ticks = 256;
    cfg->no_split = false;
    cfg->print_summary = true;
}

void qallow_lattice_config_enable(qallow_lattice_config_t* cfg, qallow_lattice_phase_t phase, bool enable) {
    if (!cfg) {
        return;
    }
    if (enable) {
        cfg->phase_mask |= phase;
    } else {
        cfg->phase_mask &= ~phase;
    }
}

int qallow_lattice_integrate(const qallow_lattice_config_t* cfg) {
    if (!cfg) {
        return -1;
    }

    qallow_state_t state;
    qallow_kernel_init(&state);

    int ticks = cfg->ticks > 0 ? cfg->ticks : 256;
    bool phase14_enabled = (cfg->phase_mask & QALLOW_LATTICE_PHASE14) != 0u;
    bool phase15_enabled = (cfg->phase_mask & QALLOW_LATTICE_PHASE15) != 0u;

    phase14_config_t phase14_cfg = {
        .enable = phase14_enabled,
        .no_split_mode = cfg->no_split,
        .share_cuda_blocks = state.cuda_enabled
    };
    phase14_configure(&phase14_cfg);

    phase15_config_t phase15_cfg = {
        .enable = phase15_enabled,
        .no_split_mode = cfg->no_split,
        .audit_unified = true
    };
    phase15_configure(&phase15_cfg);

    telemetry_t telemetry;
    telemetry_init(&telemetry);
    telemetry.mode = state.cuda_enabled ? 1 : 0;

    FILE* lattice_log = fopen("data/logs/lattice_integrations.csv", "w");
    if (lattice_log) {
        fprintf(lattice_log,
                "tick,phase14_entanglement,phase14_alignment,phase14_flux,phase14_buffer,"
                "phase15_convergence,phase15_audit,phase15_entropy,ethics_total,global_coherence,decoherence\n");
        fflush(lattice_log);
    }

    ethics_state_t ethics_snapshot;
    memset(&ethics_snapshot, 0, sizeof(ethics_snapshot));

    for (int step = 0; step < ticks; ++step) {
        qallow_kernel_tick(&state);
        qallow_ethics_check(&state, &ethics_snapshot);

        telemetry_stream_tick(&telemetry,
                              state.overlays[OVERLAY_ORBITAL].stability,
                              state.overlays[OVERLAY_RIVER_DELTA].stability,
                              state.overlays[OVERLAY_MYCELIAL].stability,
                              state.global_coherence,
                              state.decoherence_level,
                              state.cuda_enabled ? 1 : 0);

        if (lattice_log) {
            phase14_metrics_t m14;
            phase15_metrics_t m15;
            phase14_collect_metrics(&m14);
            phase15_collect_metrics(&m15);

            fprintf(lattice_log,
                    "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    state.tick_count,
                    m14.active ? m14.entanglement_strength : 0.0f,
                    m14.active ? m14.ethics_alignment : 0.0f,
                    m14.active ? m14.pocket_flux : 0.0f,
                    m14.active ? m14.decoherence_buffer : 0.0f,
                    m15.active ? m15.convergence_signal : 0.0f,
                    m15.active ? m15.audit_score : 0.0f,
                    m15.active ? m15.entropy_index : 0.0f,
                    ethics_snapshot.total_ethics_score,
                    state.global_coherence,
                    state.decoherence_level);

            if ((state.tick_count % 32) == 0) {
                fflush(lattice_log);
            }
        }
    }

    telemetry_log_benchmark(&telemetry,
                            telemetry.compile_ms,
                            telemetry.run_ms > 0.0 ? telemetry.run_ms : (double)ticks,
                            state.decoherence_level,
                            state.global_coherence,
                            telemetry.mode);
    telemetry_close(&telemetry);

    if (lattice_log) {
        fflush(lattice_log);
        fclose(lattice_log);
    }

    phase14_metrics_t final14;
    phase15_metrics_t final15;
    phase14_collect_metrics(&final14);
    phase15_collect_metrics(&final15);

    if (cfg->print_summary) {
        printf("[INTEGRATE] Unified lattice run complete (%d ticks)\n", ticks);
        if (phase14_enabled) {
            printf("[INTEGRATE] Phase14 entanglement=%.4f alignment=%.4f flux=%.4f buffer=%.4f\n",
                   final14.entanglement_strength,
                   final14.ethics_alignment,
                   final14.pocket_flux,
                   final14.decoherence_buffer);
        }
        if (phase15_enabled) {
            printf("[INTEGRATE] Phase15 convergence=%.4f audit=%.4f entropy_index=%.4f\n",
                   final15.convergence_signal,
                   final15.audit_score,
                   final15.entropy_index);
        }
        printf("[INTEGRATE] Ethics total=%.4f global_coherence=%.4f decoherence=%.6f\n",
               ethics_snapshot.total_ethics_score,
               state.global_coherence,
               state.decoherence_level);
    }

    phase14_config_t disable14 = {.enable = false, .no_split_mode = false, .share_cuda_blocks = state.cuda_enabled};
    phase14_configure(&disable14);
    phase15_config_t disable15 = {.enable = false, .no_split_mode = false, .audit_unified = false};
    phase15_configure(&disable15);

    return 0;
}
