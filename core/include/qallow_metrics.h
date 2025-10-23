#ifndef QALLOW_METRICS_H
#define QALLOW_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qallow_run_metrics {
    int tick_count;
    int cuda_enabled;
    int reached_equilibrium;
    int equilibrium_tick;
    float final_coherence;
    float final_decoherence;
} qallow_run_metrics_t;

const qallow_run_metrics_t* qallow_get_last_run_metrics(void);
void qallow_metrics_begin_run(int cuda_enabled);
void qallow_metrics_update_tick(int tick, float coherence, float decoherence, int cuda_enabled);
void qallow_metrics_mark_equilibrium(int tick);
void qallow_metrics_finalize(float coherence, float decoherence);

#ifdef __cplusplus
}
#endif

#endif
