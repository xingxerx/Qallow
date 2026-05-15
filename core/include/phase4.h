#ifndef QALLOW_PHASE4_H
#define QALLOW_PHASE4_H

#include <stdbool.h>

#include "qallow_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enable;
    bool no_split_mode;
    bool audit_unified;
} phase4_config_t;

typedef struct {
    bool active;
    float convergence_signal;
    float audit_score;
    float entropy_index;
} phase4_metrics_t;

void phase4_initialize(const qallow_state_t* state);
void phase4_configure(const phase4_config_t* cfg);
void phase4_tick(qallow_state_t* state);
void phase4_collect_metrics(phase4_metrics_t* out);
float phase4_get_convergence(void);
bool phase4_is_active(void);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE4_H */
