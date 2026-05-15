#ifndef QALLOW_PHASE15_H
#define QALLOW_PHASE15_H

#include <stdbool.h>

#include "qallow_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enable;
    bool no_split_mode;
    bool audit_unified;
} phase15_config_t;

typedef struct {
    bool active;
    float convergence_signal;
    float audit_score;
    float entropy_index;
} phase15_metrics_t;

void phase15_initialize(const qallow_state_t* state);
void phase15_configure(const phase15_config_t* cfg);
void phase15_tick(qallow_state_t* state);
void phase15_collect_metrics(phase15_metrics_t* out);
float phase15_get_convergence(void);
bool phase15_is_active(void);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE15_H */
