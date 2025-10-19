#ifndef QALLOW_PHASE15_H
#define QALLOW_PHASE15_H

#include "qallow_phase14.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int review_passes;
    float convergence_gain;
    float audit_gain;
    float bayesian_weight;
    float ethics_floor;
} phase15_config_t;

typedef struct {
    float coherence_scalar;
    float inference_entropy;
    float ethics_floor;
    float audit_score;
} phase15_status_t;

void phase15_config_default(phase15_config_t* cfg);
void phase15_singularity_converge(qallow_state_t* state,
                                  const phase15_config_t* cfg,
                                  const phase14_status_t* phase14_feedback,
                                  phase15_status_t* status);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE15_H */
