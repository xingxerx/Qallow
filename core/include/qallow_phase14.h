#ifndef QALLOW_PHASE14_H
#define QALLOW_PHASE14_H

#include "qallow_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int iterations;
    float coupling_gain;
    float harmonic_gain;
    float decoherence_damping;
    float ethics_gain;
} phase14_config_t;

typedef struct {
    float coherence_delta;
    float cross_alignment_delta;
    float ethics_projection;
    float entanglement_index;
} phase14_status_t;

void phase14_config_default(phase14_config_t* cfg);
void phase14_entanglement_integrate(qallow_state_t* state,
                                    const phase14_config_t* cfg,
                                    phase14_status_t* status);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE14_H */
