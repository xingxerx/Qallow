#ifndef QALLOW_PHASE14_H
#define QALLOW_PHASE14_H

#include <stdbool.h>

#include "qallow_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enable;
    bool no_split_mode;
    bool share_cuda_blocks;
} phase14_config_t;

typedef struct {
    bool active;
    float entanglement_strength;
    float ethics_alignment;
    float pocket_flux;
    float decoherence_buffer;
} phase14_metrics_t;

void phase14_initialize(const qallow_state_t* state);
void phase14_configure(const phase14_config_t* cfg);
void phase14_tick(qallow_state_t* state);
void phase14_collect_metrics(phase14_metrics_t* out);
float phase14_get_entanglement_strength(void);
bool phase14_is_active(void);
int phase14_gain_from_csr(const char* csv_path, int N, double* out_alpha_eff,
                          double gain_base, double gain_span);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE14_H */
