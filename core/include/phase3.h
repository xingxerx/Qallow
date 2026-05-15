#ifndef QALLOW_PHASE3_H
#define QALLOW_PHASE3_H

#include <stdbool.h>

#include "qallow_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool enable;
    bool no_split_mode;
    bool share_cuda_blocks;
} phase3_config_t;

typedef struct {
    bool active;
    float entanglement_strength;
    float ethics_alignment;
    float pocket_flux;
    float decoherence_buffer;
} phase3_metrics_t;

void phase3_initialize(const qallow_state_t* state);
void phase3_configure(const phase3_config_t* cfg);
void phase3_tick(qallow_state_t* state);
void phase3_collect_metrics(phase3_metrics_t* out);
float phase3_get_entanglement_strength(void);
bool phase3_is_active(void);
int phase3_gain_from_csr(const char* csv_path, int N, double* out_alpha_eff,
                          double gain_base, double gain_span);

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_PHASE3_H */
