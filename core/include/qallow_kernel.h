#ifndef QALLOW_KERNEL_H
#define QALLOW_KERNEL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Overlay identifiers used by the phase 3/4 engines. */
typedef enum {
    OVERLAY_ORBITAL = 0,
    OVERLAY_RIVER_DELTA = 1,
    OVERLAY_MYCELIAL = 2,
    NUM_OVERLAYS = 3
} qallow_overlay_id_t;

typedef struct {
    float stability;
} qallow_overlay_t;

/* Minimal kernel state consumed by phase3_coherence.c / phase4_convergence.c. */
typedef struct {
    qallow_overlay_t overlays[NUM_OVERLAYS];
    float ethics_S;
    float ethics_C;
    float ethics_H;
    float decoherence_level;
    float global_coherence;
    bool cuda_enabled;
} qallow_state_t;

#ifdef __cplusplus
}
#endif

#endif /* QALLOW_KERNEL_H */
