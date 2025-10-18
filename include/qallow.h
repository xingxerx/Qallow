#ifndef QALLOW_H
#define QALLOW_H

#include <stdbool.h>
#include <stddef.h>

#define QALLOW_NODES 256

typedef struct {
    double nodes[QALLOW_NODES];
    double stability;
} overlay_t;

typedef struct {
    unsigned long tick;
    overlay_t orbital, river, mycelial;
    double ethics_S, ethics_C, ethics_H;  // E = S + C + H
    double decoherence;
} qvm_t;

/* core */
void qallow_init(qvm_t* vm);
void qallow_tick(qvm_t* vm, unsigned long seed);
double qallow_global_stability(qvm_t* vm);

/* overlays */
void overlay_init(overlay_t* o, double seed);
void overlay_apply_nudge(overlay_t* o, double delta);
void overlay_propagate(overlay_t* src, overlay_t* dst, double factor);
double overlay_stability(overlay_t* o);

/* CUDA (provided by .cu) */
#ifdef __cplusplus
extern "C" {
#endif
void runPhotonicSimulation(double* hostData, int n, unsigned long seed);
void runQuantumOptimizer(double* hostData, int n);
#ifdef __cplusplus
}
#endif

#endif

