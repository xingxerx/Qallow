#ifndef POCKET_H
#define POCKET_H

#include "qallow_kernel.h"

#define MAX_POCKETS 8

typedef struct {
    qallow_state_t state;
    double result_score;
    int active;
    double memory_usage_mb;
    double memory_peak_mb;
} pocket_t;

typedef struct {
    pocket_t pockets[MAX_POCKETS];
    int count;
    double merged_score;
    double average_coherence;
    double average_decoherence;
    double memory_usage_mb;
    double memory_peak_mb;
} pocket_dimension_t;

// Spawn N parallel pocket simulations
int pocket_spawn(pocket_dimension_t* pd, int n);

// Run one tick in all active pockets
void pocket_tick_all(pocket_dimension_t* pd);

// Merge results from all pockets
double pocket_merge(pocket_dimension_t* pd);

// Get average score across pockets
double pocket_get_average_score(const pocket_dimension_t* pd);

// Capture metrics for telemetry
void pocket_capture_metrics(pocket_dimension_t* pd, int tick);

// Cleanup pockets
void pocket_cleanup(pocket_dimension_t* pd);

#endif
