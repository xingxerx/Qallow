#ifndef POCKET_H
#define POCKET_H

#include "qallow_kernel.h"

#define MAX_POCKETS 8

typedef struct {
    qallow_state_t state;
    double result_score;
    int active;
} pocket_t;

typedef struct {
    pocket_t pockets[MAX_POCKETS];
    int count;
    double merged_score;
} pocket_dimension_t;

// Spawn N parallel pocket simulations
int pocket_spawn(pocket_dimension_t* pd, int n);

// Run one tick in all active pockets
void pocket_tick_all(pocket_dimension_t* pd);

// Merge results from all pockets
double pocket_merge(pocket_dimension_t* pd);

// Get average score across pockets
double pocket_get_average_score(const pocket_dimension_t* pd);

// Cleanup pockets
void pocket_cleanup(pocket_dimension_t* pd);

#endif

