#ifndef ADAPTIVE_H
#define ADAPTIVE_H

#include <stdio.h>

typedef struct {
    double target_ms;
    double last_run_ms;
    int threads;
    double learning_rate;
    double human_score;
} adaptive_state_t;


void adaptive_load(adaptive_state_t* state);


void adaptive_save(const adaptive_state_t* state);


void adaptive_update(adaptive_state_t* state, double run_ms, double human_score);


int adaptive_get_threads(const adaptive_state_t* state);


double adaptive_get_learning_rate(const adaptive_state_t* state);

#endif

