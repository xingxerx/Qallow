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

// Load adaptive state from JSON
void adaptive_load(adaptive_state_t* state);

// Save adaptive state to JSON
void adaptive_save(const adaptive_state_t* state);

// Update parameters based on performance and human feedback
void adaptive_update(adaptive_state_t* state, double run_ms, double human_score);

// Get recommended thread count
int adaptive_get_threads(const adaptive_state_t* state);

// Get current learning rate
double adaptive_get_learning_rate(const adaptive_state_t* state);

#endif

