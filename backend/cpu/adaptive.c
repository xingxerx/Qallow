#include "adaptive.h"
#include <stdlib.h>
#include <string.h>

void adaptive_load(adaptive_state_t* state) {
    if (!state) return;
    
    // Default values
    state->target_ms = 50.0;
    state->last_run_ms = 0.0;
    state->threads = 4;
    state->learning_rate = 0.0034;
    state->human_score = 0.8;
    
    // Try to load from adapt_state.json
    FILE* f = fopen("adapt_state.json", "r");
    if (f) {
        fscanf(f, "{\n  \"target_ms\": %lf,\n  \"last_run_ms\": %lf,\n  \"threads\": %d,\n  \"learning_rate\": %lf,\n  \"human_score\": %lf\n}",
               &state->target_ms, &state->last_run_ms, &state->threads,
               &state->learning_rate, &state->human_score);
        fclose(f);
        printf("[ADAPTIVE] Loaded state from adapt_state.json\n");
    } else {
        printf("[ADAPTIVE] No prior state found, using defaults\n");
    }
}

void adaptive_save(const adaptive_state_t* state) {
    if (!state) return;
    
    FILE* f = fopen("adapt_state.json", "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"target_ms\": %.1f,\n", state->target_ms);
        fprintf(f, "  \"last_run_ms\": %.1f,\n", state->last_run_ms);
        fprintf(f, "  \"threads\": %d,\n", state->threads);
        fprintf(f, "  \"learning_rate\": %.4f,\n", state->learning_rate);
        fprintf(f, "  \"human_score\": %.2f\n", state->human_score);
        fprintf(f, "}\n");
        fclose(f);
        printf("[ADAPTIVE] State saved to adapt_state.json\n");
    }
}

void adaptive_update(adaptive_state_t* state, double run_ms, double human_score) {
    if (!state) return;
    
    state->last_run_ms = run_ms;
    state->human_score = human_score;
    
    // Adjust learning rate based on human feedback
    if (human_score < 0.7) {
        state->learning_rate *= 0.9;
        printf("[ADAPTIVE] Learning rate decreased (low score): %.4f\n", state->learning_rate);
    } else if (human_score > 0.9) {
        state->learning_rate *= 1.05;
        printf("[ADAPTIVE] Learning rate increased (high score): %.4f\n", state->learning_rate);
    }
    
    // Adjust thread count based on performance
    if (run_ms > state->target_ms) {
        state->threads++;
        printf("[ADAPTIVE] Threads increased to %d (slow run: %.2fms)\n", state->threads, run_ms);
    } else if (run_ms < state->target_ms * 0.6) {
        if (state->threads > 1) {
            state->threads--;
            printf("[ADAPTIVE] Threads decreased to %d (fast run: %.2fms)\n", state->threads, run_ms);
        }
    }
    
    // Clamp learning rate
    if (state->learning_rate < 0.001) state->learning_rate = 0.001;
    if (state->learning_rate > 0.1) state->learning_rate = 0.1;
    
    // Clamp threads
    if (state->threads < 1) state->threads = 1;
    if (state->threads > 16) state->threads = 16;
    
    adaptive_save(state);
}

int adaptive_get_threads(const adaptive_state_t* state) {
    return state ? state->threads : 4;
}

double adaptive_get_learning_rate(const adaptive_state_t* state) {
    return state ? state->learning_rate : 0.0034;
}

