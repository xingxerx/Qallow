#ifndef QALLOW_KERNEL_H
#define QALLOW_KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

// CUDA support detection
#ifdef __CUDACC__
    #define CUDA_ENABLED 1
    #include <cuda_runtime.h>
    #include <curand_kernel.h>
    #define CUDA_CALLABLE __device__ __host__
#elif defined(QALLOW_ENABLE_CUDA)
    #define CUDA_ENABLED 1
    #include <cuda_runtime.h>
    #define CUDA_CALLABLE
#else
    #define CUDA_ENABLED 0
    #define CUDA_CALLABLE
#endif

// System constants
#define MAX_NODES 256
#define MAX_TICKS 1000
#define NUM_OVERLAYS 3

// Overlay types
typedef enum {
    OVERLAY_ORBITAL = 0,
    OVERLAY_RIVER_DELTA = 1,
    OVERLAY_MYCELIAL = 2
} overlay_type_t;

// Core data structures
typedef struct {
    float values[MAX_NODES];
    float history[MAX_NODES];
    float stability;
    int node_count;
} overlay_t;

typedef struct {
    overlay_t overlays[NUM_OVERLAYS];
    float global_coherence;
    float decoherence_level;
    int tick_count;
    bool cuda_enabled;
    int gpu_device_id;
    // Phase 8-10: Adaptive-Predictive-Temporal components
    float ethics_S;  // Safety score
    float ethics_C;  // Clarity score
    float ethics_H;  // Human benefit score
} qallow_state_t;

// Ethics monitoring
typedef struct {
    float safety_score;
    float clarity_score;
    float human_benefit_score;
    float total_ethics_score;
    bool safety_check_passed;
} ethics_state_t;

// Function declarations
CUDA_CALLABLE void qallow_kernel_init(qallow_state_t* state);
CUDA_CALLABLE void qallow_kernel_tick(qallow_state_t* state);
CUDA_CALLABLE void qallow_update_decoherence(qallow_state_t* state);

// Inline implementation for CUDA compatibility
static CUDA_CALLABLE inline float qallow_calculate_stability(const overlay_t* overlay) {
    if (!overlay || overlay->node_count == 0) return 0.0f;
    
    // Calculate variance (lower variance = higher stability)
    float mean = 0.0f;
    for (int i = 0; i < overlay->node_count; i++) {
        mean += overlay->values[i];
    }
    mean /= overlay->node_count;
    
    float variance = 0.0f;
    for (int i = 0; i < overlay->node_count; i++) {
        float diff = overlay->values[i] - mean;
        variance += diff * diff;
    }
    variance /= overlay->node_count;
    
    // Stability = 1 / (1 + variance)
    return 1.0f / (1.0f + variance);
}

// VM main execution function
int qallow_vm_main(void);

// Phase 8-10: Adaptive-Predictive-Temporal functions
float qallow_global_stability(const qallow_state_t* state);
void adaptive_governance(qallow_state_t* state);
double foresight_predict(double now);
void predictive_control(qallow_state_t* state);
void temporal_alignment(qallow_state_t* state, double predicted, double actual);

// CUDA-specific functions
#if CUDA_ENABLED
void qallow_cuda_init(qallow_state_t* state);
void qallow_cuda_cleanup(qallow_state_t* state);
void qallow_cuda_process_overlays(qallow_state_t* state);
#endif

// CPU fallback functions
void qallow_cpu_process_overlays(qallow_state_t* state);

// Utility functions
void qallow_print_status(const qallow_state_t* state, int tick);
bool qallow_ethics_check(const qallow_state_t* state, ethics_state_t* ethics);

// ASCII Dashboard functions
void qallow_print_dashboard(const qallow_state_t* state, const ethics_state_t* ethics);
void qallow_print_bar(const char* label, double value, int width);

// CSV Logging functions
void qallow_csv_log_init(const char* filepath);
void qallow_csv_log_tick(const qallow_state_t* state, const ethics_state_t* ethics);
void qallow_csv_log_close(void);

#endif // QALLOW_KERNEL_H
