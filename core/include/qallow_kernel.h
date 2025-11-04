#ifndef QALLOW_KERNEL_H
#define QALLOW_KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "qallow_metrics.h"


#ifndef CUDA_ENABLED
    #ifdef __CUDACC__
        #define CUDA_ENABLED 1
    #elif defined(QALLOW_ENABLE_CUDA)
        #define CUDA_ENABLED 1
    #else
        #define CUDA_ENABLED 0
    #endif
#endif

#if CUDA_ENABLED
    #ifdef __CUDACC__
        #include <cuda_runtime.h>
        #include <curand_kernel.h>
        #ifndef CUDA_CALLABLE
            #define CUDA_CALLABLE __device__ __host__
        #endif
    #else
        #include <cuda_runtime_api.h>
        #ifndef CUDA_CALLABLE
            #define CUDA_CALLABLE
        #endif
    #endif
#else
    #ifndef CUDA_CALLABLE
        #define CUDA_CALLABLE
    #endif
#endif


#define MAX_NODES 256
#define MAX_TICKS 1000
#define NUM_OVERLAYS 3


typedef enum {
    OVERLAY_ORBITAL = 0,
    OVERLAY_RIVER_DELTA = 1,
    OVERLAY_MYCELIAL = 2
} overlay_type_t;


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

    float ethics_S;  // Safety score
    float ethics_C;  // Clarity score
    float ethics_H;  // Human benefit score
} qallow_state_t;


typedef struct {
    float safety_score;
    float clarity_score;
    float human_benefit_score;
    float reality_drift_score;
    float total_ethics_score;
    bool safety_check_passed;
    bool reality_drift_guard_passed;
} ethics_state_t;


CUDA_CALLABLE void qallow_kernel_init(qallow_state_t* state);
CUDA_CALLABLE void qallow_kernel_tick(qallow_state_t* state);
CUDA_CALLABLE void qallow_update_decoherence(qallow_state_t* state);


static CUDA_CALLABLE inline float qallow_calculate_stability(const overlay_t* overlay) {
    if (!overlay || overlay->node_count == 0) return 0.0f;
    

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
    

    return 1.0f / (1.0f + variance);
}


int qallow_vm_main(void);


float qallow_global_stability(const qallow_state_t* state);
void adaptive_governance(qallow_state_t* state);
double foresight_predict(double now);
void predictive_control(qallow_state_t* state);
void temporal_alignment(qallow_state_t* state, double predicted, double actual);


#if CUDA_ENABLED
void qallow_cuda_init(qallow_state_t* state);
void qallow_cuda_cleanup(qallow_state_t* state);
void qallow_cuda_process_overlays(qallow_state_t* state);
#endif


void qallow_cpu_process_overlays(qallow_state_t* state);


void qallow_print_status(const qallow_state_t* state, int tick);
bool qallow_ethics_check(const qallow_state_t* state, ethics_state_t* ethics);


void qallow_print_dashboard(const qallow_state_t* state, const ethics_state_t* ethics);
void qallow_print_bar(const char* label, double value, int width);


void qallow_csv_log_init(const char* filepath);
void qallow_csv_log_tick(const qallow_state_t* state, const ethics_state_t* ethics);
void qallow_csv_log_close(void);

#endif // QALLOW_KERNEL_H
