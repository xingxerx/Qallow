#ifndef MULTI_POCKET_H
#define MULTI_POCKET_H

#include "qallow_kernel.h"
#include "sandbox.h"

// Multi-Pocket Simulation Scheduler
// Runs N parallel probabilistic worldlines using CUDA streams

#define MAX_POCKETS 16
#define POCKET_TELEMETRY_INTERVAL 10

// Pocket configuration parameters
typedef struct {
    int pocket_id;
    float learning_rate;
    float noise_level;
    float stability_bias;
    int thread_count;
    char telemetry_file[128];
} pocket_params_t;

// Pocket simulation result
typedef struct {
    int pocket_id;
    qallow_state_t final_state;
    float avg_coherence;
    float avg_decoherence;
    float ethics_score;
    float confidence;
    int ticks_executed;
    double elapsed_time_ms;
} pocket_result_t;

// Multi-pocket scheduler state
typedef struct {
    int num_pockets;
    pocket_params_t params[MAX_POCKETS];
    pocket_result_t results[MAX_POCKETS];
    
#if CUDA_ENABLED
    cudaStream_t streams[MAX_POCKETS];
    bool streams_initialized;
#endif
    
    // Telemetry
    FILE* master_telemetry;
    char master_telemetry_file[128];
    
    // Timing
    double total_scheduler_time_ms;
    double max_pocket_time_ms;
    double min_pocket_time_ms;
} multi_pocket_scheduler_t;

// Pocket merge configuration
typedef struct {
    bool use_weighted_merge;
    bool filter_outliers;
    float outlier_threshold;
    float confidence_weight;
} pocket_merge_config_t;

// Function declarations

// Scheduler initialization and cleanup
void multi_pocket_init(multi_pocket_scheduler_t* scheduler, int num_pockets);
void multi_pocket_cleanup(multi_pocket_scheduler_t* scheduler);

// Pocket configuration
void multi_pocket_set_params(multi_pocket_scheduler_t* scheduler, 
                             int pocket_id, 
                             const pocket_params_t* params);
void multi_pocket_generate_random_params(multi_pocket_scheduler_t* scheduler);

// Execution
void multi_pocket_execute_all(multi_pocket_scheduler_t* scheduler, 
                              const qallow_state_t* initial_state,
                              int num_ticks);

#if CUDA_ENABLED
void multi_pocket_execute_cuda(multi_pocket_scheduler_t* scheduler,
                               const qallow_state_t* initial_state,
                               int num_ticks);
#endif

void multi_pocket_execute_cpu(multi_pocket_scheduler_t* scheduler,
                              const qallow_state_t* initial_state,
                              int num_ticks);

// Merging
void multi_pocket_merge(multi_pocket_scheduler_t* scheduler,
                       qallow_state_t* merged_state,
                       const pocket_merge_config_t* config);

// Analysis
float multi_pocket_calculate_consensus(const multi_pocket_scheduler_t* scheduler);
void multi_pocket_find_outliers(const multi_pocket_scheduler_t* scheduler, 
                                bool* is_outlier, 
                                float threshold);

// Telemetry
void multi_pocket_write_telemetry(multi_pocket_scheduler_t* scheduler, 
                                  int tick,
                                  bool include_pocket_details);
void multi_pocket_write_summary(const multi_pocket_scheduler_t* scheduler);

// Reporting
void multi_pocket_print_results(const multi_pocket_scheduler_t* scheduler);
void multi_pocket_print_statistics(const multi_pocket_scheduler_t* scheduler);

#endif // MULTI_POCKET_H