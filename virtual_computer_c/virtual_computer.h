/*
 * Virtual Computer System - Main Header
 * Unified orchestration of CUDA, Neuromorphic, and Photonic processors
 */

#ifndef VIRTUAL_COMPUTER_H
#define VIRTUAL_COMPUTER_H

#include "cuda_simulator.h"
#include "neuromorphic_simulator.h"
#include "photonic_simulator.h"
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

/* Workload Type Enumeration */
typedef enum {
    WORKLOAD_GPU_COMPUTE = 0,
    WORKLOAD_GPU_MEMORY_INTENSIVE = 1,
    WORKLOAD_NEURAL_INFERENCE = 2,
    WORKLOAD_NEURAL_TRAINING = 3,
    WORKLOAD_PHOTONIC_COMPUTE = 4,
    WORKLOAD_PHOTONIC_OPTIMIZATION = 5,
    WORKLOAD_HYBRID_PROCESSING = 6
} workload_type_t;

/* Workload Structure */
typedef struct {
    uint32_t workload_id;
    workload_type_t workload_type;
    uint32_t priority;
    size_t data_size_mb;
    uint64_t compute_ops;
    
    char status[32];
    double performance_score;
    time_t created_at;
    time_t started_at;
    time_t completed_at;
} workload_t;

/* Virtual Computer Structure */
typedef struct {
    /* Processors */
    virtual_gpu_t *cuda_gpu;
    neuromorphic_processor_t *neuromorphic;
    photonic_processor_t *photonic;
    
    /* Workload management */
    workload_t *workload_queue;
    uint32_t queue_size;
    uint32_t queue_capacity;
    
    workload_t *completed_workloads;
    uint32_t completed_size;
    uint32_t completed_capacity;
    
    uint32_t workload_counter;
    
    /* System state */
    double current_time;
    double total_energy_consumed;
    time_t started_at;
    
    double throughput_workloads_per_sec;
} virtual_computer_t;

/* Function Declarations */

/**
 * Create virtual computer system
 */
virtual_computer_t* vc_create(void);

/**
 * Destroy virtual computer system
 */
void vc_destroy(virtual_computer_t *vc);

/**
 * Create and queue a workload
 */
uint32_t vc_create_workload(virtual_computer_t *vc, workload_type_t type,
                           uint32_t priority, size_t data_size_mb, uint64_t compute_ops);

/**
 * Execute all queued workloads
 */
void vc_run_scheduled_workloads(virtual_computer_t *vc);

/**
 * Get system status
 */
void vc_get_system_status(virtual_computer_t *vc, char *buffer, size_t buffer_size);

/**
 * Print system status
 */
void vc_print_system_status(virtual_computer_t *vc);

#endif /* VIRTUAL_COMPUTER_H */
