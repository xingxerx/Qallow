/*
 * Real Computer Orchestrator - C Implementation
 * Unified interface for real CUDA GPU and Cirq quantum processing
 * Coordinates actual hardware workloads across heterogeneous architectures
 */

#ifndef REAL_COMPUTER_H
#define REAL_COMPUTER_H

#include <stdint.h>
#include <stdbool.h>
#include "real_cuda.h"
#include "cirq_quantum.h"

/* Workload Types */
typedef enum {
    WORKLOAD_GPU_COMPUTE,
    WORKLOAD_QUANTUM_CIRCUIT,
    WORKLOAD_HYBRID_OPTIMIZATION,
    WORKLOAD_GPU_ACCELERATED_NN,
    WORKLOAD_QUANTUM_OPTIMIZATION,
    WORKLOAD_MIXED_PRECISION
} workload_type_t;

/* Task Definition */
typedef struct {
    uint32_t task_id;
    workload_type_t type;
    uint32_t priority;
    char description[256];
    
    /* GPU parameters */
    uint32_t gpu_threads;
    size_t gpu_memory_mb;
    
    /* Quantum parameters */
    uint32_t quantum_qubits;
    uint32_t quantum_shots;
    
    /* Performance targets */
    double target_latency_ms;
    double energy_budget_mj;
} task_definition_t;

/* Task Execution Result */
typedef struct {
    uint32_t task_id;
    bool success;
    double execution_time_ms;
    double energy_consumed_mj;
    char hardware_used[64];
    double performance_score;
    char result_summary[512];
} task_result_t;

/* Real Computer System */
typedef struct {
    cuda_context_t *gpu;
    quantum_context_t *qpu;
    bool gpu_available;
    bool qpu_available;
    
    /* Statistics */
    uint32_t total_tasks;
    uint32_t completed_tasks;
    uint32_t failed_tasks;
    double total_energy_mj;
    double total_time_ms;
} real_computer_t;

/* Function Declarations */

/**
 * Initialize real computer system
 */
real_computer_t* real_computer_init(void);

/**
 * Cleanup real computer system
 */
void real_computer_cleanup(real_computer_t *computer);

/**
 * Check hardware availability
 */
void real_computer_check_hardware(real_computer_t *computer);

/**
 * Create task definition
 */
task_definition_t real_computer_create_task(uint32_t task_id, workload_type_t type,
                                           const char *description);

/**
 * Execute task on real hardware
 */
task_result_t* real_computer_execute_task(real_computer_t *computer,
                                         task_definition_t *task);

/**
 * Execute GPU compute workload
 */
task_result_t* real_computer_gpu_workload(real_computer_t *computer,
                                         task_definition_t *task);

/**
 * Execute quantum workload
 */
task_result_t* real_computer_quantum_workload(real_computer_t *computer,
                                             task_definition_t *task);

/**
 * Execute hybrid GPU+Quantum workload
 */
task_result_t* real_computer_hybrid_workload(real_computer_t *computer,
                                            task_definition_t *task);

/**
 * Print system status
 */
void real_computer_print_status(real_computer_t *computer);

/**
 * Print execution statistics
 */
void real_computer_print_stats(real_computer_t *computer);

#endif /* REAL_COMPUTER_H */
