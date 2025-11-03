/*
 * Virtual Computer System - C Implementation
 */

#include "virtual_computer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define INITIAL_QUEUE_CAPACITY 256
#define INITIAL_COMPLETED_CAPACITY 512

/**
 * Create virtual computer system
 */
virtual_computer_t* vc_create(void) {
    virtual_computer_t *vc = (virtual_computer_t *)malloc(sizeof(virtual_computer_t));
    if (!vc) return NULL;
    
    /* Create CUDA GPU */
    vc->cuda_gpu = gpu_create(0, 8192);
    if (!vc->cuda_gpu) {
        free(vc);
        return NULL;
    }
    
    /* Create Neuromorphic Processor */
    vc->neuromorphic = nm_create(1000, 4);
    if (!vc->neuromorphic) {
        gpu_destroy(vc->cuda_gpu);
        free(vc);
        return NULL;
    }
    
    /* Create Photonic Processor */
    vc->photonic = pp_create(64, 256);
    if (!vc->photonic) {
        nm_destroy(vc->neuromorphic);
        gpu_destroy(vc->cuda_gpu);
        free(vc);
        return NULL;
    }
    
    /* Workload management */
    vc->workload_queue = (workload_t *)malloc(INITIAL_QUEUE_CAPACITY * sizeof(workload_t));
    if (!vc->workload_queue) {
        pp_destroy(vc->photonic);
        nm_destroy(vc->neuromorphic);
        gpu_destroy(vc->cuda_gpu);
        free(vc);
        return NULL;
    }
    vc->queue_size = 0;
    vc->queue_capacity = INITIAL_QUEUE_CAPACITY;
    
    vc->completed_workloads = (workload_t *)malloc(INITIAL_COMPLETED_CAPACITY * sizeof(workload_t));
    if (!vc->completed_workloads) {
        free(vc->workload_queue);
        pp_destroy(vc->photonic);
        nm_destroy(vc->neuromorphic);
        gpu_destroy(vc->cuda_gpu);
        free(vc);
        return NULL;
    }
    vc->completed_size = 0;
    vc->completed_capacity = INITIAL_COMPLETED_CAPACITY;
    
    vc->workload_counter = 0;
    vc->current_time = 0.0;
    vc->total_energy_consumed = 0.0;
    vc->started_at = time(NULL);
    vc->throughput_workloads_per_sec = 0.0;
    
    return vc;
}

/**
 * Destroy virtual computer system
 */
void vc_destroy(virtual_computer_t *vc) {
    if (!vc) return;
    
    gpu_destroy(vc->cuda_gpu);
    nm_destroy(vc->neuromorphic);
    pp_destroy(vc->photonic);
    free(vc->workload_queue);
    free(vc->completed_workloads);
    free(vc);
}

/**
 * Create and queue a workload
 */
uint32_t vc_create_workload(virtual_computer_t *vc, workload_type_t type,
                           uint32_t priority, size_t data_size_mb, uint64_t compute_ops) {
    if (!vc) return 0;
    
    /* Resize queue if needed */
    if (vc->queue_size >= vc->queue_capacity) {
        vc->queue_capacity *= 2;
        vc->workload_queue = (workload_t *)realloc(vc->workload_queue,
            vc->queue_capacity * sizeof(workload_t));
        if (!vc->workload_queue) return 0;
    }
    
    vc->workload_counter++;
    workload_t *workload = &vc->workload_queue[vc->queue_size];
    
    workload->workload_id = vc->workload_counter;
    workload->workload_type = type;
    workload->priority = priority;
    workload->data_size_mb = data_size_mb;
    workload->compute_ops = compute_ops;
    strcpy(workload->status, "queued");
    workload->performance_score = 0.0;
    workload->created_at = time(NULL);
    workload->started_at = 0;
    workload->completed_at = 0;
    
    vc->queue_size++;
    return vc->workload_counter;
}

/**
 * Simple comparison for qsort - sort by priority (higher first)
 */
static int workload_compare(const void *a, const void *b) {
    const workload_t *wa = (const workload_t *)a;
    const workload_t *wb = (const workload_t *)b;
    return (int)wb->priority - (int)wa->priority;
}

/**
 * Execute all queued workloads
 */
void vc_run_scheduled_workloads(virtual_computer_t *vc) {
    if (!vc || vc->queue_size == 0) return;
    
    /* Sort by priority */
    qsort(vc->workload_queue, vc->queue_size, sizeof(workload_t), workload_compare);
    
    time_t start_time = time(NULL);
    
    /* Execute each workload */
    for (uint32_t i = 0; i < vc->queue_size; i++) {
        workload_t *workload = &vc->workload_queue[i];
        workload->started_at = time(NULL);
        strcpy(workload->status, "running");
        
        /* Route to appropriate processor(s) */
        double total_exec_time = 0.0;
        double total_energy = 0.0;
        
        if (workload->workload_type == WORKLOAD_GPU_COMPUTE ||
            workload->workload_type == WORKLOAD_GPU_MEMORY_INTENSIVE) {
            /* Execute on GPU */
            uint64_t addr = gpu_malloc(vc->cuda_gpu, workload->data_size_mb * 1024 * 1024, "float32");
            if (addr != 0) {
                double transfer_time;
                gpu_memcpy_to_device(vc->cuda_gpu, addr, workload->data_size_mb * 1024 * 1024, &transfer_time);
                
                uint32_t kid = gpu_launch_kernel(vc->cuda_gpu, "workload_kernel",
                                                256, 1, 1, 256, 1, 1, 49152);
                
                double exec_time;
                gpu_execute_kernel(vc->cuda_gpu, kid, workload->compute_ops, &exec_time);
                
                double transfer_back;
                gpu_memcpy_to_host(vc->cuda_gpu, addr, workload->data_size_mb * 1024 * 1024, &transfer_back);
                
                gpu_free(vc->cuda_gpu, addr);
                
                total_exec_time = transfer_time + exec_time + transfer_back;
                total_energy = exec_time * 0.5;
            }
        } else if (workload->workload_type == WORKLOAD_NEURAL_INFERENCE ||
                  workload->workload_type == WORKLOAD_NEURAL_TRAINING) {
            /* Execute on Neuromorphic Processor */
            uint32_t steps = workload->compute_ops / 1_000_000;
            for (uint32_t step = 0; step < steps && step < 100; step++) {
                nm_simulate_step(vc->neuromorphic, vc->current_time + step, (step % 10 == 0));
            }
            total_exec_time = steps * 0.1;
            total_energy = vc->neuromorphic->energy_consumed_uj;
        } else if (workload->workload_type == WORKLOAD_PHOTONIC_COMPUTE ||
                  workload->workload_type == WORKLOAD_PHOTONIC_OPTIMIZATION) {
            /* Execute on Photonic Processor */
            uint32_t num_photons = workload->compute_ops / 1_000_000;
            if (num_photons > 1000) num_photons = 1000;
            
            uint32_t *photon_ids = (uint32_t *)malloc(num_photons * sizeof(uint32_t));
            if (photon_ids) {
                pp_inject_photons(vc->photonic, num_photons, -20.0, 1550.0, photon_ids, num_photons);
                
                for (uint32_t p = 0; p < num_photons; p++) {
                    uint32_t gate_id = rand() % vc->photonic->num_gates;
                    pp_apply_gate_operation(vc->photonic, &photon_ids[p], 1, gate_id);
                }
                
                uint32_t *detected = (uint32_t *)malloc(num_photons * sizeof(uint32_t));
                if (detected) {
                    pp_detect_photons(vc->photonic, photon_ids, num_photons, detected, num_photons);
                    free(detected);
                }
                
                total_exec_time = num_photons * 0.01;
                total_energy = num_photons * 0.001;
                
                free(photon_ids);
            }
        }
        
        workload->completed_at = time(NULL);
        strcpy(workload->status, "completed");
        workload->performance_score = (double)workload->compute_ops / total_exec_time;
        
        vc->total_energy_consumed += total_energy;
        
        /* Move to completed list */
        if (vc->completed_size >= vc->completed_capacity) {
            vc->completed_capacity *= 2;
            vc->completed_workloads = (workload_t *)realloc(vc->completed_workloads,
                vc->completed_capacity * sizeof(workload_t));
            if (!vc->completed_workloads) return;
        }
        
        vc->completed_workloads[vc->completed_size] = *workload;
        vc->completed_size++;
    }
    
    time_t end_time = time(NULL);
    double elapsed_sec = difftime(end_time, start_time);
    if (elapsed_sec > 0) {
        vc->throughput_workloads_per_sec = (double)vc->queue_size / elapsed_sec;
    }
    
    /* Clear queue */
    vc->queue_size = 0;
}

/**
 * Get system status
 */
void vc_get_system_status(virtual_computer_t *vc, char *buffer, size_t buffer_size) {
    if (!vc || !buffer) return;
    
    time_t now = time(NULL);
    double uptime = difftime(now, vc->started_at);
    
    snprintf(buffer, buffer_size,
        "Virtual Computer System Status\n"
        "  Uptime: %.1f seconds\n"
        "  Total Energy: %.3f J\n"
        "  Workloads Completed: %u\n"
        "  Workloads Queued: %u\n"
        "  Throughput: %.2f workloads/sec\n",
        uptime,
        vc->total_energy_consumed,
        vc->completed_size,
        vc->queue_size,
        vc->throughput_workloads_per_sec
    );
}

/**
 * Print system status
 */
void vc_print_system_status(virtual_computer_t *vc) {
    if (!vc) return;
    
    char status_buf[512];
    vc_get_system_status(vc, status_buf, sizeof(status_buf));
    
    printf("\n");
    printf("================================================================================\n");
    printf("  VIRTUAL COMPUTER SYSTEM STATUS\n");
    printf("================================================================================\n");
    printf("%s\n", status_buf);
    printf("================================================================================\n\n");
}
