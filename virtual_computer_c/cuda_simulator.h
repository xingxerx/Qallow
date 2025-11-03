/*
 * CUDA GPU Simulator - C Implementation
 * Simulates CUDA kernel execution, device memory, and GPU operations
 * for agent optimization
 */

#ifndef CUDA_SIMULATOR_H
#define CUDA_SIMULATOR_H

#include <time.h>
#include <stdint.h>
#include <stdbool.h>

/* Kernel Status Enumeration */
typedef enum {
    KERNEL_IDLE = 0,
    KERNEL_QUEUED = 1,
    KERNEL_RUNNING = 2,
    KERNEL_COMPLETED = 3,
    KERNEL_FAILED = 4
} kernel_status_t;

/* GPU Memory Region Structure */
typedef struct {
    uint64_t address;
    size_t size;
    char data_type[32];
    bool in_use;
    time_t allocated_at;
    uint32_t access_count;
} gpu_memory_region_t;

/* CUDA Kernel Structure */
typedef struct {
    uint32_t kernel_id;
    char name[128];
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    size_t shared_memory;
    kernel_status_t status;
    time_t start_time;
    time_t end_time;
    double compute_time_ms;
    uint64_t memory_ops;
    uint32_t registers_per_thread;
} cuda_kernel_t;

/* Virtual GPU Device Structure */
typedef struct {
    uint32_t device_id;
    uint64_t device_memory;
    uint64_t free_memory;
    
    /* Memory management */
    gpu_memory_region_t *allocations;
    uint32_t num_allocations;
    uint32_t allocations_capacity;
    uint64_t memory_counter;
    
    /* Kernel execution */
    cuda_kernel_t *kernels;
    uint32_t num_kernels;
    uint32_t kernels_capacity;
    
    cuda_kernel_t *kernel_queue;
    uint32_t queue_size;
    uint32_t queue_capacity;
    
    /* Statistics */
    uint64_t total_launches;
    uint64_t total_memory_alloc;
    uint64_t total_memory_freed;
    double total_compute_time_ms;
    uint64_t peak_memory_usage;
    
    /* Performance metrics */
    double bandwidth_gbps;
    char compute_capability[8];
    uint32_t sm_count;
    uint32_t cores_per_sm;
} virtual_gpu_t;

/* Function Declarations */

/**
 * Create and initialize a virtual GPU device
 */
virtual_gpu_t* gpu_create(uint32_t device_id, uint32_t device_memory_mb);

/**
 * Destroy and free GPU device
 */
void gpu_destroy(virtual_gpu_t *gpu);

/**
 * Allocate memory on GPU device
 */
uint64_t gpu_malloc(virtual_gpu_t *gpu, size_t size, const char *data_type);

/**
 * Free GPU memory allocation
 */
bool gpu_free(virtual_gpu_t *gpu, uint64_t address);

/**
 * Copy data from host to GPU device
 */
bool gpu_memcpy_to_device(virtual_gpu_t *gpu, uint64_t address, size_t size, 
                          double *transfer_time_ms);

/**
 * Copy data from GPU device to host
 */
bool gpu_memcpy_to_host(virtual_gpu_t *gpu, uint64_t address, size_t size,
                        double *transfer_time_ms);

/**
 * Launch a CUDA kernel
 */
uint32_t gpu_launch_kernel(virtual_gpu_t *gpu, const char *kernel_name,
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          size_t shared_memory);

/**
 * Execute a queued kernel
 */
bool gpu_execute_kernel(virtual_gpu_t *gpu, uint32_t kernel_id, uint64_t compute_ops,
                        double *execution_time_ms);

/**
 * Get device properties
 */
void gpu_get_device_properties(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);

/**
 * Get memory statistics
 */
void gpu_get_memory_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);

/**
 * Get kernel statistics
 */
void gpu_get_kernel_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size);

/**
 * Print GPU status summary
 */
void gpu_print_status(virtual_gpu_t *gpu);

#endif /* CUDA_SIMULATOR_H */
