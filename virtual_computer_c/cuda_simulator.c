/*
 * CUDA GPU Simulator - C Implementation
 */

#include "cuda_simulator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define INITIAL_ALLOCATIONS_CAPACITY 1024
#define INITIAL_KERNELS_CAPACITY 256
#define INITIAL_QUEUE_CAPACITY 128

/**
 * Create and initialize a virtual GPU device
 */
virtual_gpu_t* gpu_create(uint32_t device_id, uint32_t device_memory_mb) {
    virtual_gpu_t *gpu = (virtual_gpu_t *)malloc(sizeof(virtual_gpu_t));
    if (!gpu) return NULL;
    
    gpu->device_id = device_id;
    gpu->device_memory = (uint64_t)device_memory_mb * 1024 * 1024;
    gpu->free_memory = gpu->device_memory;
    gpu->memory_counter = 0x10000000;  /* Start address */
    
    /* Initialize memory allocations */
    gpu->allocations = (gpu_memory_region_t *)malloc(
        INITIAL_ALLOCATIONS_CAPACITY * sizeof(gpu_memory_region_t)
    );
    gpu->num_allocations = 0;
    gpu->allocations_capacity = INITIAL_ALLOCATIONS_CAPACITY;
    
    /* Initialize kernel storage */
    gpu->kernels = (cuda_kernel_t *)malloc(
        INITIAL_KERNELS_CAPACITY * sizeof(cuda_kernel_t)
    );
    gpu->num_kernels = 0;
    gpu->kernels_capacity = INITIAL_KERNELS_CAPACITY;
    
    /* Initialize kernel queue */
    gpu->kernel_queue = (cuda_kernel_t *)malloc(
        INITIAL_QUEUE_CAPACITY * sizeof(cuda_kernel_t)
    );
    gpu->queue_size = 0;
    gpu->queue_capacity = INITIAL_QUEUE_CAPACITY;
    
    /* Statistics */
    gpu->total_launches = 0;
    gpu->total_memory_alloc = 0;
    gpu->total_memory_freed = 0;
    gpu->total_compute_time_ms = 0.0;
    gpu->peak_memory_usage = 0;
    
    /* Performance metrics (Ampere architecture) */
    gpu->bandwidth_gbps = 288.0;
    strcpy(gpu->compute_capability, "8.0");
    gpu->sm_count = 80;
    gpu->cores_per_sm = 128;
    
    return gpu;
}

/**
 * Destroy and free GPU device
 */
void gpu_destroy(virtual_gpu_t *gpu) {
    if (!gpu) return;
    free(gpu->allocations);
    free(gpu->kernels);
    free(gpu->kernel_queue);
    free(gpu);
}

/**
 * Allocate memory on GPU device
 */
uint64_t gpu_malloc(virtual_gpu_t *gpu, size_t size, const char *data_type) {
    if (!gpu || size > gpu->free_memory) {
        return 0;  /* Allocation failed */
    }
    
    /* Resize allocations if needed */
    if (gpu->num_allocations >= gpu->allocations_capacity) {
        gpu->allocations_capacity *= 2;
        gpu->allocations = (gpu_memory_region_t *)realloc(
            gpu->allocations,
            gpu->allocations_capacity * sizeof(gpu_memory_region_t)
        );
        if (!gpu->allocations) return 0;
    }
    
    uint64_t addr = gpu->memory_counter;
    gpu_memory_region_t *region = &gpu->allocations[gpu->num_allocations];
    
    region->address = addr;
    region->size = size;
    strncpy(region->data_type, data_type, sizeof(region->data_type) - 1);
    region->in_use = true;
    region->allocated_at = time(NULL);
    region->access_count = 0;
    
    gpu->num_allocations++;
    gpu->free_memory -= size;
    gpu->total_memory_alloc += size;
    gpu->memory_counter += size;
    
    /* Track peak memory */
    uint64_t used = gpu->device_memory - gpu->free_memory;
    if (used > gpu->peak_memory_usage) {
        gpu->peak_memory_usage = used;
    }
    
    return addr;
}

/**
 * Free GPU memory allocation
 */
bool gpu_free(virtual_gpu_t *gpu, uint64_t address) {
    if (!gpu) return false;
    
    for (uint32_t i = 0; i < gpu->num_allocations; i++) {
        if (gpu->allocations[i].address == address && gpu->allocations[i].in_use) {
            size_t size = gpu->allocations[i].size;
            gpu->free_memory += size;
            gpu->total_memory_freed += size;
            gpu->allocations[i].in_use = false;
            
            /* Compact allocations array */
            for (uint32_t j = i; j < gpu->num_allocations - 1; j++) {
                gpu->allocations[j] = gpu->allocations[j + 1];
            }
            gpu->num_allocations--;
            return true;
        }
    }
    
    return false;
}

/**
 * Copy data from host to GPU device
 */
bool gpu_memcpy_to_device(virtual_gpu_t *gpu, uint64_t address, size_t size,
                          double *transfer_time_ms) {
    if (!gpu || !transfer_time_ms) return false;
    
    /* Verify allocation exists */
    bool found = false;
    for (uint32_t i = 0; i < gpu->num_allocations; i++) {
        if (gpu->allocations[i].address == address) {
            if (size > gpu->allocations[i].size) {
                return false;
            }
            gpu->allocations[i].access_count++;
            found = true;
            break;
        }
    }
    
    if (!found) return false;
    
    /* Simulate PCIe transfer time (PCIe 4.0 = ~32 GB/s) */
    double pcie_bandwidth_gbs = 32.0;
    *transfer_time_ms = (double)size / (pcie_bandwidth_gbs * 1024.0 * 1024.0 * 1024.0) * 1000.0;
    
    return true;
}

/**
 * Copy data from GPU device to host
 */
bool gpu_memcpy_to_host(virtual_gpu_t *gpu, uint64_t address, size_t size,
                        double *transfer_time_ms) {
    /* Same as to_device for simulation purposes */
    return gpu_memcpy_to_device(gpu, address, size, transfer_time_ms);
}

/**
 * Launch a CUDA kernel
 */
uint32_t gpu_launch_kernel(virtual_gpu_t *gpu, const char *kernel_name,
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          size_t shared_memory) {
    if (!gpu || !kernel_name) return 0;
    
    /* Resize queue if needed */
    if (gpu->queue_size >= gpu->queue_capacity) {
        gpu->queue_capacity *= 2;
        gpu->kernel_queue = (cuda_kernel_t *)realloc(
            gpu->kernel_queue,
            gpu->queue_capacity * sizeof(cuda_kernel_t)
        );
        if (!gpu->kernel_queue) return 0;
    }
    
    uint32_t kernel_id = gpu->total_launches + 1;
    cuda_kernel_t *kernel = &gpu->kernel_queue[gpu->queue_size];
    
    kernel->kernel_id = kernel_id;
    strncpy(kernel->name, kernel_name, sizeof(kernel->name) - 1);
    kernel->grid_x = grid_x;
    kernel->grid_y = grid_y;
    kernel->grid_z = grid_z;
    kernel->block_x = block_x;
    kernel->block_y = block_y;
    kernel->block_z = block_z;
    kernel->shared_memory = shared_memory;
    kernel->status = KERNEL_QUEUED;
    kernel->start_time = 0;
    kernel->end_time = 0;
    kernel->compute_time_ms = 0.0;
    kernel->memory_ops = 0;
    kernel->registers_per_thread = 0;
    
    gpu->queue_size++;
    gpu->total_launches++;
    
    return kernel_id;
}

/**
 * Execute a queued kernel
 */
bool gpu_execute_kernel(virtual_gpu_t *gpu, uint32_t kernel_id, uint64_t compute_ops,
                        double *execution_time_ms) {
    if (!gpu || !execution_time_ms) return false;
    
    /* Find kernel in queue */
    cuda_kernel_t *kernel = NULL;
    int queue_idx = -1;
    for (int i = 0; i < (int)gpu->queue_size; i++) {
        if (gpu->kernel_queue[i].kernel_id == kernel_id) {
            kernel = &gpu->kernel_queue[i];
            queue_idx = i;
            break;
        }
    }
    
    if (!kernel) return false;
    
    /* Simulate kernel execution */
    kernel->status = KERNEL_RUNNING;
    kernel->start_time = time(NULL);
    
    /* Calculate execution time */
    uint32_t threads = (kernel->block_x * kernel->block_y * kernel->block_z) *
                       (kernel->grid_x * kernel->grid_y * kernel->grid_z);
    
    double gpu_tflops = 20.0;  /* ~20 TFLOPS for typical GPU */
    double theoretical_time = (double)compute_ops / (gpu_tflops * 1e12) * 1000.0;
    
    double memory_penalty = (double)kernel->memory_ops / (gpu->bandwidth_gbps * 1e9) * 1000.0;
    
    /* Add variance */
    srand(time(NULL) + kernel_id);
    double variance = 0.95 + (rand() / (double)RAND_MAX) * 0.1;
    *execution_time_ms = (theoretical_time + memory_penalty) * variance;
    
    kernel->compute_time_ms = *execution_time_ms;
    kernel->end_time = time(NULL);
    kernel->status = KERNEL_COMPLETED;
    
    gpu->total_compute_time_ms += *execution_time_ms;
    
    /* Move kernel to completed list */
    if (gpu->num_kernels >= gpu->kernels_capacity) {
        gpu->kernels_capacity *= 2;
        gpu->kernels = (cuda_kernel_t *)realloc(
            gpu->kernels,
            gpu->kernels_capacity * sizeof(cuda_kernel_t)
        );
        if (!gpu->kernels) return false;
    }
    
    gpu->kernels[gpu->num_kernels] = *kernel;
    gpu->num_kernels++;
    
    /* Remove from queue */
    for (int i = queue_idx; i < (int)gpu->queue_size - 1; i++) {
        gpu->kernel_queue[i] = gpu->kernel_queue[i + 1];
    }
    gpu->queue_size--;
    
    return true;
}

/**
 * Get device properties
 */
void gpu_get_device_properties(virtual_gpu_t *gpu, char *buffer, size_t buffer_size) {
    if (!gpu || !buffer) return;
    
    snprintf(buffer, buffer_size,
        "GPU Device %u\n"
        "  Compute Capability: %s\n"
        "  SMs: %u, Cores/SM: %u, Total Cores: %u\n"
        "  Memory: %lu MB\n"
        "  Bandwidth: %.0f GB/s\n"
        "  Kernels Launched: %lu\n",
        gpu->device_id,
        gpu->compute_capability,
        gpu->sm_count,
        gpu->cores_per_sm,
        gpu->sm_count * gpu->cores_per_sm,
        gpu->device_memory / (1024 * 1024),
        gpu->bandwidth_gbps,
        gpu->total_launches
    );
}

/**
 * Get memory statistics
 */
void gpu_get_memory_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size) {
    if (!gpu || !buffer) return;
    
    uint64_t used = gpu->device_memory - gpu->free_memory;
    
    snprintf(buffer, buffer_size,
        "GPU Memory Statistics\n"
        "  Total: %lu MB\n"
        "  Used: %lu MB\n"
        "  Free: %lu MB\n"
        "  Peak Usage: %lu MB\n"
        "  Allocations: %u\n",
        gpu->device_memory / (1024 * 1024),
        used / (1024 * 1024),
        gpu->free_memory / (1024 * 1024),
        gpu->peak_memory_usage / (1024 * 1024),
        gpu->num_allocations
    );
}

/**
 * Get kernel statistics
 */
void gpu_get_kernel_stats(virtual_gpu_t *gpu, char *buffer, size_t buffer_size) {
    if (!gpu || !buffer) return;
    
    snprintf(buffer, buffer_size,
        "GPU Kernel Statistics\n"
        "  Total Launches: %lu\n"
        "  Completed: %u\n"
        "  Queued: %u\n"
        "  Total Compute Time: %.2f ms\n",
        gpu->total_launches,
        gpu->num_kernels,
        gpu->queue_size,
        gpu->total_compute_time_ms
    );
}

/**
 * Print GPU status summary
 */
void gpu_print_status(virtual_gpu_t *gpu) {
    if (!gpu) return;
    
    printf("\n");
    printf("================================================================================\n");
    printf("  GPU Device %u Status\n", gpu->device_id);
    printf("================================================================================\n");
    printf("  Compute Capability: %s\n", gpu->compute_capability);
    printf("  SMs: %u, Cores/SM: %u, Total Cores: %u\n",
           gpu->sm_count, gpu->cores_per_sm, gpu->sm_count * gpu->cores_per_sm);
    
    uint64_t used = gpu->device_memory - gpu->free_memory;
    printf("  Memory: %lu/%lu MB (Peak: %lu MB)\n",
           used / (1024 * 1024),
           gpu->device_memory / (1024 * 1024),
           gpu->peak_memory_usage / (1024 * 1024));
    printf("  Kernels Launched: %lu, Completed: %u\n",
           gpu->total_launches, gpu->num_kernels);
    printf("  Avg Kernel Time: %.2f ms\n",
           gpu->num_kernels > 0 ? gpu->total_compute_time_ms / gpu->num_kernels : 0.0);
    printf("================================================================================\n\n");
}
