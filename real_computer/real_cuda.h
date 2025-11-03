/*
 * Real CUDA GPU Wrapper - C Implementation
 * Direct interface to NVIDIA CUDA Runtime API
 * Provides actual GPU compute capability using CUDA kernels
 */

#ifndef REAL_CUDA_H
#define REAL_CUDA_H

#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>

/* GPU Device Context */
typedef struct {
    int device_id;
    struct cudaDeviceProp device_prop;
    bool initialized;
    char device_name[256];
    size_t total_memory;
    size_t free_memory;
    
    /* Statistics */
    uint64_t kernels_launched;
    uint64_t bytes_transferred;
    double total_compute_time_ms;
} cuda_context_t;

/* GPU Memory Buffer */
typedef struct {
    void *device_ptr;
    void *host_ptr;
    size_t size;
    bool pinned;
} gpu_buffer_t;

/* GPU Kernel Configuration */
typedef struct {
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    size_t shared_memory;
    cudaStream_t stream;
} kernel_config_t;

/* Function Declarations */

/**
 * Initialize CUDA context
 */
cuda_context_t* cuda_init(int device_id);

/**
 * Cleanup CUDA context
 */
void cuda_cleanup(cuda_context_t *ctx);

/**
 * Allocate device memory
 */
gpu_buffer_t* cuda_malloc(cuda_context_t *ctx, size_t size);

/**
 * Free device memory
 */
void cuda_free(gpu_buffer_t *buffer);

/**
 * Allocate pinned host memory
 */
gpu_buffer_t* cuda_malloc_pinned(cuda_context_t *ctx, size_t size);

/**
 * Copy host to device
 */
cudaError_t cuda_h2d(gpu_buffer_t *buffer, const void *host_data, size_t size);

/**
 * Copy device to host
 */
cudaError_t cuda_d2h(void *host_data, gpu_buffer_t *buffer, size_t size);

/**
 * Async copy host to device
 */
cudaError_t cuda_h2d_async(gpu_buffer_t *buffer, const void *host_data, size_t size,
                          cudaStream_t stream);

/**
 * Async copy device to host
 */
cudaError_t cuda_d2h_async(void *host_data, gpu_buffer_t *buffer, size_t size,
                          cudaStream_t stream);

/**
 * Create kernel configuration
 */
kernel_config_t cuda_make_kernel_config(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                       size_t shared_memory);

/**
 * Get device properties
 */
void cuda_get_device_properties(cuda_context_t *ctx, char *buffer, size_t size);

/**
 * Get memory info
 */
void cuda_get_memory_info(cuda_context_t *ctx, size_t *free, size_t *total);

/**
 * Print device status
 */
void cuda_print_status(cuda_context_t *ctx);

/**
 * Check for CUDA errors
 */
bool cuda_check_error(cudaError_t error, const char *msg);

#endif /* REAL_CUDA_H */
