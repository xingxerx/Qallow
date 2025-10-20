#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/**
 * @file cuda_parallel.h
 * @brief CUDA parallel processing module for Qallow
 * 
 * Provides multi-GPU support, stream-based parallel execution,
 * and optimized memory management for GPU acceleration.
 */

/* ============================================================================
 * GPU Device Management
 * ============================================================================ */

/**
 * @brief Initialize CUDA parallel processing system
 * @return 0 on success, negative on failure
 */
int cuda_parallel_init(void);

/**
 * @brief Shutdown CUDA parallel processing system
 */
void cuda_parallel_shutdown(void);

/**
 * @brief Get number of available CUDA devices
 * @return Number of GPUs available
 */
int cuda_parallel_get_device_count(void);

/**
 * @brief Get information about a specific GPU device
 * @param device_id GPU device index
 * @param name Output buffer for device name (min 256 bytes)
 * @param compute_capability Output for compute capability (e.g., 89 for sm_89)
 * @param total_memory Output for total GPU memory in bytes
 * @return 0 on success, negative on failure
 */
int cuda_parallel_get_device_info(int device_id, char* name, int* compute_capability, 
                                   uint64_t* total_memory);

/**
 * @brief Set active GPU device
 * @param device_id GPU device index
 * @return 0 on success, negative on failure
 */
int cuda_parallel_set_device(int device_id);

/**
 * @brief Get current active GPU device
 * @return Device ID or negative on error
 */
int cuda_parallel_get_device(void);

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/**
 * @brief Allocate GPU memory
 * @param size Number of bytes to allocate
 * @return Pointer to GPU memory, NULL on failure
 */
void* cuda_parallel_malloc(size_t size);

/**
 * @brief Free GPU memory
 * @param ptr Pointer to GPU memory
 */
void cuda_parallel_free(void* ptr);

/**
 * @brief Copy data from host to GPU
 * @param dst GPU destination pointer
 * @param src Host source pointer
 * @param size Number of bytes to copy
 * @return 0 on success, negative on failure
 */
int cuda_parallel_memcpy_h2d(void* dst, const void* src, size_t size);

/**
 * @brief Copy data from GPU to host
 * @param dst Host destination pointer
 * @param src GPU source pointer
 * @param size Number of bytes to copy
 * @return 0 on success, negative on failure
 */
int cuda_parallel_memcpy_d2h(void* dst, const void* src, size_t size);

/**
 * @brief Copy data from GPU to GPU
 * @param dst GPU destination pointer
 * @param src GPU source pointer
 * @param size Number of bytes to copy
 * @return 0 on success, negative on failure
 */
int cuda_parallel_memcpy_d2d(void* dst, const void* src, size_t size);

/**
 * @brief Get available GPU memory
 * @param free_memory Output for free memory in bytes
 * @param total_memory Output for total memory in bytes
 * @return 0 on success, negative on failure
 */
int cuda_parallel_get_memory_info(uint64_t* free_memory, uint64_t* total_memory);

/* ============================================================================
 * Stream-based Parallel Execution
 * ============================================================================ */

typedef void* cuda_stream_t;

/**
 * @brief Create a CUDA stream for asynchronous operations
 * @return Stream handle, NULL on failure
 */
cuda_stream_t cuda_parallel_stream_create(void);

/**
 * @brief Destroy a CUDA stream
 * @param stream Stream handle
 */
void cuda_parallel_stream_destroy(cuda_stream_t stream);

/**
 * @brief Synchronize a CUDA stream (wait for completion)
 * @param stream Stream handle
 * @return 0 on success, negative on failure
 */
int cuda_parallel_stream_synchronize(cuda_stream_t stream);

/**
 * @brief Check if stream operations are complete
 * @param stream Stream handle
 * @return 1 if complete, 0 if still executing, negative on error
 */
int cuda_parallel_stream_query(cuda_stream_t stream);

/* ============================================================================
 * Batch Processing
 * ============================================================================ */

typedef struct {
    void* input_data;      /**< Input data pointer (GPU or host) */
    size_t input_size;     /**< Size of input data in bytes */
    void* output_data;     /**< Output data pointer (GPU or host) */
    size_t output_size;    /**< Size of output data in bytes */
    int device_id;         /**< Target GPU device */
    cuda_stream_t stream;  /**< CUDA stream for async execution */
} cuda_batch_task_t;

/**
 * @brief Submit a batch of tasks for parallel processing
 * @param tasks Array of batch tasks
 * @param task_count Number of tasks
 * @return 0 on success, negative on failure
 */
int cuda_parallel_batch_submit(const cuda_batch_task_t* tasks, int task_count);

/**
 * @brief Wait for all batch tasks to complete
 * @return 0 on success, negative on failure
 */
int cuda_parallel_batch_wait_all(void);

/* ============================================================================
 * Kernel Execution Utilities
 * ============================================================================ */

/**
 * @brief Get optimal block size for kernel execution
 * @param max_threads_per_block Maximum threads per block for the GPU
 * @return Recommended block size
 */
int cuda_parallel_get_optimal_block_size(int max_threads_per_block);

/**
 * @brief Calculate grid and block dimensions for kernel launch
 * @param total_elements Total number of elements to process
 * @param threads_per_block Threads per block
 * @param grid_size Output for grid dimension
 * @param block_size Output for block dimension
 */
void cuda_parallel_calculate_grid_block(int total_elements, int threads_per_block,
                                        int* grid_size, int* block_size);

/* ============================================================================
 * Performance Monitoring
 * ============================================================================ */

/**
 * @brief Get GPU utilization percentage
 * @param device_id GPU device index
 * @return Utilization percentage (0-100), negative on error
 */
int cuda_parallel_get_utilization(int device_id);

/**
 * @brief Get GPU temperature in Celsius
 * @param device_id GPU device index
 * @return Temperature in Celsius, negative on error
 */
int cuda_parallel_get_temperature(int device_id);

/**
 * @brief Get last CUDA error message
 * @return Error message string
 */
const char* cuda_parallel_get_last_error(void);

#ifdef __cplusplus
}
#endif

