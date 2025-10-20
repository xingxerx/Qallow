#include "cuda_parallel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Global State
 * ============================================================================ */

static int g_cuda_initialized = 0;
static int g_device_count = 0;
static char g_last_error[256] = {0};
static cuda_stream_t g_streams[16] = {NULL};
static int g_stream_count = 0;

#define MAX_STREAMS 16
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        snprintf(g_last_error, sizeof(g_last_error), \
                 "CUDA Error: %s", cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/* ============================================================================
 * GPU Device Management
 * ============================================================================ */

int cuda_parallel_init(void) {
    if (g_cuda_initialized) {
        return 0;
    }
    
    cudaError_t err = cudaGetDeviceCount(&g_device_count);
    if (err != cudaSuccess) {
        snprintf(g_last_error, sizeof(g_last_error),
                 "Failed to get device count: %s", cudaGetErrorString(err));
        return -1;
    }
    
    if (g_device_count == 0) {
        snprintf(g_last_error, sizeof(g_last_error), "No CUDA devices found");
        return -1;
    }
    
    g_cuda_initialized = 1;
    return 0;
}

void cuda_parallel_shutdown(void) {
    for (int i = 0; i < g_stream_count; i++) {
        if (g_streams[i] != NULL) {
            cudaStreamDestroy((cudaStream_t)g_streams[i]);
        }
    }
    g_stream_count = 0;
    g_cuda_initialized = 0;
}

int cuda_parallel_get_device_count(void) {
    if (!g_cuda_initialized) {
        cuda_parallel_init();
    }
    return g_device_count;
}

int cuda_parallel_get_device_info(int device_id, char* name, int* compute_capability,
                                   uint64_t* total_memory) {
    if (!g_cuda_initialized) {
        cuda_parallel_init();
    }
    
    if (device_id < 0 || device_id >= g_device_count) {
        snprintf(g_last_error, sizeof(g_last_error), "Invalid device ID: %d", device_id);
        return -1;
    }
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    if (name) {
        strncpy(name, prop.name, 255);
        name[255] = '\0';
    }
    
    if (compute_capability) {
        *compute_capability = prop.major * 10 + prop.minor;
    }
    
    if (total_memory) {
        *total_memory = (uint64_t)prop.totalGlobalMem;
    }
    
    return 0;
}

int cuda_parallel_set_device(int device_id) {
    if (!g_cuda_initialized) {
        cuda_parallel_init();
    }
    
    if (device_id < 0 || device_id >= g_device_count) {
        snprintf(g_last_error, sizeof(g_last_error), "Invalid device ID: %d", device_id);
        return -1;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    return 0;
}

int cuda_parallel_get_device(void) {
    int device_id;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return -1;
    }
    return device_id;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

void* cuda_parallel_malloc(size_t size) {
    void* ptr = NULL;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        snprintf(g_last_error, sizeof(g_last_error), "Failed to allocate %zu bytes on GPU", size);
        return NULL;
    }
    return ptr;
}

void cuda_parallel_free(void* ptr) {
    if (ptr != NULL) {
        cudaFree(ptr);
    }
}

int cuda_parallel_memcpy_h2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

int cuda_parallel_memcpy_d2h(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}

int cuda_parallel_memcpy_d2d(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return 0;
}

int cuda_parallel_get_memory_info(uint64_t* free_memory, uint64_t* total_memory) {
    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    
    if (free_memory) {
        *free_memory = (uint64_t)free_bytes;
    }
    if (total_memory) {
        *total_memory = (uint64_t)total_bytes;
    }
    
    return 0;
}

/* ============================================================================
 * Stream-based Parallel Execution
 * ============================================================================ */

cuda_stream_t cuda_parallel_stream_create(void) {
    if (g_stream_count >= MAX_STREAMS) {
        snprintf(g_last_error, sizeof(g_last_error), "Maximum streams (%d) reached", MAX_STREAMS);
        return NULL;
    }
    
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        snprintf(g_last_error, sizeof(g_last_error), "Failed to create CUDA stream");
        return NULL;
    }
    
    g_streams[g_stream_count] = (cuda_stream_t)stream;
    return g_streams[g_stream_count++];
}

void cuda_parallel_stream_destroy(cuda_stream_t stream) {
    if (stream != NULL) {
        cudaStreamDestroy((cudaStream_t)stream);
    }
}

int cuda_parallel_stream_synchronize(cuda_stream_t stream) {
    if (stream == NULL) {
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        CUDA_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
    }
    return 0;
}

int cuda_parallel_stream_query(cuda_stream_t stream) {
    cudaError_t err;
    if (stream == NULL) {
        err = cudaDeviceSynchronize();
    } else {
        err = cudaStreamQuery((cudaStream_t)stream);
    }
    
    if (err == cudaSuccess) {
        return 1;
    } else if (err == cudaErrorNotReady) {
        return 0;
    } else {
        snprintf(g_last_error, sizeof(g_last_error), "Stream query error: %s", 
                 cudaGetErrorString(err));
        return -1;
    }
}

/* ============================================================================
 * Batch Processing
 * ============================================================================ */

int cuda_parallel_batch_submit(const cuda_batch_task_t* tasks, int task_count) {
    if (tasks == NULL || task_count <= 0) {
        snprintf(g_last_error, sizeof(g_last_error), "Invalid batch parameters");
        return -1;
    }
    
    for (int i = 0; i < task_count; i++) {
        const cuda_batch_task_t* task = &tasks[i];
        
        if (task->device_id >= 0) {
            CUDA_CHECK(cudaSetDevice(task->device_id));
        }
        
        if (task->input_data && task->input_size > 0) {
            CUDA_CHECK(cudaMemcpy(task->output_data, task->input_data, 
                                  task->input_size, cudaMemcpyDeviceToDevice));
        }
    }
    
    return 0;
}

int cuda_parallel_batch_wait_all(void) {
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ============================================================================
 * Kernel Execution Utilities
 * ============================================================================ */

int cuda_parallel_get_optimal_block_size(int max_threads_per_block) {
    if (max_threads_per_block >= 1024) return 1024;
    if (max_threads_per_block >= 512) return 512;
    if (max_threads_per_block >= 256) return 256;
    return 128;
}

void cuda_parallel_calculate_grid_block(int total_elements, int threads_per_block,
                                        int* grid_size, int* block_size) {
    if (block_size) {
        *block_size = threads_per_block;
    }
    if (grid_size) {
        *grid_size = (total_elements + threads_per_block - 1) / threads_per_block;
    }
}

/* ============================================================================
 * Performance Monitoring
 * ============================================================================ */

int cuda_parallel_get_utilization(int device_id) {
    // Note: Actual utilization requires NVIDIA Management Library (NVML)
    // This is a placeholder that returns -1 (not available)
    (void)device_id;
    return -1;
}

int cuda_parallel_get_temperature(int device_id) {
    // Note: Temperature requires NVIDIA Management Library (NVML)
    // This is a placeholder that returns -1 (not available)
    (void)device_id;
    return -1;
}

const char* cuda_parallel_get_last_error(void) {
    return g_last_error[0] ? g_last_error : "No error";
}

