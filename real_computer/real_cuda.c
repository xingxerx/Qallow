/* Multi-block comment removed */

#include "real_cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Multi-block comment removed */
cuda_context_t* cuda_init(int device_id) {
    cuda_context_t *ctx = (cuda_context_t *)malloc(sizeof(cuda_context_t));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate CUDA context\n");
        return NULL;
    }

    memset(ctx, 0, sizeof(cuda_context_t));
    ctx->device_id = device_id;
    ctx->kernels_launched = 0;
    ctx->bytes_transferred = 0;
    ctx->total_compute_time_ms = 0.0;


    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to set device %d: %s\n",
                device_id, cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }


    err = cudaGetDeviceProperties(&ctx->device_prop, device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to get device properties: %s\n",
                cudaGetErrorString(err));
        free(ctx);
        return NULL;
    }

    strncpy(ctx->device_name, ctx->device_prop.name, sizeof(ctx->device_name) - 1);
    ctx->device_name[sizeof(ctx->device_name) - 1] = '\0';
    ctx->total_memory = ctx->device_prop.totalGlobalMem;

    ctx->initialized = true;

    return ctx;
}


void cuda_cleanup(cuda_context_t *ctx) {
    if (!ctx) return;
    if (ctx->initialized) {
        cudaDeviceReset();
    }
    free(ctx);
}


gpu_buffer_t* cuda_malloc(cuda_context_t *ctx, size_t size) {
    if (!ctx || !ctx->initialized || size == 0) {
        return NULL;
    }

    gpu_buffer_t *buffer = (gpu_buffer_t *)malloc(sizeof(gpu_buffer_t));
    if (!buffer) {
        fprintf(stderr, "Failed to allocate GPU buffer structure\n");
        return NULL;
    }

    memset(buffer, 0, sizeof(gpu_buffer_t));
    buffer->size = size;
    buffer->pinned = false;


    cudaError_t err = cudaMalloc(&buffer->device_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: cudaMalloc failed for %zu bytes: %s\n",
                size, cudaGetErrorString(err));
        free(buffer);
        return NULL;
    }


    buffer->host_ptr = malloc(size);
    if (!buffer->host_ptr) {
        fprintf(stderr, "Failed to allocate host memory for GPU buffer\n");
        cudaFree(buffer->device_ptr);
        free(buffer);
        return NULL;
    }

    return buffer;
}


void cuda_free(gpu_buffer_t *buffer) {
    if (!buffer) return;

    if (buffer->device_ptr) {
        cudaFree(buffer->device_ptr);
    }

    if (buffer->host_ptr) {
        if (buffer->pinned) {
            cudaFreeHost(buffer->host_ptr);
        } else {
            free(buffer->host_ptr);
        }
    }

    free(buffer);
}


gpu_buffer_t* cuda_malloc_pinned(cuda_context_t *ctx, size_t size) {
    if (!ctx || !ctx->initialized || size == 0) {
        return NULL;
    }

    gpu_buffer_t *buffer = (gpu_buffer_t *)malloc(sizeof(gpu_buffer_t));
    if (!buffer) {
        fprintf(stderr, "Failed to allocate GPU buffer structure\n");
        return NULL;
    }

    memset(buffer, 0, sizeof(gpu_buffer_t));
    buffer->size = size;
    buffer->pinned = true;


    cudaError_t err = cudaMalloc(&buffer->device_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: cudaMalloc failed for %zu bytes: %s\n",
                size, cudaGetErrorString(err));
        free(buffer);
        return NULL;
    }


    err = cudaMallocHost(&buffer->host_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: cudaMallocHost failed for %zu bytes: %s\n",
                size, cudaGetErrorString(err));
        cudaFree(buffer->device_ptr);
        free(buffer);
        return NULL;
    }

    return buffer;
}


cudaError_t cuda_h2d(gpu_buffer_t *buffer, const void *host_data, size_t size) {
    if (!buffer || !host_data || size > buffer->size) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaMemcpy(buffer->device_ptr, host_data, size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) {
        buffer->size = size;
    }
    return err;
}


cudaError_t cuda_d2h(void *host_data, gpu_buffer_t *buffer, size_t size) {
    if (!buffer || !host_data || size > buffer->size) {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpy(host_data, buffer->device_ptr, size, cudaMemcpyDeviceToHost);
}


cudaError_t cuda_h2d_async(gpu_buffer_t *buffer, const void *host_data, size_t size,
                          cudaStream_t stream) {
    if (!buffer || !host_data || size > buffer->size) {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpyAsync(buffer->device_ptr, host_data, size,
                          cudaMemcpyHostToDevice, stream);
}


cudaError_t cuda_d2h_async(void *host_data, gpu_buffer_t *buffer, size_t size,
                          cudaStream_t stream) {
    if (!buffer || !host_data || size > buffer->size) {
        return cudaErrorInvalidValue;
    }

    return cudaMemcpyAsync(host_data, buffer->device_ptr, size,
                          cudaMemcpyDeviceToHost, stream);
}


kernel_config_t cuda_make_kernel_config(uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                       size_t shared_memory) {
    kernel_config_t config;
    config.grid_x = grid_x > 0 ? grid_x : 1;
    config.grid_y = grid_y > 0 ? grid_y : 1;
    config.grid_z = grid_z > 0 ? grid_z : 1;
    config.block_x = block_x > 0 ? block_x : 1;
    config.block_y = block_y > 0 ? block_y : 1;
    config.block_z = block_z > 0 ? block_z : 1;
    config.shared_memory = shared_memory;
    config.stream = NULL;
    return config;
}


void cuda_get_device_properties(cuda_context_t *ctx, char *buffer, size_t size) {
    if (!ctx || !buffer || size == 0) return;

    struct cudaDeviceProp *prop = &ctx->device_prop;
    snprintf(buffer, size,
        "CUDA Device Properties:\n"
        "  Device: %s\n"
        "  Compute Capability: %d.%d\n"
        "  Total Memory: %.2f GB\n"
        "  Max Threads Per Block: %d\n"
        "  Max Grid Dimensions: (%d, %d, %d)\n"
        "  Max Block Dimensions: (%d, %d, %d)\n"
        "  Warp Size: %d\n"
        "  Multiprocessor Count: %d\n"
        "  Shared Memory Per Block: %zu bytes\n"
        "  Registers Per Block: %d\n"
        "  Max Threads Per Multiprocessor: %d\n"
        "  Clock Rate: %.2f GHz\n",
        ctx->device_name,
        prop->major, prop->minor,
        prop->totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
        prop->maxThreadsPerBlock,
        prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2],
        prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2],
        prop->warpSize,
        prop->multiProcessorCount,
        prop->sharedMemPerBlock,
        prop->regsPerBlock,
        prop->maxThreadsPerMultiProcessor,
        prop->clockRate / 1000000.0);
}


void cuda_get_memory_info(cuda_context_t *ctx, size_t *free, size_t *total) {
    if (!ctx) return;

    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: Failed to get memory info: %s\n",
                cudaGetErrorString(err));
        if (free) *free = 0;
        if (total) *total = 0;
    }
}


void cuda_print_status(cuda_context_t *ctx) {
    if (!ctx) return;

    char props[1024];
    cuda_get_device_properties(ctx, props, sizeof(props));
    printf("%s\n", props);

    size_t free_mem = 0, total_mem = 0;
    cuda_get_memory_info(ctx, &free_mem, &total_mem);
    printf("  Current Free Memory: %.2f MB / %.2f MB\n",
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));

    printf("  Statistics:\n");
    printf("    Kernels Launched: %" PRIu64 "\n", ctx->kernels_launched);
    printf("    Bytes Transferred: %" PRIu64 " (%.2f MB)\n",
           ctx->bytes_transferred,
           ctx->bytes_transferred / (1024.0 * 1024.0));
    printf("    Total Compute Time: %.2f ms\n", ctx->total_compute_time_ms);
}


bool cuda_check_error(cudaError_t error, const char *msg) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        return false;
    }
    return true;
}
