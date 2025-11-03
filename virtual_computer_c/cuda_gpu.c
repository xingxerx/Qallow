#include "cuda_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int cuda_gpu_init(CudaGPU *gpu, int device_id) {
    if (!gpu) {
        return -1;
    }
    memset(gpu, 0, sizeof(*gpu));
    gpu->device_id = device_id;
    return 0;
}

int cuda_gpu_alloc(CudaGPU *gpu, size_t bytes) {
    if (!gpu) {
        return -1;
    }
    cuda_gpu_free(gpu);
    gpu->device_mem = malloc(bytes);
    if (!gpu->device_mem) {
        gpu->mem_bytes = 0;
        return -2;
    }
    gpu->mem_bytes = bytes;
    return 0;
}

int cuda_gpu_free(CudaGPU *gpu) {
    if (!gpu) {
        return -1;
    }
    free(gpu->device_mem);
    gpu->device_mem = NULL;
    gpu->mem_bytes = 0;
    return 0;
}

int cuda_gpu_memcpy_to(CudaGPU *gpu, const void *host_src, size_t bytes) {
    if (!gpu || !gpu->device_mem || bytes > gpu->mem_bytes) {
        return -1;
    }
    memcpy(gpu->device_mem, host_src, bytes);
    return 0;
}

int cuda_gpu_memcpy_from(CudaGPU *gpu, void *host_dst, size_t bytes) {
    if (!gpu || !gpu->device_mem || bytes > gpu->mem_bytes) {
        return -1;
    }
    memcpy(host_dst, gpu->device_mem, bytes);
    return 0;
}

int cuda_gpu_launch_kernel(CudaGPU *gpu, CudaLaunchCfg cfg, const void *params, size_t params_size) {
    if (!gpu) {
        return -1;
    }
    (void)params;
    (void)params_size;

    double start = now_ms();
    volatile uint64_t spin = (uint64_t)cfg.grid_x * (uint64_t)cfg.block_x * 1000ULL;
    for (uint64_t i = 0; i < spin; ++i) {
        /* busy wait to simulate work */
    }
    gpu->perf.kernels_launched++;
    gpu->perf.last_exec_ms = now_ms() - start;
    return 0;
}

double cuda_gpu_last_exec_ms(const CudaGPU *gpu) {
    return gpu ? gpu->perf.last_exec_ms : 0.0;
}
