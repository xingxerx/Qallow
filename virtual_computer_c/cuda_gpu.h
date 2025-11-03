/*
 * cuda_gpu.h - Minimal CUDA GPU simulator interface
 */

#ifndef CUDA_GPU_H
#define CUDA_GPU_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int device_id;
    size_t mem_bytes;
    void *device_mem;
    struct {
        uint64_t kernels_launched;
        double last_exec_ms;
    } perf;
} CudaGPU;

typedef struct {
    uint32_t grid_x;
    uint32_t block_x;
} CudaLaunchCfg;

int cuda_gpu_init(CudaGPU *gpu, int device_id);
int cuda_gpu_alloc(CudaGPU *gpu, size_t bytes);
int cuda_gpu_free(CudaGPU *gpu);
int cuda_gpu_memcpy_to(CudaGPU *gpu, const void *host_src, size_t bytes);
int cuda_gpu_memcpy_from(CudaGPU *gpu, void *host_dst, size_t bytes);
int cuda_gpu_launch_kernel(CudaGPU *gpu, CudaLaunchCfg cfg, const void *params, size_t params_size);
double cuda_gpu_last_exec_ms(const CudaGPU *gpu);

#endif /* CUDA_GPU_H */
