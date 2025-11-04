#include "qallow/multi_gpu.h"
#include "qallow/error_codes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef QALLOW_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

/**
 * Multi-GPU Implementation
 *
 * This module provides GPU device enumeration, selection, and work distribution.
 *
 * Implementation Notes:
 * - Uses CUDA Runtime API for device enumeration (when CUDA available)
 * - Falls back to CPU when CUDA unavailable or no GPUs present
 * - Supports NVIDIA_VISIBLE_DEVICES environment variable
 * - Respects CUDA_VISIBLE_DEVICES for device filtering
 */

qallow_gpu_devices_t qallow_gpu_enumerate_devices(void) {
    qallow_gpu_devices_t devices;
    memset(&devices, 0, sizeof(devices));

#ifdef QALLOW_ENABLE_CUDA
    int cuda_device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&cuda_device_count);

    if (status != cudaSuccess || cuda_device_count == 0) {
        fprintf(stderr, "[GPU] No CUDA devices found. Falling back to CPU.\n");
        fprintf(stderr, "[GPU] Error: %s\n", cudaGetErrorString(status));
        return devices;  /* Return empty devices */
    }

    devices.device_count = cuda_device_count;
    devices.devices = (qallow_gpu_device_info_t*)malloc(
        sizeof(qallow_gpu_device_info_t) * cuda_device_count);

    if (!devices.devices) {
        fprintf(stderr, "[GPU] Memory allocation failed\n");
        devices.device_count = 0;
        return devices;
    }

    /* Query each device */
    for (int i = 0; i < cuda_device_count; ++i) {
        cudaSetDevice(i);

        qallow_gpu_device_info_t* info = &devices.devices[i];
        info->device_id = i;

        /* Get device name */
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        strncpy(info->device_name, props.name, sizeof(info->device_name) - 1);
        info->device_name[sizeof(info->device_name) - 1] = '\0';

        /* Get memory info */
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info->memory_total_mb = total_mem / (1024 * 1024);
        info->memory_available_mb = free_mem / (1024 * 1024);

        /* Get compute capability */
        info->compute_capability_major = props.major;
        info->compute_capability_minor = props.minor;
        info->max_threads_per_block = props.maxThreadsPerBlock;
        info->multiprocessor_count = props.multiProcessorCount;
        info->clock_rate_mhz = props.clockRate / 1000.0f;

        fprintf(stderr, "[GPU] Device %d: %s (%d.%d, %llu MB total)\n",
                i, info->device_name,
                info->compute_capability_major,
                info->compute_capability_minor,
                info->memory_total_mb);
    }

    /* Initialize selected devices (initially all) */
    devices.selected_devices = (int*)malloc(sizeof(int) * cuda_device_count);
    if (devices.selected_devices) {
        for (int i = 0; i < cuda_device_count; ++i) {
            devices.selected_devices[i] = i;
        }
        devices.selected_count = cuda_device_count;
    }

#else
    fprintf(stderr, "[GPU] CUDA not enabled. Falling back to CPU-only execution.\n");
#endif

    return devices;
}

qallow_gpu_device_info_t qallow_gpu_get_device_info(
    const qallow_gpu_devices_t* devices,
    int device_id) {

    qallow_gpu_device_info_t info;
    memset(&info, 0, sizeof(info));

    if (!devices || device_id < 0 || device_id >= devices->device_count) {
        return info;
    }

    return devices->devices[device_id];
}

int qallow_gpu_select_device(qallow_gpu_devices_t* devices, int device_id) {
    if (!devices || device_id < 0 || device_id >= devices->device_count) {
        return QALLOW_ERR_INVALID_PARAMETER;
    }

#ifdef QALLOW_ENABLE_CUDA
    cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        fprintf(stderr, "[GPU] Failed to select device %d: %s\n",
                device_id, cudaGetErrorString(status));
        return QALLOW_ERR_GPU_NOT_AVAILABLE;
    }
#endif

    /* Update selected devices to just this one */
    if (devices->selected_devices) {
        devices->selected_devices[0] = device_id;
        devices->selected_count = 1;
    }

    fprintf(stderr, "[GPU] Selected device %d (%s)\n",
            device_id, devices->devices[device_id].device_name);
    return QALLOW_SUCCESS;
}

int qallow_gpu_select_devices(qallow_gpu_devices_t* devices,
                              const int* device_ids,
                              int count) {
    if (!devices || !device_ids || count <= 0 || count > devices->device_count) {
        return QALLOW_ERR_INVALID_PARAMETER;
    }

    /* Validate all device IDs */
    for (int i = 0; i < count; ++i) {
        if (device_ids[i] < 0 || device_ids[i] >= devices->device_count) {
            return QALLOW_ERR_INVALID_PARAMETER;
        }
    }

    /* Update selected devices */
    if (devices->selected_devices) {
        free(devices->selected_devices);
    }
    devices->selected_devices = (int*)malloc(sizeof(int) * count);
    if (!devices->selected_devices) {
        return QALLOW_ERR_MEMORY_ALLOC;
    }

    memcpy(devices->selected_devices, device_ids, sizeof(int) * count);
    devices->selected_count = count;

    fprintf(stderr, "[GPU] Selected %d device(s):\n", count);
    for (int i = 0; i < count; ++i) {
        fprintf(stderr, "[GPU]   - Device %d (%s)\n",
                device_ids[i], devices->devices[device_ids[i]].device_name);
    }

    return QALLOW_SUCCESS;
}

int qallow_gpu_distribute_work(qallow_gpu_devices_t* devices,
                               qallow_gpu_work_unit_t* work_units,
                               int unit_count,
                               qallow_gpu_distribution_strategy_t strategy) {
    if (!devices || !work_units || unit_count <= 0) {
        return QALLOW_ERR_INVALID_PARAMETER;
    }

    if (devices->selected_count == 0) {
        fprintf(stderr, "[GPU] No devices selected. Falling back to CPU.\n");
        return QALLOW_ERR_GPU_NOT_AVAILABLE;
    }

    fprintf(stderr, "[GPU] Distributing %d work units across %d device(s) ",
            unit_count, devices->selected_count);

    switch (strategy) {
        case QALLOW_GPU_DISTRIBUTE_ROUND_ROBIN:
            fprintf(stderr, "(round-robin)\n");
            for (int i = 0; i < unit_count; ++i) {
                int dev_idx = i % devices->selected_count;
                work_units[i].target_device_id = devices->selected_devices[dev_idx];
            }
            break;

        case QALLOW_GPU_DISTRIBUTE_LOAD_BALANCED:
            fprintf(stderr, "(load-balanced)\n");
            /* TODO: Implement load-balanced distribution based on available memory */
            for (int i = 0; i < unit_count; ++i) {
                work_units[i].target_device_id = devices->selected_devices[0];
            }
            break;

        case QALLOW_GPU_DISTRIBUTE_PERFORMANCE:
            fprintf(stderr, "(performance-based)\n");
            /* TODO: Implement performance-based distribution using device speed */
            for (int i = 0; i < unit_count; ++i) {
                work_units[i].target_device_id = devices->selected_devices[0];
            }
            break;

        case QALLOW_GPU_DISTRIBUTE_AFFINITY:
            fprintf(stderr, "(affinity-aware)\n");
            /* TODO: Implement CPU affinity-aware distribution */
            for (int i = 0; i < unit_count; ++i) {
                work_units[i].target_device_id = devices->selected_devices[0];
            }
            break;

        default:
            return QALLOW_ERR_INVALID_PARAMETER;
    }

    return QALLOW_SUCCESS;
}

int qallow_gpu_get_device_affinity(int device_id,
                                    uint8_t* cpu_mask,
                                    size_t mask_size) {
    if (!cpu_mask || mask_size == 0) {
        return QALLOW_ERR_INVALID_PARAMETER;
    }

    /* TODO: Implement device affinity detection
     * 1. Use NUMA library to detect GPU-CPU affinity
     * 2. Map GPU device to closest CPU sockets
     * 3. Return CPU core mask for affinity binding
     */

    memset(cpu_mask, 0, mask_size);
    return QALLOW_SUCCESS;
}

int qallow_gpu_set_device_affinity(int device_id, const char* cpu_cores) {
    if (!cpu_cores) {
        return QALLOW_ERR_INVALID_PARAMETER;
    }

    /* TODO: Implement device affinity binding
     * 1. Parse cpu_cores string (e.g., "0-3,8-11")
     * 2. Bind GPU workloads to specified CPU cores
     * 3. Use NUMA library for memory locality
     */

    fprintf(stderr, "[GPU] Setting device %d affinity to cores: %s\n",
            device_id, cpu_cores);
    return QALLOW_SUCCESS;
}

float qallow_gpu_get_load(int device_id) {
#ifdef QALLOW_ENABLE_CUDA
    cudaSetDevice(device_id);

    /* TODO: Implement GPU load measurement
     * Currently returns placeholder
     *
     * Options:
     * 1. Use nvidia-smi output parsing
     * 2. Use CUPTI (CUDA Profiling Tools Interface)
     * 3. Use internal CUDA metrics
     */

    return 0.0f;  /* Placeholder */
#else
    return -1.0f;
#endif
}

uint64_t qallow_gpu_get_available_memory_mb(int device_id) {
#ifdef QALLOW_ENABLE_CUDA
    cudaSetDevice(device_id);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem / (1024 * 1024);
#else
    return 0;
#endif
}

void qallow_gpu_free_devices(qallow_gpu_devices_t* devices) {
    if (!devices) return;

    if (devices->devices) {
        free(devices->devices);
        devices->devices = NULL;
    }

    if (devices->selected_devices) {
        free(devices->selected_devices);
        devices->selected_devices = NULL;
    }

    devices->device_count = 0;
    devices->selected_count = 0;
}

void qallow_gpu_print_status(const qallow_gpu_devices_t* devices) {
    if (!devices) {
        printf("[GPU] No devices available\n");
        return;
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║            GPU Status Report                              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("Total GPUs: %d\n", devices->device_count);
    printf("Active GPUs: %d\n", devices->selected_count);
    printf("\n");

    for (int i = 0; i < devices->device_count; ++i) {
        const qallow_gpu_device_info_t* info = &devices->devices[i];
        printf("GPU %d: %s\n", i, info->device_name);
        printf("  Compute Capability: %d.%d\n",
               info->compute_capability_major, info->compute_capability_minor);
        printf("  Memory: %llu MB total, %llu MB available\n",
               info->memory_total_mb, info->memory_available_mb);
        printf("  Multiprocessors: %d\n", info->multiprocessor_count);
        printf("  Clock Rate: %.1f MHz\n", info->clock_rate_mhz);
        printf("  Max Threads/Block: %d\n", info->max_threads_per_block);
        printf("\n");
    }
}
