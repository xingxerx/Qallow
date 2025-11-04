#ifndef QALLOW_MULTI_GPU_H
#define QALLOW_MULTI_GPU_H

/**
 * @file multi_gpu.h
 * @brief Multi-GPU device management and work distribution
 * 
 * Provides API for:
 * - GPU device enumeration and selection
 * - Device affinity management
 * - Work load distribution across GPUs
 * - GPU-to-GPU communication
 * 
 * Usage:
 *   qallow_gpu_devices_t devices = qallow_gpu_enumerate_devices();
 *   qallow_gpu_select_device(&devices, 0);
 *   qallow_gpu_distribute_work(&devices, workload);
 */

#include <stddef.h>
#include <stdint.h>

/* GPU device info structure */
typedef struct {
    int device_id;
    char device_name[256];
    uint64_t memory_total_mb;
    uint64_t memory_available_mb;
    int compute_capability_major;
    int compute_capability_minor;
    int max_threads_per_block;
    int multiprocessor_count;
    float clock_rate_mhz;
} qallow_gpu_device_info_t;

/* GPU devices collection */
typedef struct {
    qallow_gpu_device_info_t* devices;
    int device_count;
    int* selected_devices;  /* subset for active use */
    int selected_count;
} qallow_gpu_devices_t;

/* Work distribution strategy */
typedef enum {
    QALLOW_GPU_DISTRIBUTE_ROUND_ROBIN,    /* Cycle through devices */
    QALLOW_GPU_DISTRIBUTE_LOAD_BALANCED,  /* Balance by available memory */
    QALLOW_GPU_DISTRIBUTE_PERFORMANCE,    /* Use fastest device first */
    QALLOW_GPU_DISTRIBUTE_AFFINITY        /* Respect CPU affinity */
} qallow_gpu_distribution_strategy_t;

/* Work unit for distribution */
typedef struct {
    int target_device_id;
    void* data;
    size_t data_size;
    int (*work_func)(void*);  /* Function to execute on device */
} qallow_gpu_work_unit_t;

/**
 * Enumerate available GPU devices
 * 
 * @return GPU devices structure (must be freed with qallow_gpu_free_devices)
 * 
 * Example:
 *   qallow_gpu_devices_t devices = qallow_gpu_enumerate_devices();
 *   printf("Found %d GPUs\n", devices.device_count);
 *   for (int i = 0; i < devices.device_count; i++) {
 *       printf("GPU %d: %s\n", i, devices.devices[i].device_name);
 *   }
 */
qallow_gpu_devices_t qallow_gpu_enumerate_devices(void);

/**
 * Get detailed info about a specific device
 * 
 * @param devices - GPU devices collection
 * @param device_id - device index
 * @return Device information
 */
qallow_gpu_device_info_t qallow_gpu_get_device_info(
    const qallow_gpu_devices_t* devices,
    int device_id);

/**
 * Select specific device for active use
 * 
 * @param devices - GPU devices collection
 * @param device_id - device to select
 * @return 0 on success, -1 on error
 * 
 * Note: On failure, falls back to CPU execution
 */
int qallow_gpu_select_device(qallow_gpu_devices_t* devices, int device_id);

/**
 * Select multiple devices for parallel execution
 * 
 * @param devices - GPU devices collection
 * @param device_ids - array of device IDs to select
 * @param count - number of devices to select
 * @return 0 on success, -1 on error
 * 
 * Example:
 *   int device_ids[] = {0, 1, 2};
 *   qallow_gpu_select_devices(&devices, device_ids, 3);
 */
int qallow_gpu_select_devices(qallow_gpu_devices_t* devices,
                              const int* device_ids,
                              int count);

/**
 * Distribute work across selected devices
 * 
 * @param devices - GPU devices collection
 * @param work_units - array of work units
 * @param unit_count - number of work units
 * @param strategy - distribution strategy
 * @return 0 on success, -1 on error
 */
int qallow_gpu_distribute_work(qallow_gpu_devices_t* devices,
                               qallow_gpu_work_unit_t* work_units,
                               int unit_count,
                               qallow_gpu_distribution_strategy_t strategy);

/**
 * Get current device affinity (CPU cores assigned to GPU)
 * 
 * @param device_id - GPU device ID
 * @param cpu_mask - output: bitmask of assigned CPU cores
 * @param mask_size - size of cpu_mask in bytes
 * @return 0 on success, -1 on error
 */
int qallow_gpu_get_device_affinity(int device_id,
                                    uint8_t* cpu_mask,
                                    size_t mask_size);

/**
 * Set device affinity (assign GPU to CPU cores)
 * 
 * @param device_id - GPU device ID
 * @param cpu_cores - comma-separated list (e.g., "0-3,8-11")
 * @return 0 on success, -1 on error
 */
int qallow_gpu_set_device_affinity(int device_id, const char* cpu_cores);

/**
 * Get GPU load average (0.0 - 1.0)
 * 
 * @param device_id - GPU device ID
 * @return Load average, or -1.0 on error
 */
float qallow_gpu_get_load(int device_id);

/**
 * Get available memory on device
 * 
 * @param device_id - GPU device ID
 * @return Available memory in MB, or 0 on error
 */
uint64_t qallow_gpu_get_available_memory_mb(int device_id);

/**
 * Free GPU devices structure
 * 
 * @param devices - GPU devices collection to free
 */
void qallow_gpu_free_devices(qallow_gpu_devices_t* devices);

/**
 * Print GPU status report
 * 
 * @param devices - GPU devices collection
 */
void qallow_gpu_print_status(const qallow_gpu_devices_t* devices);

#endif  /* QALLOW_MULTI_GPU_H */
