# Multi-GPU Support Guide

**Status:** Foundation layer implemented (ready for integration)  
**Last Updated:** November 4, 2025

---

## Overview

Multi-GPU support enables Qallow to:
- **Enumerate** available GPU devices
- **Select** specific GPUs for execution
- **Distribute** work across multiple GPUs
- **Manage** device affinity for optimal memory bandwidth
- **Scale** linearly with GPU count (target: 4-8x speedup on 4-8 GPUs)

---

## Quick Start

### Enable Multi-GPU

```bash
# Build with CUDA support
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel

# Run with all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 ./build/qallow run unified

# Or select specific GPUs
CUDA_VISIBLE_DEVICES=0,1 ./build/qallow run unified
```

### Check GPU Status

```c
#include "qallow/multi_gpu.h"

// Enumerate GPUs
qallow_gpu_devices_t devices = qallow_gpu_enumerate_devices();
qallow_gpu_print_status(&devices);

// Output:
// GPU 0: NVIDIA A100 80GB
//   Compute Capability: 8.0
//   Memory: 81920 MB total, 81792 MB available
//   Multiprocessors: 108
//   Clock Rate: 1410.0 MHz
//   Max Threads/Block: 1024
```

---

## API Reference

### Enumeration & Selection

```c
// Get all available GPUs
qallow_gpu_devices_t devices = qallow_gpu_enumerate_devices();

// Get info about specific GPU
qallow_gpu_device_info_t info = qallow_gpu_get_device_info(&devices, 0);
printf("GPU: %s\n", info.device_name);
printf("Memory: %llu MB\n", info.memory_total_mb);

// Select single GPU
qallow_gpu_select_device(&devices, 0);

// Select multiple GPUs
int gpu_ids[] = {0, 1};
qallow_gpu_select_devices(&devices, gpu_ids, 2);

// Free resources
qallow_gpu_free_devices(&devices);
```

### Work Distribution

```c
// Create work units
qallow_gpu_work_unit_t work[4];
for (int i = 0; i < 4; ++i) {
    work[i].data = ...;
    work[i].data_size = ...;
    work[i].work_func = &execute_on_gpu;
}

// Distribute with different strategies
qallow_gpu_distribute_work(&devices, work, 4, 
    QALLOW_GPU_DISTRIBUTE_ROUND_ROBIN);     // Simple round-robin
    // QALLOW_GPU_DISTRIBUTE_LOAD_BALANCED);  // Balance by memory
    // QALLOW_GPU_DISTRIBUTE_PERFORMANCE);    // Use fastest first
    // QALLOW_GPU_DISTRIBUTE_AFFINITY);       // Respect CPU affinity
```

### Device Affinity

```c
// Check GPU-CPU affinity (which CPU cores service this GPU)
uint8_t cpu_mask[8];
qallow_gpu_get_device_affinity(0, cpu_mask, sizeof(cpu_mask));

// Set affinity (assign GPU work to specific CPU cores)
qallow_gpu_set_device_affinity(0, "0-3,8-11");  // Cores 0-3 and 8-11

// Get current load
float load = qallow_gpu_get_load(0);  // 0.0 to 1.0

// Get available memory
uint64_t avail_mb = qallow_gpu_get_available_memory_mb(0);
```

---

## Implementation Details

### Current Status ✅

| Feature | Status | Notes |
|---------|--------|-------|
| Device enumeration | ✅ Complete | Using CUDA Runtime API |
| Device selection | ✅ Complete | Single and multiple selection |
| Memory queries | ✅ Complete | Total and available memory |
| Device info | ✅ Complete | Name, compute capability, specs |
| Print status | ✅ Complete | Formatted GPU report |

### In Progress 🔄

| Feature | Status | Target | Effort |
|---------|--------|--------|--------|
| Load-balanced distribution | 📋 Design | Week 2 | 4 hrs |
| Performance-based distribution | 📋 Design | Week 2 | 4 hrs |
| Device affinity detection | 📋 Design | Week 2 | 6 hrs |
| GPU load measurement | 📋 Design | Week 3 | 4 hrs |

### Future Enhancements 📅

| Feature | Priority | Target | Effort |
|---------|----------|--------|--------|
| NUMA-aware affinity | Medium | Week 3 | 8 hrs |
| GPU peer access | Low | Week 4 | 6 hrs |
| Device topology analysis | Low | Week 4 | 4 hrs |
| Multi-GPU synchronization | Low | Future | 10 hrs |

---

## Work Distribution Strategies

### 1. Round-Robin (Simple)

```
GPU 0: Units 0, 3, 6, ...
GPU 1: Units 1, 4, 7, ...
GPU 2: Units 2, 5, 8, ...
GPU 3: Units 3, 6, 9, ...
```

**Pros:** Simple, balanced
**Cons:** No load awareness
**Use Case:** Homogeneous workloads on identical GPUs

### 2. Load-Balanced (Recommended)

```
// Allocate based on available memory
GPU 0 (40GB free): 50 units
GPU 1 (20GB free): 25 units
GPU 2 (15GB free): 19 units
GPU 3 (5GB free):  6 units
```

**Pros:** Respects resource availability
**Cons:** Requires memory probing
**Use Case:** Variable-sized workloads, heterogeneous memory

### 3. Performance-Based (Aggressive)

```
// Sort by device speed
A100 (fast):       40 units
V100 (medium):     30 units
T4 (slow):         30 units
```

**Pros:** Maximizes throughput
**Cons:** May overload faster GPUs
**Use Case:** Performance-critical applications

### 4. Affinity-Aware (Enterprise)

```
// Respect CPU socket affinity
Cores 0-15 → GPU 0 (NVLink)
Cores 16-31 → GPU 1 (NVLink)
Cores 32-47 → GPU 2 (NVLink)
```

**Pros:** Optimal memory bandwidth
**Cons:** Complex setup
**Use Case:** NUMA systems, high-performance computing

---

## Environment Variables

```bash
# Select specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set per-GPU memory fraction (0.0-1.0)
export CUDA_PER_THREAD_DEFAULT_STREAM=1

# Enable GPU monitoring
export QALLOW_GPU_MONITOR=1

# GPU affinity string (when NUMA aware)
export QALLOW_GPU_AFFINITY="0=0-15,1=16-31,2=32-47"
```

---

## Performance Targets

### Scaling Efficiency

| GPU Count | Target Speedup | Efficiency |
|-----------|----------------|------------|
| 1 GPU | 1.0x | 100% |
| 2 GPUs | 1.8x | 90% |
| 4 GPUs | 3.5x | 87% |
| 8 GPUs | 6.5x | 81% |

### Phase Execution (4-GPU System)

| Phase | Single GPU | 4 GPU | Speedup |
|-------|-----------|-------|---------|
| Phase 12 | 15 ms | 4 ms | 3.75x |
| Phase 13 | 20 ms | 5 ms | 4.0x |
| Phase 14 | 25 ms | 7 ms | 3.6x |
| Phase 15 | 30 ms | 8 ms | 3.75x |

---

## Implementation Roadmap

### Phase 1: Foundation ✅ (Completed)
- Device enumeration
- Memory management
- Basic API structure
- Error handling integration

### Phase 2: Distribution (This Week)
- Load-balanced distribution
- Performance profiling
- Affinity detection
- Work unit scheduling

### Phase 3: Optimization (Next Week)
- GPU load measurement
- Dynamic load balancing
- Device preference optimization
- Memory migration strategies

### Phase 4: Advanced (Future)
- GPU peer-to-peer transfers
- Multi-device kernels
- Heterogeneous workload scheduling
- Automatic GPU selection

---

## Usage Examples

### Example 1: Use All Available GPUs

```c
#include "qallow/multi_gpu.h"

int main() {
    // Enumerate GPUs
    qallow_gpu_devices_t devices = qallow_gpu_enumerate_devices();
    
    if (devices.device_count == 0) {
        printf("No GPUs found. Using CPU.\n");
        return 0;
    }
    
    printf("Found %d GPUs\n", devices.device_count);
    qallow_gpu_print_status(&devices);
    
    // Select all for use
    qallow_gpu_select_devices(&devices, 
                             devices.selected_devices,
                             devices.device_count);
    
    // ... do work ...
    
    qallow_gpu_free_devices(&devices);
    return 0;
}
```

### Example 2: Load-Balanced Distribution

```c
// Create work units
qallow_gpu_work_unit_t work[100];
for (int i = 0; i < 100; ++i) {
    work[i].data = malloc(1024*1024);  // 1MB per unit
    work[i].data_size = 1024*1024;
    work[i].work_func = &kernel_function;
}

// Distribute load-balanced
qallow_gpu_distribute_work(&devices, work, 100,
    QALLOW_GPU_DISTRIBUTE_LOAD_BALANCED);

// Work will be assigned based on available GPU memory
```

### Example 3: Specific GPU Selection

```c
// Use only GPUs 0 and 2 (skip GPU 1)
int selected[] = {0, 2};
qallow_gpu_select_devices(&devices, selected, 2);

// Round-robin distribution
qallow_gpu_distribute_work(&devices, work, 100,
    QALLOW_GPU_DISTRIBUTE_ROUND_ROBIN);
```

---

## Troubleshooting

### "No CUDA devices found"

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Rebuild with CUDA
cmake -DQALLOW_ENABLE_CUDA=ON -B build
cmake --build build
```

### GPU Memory Errors

```bash
# Reduce memory per workload
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# Check available memory
nvidia-smi --query-gpu=memory.free --format=csv,nounits
```

### Uneven Load Distribution

```bash
# Enable load-balanced distribution
// In code:
qallow_gpu_distribute_work(&devices, work, count,
    QALLOW_GPU_DISTRIBUTE_LOAD_BALANCED);

# Or manually check:
nvidia-smi -l 1  # Monitor in real-time
```

---

## Contributing

To extend multi-GPU support:

1. **Add new distribution strategy:**
   - Add enum value in `qallow_gpu_distribution_strategy_t`
   - Implement logic in `qallow_gpu_distribute_work()`
   - Add tests in `tests/multi_gpu_test.c`

2. **Add device metric:**
   - Define query function like `qallow_gpu_get_*`
   - Use CUDA Runtime API or CUPTI
   - Update `qallow_gpu_device_info_t` if needed

3. **Optimize affinity:**
   - Use NUMA library for topology
   - Map GPU to CPU cores
   - Test on multi-socket systems

---

## References

- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [NVIDIA GPU Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Multi-GPU Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#multi-gpu)
- [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/a100/)

---

**Owner:** CUDA team  
**Priority:** Medium (4-8x scaling)  
**Contact:** See roadmap for assignee
