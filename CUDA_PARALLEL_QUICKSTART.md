# CUDA Parallel Processing - Quick Start Guide

## Build with CUDA

```bash
cd /root/Qallow
mkdir -p build
cd build
cmake .. -DQALLOW_ENABLE_CUDA=ON
cmake --build . --parallel
```

## Run Tests

```bash
cd /root/Qallow/build
ctest --output-on-failure
```

## Basic Usage

### Initialize CUDA
```c
#include "cuda_parallel.h"

int main() {
    // Initialize
    if (cuda_parallel_init() != 0) {
        printf("CUDA init failed\n");
        return 1;
    }
    
    // Get device count
    int device_count = cuda_parallel_get_device_count();
    printf("GPUs available: %d\n", device_count);
    
    // Cleanup
    cuda_parallel_shutdown();
    return 0;
}
```

### Allocate and Copy Memory

```c
// Allocate GPU memory
size_t size = 1024 * 1024; // 1 MB
float* gpu_data = (float*)cuda_parallel_malloc(size);

// Copy from host to GPU
float* host_data = (float*)malloc(size);
cuda_parallel_memcpy_h2d(gpu_data, host_data, size);

// Copy from GPU to host
cuda_parallel_memcpy_d2h(host_data, gpu_data, size);

// Free GPU memory
cuda_parallel_free(gpu_data);
free(host_data);
```

### Use Streams for Async Operations

```c
// Create stream
cuda_stream_t stream = cuda_parallel_stream_create();

// Submit work to stream
// ... kernel launches ...

// Wait for completion
cuda_parallel_stream_synchronize(stream);

// Cleanup
cuda_parallel_stream_destroy(stream);
```

### Multi-GPU Processing

```c
int device_count = cuda_parallel_get_device_count();

for (int i = 0; i < device_count; i++) {
    // Set device
    cuda_parallel_set_device(i);
    
    // Get device info
    char name[256];
    int compute_cap;
    uint64_t total_mem;
    cuda_parallel_get_device_info(i, name, &compute_cap, &total_mem);
    
    printf("GPU %d: %s (Compute: %d.%d, Memory: %llu MB)\n",
           i, name, compute_cap/10, compute_cap%10, 
           total_mem/(1024*1024));
    
    // Process on this GPU
    // ...
}
```

### Check GPU Memory

```c
uint64_t free_mem, total_mem;
cuda_parallel_get_memory_info(&free_mem, &total_mem);

printf("GPU Memory: %llu MB free / %llu MB total\n",
       free_mem/(1024*1024), total_mem/(1024*1024));
```

### Error Handling

```c
int result = cuda_parallel_set_device(invalid_id);
if (result != 0) {
    const char* error = cuda_parallel_get_last_error();
    printf("Error: %s\n", error);
}
```

## API Reference

### Device Management
- `cuda_parallel_init()` - Initialize CUDA
- `cuda_parallel_shutdown()` - Cleanup CUDA
- `cuda_parallel_get_device_count()` - Get GPU count
- `cuda_parallel_get_device_info()` - Get GPU properties
- `cuda_parallel_set_device()` - Select GPU
- `cuda_parallel_get_device()` - Get current GPU

### Memory Operations
- `cuda_parallel_malloc()` - Allocate GPU memory
- `cuda_parallel_free()` - Free GPU memory
- `cuda_parallel_memcpy_h2d()` - Host → Device
- `cuda_parallel_memcpy_d2h()` - Device → Host
- `cuda_parallel_memcpy_d2d()` - Device → Device
- `cuda_parallel_get_memory_info()` - Query memory

### Streams
- `cuda_parallel_stream_create()` - Create stream
- `cuda_parallel_stream_destroy()` - Destroy stream
- `cuda_parallel_stream_synchronize()` - Wait for stream
- `cuda_parallel_stream_query()` - Check stream status

### Batch Processing
- `cuda_parallel_batch_submit()` - Submit tasks
- `cuda_parallel_batch_wait_all()` - Wait for all

### Utilities
- `cuda_parallel_get_optimal_block_size()` - Block size
- `cuda_parallel_calculate_grid_block()` - Grid/block dims
- `cuda_parallel_get_last_error()` - Get error message

## Environment Variables

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### CUDA Not Found
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Ensure CUDA is in PATH
export PATH=/opt/cuda/bin:$PATH
```

### Memory Allocation Failed
```c
// Check available memory
uint64_t free_mem, total_mem;
cuda_parallel_get_memory_info(&free_mem, &total_mem);

// Reduce allocation size if needed
size_t safe_size = free_mem / 2; // Use half of available
```

### Segmentation Fault
- Ensure CUDA is initialized before use
- Check for NULL pointers
- Verify GPU is available
- Check error messages with `cuda_parallel_get_last_error()`

## Performance Tips

1. **Batch Operations**: Group small transfers
2. **Use Streams**: Overlap computation and transfer
3. **Memory Pooling**: Reuse allocations
4. **Device Selection**: Distribute work across GPUs
5. **Minimize Sync**: Reduce synchronization points

## Files

- `runtime/cuda_parallel.h` - Header file
- `runtime/cuda_parallel.cu` - Implementation
- `tests/unit/test_cuda_parallel.cu` - Tests
- `CUDA_PARALLEL_PROCESSING_GUIDE.md` - Full guide
- `CUDA_PARALLEL_IMPLEMENTATION_SUMMARY.md` - Summary

## Support

For detailed information, see:
- `CUDA_PARALLEL_PROCESSING_GUIDE.md` - Complete documentation
- `CUDA_PARALLEL_IMPLEMENTATION_SUMMARY.md` - Implementation details
- NVIDIA CUDA Documentation: https://docs.nvidia.com/cuda/

## Status

✅ **CUDA Parallel Processing Ready**
- All tests passing
- Multi-GPU support enabled
- Stream-based execution functional
- Memory management optimized
- Error handling robust

