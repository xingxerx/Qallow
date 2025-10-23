# CUDA Parallel Processing Guide - Qallow

## Overview

The Qallow project now includes comprehensive CUDA support for parallel processing with:

- **Multi-GPU Support**: Manage multiple NVIDIA GPUs simultaneously
- **Stream-based Execution**: Asynchronous kernel execution with CUDA streams
- **Optimized Memory Management**: Efficient GPU memory allocation and transfers
- **Batch Processing**: Submit multiple tasks for parallel execution
- **Performance Monitoring**: Track GPU utilization and memory usage

## Quick Start

### Build with CUDA Support

```bash
cd /root/Qallow
mkdir -p build
cd build
cmake .. -DQALLOW_ENABLE_CUDA=ON
cmake --build . --parallel
```

### Run CUDA Tests

```bash
cd /root/Qallow/build
ctest --output-on-failure -V
```

## API Reference

### GPU Device Management

#### Initialize CUDA System
```c
int cuda_parallel_init(void);
```
Initializes the CUDA parallel processing system. Must be called before using other functions.

**Returns**: 0 on success, negative on failure

#### Get Device Count
```c
int cuda_parallel_get_device_count(void);
```
Returns the number of available CUDA devices.

#### Get Device Information
```c
int cuda_parallel_get_device_info(int device_id, char* name, 
                                   int* compute_capability, 
                                   uint64_t* total_memory);
```
Retrieves information about a specific GPU device.

**Parameters**:
- `device_id`: GPU device index (0-based)
- `name`: Output buffer for device name (min 256 bytes)
- `compute_capability`: Output for compute capability (e.g., 89 for sm_89)
- `total_memory`: Output for total GPU memory in bytes

**Returns**: 0 on success, negative on failure

#### Set Active Device
```c
int cuda_parallel_set_device(int device_id);
```
Sets the active GPU device for subsequent operations.

### Memory Management

#### Allocate GPU Memory
```c
void* cuda_parallel_malloc(size_t size);
```
Allocates memory on the GPU.

**Returns**: Pointer to GPU memory, NULL on failure

#### Copy Data Host to Device
```c
int cuda_parallel_memcpy_h2d(void* dst, const void* src, size_t size);
```
Copies data from host (CPU) memory to GPU memory.

#### Copy Data Device to Host
```c
int cuda_parallel_memcpy_d2h(void* dst, const void* src, size_t size);
```
Copies data from GPU memory to host (CPU) memory.

#### Copy Data Device to Device
```c
int cuda_parallel_memcpy_d2d(void* dst, const void* src, size_t size);
```
Copies data between GPU memories (useful for multi-GPU scenarios).

#### Get Memory Information
```c
int cuda_parallel_get_memory_info(uint64_t* free_memory, 
                                   uint64_t* total_memory);
```
Retrieves current GPU memory usage.

### Stream-based Parallel Execution

#### Create Stream
```c
cuda_stream_t cuda_parallel_stream_create(void);
```
Creates a new CUDA stream for asynchronous operations.

**Returns**: Stream handle, NULL on failure

#### Synchronize Stream
```c
int cuda_parallel_stream_synchronize(cuda_stream_t stream);
```
Waits for all operations in the stream to complete.

#### Query Stream Status
```c
int cuda_parallel_stream_query(cuda_stream_t stream);
```
Checks if stream operations are complete without blocking.

**Returns**: 1 if complete, 0 if still executing, negative on error

### Batch Processing

#### Submit Batch Tasks
```c
int cuda_parallel_batch_submit(const cuda_batch_task_t* tasks, 
                                int task_count);
```
Submits multiple tasks for parallel processing.

**Task Structure**:
```c
typedef struct {
    void* input_data;      // Input data pointer
    size_t input_size;     // Size of input data
    void* output_data;     // Output data pointer
    size_t output_size;    // Size of output data
    int device_id;         // Target GPU device
    cuda_stream_t stream;  // CUDA stream for async execution
} cuda_batch_task_t;
```

#### Wait for All Batch Tasks
```c
int cuda_parallel_batch_wait_all(void);
```
Waits for all submitted batch tasks to complete.

## Usage Examples

### Example 1: Basic GPU Memory Operations

```c
#include "cuda_parallel.h"

int main() {
    // Initialize CUDA
    if (cuda_parallel_init() != 0) {
        printf("CUDA initialization failed\n");
        return 1;
    }
    
    // Get device info
    char device_name[256];
    int compute_cap;
    uint64_t total_mem;
    cuda_parallel_get_device_info(0, device_name, &compute_cap, &total_mem);
    printf("GPU: %s (Compute Capability: %d.%d, Memory: %llu MB)\n",
           device_name, compute_cap/10, compute_cap%10, total_mem/(1024*1024));
    
    // Allocate GPU memory
    size_t data_size = 1024 * 1024; // 1 MB
    float* gpu_data = (float*)cuda_parallel_malloc(data_size);
    
    // Copy data to GPU
    float* host_data = (float*)malloc(data_size);
    cuda_parallel_memcpy_h2d(gpu_data, host_data, data_size);
    
    // ... perform GPU operations ...
    
    // Copy results back
    cuda_parallel_memcpy_d2h(host_data, gpu_data, data_size);
    
    // Cleanup
    cuda_parallel_free(gpu_data);
    free(host_data);
    cuda_parallel_shutdown();
    
    return 0;
}
```

### Example 2: Multi-GPU Processing

```c
#include "cuda_parallel.h"

int main() {
    cuda_parallel_init();
    
    int device_count = cuda_parallel_get_device_count();
    printf("Available GPUs: %d\n", device_count);
    
    // Process on each GPU
    for (int i = 0; i < device_count; i++) {
        cuda_parallel_set_device(i);
        
        // Allocate and process on GPU i
        void* gpu_data = cuda_parallel_malloc(1024 * 1024);
        // ... perform operations ...
        cuda_parallel_free(gpu_data);
    }
    
    cuda_parallel_shutdown();
    return 0;
}
```

### Example 3: Asynchronous Stream Processing

```c
#include "cuda_parallel.h"

int main() {
    cuda_parallel_init();
    
    // Create multiple streams for parallel execution
    cuda_stream_t stream1 = cuda_parallel_stream_create();
    cuda_stream_t stream2 = cuda_parallel_stream_create();
    
    // Submit work to streams
    void* gpu_data1 = cuda_parallel_malloc(1024);
    void* gpu_data2 = cuda_parallel_malloc(1024);
    
    // ... submit work to streams ...
    
    // Wait for completion
    cuda_parallel_stream_synchronize(stream1);
    cuda_parallel_stream_synchronize(stream2);
    
    // Cleanup
    cuda_parallel_free(gpu_data1);
    cuda_parallel_free(gpu_data2);
    cuda_parallel_stream_destroy(stream1);
    cuda_parallel_stream_destroy(stream2);
    cuda_parallel_shutdown();
    
    return 0;
}
```

## Performance Tips

1. **Batch Operations**: Group multiple small transfers into larger ones
2. **Use Streams**: Overlap computation and data transfer with streams
3. **Memory Pooling**: Reuse GPU memory allocations when possible
4. **Device Selection**: Use `cuda_parallel_set_device()` to distribute work
5. **Synchronization**: Minimize synchronization points for better performance

## Troubleshooting

### CUDA Not Available
If CUDA is not detected, ensure:
- NVIDIA GPU drivers are installed
- CUDA Toolkit is installed
- `nvcc` compiler is in PATH

### Memory Allocation Failures
- Check available GPU memory with `cuda_parallel_get_memory_info()`
- Reduce allocation size or free unused memory
- Check for memory leaks in your code

### Performance Issues
- Use `nvidia-smi` to monitor GPU utilization
- Profile with NVIDIA Nsight Compute
- Check for excessive synchronization

## Building with CUDA

### CMake Configuration

```bash
cmake .. -DQALLOW_ENABLE_CUDA=ON
```

### Environment Variables

```bash
# Specify GPU to use
export CUDA_VISIBLE_DEVICES=0

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## Files

- `runtime/cuda_parallel.h` - Header file with API definitions
- `runtime/cuda_parallel.cu` - Implementation
- `tests/unit/test_cuda_parallel.cu` - Unit tests

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

