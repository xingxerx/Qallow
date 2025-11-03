"""
Real CUDA GPU Processor - Direct GPU kernel execution with PyCUDA
Uses actual CUDA 12.6 installed on system.
"""


class RealCudaProcessor:
    """Direct interface to NVIDIA GPU using PyCUDA."""
    
    def __init__(self):
        """Initialize CUDA context and GPU."""
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.properties = self.device.get_attributes()
        self._init_kernels()
        
    def _init_kernels(self):
        """Compile and initialize CUDA kernels."""
        # Simple vector addition kernel
        self.kernel_code = """
        __global__ void vector_add(float *a, float *b, float *c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        
        __global__ void matrix_multiply(float *A, float *B, float *C,
                                       int rows_A, int cols_A, int cols_B) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (row < rows_A && col < cols_B) {
                float sum = 0.0f;
                for (int k = 0; k < cols_A; k++) {
                    sum += A[row * cols_A + k] * B[k * cols_B + col];
                }
                C[row * cols_B + col] = sum;
            }
        }
        
        __global__ void parallel_reduce(float *data, float *result, int n) {
            extern __shared__ float shared[];
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Load data into shared memory
            if (idx < n) {
                shared[threadIdx.x] = data[idx];
            } else {
                shared[threadIdx.x] = 0.0f;
            }
            __syncthreads();
            
            // Parallel reduction in shared memory
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    shared[threadIdx.x] += shared[threadIdx.x + stride];
                }
                __syncthreads();
            }
            
            // Write result
            if (threadIdx.x == 0) {
                result[blockIdx.x] = shared[0];
            }
        }
        """
        
        # Compile kernels
        from pycuda.compiler import SourceModule
        self.module = SourceModule(self.kernel_code)
        self.kernel_add = self.module.get_function("vector_add")
        self.kernel_matmul = self.module.get_function("matrix_multiply")
        self.kernel_reduce = self.module.get_function("parallel_reduce")
        
    def get_gpu_info(self) -> Dict:
        """Get GPU device information."""
        self.ctx.push()
        try:
            driver_version, runtime_version = cuda.get_version()
            free_mem, total_mem = cuda.mem_get_info()
            
            return {
                "device_name": self.device.name().decode('utf-8'),
                "compute_capability": self.device.compute_capability(),
                "max_threads_per_block": int(self.properties[cuda.device_attribute.MAX_THREADS_PER_BLOCK]),
                "total_memory_gb": total_mem / (1024**3),
                "free_memory_gb": free_mem / (1024**3),
                "cuda_driver_version": f"{driver_version // 1000}.{(driver_version % 1000) // 10}",
                "cuda_runtime_version": f"{runtime_version // 1000}.{(runtime_version % 1000) // 10}",
                "multiprocessor_count": int(self.properties[cuda.device_attribute.MULTIPROCESSOR_COUNT]),
            }
        finally:
            self.ctx.pop()
            
    def vector_add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Add two vectors on GPU."""
        self.ctx.push()
        try:
            n = len(a)
            a = a.astype(np.float32)
            b = b.astype(np.float32)
            
            a_gpu = cuda.mem_alloc(a.nbytes)
            b_gpu = cuda.mem_alloc(b.nbytes)
            c_gpu = cuda.mem_alloc(a.nbytes)
            
            cuda.memcpy_htod(a_gpu, a)
            cuda.memcpy_htod(b_gpu, b)
            
            threads_per_block = 256
            blocks = (n + threads_per_block - 1) // threads_per_block
            
            self.kernel_add.prepared_call(
                (blocks, 1), (threads_per_block, 1, 1),
                a_gpu, b_gpu, c_gpu, np.int32(n)
            )
            
            result = np.empty_like(a)
            cuda.memcpy_dtoh(result, c_gpu)
            
            a_gpu.free()
            b_gpu.free()
            c_gpu.free()
            
            return result
        finally:
            self.ctx.pop()
            
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices on GPU."""
        self.ctx.push()
        try:
            A = A.astype(np.float32)
            B = B.astype(np.float32)
            
            rows_A, cols_A = A.shape
            rows_B, cols_B = B.shape
            
            if cols_A != rows_B:
                raise ValueError(f"Incompatible dimensions: {A.shape} x {B.shape}")
            
            C = np.zeros((rows_A, cols_B), dtype=np.float32)
            
            A_gpu = cuda.mem_alloc(A.nbytes)
            B_gpu = cuda.mem_alloc(B.nbytes)
            C_gpu = cuda.mem_alloc(C.nbytes)
            
            cuda.memcpy_htod(A_gpu, A)
            cuda.memcpy_htod(B_gpu, B)
            
            block_x, block_y = 16, 16
            grid_x = (cols_B + block_x - 1) // block_x
            grid_y = (rows_A + block_y - 1) // block_y
            
            self.kernel_matmul.prepared_call(
                (grid_x, grid_y), (block_x, block_y, 1),
                A_gpu, B_gpu, C_gpu,
                np.int32(rows_A), np.int32(cols_A), np.int32(cols_B)
            )
            
            cuda.memcpy_dtoh(C, C_gpu)
            
            A_gpu.free()
            B_gpu.free()
            C_gpu.free()
            
            return C
        finally:
            self.ctx.pop()
            
    def parallel_sum(self, data: np.ndarray) -> float:
        """Sum array elements using parallel reduction on GPU."""
        self.ctx.push()
        try:
            data = data.astype(np.float32)
            n = len(data)
            
            threads_per_block = 256
            blocks = (n + threads_per_block - 1) // threads_per_block
            
            data_gpu = cuda.mem_alloc(data.nbytes)
            result_gpu = cuda.mem_alloc(blocks * 4)
            
            cuda.memcpy_htod(data_gpu, data)
            
            shared_size = threads_per_block * 4
            self.kernel_reduce.prepared_call(
                (blocks, 1), (threads_per_block, 1, 1),
                data_gpu, result_gpu, np.int32(n),
                shared_size=shared_size
            )
            
            partial_results = np.empty(blocks, dtype=np.float32)
            cuda.memcpy_dtoh(partial_results, result_gpu)
            
            data_gpu.free()
            result_gpu.free()
            
            return float(np.sum(partial_results))
        finally:
            self.ctx.pop()
            
    def benchmark_operation(self, operation: str, size: int = 1000000) -> Dict:
        """Benchmark a GPU operation."""
        self.ctx.push()
        try:
            if operation == "vector_add":
                a = np.random.randn(size).astype(np.float32)
                b = np.random.randn(size).astype(np.float32)
                
                start = cuda.Event()
                end = cuda.Event()
                start.record()
                
                self.vector_add(a, b)
                
                end.record()
                end.synchronize()
                
                elapsed_ms = start.time_till(end) * 1e-3
                throughput_gbs = (a.nbytes * 3) / elapsed_ms / 1e9
                
                return {
                    "operation": operation,
                    "size": size,
                    "time_ms": elapsed_ms,
                    "throughput_gb_s": throughput_gbs,
                }
                
            elif operation == "matmul":
                # NxN matrix
                n = int(np.sqrt(size))
                A = np.random.randn(n, n).astype(np.float32)
                B = np.random.randn(n, n).astype(np.float32)
                
                start = cuda.Event()
                end = cuda.Event()
                start.record()
                
                self.matrix_multiply(A, B)
                
                end.record()
                end.synchronize()
                
                elapsed_ms = start.time_till(end) * 1e-3
                flops = 2 * n * n * n / elapsed_ms / 1e9
                
                return {
                    "operation": operation,
                    "matrix_size": (n, n),
                    "time_ms": elapsed_ms,
                    "gflops": flops,
                }
                
            elif operation == "reduction":
                data = np.random.randn(size).astype(np.float32)
                
                start = cuda.Event()
                end = cuda.Event()
                start.record()
                
                self.parallel_sum(data)
                
                end.record()
                end.synchronize()
                
                elapsed_ms = start.time_till(end) * 1e-3
                throughput_gbs = (data.nbytes * 2) / elapsed_ms / 1e9
                
                return {
                    "operation": operation,
                    "size": size,
                    "time_ms": elapsed_ms,
                    "throughput_gb_s": throughput_gbs,
                }
        finally:
            self.ctx.pop()
            
    def cleanup(self):
        """Free GPU context."""
        self.ctx.pop()


if __name__ == "__main__":
    processor = RealCudaProcessor()
    print("GPU Info:", processor.get_gpu_info())
    
    # Test operations
    a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    result = processor.vector_add(a, b)
    print("Vector add result:", result)
    
    # Benchmark
    print("\nBenchmarks:")
    print("Vector add (1M):", processor.benchmark_operation("vector_add", 1000000))
    print("Matrix mul (1000x1000):", processor.benchmark_operation("matmul", 1000000))
    print("Reduction (1M):", processor.benchmark_operation("reduction", 1000000))
    
    processor.cleanup()
