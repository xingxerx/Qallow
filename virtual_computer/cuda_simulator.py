#!/usr/bin/env python3
"""
Virtual CUDA GPU Simulator
Simulates CUDA kernel execution, device memory, and GPU operations for agent optimization
"""

import random
import time


class KernelStatus(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GPUMemoryRegion:
    """Represents a memory allocation on GPU device"""
    address: int
    size: int
    data_type: str
    in_use: bool = True
    allocated_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def __hash__(self):
        return hash(self.address)


@dataclass
class CUDAKernel:
    """Represents a CUDA kernel execution"""
    kernel_id: int
    name: str
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]
    shared_memory: int
    status: KernelStatus = KernelStatus.QUEUED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    compute_time: float = 0.0
    memory_ops: int = 0
    registers_per_thread: int = 0
    
    def get_thread_count(self) -> int:
        return self.block_size[0] * self.block_size[1] * self.block_size[2]
    
    def get_block_count(self) -> int:
        return self.grid_size[0] * self.grid_size[1] * self.grid_size[2]


class VirtualGPU:
    """Simulates a NVIDIA GPU with device memory and kernel execution"""
    
    def __init__(self, device_id: int = 0, device_memory_mb: int = 8192):
        self.device_id = device_id
        self.device_memory = device_memory_mb * 1024 * 1024  # Convert to bytes
        self.free_memory = self.device_memory
        self.allocations: Dict[int, GPUMemoryRegion] = {}
        self.memory_counter = 0x10000000  # Start address
        
        # Kernel execution
        self.kernels: List[CUDAKernel] = []
        self.kernel_counter = 0
        self.kernel_queue: List[CUDAKernel] = []
        self.current_kernel: Optional[CUDAKernel] = None
        
        # Statistics
        self.total_launches = 0
        self.total_memory_alloc = 0
        self.total_memory_freed = 0
        self.total_compute_time = 0.0
        self.peak_memory_usage = 0
        self.kernel_execution_times: List[float] = []
        
        # Performance metrics
        self.bandwidth_gbps = 288  # GB/s for typical GPU
        self.compute_capability = "8.0"  # Ampere architecture
        self.sm_count = 80
        self.cores_per_sm = 128
        
    def malloc(self, size: int, data_type: str = "float32") -> Optional[int]:
        """Allocate memory on GPU device"""
        if size > self.free_memory:
            return None  # Allocation failed
        
        addr = self.memory_counter
        region = GPUMemoryRegion(
            address=addr,
            size=size,
            data_type=data_type
        )
        
        self.allocations[addr] = region
        self.free_memory -= size
        self.total_memory_alloc += size
        self.memory_counter += size
        
        # Track peak memory
        current_used = self.device_memory - self.free_memory
        if current_used > self.peak_memory_usage:
            self.peak_memory_usage = current_used
        
        return addr
    
    def free(self, address: int) -> bool:
        """Free GPU memory allocation"""
        if address not in self.allocations:
            return False
        
        region = self.allocations[address]
        self.free_memory += region.size
        self.total_memory_freed += region.size
        del self.allocations[address]
        return True
    
    def memcpy_to_device(self, address: int, size: int, host_data_size: int) -> Tuple[bool, float]:
        """Copy data from host to GPU device"""
        if address not in self.allocations:
            return False, 0.0
        
        region = self.allocations[address]
        if size > region.size:
            return False, 0.0
        
        # Simulate PCIe transfer time (PCIe 4.0 = ~32 GB/s)
        transfer_time = (size / (32 * 1024 * 1024 * 1024)) * 1000  # milliseconds
        
        region.access_count += 1
        return True, transfer_time
    
    def memcpy_to_host(self, address: int, size: int) -> Tuple[bool, float]:
        """Copy data from GPU device to host"""
        if address not in self.allocations:
            return False, 0.0
        
        region = self.allocations[address]
        if size > region.size:
            return False, 0.0
        
        # Simulate PCIe transfer time
        transfer_time = (size / (32 * 1024 * 1024 * 1024)) * 1000  # milliseconds
        
        region.access_count += 1
        return True, transfer_time
    
    def launch_kernel(self, kernel_name: str, grid_size: Tuple[int, int, int],
                     block_size: Tuple[int, int, int], shared_memory: int = 0) -> int:
        """Queue a CUDA kernel for execution"""
        self.kernel_counter += 1
        
        kernel = CUDAKernel(
            kernel_id=self.kernel_counter,
            name=kernel_name,
            grid_size=grid_size,
            block_size=block_size,
            shared_memory=shared_memory
        )
        
        self.kernel_queue.append(kernel)
        self.total_launches += 1
        
        return kernel.kernel_id
    
    def execute_kernel(self, kernel_id: int, compute_ops: int) -> Tuple[bool, float]:
        """Execute a queued kernel and return execution time"""
        # Find kernel in queue
        kernel = None
        for k in self.kernel_queue:
            if k.kernel_id == kernel_id:
                kernel = k
                break
        
        if not kernel:
            return False, 0.0
        
        # Simulate kernel execution
        kernel.status = KernelStatus.RUNNING
        kernel.start_time = time.time()
        
        # Calculate execution time based on:
        # 1. Number of threads
        # 2. Compute operations
        # 3. GPU compute capability
        threads = kernel.get_thread_count() * kernel.get_block_count()
        
        # Simulate GPU computation (TFLOPS for FP32)
        gpu_tflops = 20.0  # ~20 TFLOPS for typical GPU
        theoretical_time = (compute_ops / (gpu_tflops * 1e12)) * 1000  # ms
        
        # Add memory access penalty
        memory_penalty = (kernel.memory_ops / (self.bandwidth_gbps * 1e9)) * 1000  # ms
        
        # Simulate execution with some variance
        variance = random.uniform(0.95, 1.05)
        execution_time = (theoretical_time + memory_penalty) * variance
        
        # Simulate execution delay
        time.sleep(execution_time / 1000)
        
        kernel.end_time = time.time()
        kernel.compute_time = execution_time
        kernel.status = KernelStatus.COMPLETED
        
        self.kernel_execution_times.append(execution_time)
        self.total_compute_time += execution_time
        
        # Move to completed kernels
        self.kernel_queue.remove(kernel)
        self.kernels.append(kernel)
        
        return True, execution_time
    
    def get_device_properties(self) -> Dict:
        """Get GPU device properties"""
        return {
            "device_id": self.device_id,
            "device_memory_mb": self.device_memory // (1024 * 1024),
            "free_memory_mb": self.free_memory // (1024 * 1024),
            "peak_memory_usage_mb": self.peak_memory_usage // (1024 * 1024),
            "compute_capability": self.compute_capability,
            "sm_count": self.sm_count,
            "cores_per_sm": self.cores_per_sm,
            "total_cores": self.sm_count * self.cores_per_sm,
            "bandwidth_gbps": self.bandwidth_gbps,
            "total_launches": self.total_launches,
            "kernels_completed": len(self.kernels),
            "total_compute_time_ms": self.total_compute_time,
        }
    
    def get_memory_stats(self) -> Dict:
        """Get memory allocation statistics"""
        return {
            "total_memory_mb": self.device_memory // (1024 * 1024),
            "free_memory_mb": self.free_memory // (1024 * 1024),
            "used_memory_mb": (self.device_memory - self.free_memory) // (1024 * 1024),
            "allocations_count": len(self.allocations),
            "total_allocated_mb": self.total_memory_alloc // (1024 * 1024),
            "total_freed_mb": self.total_memory_freed // (1024 * 1024),
            "peak_memory_usage_mb": self.peak_memory_usage // (1024 * 1024),
        }
    
    def get_kernel_stats(self) -> Dict:
        """Get kernel execution statistics"""
        if not self.kernel_execution_times:
            return {
                "total_launches": 0,
                "avg_execution_time_ms": 0.0,
                "min_execution_time_ms": 0.0,
                "max_execution_time_ms": 0.0,
            }
        
        return {
            "total_launches": self.total_launches,
            "completed_kernels": len(self.kernels),
            "avg_execution_time_ms": sum(self.kernel_execution_times) / len(self.kernel_execution_times),
            "min_execution_time_ms": min(self.kernel_execution_times),
            "max_execution_time_ms": max(self.kernel_execution_times),
            "total_compute_time_ms": self.total_compute_time,
        }
    
    def print_status(self):
        """Print GPU status summary"""
        props = self.get_device_properties()
        mem = self.get_memory_stats()
        kern = self.get_kernel_stats()
        
        print(f"\n{'='*70}")
        print(f"  GPU Device {props['device_id']} Status")
        print(f"{'='*70}")
        print(f"  Compute Capability: {props['compute_capability']}")
        print(f"  SMs: {props['sm_count']}, Cores/SM: {props['cores_per_sm']}, Total Cores: {props['total_cores']}")
        print(f"  Memory: {mem['used_memory_mb']}/{mem['total_memory_mb']} MB (Peak: {mem['peak_memory_usage_mb']} MB)")
        print(f"  Kernels Launched: {kern['total_launches']}, Completed: {kern['completed_kernels']}")
        print(f"  Avg Kernel Time: {kern['avg_execution_time_ms']:.2f} ms")
        print(f"{'='*70}\n")
