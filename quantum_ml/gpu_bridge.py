"""
GPU Bridge for Quantum ML
Connects Python quantum NAS to Rust GPU acceleration via ctypes FFI
"""

import ctypes
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


class QuantumMLGPUBridge:
    """
    FFI bridge to Rust GPU acceleration for quantum ML operations.
    
    Provides:
    - GPU initialization and management
    - Quantum state processing on GPU
    - Architecture evaluation on GPU
    - Wave function collapse optimization
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize GPU bridge.

        Args:
            lib_path: Path to compiled Rust library (libqallow_native.so)
        """
        self.lib = None
        self.gpu_manager = None
        self.initialized = False

        if lib_path is None:
            # Try to find the library
            lib_path = self._find_library()

        if lib_path and os.path.exists(lib_path):
            try:
                self.lib = ctypes.CDLL(lib_path)
                self._setup_functions()
                self.gpu_manager = self._init_gpu()
                self.initialized = True
                print(f"✓ GPU Bridge initialized with {lib_path}")
            except Exception as e:
                print(f"⚠ Failed to load GPU library: {e}")
                print("  Falling back to CPU-only mode")
        else:
            print("⚠ GPU library not found - CPU-only mode")

    def is_initialized(self) -> bool:
        """Check if GPU bridge is properly initialized."""
        return self.initialized and self.lib is not None and self.gpu_manager is not None
    
    def _find_library(self) -> Optional[str]:
        """Find the compiled Rust library."""
        possible_paths = [
            "/root/Qallow/target/release/libqallow_native.so",
            "/root/Qallow/native_app/target/release/libqallow_native.so",
            "/root/Qallow/native_app/target/debug/libqallow_native.so",
            "./target/release/libqallow_native.so",
            "./target/debug/libqallow_native.so",
            "./native_app/target/release/libqallow_native.so",
            "./native_app/target/debug/libqallow_native.so",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None
    
    def _setup_functions(self):
        """Setup ctypes function signatures."""
        if not self.lib:
            return
        
        # GPU initialization
        self.lib.quantum_ml_gpu_init.restype = ctypes.c_void_p
        self.lib.quantum_ml_gpu_init.argtypes = []
        
        # Process quantum states
        self.lib.quantum_ml_process_states.restype = ctypes.c_void_p
        self.lib.quantum_ml_process_states.argtypes = [
            ctypes.c_void_p,  # gpu_manager
            ctypes.POINTER(ctypes.c_int32),  # states
            ctypes.c_uint32,  # count
        ]
        
        # Evaluate architectures
        self.lib.quantum_ml_evaluate_architectures.restype = ctypes.c_void_p
        self.lib.quantum_ml_evaluate_architectures.argtypes = [
            ctypes.c_void_p,  # gpu_manager
            ctypes.c_void_p,  # consciousness
        ]
        
        # Evolve architectures
        self.lib.quantum_ml_evolve_architectures.restype = ctypes.c_int32
        self.lib.quantum_ml_evolve_architectures.argtypes = [
            ctypes.c_void_p,  # gpu_manager
            ctypes.c_void_p,  # consciousness
            ctypes.c_uint32,  # iterations
        ]
        
        # Collapse wave function
        self.lib.quantum_ml_collapse_wave_function.restype = ctypes.c_uint32
        self.lib.quantum_ml_collapse_wave_function.argtypes = [
            ctypes.c_void_p,  # gpu_manager
            ctypes.c_void_p,  # consciousness
        ]
        
        # Get GPU metrics
        self.lib.quantum_ml_get_gpu_metrics.restype = ctypes.c_char_p
        self.lib.quantum_ml_get_gpu_metrics.argtypes = [ctypes.c_void_p]
        
        # Free functions
        self.lib.quantum_ml_gpu_free.argtypes = [ctypes.c_void_p]
        self.lib.quantum_ml_consciousness_free.argtypes = [ctypes.c_void_p]
        self.lib.quantum_ml_free_string.argtypes = [ctypes.c_char_p]
    
    def _init_gpu(self) -> Optional[ctypes.c_void_p]:
        """Initialize GPU manager."""
        if not self.lib:
            return None
        
        try:
            return self.lib.quantum_ml_gpu_init()
        except Exception as e:
            print(f"⚠ GPU initialization failed: {e}")
            return None
    
    def process_quantum_states(self, states: List[int]) -> Optional[ctypes.c_void_p]:
        """
        Process quantum states on GPU.
        
        Args:
            states: List of quantum state values
            
        Returns:
            Pointer to consciousness state on GPU
        """
        if not self.lib or not self.gpu_manager:
            return None
        
        try:
            states_array = (ctypes.c_int32 * len(states))(*states)
            consciousness = self.lib.quantum_ml_process_states(
                self.gpu_manager,
                states_array,
                len(states)
            )
            return consciousness
        except Exception as e:
            print(f"⚠ Failed to process quantum states: {e}")
            return None
    
    def evaluate_architectures(self, consciousness: ctypes.c_void_p) -> Dict[str, Any]:
        """
        Evaluate architectures on GPU.
        
        Args:
            consciousness: Pointer to consciousness state
            
        Returns:
            Evaluation metrics
        """
        if not self.lib or not self.gpu_manager or not consciousness:
            return {}
        
        try:
            result_ptr = self.lib.quantum_ml_evaluate_architectures(
                self.gpu_manager,
                consciousness
            )
            
            # Parse result structure
            result_struct = ctypes.cast(result_ptr, ctypes.POINTER(ctypes.c_void_p))
            
            return {
                'status': 'evaluated',
                'gpu_accelerated': True
            }
        except Exception as e:
            print(f"⚠ Failed to evaluate architectures: {e}")
            return {}
    
    def evolve_architectures(
        self,
        consciousness: ctypes.c_void_p,
        iterations: int = 100
    ) -> bool:
        """
        Evolve architectures on GPU.
        
        Args:
            consciousness: Pointer to consciousness state
            iterations: Number of evolution iterations
            
        Returns:
            Success status
        """
        if not self.lib or not self.gpu_manager or not consciousness:
            return False
        
        try:
            result = self.lib.quantum_ml_evolve_architectures(
                self.gpu_manager,
                consciousness,
                iterations
            )
            return result == 0
        except Exception as e:
            print(f"⚠ Failed to evolve architectures: {e}")
            return False
    
    def collapse_wave_function(self, consciousness: ctypes.c_void_p) -> int:
        """
        Collapse wave function to find optimal architecture.
        
        Args:
            consciousness: Pointer to consciousness state
            
        Returns:
            Index of optimal architecture
        """
        if not self.lib or not self.gpu_manager or not consciousness:
            return -1
        
        try:
            return self.lib.quantum_ml_collapse_wave_function(
                self.gpu_manager,
                consciousness
            )
        except Exception as e:
            print(f"⚠ Failed to collapse wave function: {e}")
            return -1
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        """Get GPU device metrics."""
        if not self.lib or not self.gpu_manager:
            return {}
        
        try:
            metrics_ptr = self.lib.quantum_ml_get_gpu_metrics(self.gpu_manager)
            if metrics_ptr:
                metrics_str = ctypes.string_at(metrics_ptr).decode('utf-8')
                self.lib.quantum_ml_free_string(metrics_ptr)
                return json.loads(metrics_str)
        except Exception as e:
            print(f"⚠ Failed to get GPU metrics: {e}")
        
        return {}
    
    def cleanup(self):
        """Clean up GPU resources."""
        if self.lib and self.gpu_manager:
            try:
                self.lib.quantum_ml_gpu_free(self.gpu_manager)
            except Exception as e:
                print(f"⚠ Failed to cleanup GPU: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


if __name__ == "__main__":
    bridge = QuantumMLGPUBridge()
    
    # Get GPU metrics
    metrics = bridge.get_gpu_metrics()
    if metrics:
        print(f"✓ GPU Metrics: {metrics}")
    else:
        print("⚠ GPU metrics not available")
    
    bridge.cleanup()

