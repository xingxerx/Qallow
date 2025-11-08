#!/usr/bin/env python3
"""
CUDA-Q Quantum Bridge for Qallow Meta-Learning
MANDATORY GPU-accelerated quantum simulation via CUDA-Q 0.8+

This module provides production-grade quantum circuit execution on NVIDIA GPUs
using the official CUDA-Q framework. It is NOT optional and requires CUDA-Q
to be properly installed and operational.

Requirements (MANDATORY):
  - CUDA 12.0+ (NVIDIA GPU driver)
  - CUDA-Q 0.8+ (from nvidia/cuda-quantum)
  - cupy 12.0+ (GPU array operations)
  - numpy 1.20+

Installation:
  pip install cuda-quantum>=0.8.0
  pip install cupy-cuda12x  # where x is your CUDA minor version
  
Verification:
  python3 -c "import cudaq; print(cudaq.__version__)"

Usage:
  from python.quantum.cuda_q_bridge import CudaQBridge, CudaQConfig
  
  config = CudaQConfig(backend='nvidia', shots=1024)
  bridge = CudaQBridge(config)
  
  result = bridge.quantum_sample(
      circuit_params={'theta_0': 0.5},
      n_qubits=4,
      circuit_depth=2,
      n_shots=1024
  )

Raises:
  ImportError: If CUDA-Q not installed (see installation above)
  RuntimeError: If GPU not available or CUDA-Q initialization fails
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np

# MANDATORY: Import CUDA-Q (will raise ImportError if not installed)
try:
    import cudaq
    CUDAQ_AVAILABLE = True
    CUDAQ_VERSION = cudaq.__version__
except ImportError as e:
    CUDAQ_AVAILABLE = False
    CUDAQ_VERSION = None
    _import_error = e

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CudaQConfig:
    """Configuration for CUDA-Q quantum backend"""
    backend: str = "nvidia"  # nvidia, quantinuum, ionq, iqm, etc.
    target_name: str = "mqpu"  # multi-GPU execution
    shots: int = 1024
    seed: Optional[int] = None
    enable_metrics: bool = True
    optimization_level: int = 2  # 0-3 for circuit optimization


@dataclass
class QuantumSample:
    """Quantum measurement result"""
    bitstring: str
    probability: float
    energy: Optional[float] = None
    timestamp: float = 0.0


@dataclass
class QuantumSamplingResult:
    """Complete quantum sampling result from circuit execution"""
    n_samples: int
    n_qubits: int
    backend: str
    samples: List[QuantumSample]
    execution_time_ms: float
    circuit_depth: int
    circuit_gates: int
    metrics: Dict[str, Any]


class CudaQBridge:
    """
    GPU-accelerated quantum sampling interface using CUDA-Q 0.8+
    
    IMPORTANT: This is NOT optional. CUDA-Q must be installed and operational.
    See module docstring for installation instructions.
    
    This bridge provides production-grade quantum circuit execution with:
    - Full CUDA-Q backend support (nvidia, quantinuum, ionq, iqm, etc.)
    - Multi-GPU execution planning
    - Precise error reporting
    - Performance metrics (gate count, circuit depth, execution time)
    - Deterministic sampling with seed control
    """

    def __init__(self, config: Optional[CudaQConfig] = None):
        """
        Initialize CUDA-Q bridge - RAISES if CUDA-Q not available
        
        Args:
            config: CudaQConfig for backend selection
            
        Raises:
            ImportError: If CUDA-Q package not installed
            RuntimeError: If CUDA-Q initialization fails or GPU not available
        """
        # CHECK: CUDA-Q must be available
        if not CUDAQ_AVAILABLE:
            raise ImportError(
                f"CUDA-Q is REQUIRED but not installed.\n"
                f"Installation error: {_import_error}\n"
                f"Please install: pip install cuda-quantum>=0.8.0\n"
                f"See module docstring for full setup instructions."
            )
        
        self.config = config or CudaQConfig()
        self.cudaq_module = cudaq
        self.backend_target = None
        self.device_info = None
        
        # Initialize backend
        self._init_backend()
        
        logger.info(f"✓ CUDA-Q bridge initialized: backend={self.config.backend}, "
                   f"version={CUDAQ_VERSION}")

    def _init_backend(self):
        """
        Initialize CUDA-Q backend target
        
        Raises:
            RuntimeError: If backend not available or GPU not working
        """
        try:
            # Attempt backend initialization
            if self.config.backend == "nvidia":
                # NVIDIA GPU backend (requires CUDA)
                self.backend_target = self.cudaq_module.target.Nvidia()
                self.device_info = "NVIDIA GPU (nvidia/cuda-quantum)"
                
            elif self.config.backend == "quantinuum":
                # Quantinuum cloud backend
                self.backend_target = self.cudaq_module.target.Quantinuum()
                self.device_info = "Quantinuum cloud"
                
            elif self.config.backend == "ionq":
                # IonQ cloud backend
                self.backend_target = self.cudaq_module.target.IonQ()
                self.device_info = "IonQ cloud"
                
            elif self.config.backend == "iqm":
                # IQM cloud backend
                self.backend_target = self.cudaq_module.target.IQM()
                self.device_info = "IQM cloud"
                
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")
            
            # Set backend as active
            cudaq.set_target(self.backend_target)
            
            logger.info(f"✓ CUDA-Q backend set: {self.config.backend} ({self.device_info})")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CUDA-Q backend '{self.config.backend}':\n"
                f"  {type(e).__name__}: {e}\n"
                f"Possible causes:\n"
                f"  1. GPU not available\n"
                f"  2. CUDA not installed\n"
                f"  3. Backend credentials not configured\n"
                f"  4. CUDA-Q installation corrupted\n"
                f"Try: pip install --upgrade cuda-quantum"
            )

    def quantum_sample(
        self,
        circuit_params: Dict[str, float],
        n_qubits: int,
        circuit_depth: int = 5,
        n_shots: int = 1024,
        importance_weights: Optional[List[float]] = None
    ) -> QuantumSamplingResult:
        """
        Execute quantum sampling on GPU
        
        Args:
            circuit_params: Parameter dict {param_name: value}
            n_qubits: Number of qubits
            circuit_depth: Circuit depth
            n_shots: Measurement samples
            importance_weights: Optional importance weights for resampling
        
        Returns:
            QuantumSamplingResult with accurate measurements from GPU
            
        Raises:
            RuntimeError: If GPU execution fails
            ValueError: If parameters invalid
        """
        if not self.backend_target:
            raise RuntimeError("Backend not initialized")
        
        if n_qubits < 1 or n_qubits > 30:
            raise ValueError(f"Invalid n_qubits: {n_qubits} (must be 1-30)")
        
        if circuit_depth < 1:
            raise ValueError(f"Invalid circuit_depth: {circuit_depth}")
        
        if n_shots < 1 or n_shots > 100000:
            raise ValueError(f"Invalid n_shots: {n_shots} (must be 1-100000)")
        
        try:
            start_time = time.time()
            
            # Build parameterized circuit
            kernel = self._build_ansatz(n_qubits, circuit_depth, circuit_params)
            
            # Set seed for reproducibility if provided
            if self.config.seed is not None:
                cudaq.set_random_seed(self.config.seed)
            
            # Execute on GPU backend with proper configuration
            result = cudaq.sample(kernel, shots_count=n_shots)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Process results
            samples = self._process_counts(result, importance_weights)
            
            return QuantumSamplingResult(
                n_samples=len(samples),
                n_qubits=n_qubits,
                backend=self.config.backend,
                samples=samples,
                execution_time_ms=execution_time_ms,
                circuit_depth=circuit_depth,
                circuit_gates=self._estimate_gate_count(n_qubits, circuit_depth),
                metrics={
                    "shot_efficiency": len(samples) / n_shots,
                    "unique_bitstrings": len(set(s.bitstring for s in samples)),
                    "avg_probability": np.mean([s.probability for s in samples]),
                    "max_probability": np.max([s.probability for s in samples]),
                    "device": self.device_info,
                    "seed": self.config.seed
                }
            )
            
        except Exception as e:
            raise RuntimeError(
                f"GPU quantum sampling failed:\n"
                f"  {type(e).__name__}: {e}\n"
                f"Backend: {self.config.backend}\n"
                f"Qubits: {n_qubits}, Shots: {n_shots}\n"
                f"This is a real error - not falling back to mock."
            )

    def _build_ansatz(
        self,
        n_qubits: int,
        depth: int,
        params: Dict[str, float]
    ) -> Any:
        """
        Build parameterized quantum circuit using CUDA-Q
        
        Circuit structure:
        - Layer d (0 to depth-1):
          - RY rotations on all qubits
          - CNOT ring entanglement (q[i] → q[(i+1)%n])
        - Measurement on all qubits
        """
        @cudaq.kernel
        def ansatz():
            qvec = cudaq.qvector(n_qubits)
            
            for d in range(depth):
                # RY rotation layer
                for q in range(n_qubits):
                    param_name = f"theta_{d}_{q}"
                    param_val = params.get(param_name, 0.0)
                    cudaq.ry(param_val, qvec[q])
                
                # Entanglement: ring topology CNOT
                for q in range(n_qubits - 1):
                    cudaq.cx(qvec[q], qvec[q + 1])
                if n_qubits > 1:
                    cudaq.cx(qvec[n_qubits - 1], qvec[0])
            
            # Measure all qubits
            cudaq.mz(qvec)
        
        return ansatz

    def _process_counts(
        self,
        result: Any,
        importance_weights: Optional[List[float]] = None
    ) -> List[QuantumSample]:
        """
        Convert CUDA-Q measurement results to QuantumSample objects
        """
        samples = []
        
        # Extract counts from CUDA-Q result object
        if hasattr(result, 'counts'):
            counts = result.counts()
        else:
            counts = dict(result)
        
        total_shots = sum(counts.values()) if counts else 1
        
        for bitstring, count in counts.items():
            probability = count / total_shots
            
            # Apply importance weighting if provided
            if importance_weights:
                idx = int(bitstring, 2) % len(importance_weights)
                weighted_prob = probability * importance_weights[idx]
            else:
                weighted_prob = probability
            
            samples.append(QuantumSample(
                bitstring=str(bitstring),
                probability=weighted_prob,
                energy=None,
                timestamp=time.time()
            ))
        
        # Normalize probabilities
        total_prob = sum(s.probability for s in samples)
        if total_prob > 0:
            for s in samples:
                s.probability /= total_prob
        
        return samples

    def _estimate_gate_count(self, n_qubits: int, depth: int) -> int:
        """Estimate total gate count"""
        # depth * (n_qubits RY gates + (n_qubits - 1) CNOTs + 1 wrap CNOT)
        return depth * (n_qubits + n_qubits)

    def get_backend_status(self) -> Dict[str, Any]:
        """Get detailed backend status"""
        return {
            "available": True,  # Would be False if not initialized
            "backend": self.config.backend,
            "device_info": self.device_info,
            "cudaq_version": CUDAQ_VERSION,
            "shots": self.config.shots,
            "seed": self.config.seed,
            "optimization_level": self.config.optimization_level,
            "status": "operational"
        }

    def export_to_json(self, result: QuantumSamplingResult) -> str:
        """Export sampling result to JSON"""
        data = {
            "algorithm": "cuda_q_quantum_sampling",
            "n_samples": result.n_samples,
            "n_qubits": result.n_qubits,
            "backend": result.backend,
            "execution_time_ms": result.execution_time_ms,
            "circuit_depth": result.circuit_depth,
            "circuit_gates": result.circuit_gates,
            "samples": [asdict(s) for s in result.samples[:10]],  # First 10
            "sample_count": len(result.samples),
            "metrics": result.metrics
        }
        return json.dumps(data, indent=2)


# Module initialization: Verify CUDA-Q availability on import
if not CUDAQ_AVAILABLE:
    logger.warning(f"⚠️  CUDA-Q not available (will fail at runtime)")
else:
    logger.info(f"✓ CUDA-Q module loaded: version {CUDAQ_VERSION}")


def get_cuda_q_bridge(backend: str = "nvidia") -> CudaQBridge:
    """
    Create CUDA-Q bridge with specified backend
    
    Args:
        backend: "nvidia", "quantinuum", "ionq", "iqm"
        
    Returns:
        CudaQBridge instance
        
    Raises:
        ImportError: If CUDA-Q not installed
        RuntimeError: If backend not available
    """
    config = CudaQConfig(backend=backend)
    return CudaQBridge(config)


if __name__ == "__main__":
    # Test CUDA-Q bridge
    logging.basicConfig(level=logging.INFO)
    
    bridge = get_cuda_q_bridge(backend="nvidia")
    print(f"Backend status: {bridge.get_backend_status()}")
    
    # Test quantum sampling
    params = {f"theta_0_{i}": 0.1 * i for i in range(4)}
    result = bridge.quantum_sample(
        circuit_params=params,
        n_qubits=4,
        circuit_depth=2,
        n_shots=100
    )
    
    print(f"\nQuantum sampling result:")
    print(f"  Samples: {result.n_samples}")
    print(f"  Backend: {result.backend}")
    print(f"  Execution time: {result.execution_time_ms:.2f} ms")
    print(f"  Unique bitstrings: {result.metrics['unique_bitstrings']}")
