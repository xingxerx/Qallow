#!/usr/bin/env python3
"""
Unified Backend Orchestrator for Qallow Meta-Learning
Coordinates CPU, CUDA, CUDA-Q, and Cirq quantum backends

This module provides:
- Automatic backend selection based on availability
- Transparent switching between CPU/CUDA/CUDA-Q/Cirq
- Integrated telemetry and performance monitoring
- Unified interface for quantum + classical optimization

Usage:
  from python.quantum.orchestrator import QuantumOrchestrator
  orchestra = QuantumOrchestrator(preferred_backend='auto')
  result = orchestra.execute_optimization(n_qubits=4, n_iterations=100)
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import numpy as np

# Ensure project root is importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger = logging.getLogger(__name__)


class Backend(Enum):
    """Available quantum backends"""
    CPU = "cpu"           # Classical-only Bayesian optimization
    CUDA = "cuda"         # GPU-accelerated quantum sampling
    CUDA_Q = "cuda_q"     # CUDA-Q 0.8+ quantum simulation
    CIRQ = "cirq"         # Cirq CPU-based quantum simulation


@dataclass
class BackendStatus:
    """Status information for a backend"""
    name: str
    available: bool
    priority: int  # Lower = preferred
    version: Optional[str]
    device_info: Optional[str]
    error_message: Optional[str] = None


@dataclass
class OptimizationStep:
    """Single optimization step result"""
    iteration: int
    best_loss: float
    current_loss: float
    improvement: float
    backend_used: str
    n_samples: int
    execution_time_ms: float
    metrics: Dict[str, Any]


@dataclass
class OptimizationResult:
    """Complete optimization result"""
    n_iterations: int
    n_qubits: int
    best_loss: float
    best_parameters: List[float]
    steps: List[OptimizationStep]
    backend_sequence: List[str]  # Backends used in order
    total_time_ms: float
    convergence_iteration: Optional[int]
    metrics: Dict[str, Any]


class QuantumOrchestrator:
    """
    Unified orchestrator for quantum-classical hybrid optimization
    
    Automatically selects and manages available quantum backends,
    falling back gracefully when backends are unavailable.
    """

    def __init__(self, preferred_backend: str = "auto"):
        """
        Initialize orchestrator
        
        Args:
            preferred_backend: "auto", "cpu", "cuda", "cuda_q", "cirq"
        """
        self.preferred_backend = preferred_backend
        self.backends = {}
        self.active_backend = None
        self.backend_priority = [
            Backend.CUDA_Q,
            Backend.CUDA,
            Backend.CIRQ,
            Backend.CPU
        ]
        
        # Initialize backend detection
        self._detect_backends()
        self._select_backend()
        
        logger.info(f"Orchestrator initialized with backend: {self.active_backend}")

    def _detect_backends(self):
        """Detect which backends are available"""
        # Check CUDA-Q
        try:
            import cudaq
            self.backends[Backend.CUDA_Q] = BackendStatus(
                name="CUDA-Q",
                available=True,
                priority=0,
                version=getattr(cudaq, '__version__', 'unknown'),
                device_info="GPU quantum simulation"
            )
            logger.info("✓ CUDA-Q backend available")
        except ImportError:
            self.backends[Backend.CUDA_Q] = BackendStatus(
                name="CUDA-Q",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message="cudaq package not installed"
            )
        
        # Check CUDA (check for CUDA runtime)
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(0)
            props = device.get_attributes()
            self.backends[Backend.CUDA] = BackendStatus(
                name="CUDA",
                available=True,
                priority=1,
                version=f"CUDA {cuda.get_version()}",
                device_info=f"{device.name()}"
            )
            logger.info(f"✓ CUDA backend available: {device.name()}")
        except Exception as e:
            self.backends[Backend.CUDA] = BackendStatus(
                name="CUDA",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message=str(e)
            )
        
        # Check Cirq
        try:
            import cirq
            self.backends[Backend.CIRQ] = BackendStatus(
                name="Cirq",
                available=True,
                priority=2,
                version=getattr(cirq, '__version__', 'unknown'),
                device_info="CPU quantum simulator"
            )
            logger.info("✓ Cirq backend available")
        except ImportError:
            self.backends[Backend.CIRQ] = BackendStatus(
                name="Cirq",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message="cirq package not installed"
            )
        
        # CPU backend always available
        self.backends[Backend.CPU] = BackendStatus(
            name="CPU",
            available=True,
            priority=3,
            version="1.0 (Classical only)",
            device_info="CPU Bayesian optimization"
        )
        logger.info("✓ CPU backend always available")

    def _select_backend(self):
        """Select active backend based on preferences and availability"""
        if self.preferred_backend == "auto":
            # Select highest priority available backend
            available = [
                (b, self.backends[b]) 
                for b in self.backend_priority 
                if b in self.backends and self.backends[b].available
            ]
            
            if available:
                self.active_backend = available[0][0].value
                logger.info(f"Auto-selected backend: {self.active_backend}")
            else:
                self.active_backend = Backend.CPU.value
                logger.warning("No quantum backends available, using CPU only")
        else:
            # Use requested backend
            for b in Backend:
                if b.value == self.preferred_backend:
                    if self.backends[b].available:
                        self.active_backend = b.value
                    else:
                        logger.warning(f"Requested backend '{self.preferred_backend}' unavailable, falling back to CPU")
                        self.active_backend = Backend.CPU.value
                    break

    def get_backend_status(self) -> Dict[str, Any]:
        """Get status of all backends"""
        status = {}
        for backend, info in self.backends.items():
            status[backend.value] = {
                "available": info.available,
                "priority": info.priority,
                "version": info.version,
                "device_info": info.device_info,
                "error": info.error_message
            }
        return status

    def execute_optimization(
        self,
        n_qubits: int = 4,
        n_iterations: int = 50,
        circuit_depth: int = 3,
        param_bounds: Optional[Tuple[float, float]] = None,
        convergence_threshold: float = 0.01
    ) -> OptimizationResult:
        """
        Execute quantum-classical hybrid optimization
        
        Args:
            n_qubits: Number of qubits for quantum circuits
            n_iterations: Maximum optimization iterations
            circuit_depth: Depth of parameterized quantum circuits
            param_bounds: (min, max) bounds for parameters
            convergence_threshold: Stop when improvements < threshold
        
        Returns:
            OptimizationResult with full history and metrics
        """
        import time
        start_time = time.time()
        
        if param_bounds is None:
            param_bounds = (0.0, 2 * np.pi)
        
        n_params = n_qubits * circuit_depth
        
        # Initialize optimization state
        best_loss = 1e10
        best_parameters = np.random.uniform(param_bounds[0], param_bounds[1], n_params)
        steps = []
        backend_sequence = []
        
        logger.info(f"Starting optimization: backend={self.active_backend}, "
                   f"n_qubits={n_qubits}, n_iterations={n_iterations}")
        
        # Main optimization loop
        for iteration in range(n_iterations):
            step_start = time.time()
            
            # Sample quantum state using active backend
            if self.active_backend in [Backend.CUDA_Q.value, Backend.CIRQ.value]:
                n_samples = 1024
                samples = self._quantum_sample(n_qubits, circuit_depth, best_parameters, n_samples)
            else:
                n_samples = 0
                samples = []
            
            # Evaluate objective function (mock: energy of quantum state)
            current_loss = self._evaluate_objective(best_parameters, samples)
            improvement = max(0.0, best_loss - current_loss)
            
            # Update best
            if current_loss < best_loss:
                best_loss = current_loss
                best_parameters = best_parameters.copy()  # Would update in real optimization
            
            step_time_ms = (time.time() - step_start) * 1000
            
            step = OptimizationStep(
                iteration=iteration,
                best_loss=best_loss,
                current_loss=current_loss,
                improvement=improvement,
                backend_used=self.active_backend,
                n_samples=n_samples,
                execution_time_ms=step_time_ms,
                metrics={
                    "param_magnitude": float(np.linalg.norm(best_parameters)),
                    "exploration_ratio": 0.3 + 0.7 * (1.0 - iteration / n_iterations)
                }
            )
            
            steps.append(step)
            backend_sequence.append(self.active_backend)
            
            if iteration % max(1, n_iterations // 10) == 0:
                logger.info(f"Iteration {iteration}: best_loss={best_loss:.6f}, "
                           f"improvement={improvement:.6f}")
            
            # Check convergence
            if improvement < convergence_threshold and iteration > 10:
                logger.info(f"Converged at iteration {iteration}")
                convergence_iteration = iteration
                break
        else:
            convergence_iteration = None
        
        total_time_ms = (time.time() - start_time) * 1000
        
        result = OptimizationResult(
            n_iterations=len(steps),
            n_qubits=n_qubits,
            best_loss=best_loss,
            best_parameters=best_parameters.tolist(),
            steps=steps,
            backend_sequence=backend_sequence,
            total_time_ms=total_time_ms,
            convergence_iteration=convergence_iteration,
            metrics={
                "avg_step_time_ms": np.mean([s.execution_time_ms for s in steps]),
                "max_step_time_ms": np.max([s.execution_time_ms for s in steps]),
                "total_samples": sum(s.n_samples for s in steps),
                "backend_switches": sum(1 for i in range(1, len(backend_sequence)) 
                                      if backend_sequence[i] != backend_sequence[i-1])
            }
        )
        
        logger.info(f"Optimization complete: best_loss={best_loss:.6f}, "
                   f"total_time={total_time_ms:.1f}ms")
        
        return result

    def _quantum_sample(
        self,
        n_qubits: int,
        circuit_depth: int,
        parameters: np.ndarray,
        n_shots: int
    ) -> np.ndarray:
        """Sample from quantum circuit using active backend"""
        if self.active_backend == Backend.CUDA_Q.value:
            try:
                from python.quantum.cuda_q_bridge import get_cuda_q_bridge
                bridge = get_cuda_q_bridge(backend="nvidia")
                params_dict = {f"theta_{d}_{q}": parameters[d * n_qubits + q]
                             for d in range(circuit_depth) for q in range(n_qubits)}
                result = bridge.quantum_sample(params_dict, n_qubits, circuit_depth, n_shots)
                return np.array([int(s.bitstring, 2) for s in result.samples])
            except Exception as e:
                logger.warning(f"CUDA-Q sampling failed: {e}")
                return np.array([])
        
        elif self.active_backend == Backend.CIRQ.value:
            try:
                from python.quantum.cirq_bridge import get_cirq_bridge
                bridge = get_cirq_bridge(backend="simulator")
                params_dict = {f"theta_{d}_{q}": parameters[d * n_qubits + q]
                             for d in range(circuit_depth) for q in range(n_qubits)}
                result = bridge.quantum_sample(params_dict, n_qubits, circuit_depth, n_shots)
                return np.array([int(s.bitstring, 2) for s in result.samples])
            except Exception as e:
                logger.warning(f"Cirq sampling failed: {e}")
                return np.array([])
        
        else:  # CPU
            return np.array([])

    def _evaluate_objective(
        self,
        parameters: np.ndarray,
        samples: np.ndarray
    ) -> float:
        """Evaluate objective function (mock: quadratic with noise)"""
        # Mock objective: minimize ||parameters|| with penalty
        loss = np.sum(parameters ** 2)
        
        # Add quantum sampling contribution if available
        if len(samples) > 0:
            entropy = -np.mean(np.log(np.bincount(samples) / len(samples) + 1e-10))
            loss += 0.1 * entropy  # Encourage exploration
        
        return float(loss + np.random.normal(0, 0.01))

    def export_result_json(self, result: OptimizationResult) -> str:
        """Export optimization result to JSON"""
        data = {
            "n_iterations": result.n_iterations,
            "n_qubits": result.n_qubits,
            "best_loss": result.best_loss,
            "best_parameters": result.best_parameters,
            "convergence_iteration": result.convergence_iteration,
            "total_time_ms": result.total_time_ms,
            "backend_sequence": result.backend_sequence,
            "metrics": result.metrics,
            "steps": [asdict(s) for s in result.steps[:5]]  # First 5 steps
        }
        return json.dumps(data, indent=2)


def main():
    """Test orchestrator"""
    logging.basicConfig(level=logging.INFO)
    
    # Create orchestrator
    orchestra = QuantumOrchestrator(preferred_backend="auto")
    
    # Print backend status
    print("\n=== Backend Status ===")
    status = orchestra.get_backend_status()
    for backend, info in status.items():
        avail = "✓" if info['available'] else "✗"
        print(f"{avail} {backend:10} ({info['version']})")
    
    print(f"\nActive backend: {orchestra.active_backend}")
    
    # Run optimization
    print("\n=== Running Optimization ===")
    result = orchestra.execute_optimization(
        n_qubits=4,
        n_iterations=20,
        circuit_depth=2,
        convergence_threshold=0.01
    )
    
    # Print results
    print(f"\nFinal Results:")
    print(f"  Best loss: {result.best_loss:.8f}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Time: {result.total_time_ms:.1f} ms")
    print(f"  Backend switches: {result.metrics['backend_switches']}")
    
    # Export JSON
    json_result = orchestra.export_result_json(result)
    print(f"\nJSON Export:\n{json_result}")


if __name__ == "__main__":
    main()
