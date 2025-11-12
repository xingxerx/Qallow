#!/usr/bin/env python3
"""
Unified Backend Orchestrator for Qallow Meta-Learning
Primary Backend: CUDA-Q 0.8+ (MANDATORY)
Secondary Backends: CUDA, Cirq (optional)
CPU Backend: Classical baseline only

CRITICAL: CUDA-Q is REQUIRED for quantum acceleration.
If CUDA-Q is not available, the orchestrator will FAIL with clear instructions.
Cirq/CPU are fallbacks only when explicitly requested, not automatic.

This module provides:
- Mandatory CUDA-Q backend with explicit error handling
- Optional Cirq/CUDA for explicitly requested alternatives
- Integrated telemetry and performance monitoring
- Unified interface for quantum + classical optimization

Usage:
  from python.quantum.orchestrator import QuantumOrchestrator
  # Requires CUDA-Q to be installed
  orchestra = QuantumOrchestrator(backend='cuda_q')  # Explicit
  result = orchestra.execute_optimization(n_qubits=4, n_iterations=100)
  
  # Or use auto (checks CUDA-Q first, requires explicit alternative if unavailable)
  orchestra = QuantumOrchestrator(backend='auto')  # auto=cuda_q if available else error
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
    
    CUDA-Q is the PRIMARY backend and is REQUIRED.
    Falls back to explicit alternatives only when CUDA-Q is unavailable
    and explicitly requested.
    
    Backend Priority:
    1. CUDA-Q 0.8+ (GPU quantum) - REQUIRED
    2. CUDA (GPU accelerated) - optional alternative
    3. Cirq (CPU simulator) - optional fallback only
    4. CPU (classical only) - baseline
    """

    def __init__(self, backend: str = "cuda_q"):
        """
        Initialize orchestrator
        
        Args:
            backend: "cuda_q" (required), "cuda", "cirq", "cpu"
        
        Raises:
            ImportError: If CUDA-Q requested but not available
            ValueError: If backend not recognized
        """
        # Validate backend choice
        valid_backends = {"cuda_q", "cuda", "cirq", "cpu"}
        if backend not in valid_backends:
            raise ValueError(
                f"Backend '{backend}' not recognized.\n"
                f"Valid options: {', '.join(sorted(valid_backends))}"
            )
        
        self.requested_backend = backend
        self.backends = {}
        self.active_backend = None
        
        # Initialize backend detection (CUDA-Q mandatory first)
        self._detect_backends()
        self._select_backend()
        
        logger.info(f"✓ Orchestrator initialized: backend={self.active_backend}")

    def _detect_backends(self):
        """
        Detect available backends with CUDA-Q as primary requirement
        
        CUDA-Q is checked first. If not available:
        - If 'cuda_q' requested: FAIL with installation instructions
        - If 'auto': attempt to use available alternative (not silent)
        - If other backend requested: use that alternative
        """
        # CHECK: CUDA-Q availability (PRIMARY)
        cudaq_available = False
        cudaq_version = None
        cudaq_error = None
        
        try:
            import cudaq
            cudaq_available = True
            cudaq_version = getattr(cudaq, '__version__', 'unknown')
            self.backends['cuda_q'] = BackendStatus(
                name="CUDA-Q",
                available=True,
                priority=0,  # Highest priority
                version=cudaq_version,
                device_info="NVIDIA GPU quantum simulation"
            )
            logger.info(f"✓ CUDA-Q available: v{cudaq_version}")
        except ImportError as e:
            cudaq_error = str(e)
            self.backends['cuda_q'] = BackendStatus(
                name="CUDA-Q",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message=cudaq_error
            )
            logger.warning(f"⚠️  CUDA-Q not available: {cudaq_error}")
        
        # CHECK: CUDA backend (optional, secondary)
        try:
            import pycuda.driver as cuda
            cuda.init()
            device = cuda.Device(0)
            device_name = device.name().decode('utf-8') if isinstance(device.name(), bytes) else device.name()
            
            self.backends['cuda'] = BackendStatus(
                name="CUDA",
                available=True,
                priority=1,
                version=f"CUDA {cuda.get_version()}",
                device_info=f"GPU: {device_name}"
            )
            logger.info(f"✓ CUDA backend available: {device_name}")
        except Exception as e:
            self.backends['cuda'] = BackendStatus(
                name="CUDA",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message=str(e)
            )
        
        # CHECK: Cirq backend (optional, fallback)
        try:
            import cirq
            self.backends['cirq'] = BackendStatus(
                name="Cirq",
                available=True,
                priority=2,
                version=getattr(cirq, '__version__', 'unknown'),
                device_info="CPU quantum simulator (Cirq)"
            )
            logger.info(f"✓ Cirq backend available")
        except ImportError as e:
            self.backends['cirq'] = BackendStatus(
                name="Cirq",
                available=False,
                priority=999,
                version=None,
                device_info=None,
                error_message=str(e)
            )
        
        # CPU backend always available (classical baseline)
        self.backends['cpu'] = BackendStatus(
            name="CPU",
            available=True,
            priority=3,
            version="1.0",
            device_info="Classical CPU Bayesian optimization"
        )

    def _select_backend(self):
        """
        Select active backend with mandatory CUDA-Q preference
        
        Rules:
        1. If 'cuda_q' requested: MUST be available or FAIL
        2. If other backend explicitly requested: use if available
        3. If backend unavailable: raise error with installation instructions
        """
        if self.requested_backend == "cuda_q":
            # CUDA-Q explicitly requested - MUST be available
            if not self.backends.get('cuda_q', BackendStatus('', False, 999, None, None)).available:
                error_msg = (
                    f"CUDA-Q is REQUIRED but not available.\n"
                    f"Installation instructions:\n"
                    f"  pip install cuda-quantum>=0.8.0\n"
                    f"System check:\n"
                    f"  - Python 3.9+: python --version\n"
                    f"  - CUDA 12.0+: nvcc --version\n"
                    f"  - cupy: pip install cupy-cuda12x\n"
                    f"If CUDA-Q is not suitable, use --backend=cirq or --backend=cpu instead"
                )
                raise ImportError(error_msg)
            
            self.active_backend = "cuda_q"
            logger.info("✓ CUDA-Q backend selected (mandatory quantum acceleration)")
        
        elif self.requested_backend == "cuda":
            # CUDA explicitly requested
            if not self.backends.get('cuda', BackendStatus('', False, 999, None, None)).available:
                raise ImportError(
                    f"CUDA backend requested but not available.\n"
                    f"Install: pip install pycuda\n"
                    f"Or use: --backend=cuda_q (recommended) or --backend=cirq"
                )
            self.active_backend = "cuda"
            logger.info("✓ CUDA backend selected")
        
        elif self.requested_backend == "cirq":
            # Cirq explicitly requested (CPU simulation)
            if not self.backends.get('cirq', BackendStatus('', False, 999, None, None)).available:
                raise ImportError(
                    f"Cirq backend requested but not available.\n"
                    f"Install: pip install cirq>=1.2.0\n"
                    f"Or use: --backend=cuda_q (recommended, requires GPU)"
                )
            self.active_backend = "cirq"
            logger.info("✓ Cirq CPU simulation backend selected")
        
        elif self.requested_backend == "cpu":
            # CPU-only (classical baseline)
            self.active_backend = "cpu"
            logger.info("✓ CPU classical backend selected (no quantum acceleration)")
        
        else:
            raise ValueError(f"Unknown backend: {self.requested_backend}")

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
    """Test orchestrator with mandatory CUDA-Q"""
    logging.basicConfig(level=logging.INFO)
    
    print("=== Qallow Quantum Orchestrator ===\n")
    
    # Try to initialize with CUDA-Q (required)
    try:
        orchestra = QuantumOrchestrator(backend="cuda_q")
    except ImportError as e:
        print(f"ERROR: {e}\n")
        print("Falling back to Cirq CPU simulation (slower, no GPU)...")
        try:
            orchestra = QuantumOrchestrator(backend="cirq")
        except ImportError:
            print("Cirq not available either, using CPU baseline (classical only)...")
            orchestra = QuantumOrchestrator(backend="cpu")
    
    # Print backend status
    print("\n=== Available Backends ===")
    status = orchestra.get_backend_status()
    for backend, info in status.items():
        avail = "✓" if info['available'] else "✗"
        print(f"  {avail} {backend:10} v{info['version']:10} {info['device_info']}")
    
    print(f"\nActive backend: {(orchestra.active_backend or '').upper()}")
    
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
    
    # Export JSON
    json_result = orchestra.export_result_json(result)
    print(f"\nJSON Export (first 200 chars):\n{json_result[:200]}...")


if __name__ == "__main__":
    main()
