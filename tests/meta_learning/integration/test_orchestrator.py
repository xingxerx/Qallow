#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for Qallow Meta-Learning
Tests all backends and acceleration paths (CPU, CUDA, CUDA-Q, Cirq)

Run with:
  pytest tests/meta_learning/integration/test_orchestrator.py -v
"""

import pytest
import numpy as np
import json
import logging
import time
from pathlib import Path

# Import quantum modules
from python.quantum.cuda_q_bridge import CudaQBridge, CudaQConfig, get_cuda_q_bridge
from python.quantum.cirq_bridge import CirqBridge, CirqConfig, get_cirq_bridge
from python.quantum.orchestrator import QuantumOrchestrator, Backend


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCudaQBridge:
    """Test CUDA-Q quantum bridge"""

    def test_cuda_q_initialization(self):
        """Test CUDA-Q bridge creation"""
        bridge = get_cuda_q_bridge(backend="nvidia")
        assert bridge is not None
        logger.info(f"CUDA-Q bridge status: {bridge.get_backend_status()}")

    def test_cuda_q_quantum_sample(self):
        """Test CUDA-Q quantum sampling"""
        bridge = get_cuda_q_bridge(backend="nvidia")
        
        params = {f"theta_0_{i}": 0.1 * i for i in range(4)}
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=4,
            circuit_depth=2,
            n_shots=100
        )
        
        assert result is not None
        assert result.n_samples > 0
        assert result.n_qubits == 4
        logger.info(f"CUDA-Q samples: {result.n_samples}, "
                   f"execution_time: {result.execution_time_ms:.2f}ms")

    def test_cuda_q_with_importance_weights(self):
        """Test CUDA-Q with importance weighting"""
        bridge = get_cuda_q_bridge(backend="nvidia")
        
        params = {f"theta_0_{i}": 0.5 for i in range(4)}
        importance_weights = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02,
                            1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]
        
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=4,
            circuit_depth=1,
            n_shots=200,
            importance_weights=importance_weights
        )
        
        assert result is not None
        logger.info(f"CUDA-Q with importance weights: {result.n_samples} samples")

    def test_cuda_q_json_export(self):
        """Test CUDA-Q result JSON export"""
        bridge = get_cuda_q_bridge(backend="nvidia")
        
        params = {f"theta_0_{i}": 0.2 * i for i in range(3)}
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=3,
            circuit_depth=1,
            n_shots=50
        )
        
        json_str = bridge.export_to_json(result)
        assert json_str is not None
        
        data = json.loads(json_str)
        assert data["n_qubits"] == 3
        assert data["backend"] == "mock" or "nvidia" in data["backend"]
        logger.info(f"JSON export successful: {len(json_str)} bytes")


class TestCirqBridge:
    """Test Cirq quantum bridge"""

    def test_cirq_initialization(self):
        """Test Cirq bridge creation"""
        bridge = get_cirq_bridge(backend="simulator")
        assert bridge is not None
        logger.info(f"Cirq bridge status: {bridge.get_backend_status()}")

    def test_cirq_quantum_sample(self):
        """Test Cirq quantum sampling"""
        bridge = get_cirq_bridge(backend="simulator")
        
        params = {f"theta_0_{i}": 0.15 * i for i in range(4)}
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=4,
            circuit_depth=2,
            n_shots=100
        )
        
        assert result is not None
        assert result.n_samples > 0
        logger.info(f"Cirq samples: {result.n_samples}, "
                   f"unique: {result.unique_bitstrings}, "
                   f"entropy: {result.metrics.get('entropy', 0):.4f}")

    def test_cirq_circuit_metrics(self):
        """Test Cirq circuit analysis"""
        bridge = get_cirq_bridge(backend="simulator")
        
        params = {f"theta_0_{i}": 0.1 for i in range(3)}
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=3,
            circuit_depth=2,
            n_shots=50
        )
        
        metrics = result.circuit_metrics
        assert metrics.n_qubits == 3
        assert metrics.depth >= 0
        assert metrics.n_gates >= 0
        logger.info(f"Cirq circuit: depth={metrics.depth}, "
                   f"gates={metrics.n_gates}, "
                   f"2-qubit_gates={metrics.n_two_qubit_gates}")

    def test_cirq_json_export(self):
        """Test Cirq result JSON export"""
        bridge = get_cirq_bridge(backend="simulator")
        
        params = {f"theta_0_{i}": 0.25 * i for i in range(3)}
        result = bridge.quantum_sample(
            circuit_params=params,
            n_qubits=3,
            circuit_depth=1,
            n_shots=75
        )
        
        json_str = bridge.export_to_json(result)
        assert json_str is not None
        
        data = json.loads(json_str)
        assert data["n_qubits"] == 3
        assert data["unique_bitstrings"] > 0
        logger.info(f"Cirq JSON export: {len(json_str)} bytes")


class TestQuantumOrchestrator:
    """Test unified quantum orchestrator"""

    def test_orchestrator_creation(self):
        """Test orchestrator initialization"""
        orchestra = QuantumOrchestrator(preferred_backend="auto")
        assert orchestra is not None
        assert orchestra.active_backend is not None
        logger.info(f"Orchestrator created with backend: {orchestra.active_backend}")

    def test_backend_detection(self):
        """Test backend availability detection"""
        orchestra = QuantumOrchestrator()
        status = orchestra.get_backend_status()
        
        assert status is not None
        assert len(status) >= 1  # At least CPU should be available
        
        available_backends = [b for b, s in status.items() if s['available']]
        logger.info(f"Available backends: {available_backends}")

    def test_orchestrator_auto_fallback(self):
        """Test automatic fallback to CPU"""
        # Request CUDA-Q, should fallback to CPU if not available
        orchestra = QuantumOrchestrator(preferred_backend="cuda_q")
        assert orchestra.active_backend is not None
        logger.info(f"Backend selection: {orchestra.active_backend}")

    def test_orchestrator_optimization_small(self):
        """Test small-scale optimization"""
        orchestra = QuantumOrchestrator(preferred_backend="cpu")
        
        result = orchestra.execute_optimization(
            n_qubits=2,
            n_iterations=5,
            circuit_depth=1,
            convergence_threshold=1.0
        )
        
        assert result is not None
        assert result.n_iterations > 0
        assert result.best_loss < 1e10
        assert len(result.steps) == result.n_iterations
        logger.info(f"Optimization result: loss={result.best_loss:.6f}, "
                   f"iterations={result.n_iterations}")

    def test_orchestrator_optimization_medium(self):
        """Test medium-scale optimization with quantum backend"""
        orchestra = QuantumOrchestrator(preferred_backend="auto")
        
        start_time = time.time()
        result = orchestra.execute_optimization(
            n_qubits=4,
            n_iterations=15,
            circuit_depth=2,
            param_bounds=(0.0, 2 * np.pi),
            convergence_threshold=0.01
        )
        elapsed = time.time() - start_time
        
        assert result is not None
        assert result.n_qubits == 4
        assert result.n_iterations <= 15
        assert result.total_time_ms > 0
        logger.info(f"Medium optimization: "
                   f"loss={result.best_loss:.6f}, "
                   f"iterations={result.n_iterations}, "
                   f"time={elapsed:.2f}s")

    def test_orchestrator_convergence(self):
        """Test convergence detection"""
        orchestra = QuantumOrchestrator(preferred_backend="cpu")
        
        result = orchestra.execute_optimization(
            n_qubits=3,
            n_iterations=50,
            circuit_depth=1,
            convergence_threshold=0.1  # Loose threshold for testing
        )
        
        # Check convergence state
        if result.convergence_iteration is not None:
            assert result.convergence_iteration < result.n_iterations
            logger.info(f"Convergence detected at iteration {result.convergence_iteration}")
        else:
            logger.info("No convergence within iterations")

    def test_orchestrator_json_export(self):
        """Test orchestrator result JSON export"""
        orchestra = QuantumOrchestrator(preferred_backend="cpu")
        
        result = orchestra.execute_optimization(
            n_qubits=2,
            n_iterations=3,
            circuit_depth=1
        )
        
        json_str = orchestra.export_result_json(result)
        assert json_str is not None
        
        data = json.loads(json_str)
        assert data["n_qubits"] == 2
        assert data["best_loss"] < 1e10
        assert len(data["backend_sequence"]) > 0
        logger.info(f"JSON export: {len(json_str)} bytes")

    def test_orchestrator_metrics_accuracy(self):
        """Test metrics computation accuracy"""
        orchestra = QuantumOrchestrator(preferred_backend="cpu")
        
        result = orchestra.execute_optimization(
            n_qubits=2,
            n_iterations=5,
            circuit_depth=1
        )
        
        # Validate metrics
        metrics = result.metrics
        assert metrics["avg_step_time_ms"] > 0
        assert metrics["max_step_time_ms"] >= metrics["avg_step_time_ms"]
        assert metrics["backend_switches"] >= 0
        logger.info(f"Metrics: avg_step={metrics['avg_step_time_ms']:.2f}ms, "
                   f"max_step={metrics['max_step_time_ms']:.2f}ms, "
                   f"switches={metrics['backend_switches']}")


class TestBackendPerformance:
    """Performance comparison tests"""

    def test_cpu_performance_baseline(self):
        """Benchmark CPU-only optimization"""
        orchestra = QuantumOrchestrator(preferred_backend="cpu")
        
        start = time.time()
        result = orchestra.execute_optimization(
            n_qubits=3,
            n_iterations=10,
            circuit_depth=1
        )
        elapsed = time.time() - start
        
        avg_iter_time = elapsed / result.n_iterations
        logger.info(f"CPU baseline: {elapsed:.3f}s total, "
                   f"{avg_iter_time:.3f}s per iteration")

    def test_quantum_backend_acceleration(self):
        """Compare quantum vs CPU performance"""
        # Try quantum backend if available
        orchestra_quantum = QuantumOrchestrator(preferred_backend="auto")
        orchestra_cpu = QuantumOrchestrator(preferred_backend="cpu")
        
        if orchestra_quantum.active_backend == orchestra_cpu.active_backend:
            logger.info("Quantum backend not available, skipping comparison")
            return
        
        # Run on quantum backend
        start_q = time.time()
        result_q = orchestra_quantum.execute_optimization(
            n_qubits=3,
            n_iterations=5,
            circuit_depth=1
        )
        time_q = time.time() - start_q
        
        # Run on CPU
        start_cpu = time.time()
        result_cpu = orchestra_cpu.execute_optimization(
            n_qubits=3,
            n_iterations=5,
            circuit_depth=1
        )
        time_cpu = time.time() - start_cpu
        
        speedup = time_cpu / time_q
        logger.info(f"Speedup comparison: "
                   f"{orchestra_quantum.active_backend}={time_q:.3f}s, "
                   f"CPU={time_cpu:.3f}s, "
                   f"speedup={speedup:.2f}x")


class TestIntegration:
    """End-to-end integration tests"""

    def test_full_workflow(self):
        """Test complete optimization workflow"""
        # Create orchestrator
        orchestra = QuantumOrchestrator(preferred_backend="auto")
        logger.info(f"Step 1: Created orchestrator with {orchestra.active_backend}")
        
        # Run optimization
        result = orchestra.execute_optimization(
            n_qubits=4,
            n_iterations=10,
            circuit_depth=2,
            convergence_threshold=0.05
        )
        logger.info(f"Step 2: Completed optimization (loss={result.best_loss:.6f})")
        
        # Export results
        json_result = orchestra.export_result_json(result)
        data = json.loads(json_result)
        logger.info(f"Step 3: Exported JSON result ({len(json_result)} bytes)")
        
        # Verify data integrity
        assert data["best_loss"] == result.best_loss
        assert len(data["best_parameters"]) == 8  # 4 qubits * 2 depth
        logger.info("Step 4: Data integrity verified ✓")

    def test_multi_backend_execution(self):
        """Test running on multiple backends"""
        backends = ["cpu", "cirq"]
        results = {}
        
        for backend in backends:
            try:
                orchestra = QuantumOrchestrator(preferred_backend=backend)
                result = orchestra.execute_optimization(
                    n_qubits=2,
                    n_iterations=3,
                    circuit_depth=1
                )
                results[backend] = result.best_loss
                logger.info(f"{backend}: loss={result.best_loss:.6f}")
            except Exception as e:
                logger.warning(f"Backend {backend} failed: {e}")
        
        assert len(results) > 0
        logger.info(f"Multi-backend results: {results}")


# Pytest markers
@pytest.mark.slow
def test_long_optimization():
    """Long-running optimization test"""
    orchestra = QuantumOrchestrator(preferred_backend="cpu")
    result = orchestra.execute_optimization(
        n_qubits=5,
        n_iterations=100,
        circuit_depth=3,
        convergence_threshold=0.001
    )
    logger.info(f"Long optimization: loss={result.best_loss:.6f}, "
               f"iterations={result.n_iterations}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
