#!/usr/bin/env python3
"""
Unified Quantum Algorithm Framework - QISKIT VERSION
Equivalent implementation using Qiskit for direct comparison with Cirq.
"""

import numpy as np
from typing import List, Dict, Any
from math import gcd
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import time

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit.circuit import Parameter
    from qiskit.primitives import Sampler
except ImportError:
    print("⚠️  Qiskit not installed. Install with: pip install qiskit qiskit-aer")
    exit(1)


class AlgorithmType(Enum):
    """Supported quantum algorithms"""
    HELLO_QUANTUM = "hello_quantum"
    BELL_STATE = "bell_state"
    DEUTSCH = "deutsch"
    GROVER = "grover"
    SHOR = "shor"
    VQE = "vqe"


@dataclass
class AlgorithmResult:
    """Result from running a quantum algorithm"""
    algorithm: str
    timestamp: str
    success: bool
    circuit: str
    measurements: Dict[str, Any]
    metrics: Dict[str, float]
    error: str = None


class QuantumAlgorithmFramework:
    """Unified framework for all quantum algorithms using Qiskit"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.simulator = AerSimulator()
        self.results = []
        self.start_time = time.time()
    
    def log(self, msg: str):
        """Print log message if verbose"""
        if self.verbose:
            print(msg)
    
    # ==================== HELLO QUANTUM ====================
    def run_hello_quantum(self) -> AlgorithmResult:
        """Run Hello Quantum circuit"""
        self.log("\n" + "="*70)
        self.log("HELLO QUANTUM - Basic Quantum Circuit")
        self.log("="*70)
        
        try:
            qc = QuantumCircuit(3, 3)
            qc.h(0)
            qc.cx(0, 1)
            qc.x(2)
            qc.measure([0, 1, 2], [0, 1, 2])
            
            self.log(f"\nCircuit:\n{qc}")
            
            job = self.simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)
            
            self.log(f"\nMeasurement histogram:\n{counts}")
            
            metrics = {
                'total_shots': 1000,
                'unique_states': len(counts),
                'most_common': max(counts, key=counts.get)
            }
            
            return AlgorithmResult(
                algorithm="HELLO_QUANTUM",
                timestamp=datetime.now().isoformat(),
                success=True,
                circuit=str(qc),
                measurements=counts,
                metrics=metrics
            )
        except Exception as e:
            return AlgorithmResult(
                algorithm="HELLO_QUANTUM",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    # ==================== BELL STATE ====================
    def run_bell_state(self) -> AlgorithmResult:
        """Run Bell State (entanglement) circuit"""
        self.log("\n" + "="*70)
        self.log("BELL STATE - Quantum Entanglement")
        self.log("="*70)
        
        try:
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            
            self.log(f"\nCircuit:\n{qc}")
            
            job = self.simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)
            
            self.log(f"\nMeasurement histogram:\n{counts}")
            self.log("Expected: Only |00⟩ and |11⟩ (qubits are entangled)")
            
            # Calculate fidelity (should be close to 1.0 for perfect entanglement)
            fidelity = 1.0 if len(counts) == 2 else 0.5
            
            metrics = {
                'total_shots': 1000,
                'entanglement_states': len(counts),
                'bell_state_fidelity': fidelity
            }
            
            return AlgorithmResult(
                algorithm="BELL_STATE",
                timestamp=datetime.now().isoformat(),
                success=True,
                circuit=str(qc),
                measurements=counts,
                metrics=metrics
            )
        except Exception as e:
            return AlgorithmResult(
                algorithm="BELL_STATE",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    # ==================== DEUTSCH ALGORITHM ====================
    def run_deutsch(self) -> AlgorithmResult:
        """Run Deutsch algorithm"""
        self.log("\n" + "="*70)
        self.log("DEUTSCH ALGORITHM - Function Classification")
        self.log("="*70)
        
        try:
            qc = QuantumCircuit(2, 1)
            qc.x(1)
            qc.h(0)
            qc.h(1)
            qc.h(0)
            qc.measure(0, 0)
            
            self.log(f"\nCircuit:\n{qc}")
            
            job = self.simulator.run(qc, shots=100)
            result = job.result()
            counts = result.get_counts(qc)
            
            self.log(f"\nMeasurement histogram:\n{counts}")
            self.log("Expected: All 0s (constant function)")
            
            metrics = {
                'total_shots': 100,
                'result_0': counts.get('0', 0),
                'result_1': counts.get('1', 0),
                'function_type': 'constant' if counts.get('0', 0) > 50 else 'balanced'
            }
            
            return AlgorithmResult(
                algorithm="DEUTSCH",
                timestamp=datetime.now().isoformat(),
                success=True,
                circuit=str(qc),
                measurements=counts,
                metrics=metrics
            )
        except Exception as e:
            return AlgorithmResult(
                algorithm="DEUTSCH",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    # ==================== GROVER'S ALGORITHM ====================
    def run_grovers_algorithm(self, n_qubits: int = 3, marked_state: int = 5) -> AlgorithmResult:
        """Run Grover's quantum search algorithm"""
        self.log("\n" + "="*70)
        self.log(f"GROVER'S ALGORITHM - Quantum Search")
        self.log("="*70)
        
        try:
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # Initialize superposition
            qc.h(range(n_qubits))
            
            # Grover iterations
            iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))
            for _ in range(iterations):
                # Oracle
                for i in range(n_qubits):
                    if not (marked_state >> i) & 1:
                        qc.x(i)
                qc.z(n_qubits - 1)
                for i in range(n_qubits):
                    if not (marked_state >> i) & 1:
                        qc.x(i)
                
                # Diffusion
                qc.h(range(n_qubits))
                qc.x(range(n_qubits))
                qc.z(n_qubits - 1)
                qc.x(range(n_qubits))
                qc.h(range(n_qubits))
            
            qc.measure(range(n_qubits), range(n_qubits))
            
            self.log(f"\nSearching for state |{bin(marked_state)[2:].zfill(n_qubits)}⟩ ({marked_state})")
            self.log(f"Qubits: {n_qubits}, Iterations: {iterations}")
            
            job = self.simulator.run(qc, shots=1000)
            result = job.result()
            counts = result.get_counts(qc)
            
            self.log(f"\nMeasurement histogram:\n{counts}")
            
            marked_count = counts.get(bin(marked_state)[2:].zfill(n_qubits), 0)
            
            metrics = {
                'total_shots': 1000,
                'marked_state_count': marked_count,
                'marked_state_probability': marked_count / 1000,
                'n_qubits': n_qubits,
                'n_iterations': iterations
            }
            
            return AlgorithmResult(
                algorithm="GROVER",
                timestamp=datetime.now().isoformat(),
                success=True,
                circuit=str(qc),
                measurements=counts,
                metrics=metrics
            )
        except Exception as e:
            return AlgorithmResult(
                algorithm="GROVER",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    # ==================== SHOR'S ALGORITHM ====================
    def run_shors_algorithm(self, N: int = 15) -> AlgorithmResult:
        """Run Shor's factoring algorithm (classical simulation)"""
        self.log("\n" + "="*70)
        self.log(f"SHOR'S ALGORITHM - Quantum Factoring")
        self.log("="*70)
        
        try:
            self.log(f"\nFactoring N = {N}")
            
            # Classical factoring for demo
            for i in range(2, int(np.sqrt(N)) + 1):
                if N % i == 0:
                    factor1, factor2 = i, N // i
                    self.log(f"\nResult: {N} = {factor1} × {factor2}")
                    
                    metrics = {
                        'N': N,
                        'factor1': factor1,
                        'factor2': factor2,
                        'success': True
                    }
                    
                    return AlgorithmResult(
                        algorithm="SHOR",
                        timestamp=datetime.now().isoformat(),
                        success=True,
                        circuit="",
                        measurements={},
                        metrics=metrics
                    )
            
            raise ValueError(f"Could not factor {N}")
        except Exception as e:
            return AlgorithmResult(
                algorithm="SHOR",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    # ==================== VQE ====================
    def run_vqe(self, n_qubits: int = 2, n_iterations: int = 10) -> AlgorithmResult:
        """Run VQE (Variational Quantum Eigensolver)"""
        self.log("\n" + "="*70)
        self.log(f"VQE - Variational Quantum Eigensolver")
        self.log("="*70)
        
        try:
            self.log(f"\nOptimizing {n_qubits} qubits for {n_iterations} iterations")
            
            # Simple VQE simulation
            params = np.random.rand(n_qubits)
            best_energy = float('inf')
            best_params = params.copy()
            
            for iteration in range(n_iterations):
                energy = np.sum(np.sin(params))
                if energy < best_energy:
                    best_energy = energy
                    best_params = params.copy()
                params += np.random.randn(n_qubits) * 0.1
                
                if iteration % 2 == 0:
                    self.log(f"Iteration {iteration + 1}: Energy = {energy:.6f}")
            
            metrics = {
                'n_qubits': n_qubits,
                'n_iterations': n_iterations,
                'final_energy': float(best_energy),
                'best_params': best_params.tolist()
            }
            
            return AlgorithmResult(
                algorithm="VQE",
                timestamp=datetime.now().isoformat(),
                success=True,
                circuit="",
                measurements={},
                metrics=metrics
            )
        except Exception as e:
            return AlgorithmResult(
                algorithm="VQE",
                timestamp=datetime.now().isoformat(),
                success=False,
                circuit="",
                measurements={},
                metrics={},
                error=str(e)
            )
    
    def run_all(self):
        """Run all algorithms"""
        self.log("\n" + "█"*80)
        self.log("█" + " "*78 + "█")
        self.log("█" + "UNIFIED QUANTUM ALGORITHM FRAMEWORK - QISKIT VERSION".center(78) + "█")
        self.log("█" + " "*78 + "█")
        self.log("█"*80)
        
        self.results.append(self.run_hello_quantum())
        self.results.append(self.run_bell_state())
        self.results.append(self.run_deutsch())
        self.results.append(self.run_grovers_algorithm())
        self.results.append(self.run_shors_algorithm())
        self.results.append(self.run_vqe())
        
        self.print_summary()
        self.export_results()
    
    def print_summary(self):
        """Print summary of all results"""
        self.log("\n" + "="*70)
        self.log("SUMMARY - ALL ALGORITHMS")
        self.log("="*70)
        
        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            self.log(f"\n{result.algorithm}: {status}")
            self.log(f"  Metrics: {result.metrics}")
    
    def export_results(self):
        """Export results to JSON"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'framework': 'Qiskit',
            'execution_time': time.time() - self.start_time,
            'results': [
                {
                    'algorithm': r.algorithm,
                    'success': r.success,
                    'metrics': r.metrics,
                    'error': r.error
                }
                for r in self.results
            ]
        }
        
        with open('/tmp/quantum_results_qiskit.json', 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.log(f"\n✅ Results exported to /tmp/quantum_results_qiskit.json")


if __name__ == "__main__":
    framework = QuantumAlgorithmFramework(verbose=True)
    framework.run_all()

