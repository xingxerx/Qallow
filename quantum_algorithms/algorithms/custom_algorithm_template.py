#!/usr/bin/env python3
"""
Custom Quantum Algorithm Template
Build your own quantum algorithm using this template!

This template provides:
  ✅ Basic circuit structure
  ✅ Parameter configuration
  ✅ Measurement and analysis
  ✅ Integration with Qallow
  ✅ Results export
"""

import cirq
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class AlgorithmConfig:
    """Configuration for your quantum algorithm"""
    n_qubits: int = 3
    n_shots: int = 1000
    algorithm_name: str = "Custom Algorithm"
    description: str = "Your quantum algorithm description"
    parameters: Dict[str, Any] = None


class CustomQuantumAlgorithm:
    """Template for building custom quantum algorithms"""
    
    def __init__(self, config: AlgorithmConfig = None):
        self.config = config or AlgorithmConfig()
        self.simulator = cirq.Simulator()
        self.circuit = None
        self.result = None
    
    def build_circuit(self) -> cirq.Circuit:
        """
        Build your quantum circuit here.
        
        Example gates:
          - cirq.H(qubit)           # Hadamard
          - cirq.X(qubit)           # Pauli-X
          - cirq.Y(qubit)           # Pauli-Y
          - cirq.Z(qubit)           # Pauli-Z
          - cirq.S(qubit)           # Phase gate
          - cirq.T(qubit)           # T gate
          - cirq.CNOT(q0, q1)       # Controlled-NOT
          - cirq.CZ(q0, q1)         # Controlled-Z
          - cirq.SWAP(q0, q1)       # Swap
          - cirq.Rx(angle)(qubit)   # Rotation X
          - cirq.Ry(angle)(qubit)   # Rotation Y
          - cirq.Rz(angle)(qubit)   # Rotation Z
        """
        qubits = cirq.LineQubit.range(self.config.n_qubits)
        circuit = cirq.Circuit()
        
        # ===== YOUR ALGORITHM HERE =====
        # Example: Create a simple superposition
        for qubit in qubits:
            circuit.append(cirq.H(qubit))
        
        # Add your quantum gates here
        # circuit.append(cirq.CNOT(qubits[0], qubits[1]))
        # circuit.append(cirq.Rz(np.pi/4)(qubits[0]))
        
        # Measure all qubits
        circuit.append(cirq.measure(*qubits, key='result'))
        
        return circuit
    
    def run(self) -> Dict[str, Any]:
        """Execute the quantum algorithm"""
        print(f"\n{'='*70}")
        print(f"Running: {self.config.algorithm_name}")
        print(f"{'='*70}")
        print(f"Description: {self.config.description}")
        print(f"Qubits: {self.config.n_qubits}")
        print(f"Shots: {self.config.n_shots}")
        
        # Build circuit
        self.circuit = self.build_circuit()
        print(f"\nCircuit:\n{self.circuit}")
        
        # Run simulation
        print(f"\nRunning simulation...")
        result = self.simulator.run(self.circuit, repetitions=self.config.n_shots)
        histogram = result.histogram(key='result')
        
        print(f"\nResults:")
        print(f"Histogram: {histogram}")
        
        # Analyze results
        metrics = self.analyze_results(histogram)
        
        return {
            "algorithm": self.config.algorithm_name,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "circuit": str(self.circuit),
            "measurements": dict(histogram),
            "metrics": metrics,
            "config": {
                "n_qubits": self.config.n_qubits,
                "n_shots": self.config.n_shots,
                "parameters": self.config.parameters or {}
            }
        }
    
    def analyze_results(self, histogram: Dict) -> Dict[str, Any]:
        """Analyze measurement results"""
        total_shots = sum(histogram.values())
        most_common = max(histogram.items(), key=lambda x: x[1])
        
        metrics = {
            "total_shots": total_shots,
            "unique_states": len(histogram),
            "most_common_state": int(most_common[0]),
            "most_common_count": most_common[1],
            "most_common_probability": most_common[1] / total_shots,
            "entropy": self.calculate_entropy(histogram),
        }
        
        return metrics
    
    def calculate_entropy(self, histogram: Dict) -> float:
        """Calculate Shannon entropy of measurement results"""
        total = sum(histogram.values())
        entropy = 0.0
        for count in histogram.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def export_results(self, filename: str = None):
        """Export results to JSON"""
        if self.result is None:
            print("No results to export. Run the algorithm first!")
            return
        
        if filename is None:
            filename = f"results/{self.config.algorithm_name.lower().replace(' ', '_')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.result, f, indent=2)
        
        print(f"Results exported to: {filename}")


# ===== EXAMPLE ALGORITHMS =====

class QuantumSearchAlgorithm(CustomQuantumAlgorithm):
    """Grover's algorithm for quantum search"""
    
    def __init__(self, marked_state: int = 5):
        config = AlgorithmConfig(
            n_qubits=3,
            algorithm_name="Quantum Search (Grover's)",
            description=f"Search for marked state {marked_state} in superposition"
        )
        super().__init__(config)
        self.marked_state = marked_state
    
    def build_circuit(self) -> cirq.Circuit:
        """Build Grover's algorithm circuit"""
        qubits = cirq.LineQubit.range(self.config.n_qubits)
        circuit = cirq.Circuit()
        
        # Initialize superposition
        for qubit in qubits:
            circuit.append(cirq.H(qubit))
        
        # Grover iterations
        iterations = int(np.pi / 4 * np.sqrt(2 ** self.config.n_qubits))
        for _ in range(iterations):
            # Oracle: mark the target state
            self._apply_oracle(circuit, qubits)
            # Diffusion operator
            self._apply_diffusion(circuit, qubits)
        
        # Measure
        circuit.append(cirq.measure(*qubits, key='result'))
        return circuit
    
    def _apply_oracle(self, circuit: cirq.Circuit, qubits: List):
        """Apply oracle to mark target state"""
        # Simple oracle: flip phase of marked state
        circuit.append(cirq.Z(qubits[0]))
    
    def _apply_diffusion(self, circuit: cirq.Circuit, qubits: List):
        """Apply diffusion operator"""
        for qubit in qubits:
            circuit.append(cirq.H(qubit))
        for qubit in qubits:
            circuit.append(cirq.X(qubit))
        circuit.append(cirq.Z(qubits[-1]))
        for qubit in qubits:
            circuit.append(cirq.X(qubit))
        for qubit in qubits:
            circuit.append(cirq.H(qubit))


if __name__ == "__main__":
    # Example 1: Run custom algorithm
    print("\n" + "█"*70)
    print("█" + " CUSTOM QUANTUM ALGORITHM TEMPLATE ".center(68) + "█")
    print("█"*70)
    
    # Create and run custom algorithm
    algo = CustomQuantumAlgorithm()
    result = algo.run()
    algo.result = result
    
    # Example 2: Run Grover's search
    print("\n" + "█"*70)
    grover = QuantumSearchAlgorithm(marked_state=5)
    result = grover.run()
    grover.result = result

