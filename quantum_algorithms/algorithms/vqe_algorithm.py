#!/usr/bin/env python3
"""
VQE (Variational Quantum Eigensolver) - Hybrid Quantum-Classical Algorithm
Finds ground state energy of quantum systems
"""

import cirq
import numpy as np
from typing import Callable, List, Tuple


def ansatz_circuit(qubits: List[cirq.Qid], params: np.ndarray) -> cirq.Circuit:
    """
    Parameterized ansatz circuit for VQE
    
    Args:
        qubits: List of qubits
        params: Parameters for rotation gates
    
    Returns:
        Parameterized circuit
    """
    circuit = cirq.Circuit()
    
    # Initial superposition
    circuit.append(cirq.H.on_each(*qubits))
    
    # Parameterized rotations
    for i, qubit in enumerate(qubits):
        if i < len(params):
            circuit.append(cirq.rz(params[i])(qubit))
            circuit.append(cirq.ry(params[i])(qubit))
    
    # Entangling layer
    for i in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    
    return circuit


def hamiltonian_expectation(circuit: cirq.Circuit, 
                           qubits: List[cirq.Qid],
                           hamiltonian: List[Tuple]) -> float:
    """
    Calculate expectation value of Hamiltonian
    
    Args:
        circuit: Quantum circuit
        qubits: List of qubits
        hamiltonian: List of (coefficient, operator_list) tuples
    
    Returns:
        Expectation value
    """
    simulator = cirq.Simulator()
    expectation = 0.0
    
    # For simplicity, measure Z on all qubits
    full_circuit = circuit.copy()
    full_circuit.append(cirq.measure(*qubits, key='result'))
    
    result = simulator.run(full_circuit, repetitions=1000)
    measurements = result.measurements['result']
    
    # Calculate expectation value
    for measurement in measurements:
        # Convert binary to energy
        energy = sum((-1) ** bit for bit in measurement)
        expectation += energy
    
    return expectation / len(measurements)


def vqe_optimization(n_qubits: int = 2, n_iterations: int = 10) -> Tuple[float, np.ndarray]:
    """
    Run VQE optimization
    
    Args:
        n_qubits: Number of qubits
        n_iterations: Number of optimization iterations
    
    Returns:
        Tuple of (minimum_energy, optimal_parameters)
    """
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     VQE (Variational Quantum Eigensolver) - Optimization       ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    qubits = cirq.LineQubit.range(n_qubits)
    
    # Initialize parameters
    params = np.random.randn(n_qubits) * 0.1
    learning_rate = 0.1
    
    print(f"Configuration:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Learning rate: {learning_rate}\n")
    
    energies = []
    
    for iteration in range(n_iterations):
        # Create circuit with current parameters
        circuit = ansatz_circuit(qubits, params)
        
        # Calculate energy
        energy = hamiltonian_expectation(circuit, qubits, [])
        energies.append(energy)
        
        # Parameter update (simplified gradient descent)
        for i in range(len(params)):
            # Numerical gradient
            params_plus = params.copy()
            params_plus[i] += 0.01
            circuit_plus = ansatz_circuit(qubits, params_plus)
            energy_plus = hamiltonian_expectation(circuit_plus, qubits, [])
            
            gradient = (energy_plus - energy) / 0.01
            params[i] -= learning_rate * gradient
        
        if (iteration + 1) % 2 == 0:
            print(f"Iteration {iteration + 1:2d}: Energy = {energy:.6f}")
    
    print(f"\n✅ Optimization complete!")
    print(f"   Final energy: {energies[-1]:.6f}")
    print(f"   Optimal parameters: {params}")
    
    return energies[-1], params


def show_vqe_circuit():
    """Display a VQE ansatz circuit"""
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║              VQE Ansatz Circuit (2 qubits)                    ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    qubits = cirq.LineQubit.range(2)
    params = np.array([0.5, 0.3])
    
    circuit = ansatz_circuit(qubits, params)
    print(circuit)
    print()


def vqe_for_h2():
    """VQE for H2 molecule ground state energy"""
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║        VQE for H2 Molecule Ground State Energy                ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    print("Finding ground state energy of H2 molecule...")
    print("(Simplified demonstration)\n")
    
    energy, params = vqe_optimization(n_qubits=2, n_iterations=5)
    
    print(f"\nH2 Ground State Energy (approximate): {energy:.6f} Hartree")
    print("(Experimental value: -1.174 Hartree)")
    print()


if __name__ == "__main__":
    # Run VQE optimization
    energy, params = vqe_optimization(n_qubits=2, n_iterations=10)
    
    # Show circuit
    show_vqe_circuit()
    
    # VQE for H2
    vqe_for_h2()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    VQE Algorithm Complete!                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")

