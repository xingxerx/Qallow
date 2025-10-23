#!/usr/bin/env python3
"""
Grover's Algorithm - Quantum Search Algorithm
Searches an unsorted database in O(√N) time
"""

import cirq
import numpy as np
from typing import List, Callable


def grover_oracle(qubits: List[cirq.Qid], marked_state: int) -> cirq.Circuit:
    """
    Create an oracle that marks the target state
    
    Args:
        qubits: List of qubits
        marked_state: The state to mark (as integer)
    
    Returns:
        Circuit implementing the oracle
    """
    circuit = cirq.Circuit()
    n = len(qubits)
    
    # Convert marked_state to binary and apply X gates where bit is 0
    for i in range(n):
        if not (marked_state >> i) & 1:
            circuit.append(cirq.X(qubits[i]))
    
    # Multi-controlled Z gate
    if n == 1:
        circuit.append(cirq.Z(qubits[0]))
    elif n == 2:
        circuit.append(cirq.CZ(qubits[0], qubits[1]))
    else:
        # For more qubits, use multi-controlled Z
        circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
    
    # Undo X gates
    for i in range(n):
        if not (marked_state >> i) & 1:
            circuit.append(cirq.X(qubits[i]))
    
    return circuit


def grover_diffusion(qubits: List[cirq.Qid]) -> cirq.Circuit:
    """
    Create the diffusion operator (inversion about average)
    
    Args:
        qubits: List of qubits
    
    Returns:
        Circuit implementing the diffusion operator
    """
    circuit = cirq.Circuit()
    n = len(qubits)
    
    # Apply Hadamard to all qubits
    circuit.append(cirq.H.on_each(*qubits))
    
    # Apply X to all qubits
    circuit.append(cirq.X.on_each(*qubits))
    
    # Multi-controlled Z gate
    if n == 1:
        circuit.append(cirq.Z(qubits[0]))
    elif n == 2:
        circuit.append(cirq.CZ(qubits[0], qubits[1]))
    else:
        circuit.append(cirq.Z(qubits[-1]).controlled_by(*qubits[:-1]))
    
    # Apply X to all qubits
    circuit.append(cirq.X.on_each(*qubits))
    
    # Apply Hadamard to all qubits
    circuit.append(cirq.H.on_each(*qubits))
    
    return circuit


def grovers_algorithm(n_qubits: int, marked_state: int) -> cirq.Circuit:
    """
    Implement Grover's algorithm
    
    Args:
        n_qubits: Number of qubits
        marked_state: The state to search for
    
    Returns:
        Complete Grover circuit
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Initialize superposition
    circuit.append(cirq.H.on_each(*qubits))
    
    # Calculate number of iterations
    n_iterations = int(np.pi / 4 * np.sqrt(2 ** n_qubits))
    
    # Apply Grover operator n_iterations times
    for _ in range(n_iterations):
        circuit.append(grover_oracle(qubits, marked_state))
        circuit.append(grover_diffusion(qubits))
    
    # Measure
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit


def run_grovers():
    """Run Grover's algorithm example"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              Grover's Algorithm - Quantum Search               ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    n_qubits = 3
    marked_state = 5  # Search for state |101⟩
    
    print(f"Searching for state |{bin(marked_state)[2:].zfill(n_qubits)}⟩ ({marked_state})")
    print(f"Using {n_qubits} qubits\n")
    
    # Create circuit
    circuit = grovers_algorithm(n_qubits, marked_state)
    print("Grover's Circuit:")
    print(circuit)
    print()
    
    # Simulate
    simulator = cirq.Simulator()
    print("Running 1000 shots:")
    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')
    
    print(histogram)
    print(f"\n✅ Most frequent result should be {marked_state}")
    print("   (Grover's algorithm amplifies the marked state)")
    print()


if __name__ == "__main__":
    run_grovers()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    Grover's Algorithm Complete!               ║")
    print("╚════════════════════════════════════════════════════════════════╝")

