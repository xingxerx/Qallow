#!/usr/bin/env python3
"""
Hello Quantum - Basic Cirq Example
Demonstrates fundamental quantum operations with Cirq
"""

import cirq
import numpy as np


def hello_quantum():
    """Create and run a simple quantum circuit"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              Hello Quantum - Cirq Example                      ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    # Create qubits
    q0, q1, q2 = cirq.LineQubit.range(3)
    print(f"✅ Created 3 qubits: {q0}, {q1}, {q2}\n")

    # Create a simple circuit
    circuit = cirq.Circuit(
        cirq.H(q0),                    # Hadamard on q0
        cirq.CNOT(q0, q1),             # CNOT: q0 controls q1
        cirq.X(q2),                    # Pauli-X on q2
        cirq.measure(q0, q1, q2, key='result')  # Measure all qubits
    )

    print("Circuit:")
    print(circuit)
    print()

    # Simulate the circuit
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    print(f"Simulation result:\n{result}\n")

    # Run multiple times to see statistics
    print("Running 1000 shots:")
    result = simulator.run(circuit, repetitions=1000)
    print(result.histogram(key='result'))
    print()


def bell_state():
    """Create a Bell state (entangled state)"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    Bell State (Entanglement)                   ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    q0, q1 = cirq.LineQubit.range(2)

    # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result')
    )

    print("Bell State Circuit:")
    print(circuit)
    print()

    simulator = cirq.Simulator()
    print("Running 1000 shots:")
    result = simulator.run(circuit, repetitions=1000)
    histogram = result.histogram(key='result')
    print(histogram)
    print("\nNote: You should see only |00⟩ and |11⟩ states (qubits are entangled)")
    print()


def deutsch_algorithm():
    """Deutsch Algorithm - Determine if function is constant or balanced"""
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    Deutsch Algorithm                          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    q0, q1 = cirq.LineQubit.range(2)

    # Test with identity function (constant)
    print("Testing with Identity function (constant):")
    circuit = cirq.Circuit(
        cirq.X(q1),                    # Initialize q1 to |1⟩
        cirq.H(q0),
        cirq.H(q1),
        cirq.I(q0),                    # Identity (constant function)
        cirq.H(q0),
        cirq.measure(q0, key='result')
    )

    print(circuit)
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    histogram = result.histogram(key='result')
    print(f"Result: {histogram}")
    print("Expected: All 0s (constant function)\n")


if __name__ == "__main__":
    hello_quantum()
    bell_state()
    deutsch_algorithm()

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                    Examples Complete!                         ║")
    print("╚════════════════════════════════════════════════════════════════╝")

