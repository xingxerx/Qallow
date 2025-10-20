#!/usr/bin/env python3
"""
Example: simulate a Bell state circuit locally with Qiskit Aer.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def build_bell_circuit() -> QuantumCircuit:
    """Return a simple Bell state circuit."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def main() -> None:
    circuit = build_bell_circuit()
    simulator = AerSimulator()
    counts = simulator.run(circuit, shots=1024).result().get_counts()
    print("Measurement counts:", counts)


if __name__ == "__main__":
    main()
