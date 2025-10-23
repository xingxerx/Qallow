#!/usr/bin/env python3
"""
Shor's Algorithm - Quantum Factoring Algorithm
Factors large numbers exponentially faster than classical algorithms
"""

import cirq
import numpy as np
from math import gcd
from typing import Tuple


def quantum_phase_estimation(n_qubits: int, phase: float) -> cirq.Circuit:
    """
    Simplified Quantum Phase Estimation circuit
    
    Args:
        n_qubits: Number of counting qubits
        phase: The phase to estimate
    
    Returns:
        Circuit for phase estimation
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Initialize superposition
    circuit.append(cirq.H.on_each(*qubits))
    
    # Apply controlled unitary operations
    for i in range(n_qubits):
        angle = 2 * np.pi * phase * (2 ** i)
        circuit.append(cirq.rz(angle)(qubits[i]))
    
    # Inverse QFT (simplified)
    for i in range(n_qubits):
        circuit.append(cirq.H(qubits[i]))
    
    return circuit


def order_finding_circuit(n_qubits: int, a: int, N: int) -> cirq.Circuit:
    """
    Quantum circuit for order finding (simplified)
    Finds r such that a^r ≡ 1 (mod N)
    
    Args:
        n_qubits: Number of qubits
        a: Base
        N: Modulus
    
    Returns:
        Order finding circuit
    """
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # Initialize superposition
    circuit.append(cirq.H.on_each(*qubits))
    
    # Simplified modular exponentiation simulation
    # In a real implementation, this would be a controlled modular exponentiation
    for i in range(n_qubits):
        circuit.append(cirq.rz(2 * np.pi * (a ** (2 ** i)) / N)(qubits[i]))
    
    # Measure
    circuit.append(cirq.measure(*qubits, key='result'))
    
    return circuit


def classical_order_finding(a: int, N: int) -> int:
    """
    Classical order finding (for demonstration)
    Finds r such that a^r ≡ 1 (mod N)
    """
    r = 1
    result = a % N
    while result != 1:
        result = (result * a) % N
        r += 1
        if r > 1000:  # Safety limit
            return -1
    return r


def shors_algorithm_demo(N: int = 15) -> Tuple[int, int]:
    """
    Simplified Shor's Algorithm demonstration
    
    Args:
        N: Number to factor
    
    Returns:
        Tuple of factors
    """
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              Shor's Algorithm - Quantum Factoring              ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    print(f"Factoring N = {N}\n")
    
    # Step 1: Check if N is even
    if N % 2 == 0:
        print(f"✅ N is even: {N} = 2 × {N // 2}")
        return 2, N // 2
    
    # Step 2: Pick random a
    a = 7  # Fixed for demo
    print(f"Step 1: Pick random a = {a}")
    
    # Step 3: Check gcd(a, N)
    g = gcd(a, N)
    print(f"Step 2: gcd({a}, {N}) = {g}")
    
    if g != 1:
        print(f"✅ Found factor: {g}")
        return g, N // g
    
    # Step 4: Find order (quantum part)
    print(f"\nStep 3: Find order r such that {a}^r ≡ 1 (mod {N})")
    print("        (In real Shor's, this uses Quantum Phase Estimation)")
    
    r = classical_order_finding(a, N)
    print(f"        Order r = {r}")
    
    if r % 2 != 0:
        print("        Order is odd, retry with different a")
        return -1, -1
    
    # Step 5: Compute factors
    print(f"\nStep 4: Compute factors using r = {r}")
    x = pow(a, r // 2, N)
    print(f"        {a}^({r}//2) mod {N} = {x}")
    
    factor1 = gcd(x - 1, N)
    factor2 = gcd(x + 1, N)
    
    print(f"        gcd({x} - 1, {N}) = {factor1}")
    print(f"        gcd({x} + 1, {N}) = {factor2}")
    
    if factor1 > 1 and factor1 < N:
        print(f"\n✅ Found factors: {factor1} × {N // factor1} = {N}")
        return factor1, N // factor1
    elif factor2 > 1 and factor2 < N:
        print(f"\n✅ Found factors: {factor2} × {N // factor2} = {N}")
        return factor2, N // factor2
    else:
        print("\n❌ Failed to find factors")
        return -1, -1


def show_quantum_circuit():
    """Show a simplified quantum circuit for Shor's algorithm"""
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║          Simplified Quantum Phase Estimation Circuit           ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    circuit = quantum_phase_estimation(3, 0.5)
    print(circuit)
    print()


if __name__ == "__main__":
    # Run Shor's algorithm demo
    factor1, factor2 = shors_algorithm_demo(15)
    
    # Show quantum circuit
    show_quantum_circuit()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                  Shor's Algorithm Complete!                   ║")
    print("╚════════════════════════════════════════════════════════════════╝")

