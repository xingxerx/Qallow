#!/usr/bin/env python3
"""
CUDA-Q Quick Start Examples for Qallow
Demonstrates basic quantum circuits and integration with Qallow
"""

import sys
sys.path.insert(0, '/root/Qallow/third_party/cuda-quantum/python')

try:
    import cudaq
    print("✅ CUDA-Q imported successfully!")
except ImportError as e:
    print(f"❌ Failed to import CUDA-Q: {e}")
    print("Run: pip install cuda-quantum")
    sys.exit(1)

# ============================================================================
# Example 1: Bell State (Entanglement)
# ============================================================================
print("\n" + "="*70)
print("Example 1: Bell State (Entanglement)")
print("="*70)

@cudaq.kernel
def bell_state():
    """Create a Bell state (maximally entangled pair)"""
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cx(qubits[0], qubits[1])
    mz(qubits)

try:
    result = cudaq.sample(bell_state, shots=1000)
    print("\nBell State Results (1000 shots):")
    print(result)
    print("\nExpected: ~500 '00' and ~500 '11' (maximally entangled)")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 2: Superposition
# ============================================================================
print("\n" + "="*70)
print("Example 2: Superposition")
print("="*70)

@cudaq.kernel
def superposition():
    """Create equal superposition of all states"""
    qubits = cudaq.qvector(3)
    for q in qubits:
        h(q)
    mz(qubits)

try:
    result = cudaq.sample(superposition, shots=1000)
    print("\nSuperposition Results (1000 shots):")
    print(result)
    print("\nExpected: ~125 counts for each of 8 possible states")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 3: Quantum Phase Estimation
# ============================================================================
print("\n" + "="*70)
print("Example 3: Quantum Phase Estimation")
print("="*70)

@cudaq.kernel
def phase_estimation(angle: float):
    """Estimate phase using quantum circuit"""
    q = cudaq.qvector(2)
    h(q[0])
    rz(angle, q[1])
    cx(q[0], q[1])
    h(q[0])
    mz(q)

try:
    angle = 1.57  # π/2
    result = cudaq.sample(phase_estimation, angle, shots=100)
    print(f"\nPhase Estimation Results (angle={angle}):")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 4: Grover's Algorithm (2-qubit)
# ============================================================================
print("\n" + "="*70)
print("Example 4: Grover's Algorithm")
print("="*70)

@cudaq.kernel
def grovers_algorithm():
    """Simple 2-qubit Grover's algorithm"""
    qubits = cudaq.qvector(2)
    
    # Initialize superposition
    for q in qubits:
        h(q)
    
    # Oracle: mark |11⟩
    z(qubits[0])
    z(qubits[1])
    cx(qubits[0], qubits[1])
    z(qubits[0])
    z(qubits[1])
    
    # Diffusion operator
    for q in qubits:
        h(q)
    for q in qubits:
        x(q)
    cx(qubits[0], qubits[1])
    for q in qubits:
        x(q)
    for q in qubits:
        h(q)
    
    mz(qubits)

try:
    result = cudaq.sample(grovers_algorithm, shots=1000)
    print("\nGrover's Algorithm Results (1000 shots):")
    print(result)
    print("\nExpected: High probability for |11⟩ (marked state)")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 5: Available Targets
# ============================================================================
print("\n" + "="*70)
print("Example 5: Available Quantum Backends")
print("="*70)

try:
    targets = cudaq.get_targets()
    print("\nAvailable CUDA-Q targets:")
    for target in targets:
        print(f"  • {target}")
    
    current = cudaq.get_target()
    print(f"\nCurrent target: {current}")
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Example 6: Parameterized Circuit
# ============================================================================
print("\n" + "="*70)
print("Example 6: Parameterized Circuit")
print("="*70)

@cudaq.kernel
def parameterized_circuit(theta: float, phi: float):
    """Circuit with parameters"""
    q = cudaq.qvector(2)
    ry(theta, q[0])
    rz(phi, q[1])
    cx(q[0], q[1])
    mz(q)

try:
    import math
    theta = math.pi / 4
    phi = math.pi / 3
    result = cudaq.sample(parameterized_circuit, theta, phi, shots=100)
    print(f"\nParameterized Circuit Results (θ={theta:.3f}, φ={phi:.3f}):")
    print(result)
except Exception as e:
    print(f"Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ CUDA-Q Quick Start Complete!")
print("="*70)
print("""
Next steps:
1. Explore more examples in /root/Qallow/third_party/cuda-quantum/examples/
2. Read the documentation: https://nvidia.github.io/cuda-quantum/
3. Integrate CUDA-Q with Qallow phases
4. Build hybrid quantum-classical algorithms

For more information, see: /root/Qallow/CUDA_Q_GUIDE.md
""")

