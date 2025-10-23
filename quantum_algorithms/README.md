# Quantum Algorithms with Cirq

This directory contains quantum algorithm implementations using Google's Cirq framework, integrated with the Qallow quantum virtual machine.

## Setup

### Prerequisites
- Python 3.13+
- Virtual environment (venv)

### Installation

```bash
# Run the setup script
bash quantum_algorithms/setup.sh

# Or manually activate the environment
source venv/bin/activate
```

## Available Algorithms

### 1. Hello Quantum (`algorithms/hello_quantum.py`)
Basic introduction to Cirq with fundamental quantum operations:
- Simple quantum circuits
- Bell states (entanglement)
- Deutsch algorithm

**Run:**
```bash
source venv/bin/activate
python3 quantum_algorithms/algorithms/hello_quantum.py
```

### 2. Grover's Algorithm (`algorithms/grovers_algorithm.py`)
Quantum search algorithm that searches unsorted databases in O(√N) time:
- Oracle construction
- Diffusion operator
- Amplitude amplification

**Run:**
```bash
python3 quantum_algorithms/algorithms/grovers_algorithm.py
```

### 3. Shor's Algorithm (`algorithms/shors_algorithm.py`)
Quantum factoring algorithm for breaking RSA encryption:
- Order finding
- Quantum phase estimation
- Classical post-processing

**Run:**
```bash
python3 quantum_algorithms/algorithms/shors_algorithm.py
```

### 4. VQE Algorithm (`algorithms/vqe_algorithm.py`)
Variational Quantum Eigensolver - hybrid quantum-classical algorithm:
- Parameterized ansatz circuits
- Hamiltonian expectation values
- Gradient-based optimization
- H2 molecule ground state energy

**Run:**
```bash
python3 quantum_algorithms/algorithms/vqe_algorithm.py
```

## Integration with Qallow

These quantum algorithms can be integrated with Qallow's quantum bridge:

```python
import cirq
from qallow_quantum_bridge import execute_on_qallow

# Create a Cirq circuit
circuit = cirq.Circuit(
    cirq.H(cirq.LineQubit(0)),
    cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1))
)

# Execute on Qallow
result = execute_on_qallow(circuit, shots=1000)
```

## Key Concepts

### Qubits
- Basic unit of quantum information
- Can be in superposition of |0⟩ and |1⟩
- Measured to get classical bits

### Quantum Gates
- **H (Hadamard)**: Creates superposition
- **X, Y, Z (Pauli)**: Single-qubit rotations
- **CNOT**: Two-qubit entangling gate
- **RZ, RY, RX**: Parameterized rotations

### Quantum Circuits
- Sequence of quantum gates
- Can be simulated classically (for small systems)
- Can be executed on quantum hardware

### Measurement
- Collapses quantum state to classical bits
- Probabilistic outcome based on amplitudes

## Resources

- [Cirq Documentation](https://quantumai.google/cirq)
- [Quantum Computing Basics](https://quantumai.google/learn)
- [Qallow Documentation](../README.md)

## Next Steps

1. Run the example algorithms
2. Modify parameters and observe results
3. Create your own quantum algorithms
4. Integrate with Qallow's quantum bridge
5. Execute on real quantum hardware (via Cirq-Google)

## Contributing

To add new quantum algorithms:

1. Create a new file in `algorithms/`
2. Follow the existing code structure
3. Include docstrings and comments
4. Add examples and test cases
5. Update this README

## License

Same as Qallow project

