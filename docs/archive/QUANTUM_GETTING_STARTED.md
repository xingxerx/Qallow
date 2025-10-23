# Getting Started with Quantum Algorithm Development

## 5-Minute Quick Start

### 1. Activate Environment
```bash
cd /root/Qallow
source venv/bin/activate
```

### 2. Run Your First Algorithm
```bash
python3 quantum_algorithms/algorithms/hello_quantum.py
```

### 3. Explore Results
```bash
# Run Grover's Algorithm
python3 quantum_algorithms/algorithms/grovers_algorithm.py

# Run Shor's Algorithm
python3 quantum_algorithms/algorithms/shors_algorithm.py

# Run VQE
python3 quantum_algorithms/algorithms/vqe_algorithm.py
```

---

## Understanding the Basics

### What is a Quantum Circuit?

A quantum circuit is a sequence of quantum gates applied to qubits:

```python
import cirq

# Create qubits
q0, q1 = cirq.LineQubit.range(2)

# Build circuit
circuit = cirq.Circuit(
    cirq.H(q0),              # Hadamard gate
    cirq.CNOT(q0, q1),       # CNOT gate
    cirq.measure(q0, q1, key='result')  # Measurement
)

# Print circuit
print(circuit)
```

### Key Quantum Concepts

**Superposition**: A qubit can be in multiple states simultaneously
```python
cirq.H(qubit)  # Creates superposition
```

**Entanglement**: Qubits can be correlated
```python
cirq.CNOT(q0, q1)  # Entangles q0 and q1
```

**Measurement**: Collapses quantum state to classical bits
```python
cirq.measure(qubit, key='result')
```

---

## Your First Custom Algorithm

### Step 1: Create a New File

```bash
cat > quantum_algorithms/algorithms/my_first_algorithm.py << 'EOF'
#!/usr/bin/env python3
"""My First Quantum Algorithm"""

import cirq

def my_algorithm():
    # Create qubits
    q0, q1, q2 = cirq.LineQubit.range(3)
    
    # Build circuit
    circuit = cirq.Circuit(
        cirq.H.on_each(q0, q1, q2),  # Superposition
        cirq.CNOT(q0, q1),            # Entangle
        cirq.CNOT(q1, q2),            # Entangle
        cirq.measure(q0, q1, q2, key='result')
    )
    
    print("My First Quantum Circuit:")
    print(circuit)
    print()
    
    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    print("Results:")
    print(result.histogram(key='result'))

if __name__ == "__main__":
    my_algorithm()
EOF
```

### Step 2: Run It

```bash
source venv/bin/activate
python3 quantum_algorithms/algorithms/my_first_algorithm.py
```

### Step 3: Modify and Experiment

Try changing:
- Number of qubits
- Gate sequences
- Measurement basis
- Number of repetitions

---

## Common Quantum Gates

### Single-Qubit Gates

```python
cirq.H(q)      # Hadamard - creates superposition
cirq.X(q)      # Pauli-X - bit flip
cirq.Y(q)      # Pauli-Y - bit and phase flip
cirq.Z(q)      # Pauli-Z - phase flip
cirq.S(q)      # S gate - phase gate
cirq.T(q)      # T gate - π/8 gate
cirq.rx(angle)(q)  # Rotation around X
cirq.ry(angle)(q)  # Rotation around Y
cirq.rz(angle)(q)  # Rotation around Z
```

### Two-Qubit Gates

```python
cirq.CNOT(q0, q1)   # Controlled-NOT
cirq.CZ(q0, q1)     # Controlled-Z
cirq.SWAP(q0, q1)   # Swap qubits
cirq.XX(q0, q1)     # XX interaction
cirq.YY(q0, q1)     # YY interaction
cirq.ZZ(q0, q1)     # ZZ interaction
```

### Multi-Qubit Gates

```python
cirq.CCX(q0, q1, q2)  # Toffoli (controlled-controlled-X)
cirq.CCZ(q0, q1, q2)  # Controlled-controlled-Z
```

---

## Simulating Circuits

### Basic Simulation

```python
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
print(result)
```

### Running Multiple Shots

```python
result = simulator.run(circuit, repetitions=1000)
histogram = result.histogram(key='result')
print(histogram)
```

### Getting Measurement Results

```python
result = simulator.run(circuit, repetitions=100)
measurements = result.measurements['result']
print(measurements)  # Array of measurement outcomes
```

---

## Debugging Tips

### Print Circuit

```python
print(circuit)  # Shows circuit diagram
```

### Check Circuit Depth

```python
print(f"Circuit depth: {len(circuit)}")
```

### Validate Circuit

```python
# Check if circuit is valid
try:
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
except Exception as e:
    print(f"Error: {e}")
```

### Add Comments

```python
circuit = cirq.Circuit(
    cirq.H(q0),  # Create superposition
    cirq.CNOT(q0, q1),  # Entangle qubits
    cirq.measure(q0, q1, key='result')  # Measure
)
```

---

## Next Steps

### 1. Understand Existing Algorithms
- Read through `hello_quantum.py`
- Understand each gate and operation
- Run with different parameters

### 2. Implement QAOA
- Quantum Approximate Optimization Algorithm
- Great for learning parameterized circuits
- Useful for optimization problems

### 3. Create Benchmarks
- Compare quantum vs classical
- Measure performance
- Analyze results

### 4. Integrate with Qallow
- Learn Qallow architecture
- Create quantum bridge
- Execute on Qallow backend

---

## Resources

### Documentation
- Cirq Docs: https://quantumai.google/cirq
- Quantum Computing Basics: https://quantumai.google/learn
- QuTiP Docs: http://qutip.org/

### Local Files
- `/root/Qallow/quantum_algorithms/README.md`
- `/root/Qallow/QUANTUM_SETUP_GUIDE.md`
- `/root/Qallow/quantum_algorithms/algorithms/`

### Learning Resources
- IBM Quantum: https://quantum-computing.ibm.com/
- Qiskit Textbook: https://qiskit.org/textbook/
- Quantum Computing Playground: https://www.quantum-playground.com/

---

## Troubleshooting

### Import Error
```bash
source venv/bin/activate
pip install cirq cirq-google
```

### Circuit Too Large
- Reduce number of qubits
- Use fewer gates
- Consider approximations

### Simulation Slow
- Reduce repetitions
- Use fewer qubits
- Optimize circuit

### Results Unexpected
- Check gate order
- Verify qubit indices
- Print circuit diagram
- Add debug measurements

---

## Quick Reference

### Create Qubits
```python
q = cirq.LineQubit(0)           # Single qubit
qubits = cirq.LineQubit.range(5)  # 5 qubits
```

### Build Circuit
```python
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)
```

### Simulate
```python
simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
```

### Get Results
```python
histogram = result.histogram(key='result')
print(histogram)
```

---

**Ready to start? Run this:**
```bash
cd /root/Qallow
source venv/bin/activate
python3 quantum_algorithms/algorithms/hello_quantum.py
```

**Happy quantum computing! 🚀**

