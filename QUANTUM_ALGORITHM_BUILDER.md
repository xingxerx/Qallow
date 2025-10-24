# 🚀 Qallow Quantum Algorithm Builder

Build and run quantum algorithms on the Qallow quantum engine!

## Quick Start

### 1. Run Existing Algorithms

```bash
# Activate virtual environment
source venv/bin/activate

# Run Hello Quantum
python3 quantum_algorithms/algorithms/hello_quantum.py

# Run Grover's Search
python3 quantum_algorithms/algorithms/grovers_algorithm.py

# Run VQE (Variational Quantum Eigensolver)
python3 quantum_algorithms/algorithms/vqe_algorithm.py

# Run Shor's Algorithm
python3 quantum_algorithms/algorithms/shors_algorithm.py

# Run all algorithms
python3 quantum_algorithms/unified_quantum_framework.py
```

### 2. Build Your Own Algorithm

```bash
# Copy the template
cp quantum_algorithms/algorithms/custom_algorithm_template.py \
   quantum_algorithms/algorithms/my_algorithm.py

# Edit my_algorithm.py and implement build_circuit()

# Run your algorithm
python3 quantum_algorithms/algorithms/my_algorithm.py
```

## Available Quantum Gates

### Single-Qubit Gates
```python
cirq.H(qubit)              # Hadamard - creates superposition
cirq.X(qubit)              # Pauli-X - bit flip
cirq.Y(qubit)              # Pauli-Y
cirq.Z(qubit)              # Pauli-Z - phase flip
cirq.S(qubit)              # Phase gate
cirq.T(qubit)              # T gate
cirq.Rx(angle)(qubit)      # Rotation around X-axis
cirq.Ry(angle)(qubit)      # Rotation around Y-axis
cirq.Rz(angle)(qubit)      # Rotation around Z-axis
```

### Two-Qubit Gates
```python
cirq.CNOT(q0, q1)          # Controlled-NOT
cirq.CZ(q0, q1)            # Controlled-Z
cirq.SWAP(q0, q1)          # Swap qubits
cirq.XX(q0, q1)            # XX interaction
cirq.YY(q0, q1)            # YY interaction
cirq.ZZ(q0, q1)            # ZZ interaction
```

### Measurement
```python
cirq.measure(q0, q1, key='result')  # Measure qubits
```

## Algorithm Examples

### Example 1: Bell State (Entanglement)

```python
def build_circuit(self):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.H(q0),                    # Hadamard on q0
        cirq.CNOT(q0, q1),             # Entangle q0 and q1
        cirq.measure(q0, q1, key='result')
    )
    return circuit
```

### Example 2: Deutsch Algorithm

```python
def build_circuit(self):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.X(q1),                    # Initialize q1 to |1⟩
        cirq.H(q0),
        cirq.H(q1),
        cirq.I(q0),                    # Oracle (identity = constant)
        cirq.H(q0),
        cirq.measure(q0, key='result')
    )
    return circuit
```

### Example 3: Quantum Fourier Transform

```python
def build_circuit(self):
    qubits = cirq.LineQubit.range(3)
    circuit = cirq.Circuit()
    
    # Initialize superposition
    for q in qubits:
        circuit.append(cirq.H(q))
    
    # QFT
    for i, q in enumerate(qubits):
        circuit.append(cirq.H(q))
        for j in range(i+1, len(qubits)):
            angle = 2 * np.pi / (2 ** (j - i + 1))
            circuit.append(cirq.CZPowGate(exponent=angle/np.pi)(qubits[i], qubits[j]))
    
    circuit.append(cirq.measure(*qubits, key='result'))
    return circuit
```

## Integration with Qallow Phases

Run your algorithm with Qallow phases:

```bash
# Phase 14: Coherence-Lattice Integration
./build/qallow phase 14 --ticks=500 --target_fidelity=0.981

# Phase 13: Harmonic Propagation
./build/qallow phase 13 --ticks=400

# Phase 15: Convergence & Lock-In
./build/qallow phase 15 --ticks=300
```

## Monitor with GUI

Run the GUI to monitor your algorithm:

```bash
cargo run
```

Then click "▶️ RUN VM" to execute your algorithm and watch:
- Real-time metrics
- Terminal output
- Fidelity progress
- Coherence levels

## Algorithm Performance Metrics

Your algorithm will report:
- **Total Shots**: Number of measurements
- **Unique States**: Different measurement outcomes
- **Most Common State**: Most frequently measured state
- **Probability**: Probability of most common state
- **Entropy**: Shannon entropy of results

## Advanced: Custom Oracles

For Grover's algorithm, implement custom oracles:

```python
def _apply_oracle(self, circuit, qubits):
    """Mark states where sum of qubits is even"""
    # Apply phase flip to marked states
    circuit.append(cirq.Z(qubits[0]))
    circuit.append(cirq.Z(qubits[1]))
```

## Export Results

Results are automatically exported to JSON:

```python
algo = MyAlgorithm()
result = algo.run()
algo.result = result
algo.export_results("my_results.json")
```

## Troubleshooting

**ImportError: No module named 'cirq'**
```bash
source venv/bin/activate
pip install cirq
```

**Circuit too large**
- Reduce number of qubits
- Simplify circuit
- Use fewer iterations

**Results don't match expected**
- Check oracle implementation
- Verify gate sequence
- Increase number of shots

## Next Steps

1. ✅ Run existing algorithms
2. ✅ Modify template for your problem
3. ✅ Test with small qubit counts
4. ✅ Scale up gradually
5. ✅ Monitor with GUI
6. ✅ Export and analyze results

## Resources

- Cirq Documentation: https://quantumai.google/cirq
- Quantum Computing Basics: https://quantum.ibm.com/
- Qallow Documentation: See README.md

Happy quantum computing! 🎉

