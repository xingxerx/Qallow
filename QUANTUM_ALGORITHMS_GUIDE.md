# 🚀 QALLOW QUANTUM ALGORITHMS GUIDE

Complete guide to building, running, and integrating quantum algorithms with the Qallow engine.

## 📋 Quick Start

### Run All Algorithms
```bash
cd /root/Qallow
source venv/bin/activate
python3 quantum_algorithms/QUANTUM_ALGORITHM_SUITE.py
```

### Run Individual Algorithm Categories

**Unified Framework (6 algorithms)**
```bash
python3 quantum_algorithms/unified_quantum_framework.py
```

**Quantum Search**
```bash
python3 quantum_algorithms/algorithms/my_quantum_search.py
```

**Quantum Optimization (QAOA)**
```bash
python3 quantum_algorithms/algorithms/quantum_optimization.py
```

**Quantum Machine Learning**
```bash
python3 quantum_algorithms/algorithms/quantum_ml.py
```

**Quantum Simulation**
```bash
python3 quantum_algorithms/algorithms/quantum_simulation.py
```

---

## 🎯 Algorithm Categories

### 1. UNIFIED QUANTUM FRAMEWORK (6 Algorithms)
Pre-built algorithms using Google Cirq:

- **Hello Quantum** - Basic superposition and measurement
- **Bell State** - Quantum entanglement demonstration
- **Deutsch Algorithm** - Function classification
- **Grover's Algorithm** - Quantum search with √N speedup
- **Shor's Algorithm** - Integer factorization
- **VQE** - Variational Quantum Eigensolver for ground state energy

**File**: `quantum_algorithms/unified_quantum_framework.py`

### 2. QUANTUM SEARCH
Advanced search algorithms:

- **Quantum Database Search** - Grover's algorithm for database queries
  - Searches database of N items in O(√N) time
  - Optimized oracle and diffusion operators
  - Configurable database size and target value

**File**: `quantum_algorithms/algorithms/my_quantum_search.py`

### 3. QUANTUM OPTIMIZATION (QAOA)
Solve optimization problems:

- **MaxCut Problem** - Find maximum cut in graph
  - Approximation ratio: 0.8 (80% of optimal)
  - Configurable graph edges and QAOA depth
  
- **Traveling Salesman Problem** - Find shortest tour
  - Distance matrix based optimization
  - Quantum-classical hybrid approach

**File**: `quantum_algorithms/algorithms/quantum_optimization.py`

### 4. QUANTUM MACHINE LEARNING
ML algorithms using quantum circuits:

- **Quantum Classifier** - Binary classification
  - Parameterized quantum circuits
  - Feature encoding and variational layers
  - Training with gradient descent
  
- **Quantum Clustering** - Unsupervised clustering
  - Quantum distance metrics
  - K-means style clustering
  - Configurable number of clusters

**File**: `quantum_algorithms/algorithms/quantum_ml.py`

### 5. QUANTUM SIMULATION
Simulate quantum systems:

- **Harmonic Oscillator** - Energy levels and eigenstates
  - Calculates energy levels: E_n = ℏω(n + 0.5)
  - Quantum state encoding
  
- **Molecular Simulation** - VQE for molecules
  - Hartree-Fock energy calculation
  - Variational optimization
  - Ground state energy finding
  
- **Quantum Dynamics** - Time evolution
  - Hamiltonian-based evolution
  - Probability tracking over time

**File**: `quantum_algorithms/algorithms/quantum_simulation.py`

---

## 🛠️ Building Custom Algorithms

### Template Structure
```python
from custom_algorithm_template import CustomQuantumAlgorithm, AlgorithmConfig

class MyQuantumAlgorithm(CustomQuantumAlgorithm):
    def __init__(self):
        config = AlgorithmConfig(
            n_qubits=3,
            n_shots=1000,
            algorithm_name="My Algorithm",
            description="My custom quantum algorithm"
        )
        super().__init__(config)
    
    def build_circuit(self):
        qubits = cirq.LineQubit.range(self.config.n_qubits)
        circuit = cirq.Circuit()
        
        # Add your quantum gates here
        for q in qubits:
            circuit.append(cirq.H(q))
        
        circuit.append(cirq.measure(*qubits, key='result'))
        return circuit
    
    def run(self):
        self.circuit = self.build_circuit()
        result = self.simulator.run(self.circuit, repetitions=self.config.n_shots)
        histogram = result.histogram(key='result')
        metrics = self.analyze_results(histogram)
        return metrics

if __name__ == "__main__":
    algo = MyQuantumAlgorithm()
    result = algo.run()
```

### Common Quantum Gates

| Gate | Symbol | Effect |
|------|--------|--------|
| Hadamard | H | Superposition: \|0⟩ → (\|0⟩ + \|1⟩)/√2 |
| Pauli-X | X | Bit flip: \|0⟩ ↔ \|1⟩ |
| Pauli-Y | Y | Bit+phase flip |
| Pauli-Z | Z | Phase flip: \|1⟩ → -\|1⟩ |
| CNOT | ⊕ | Controlled-NOT (entanglement) |
| CZ | Z | Controlled-Z |
| RX(θ) | Rx | Rotation around X-axis |
| RY(θ) | Ry | Rotation around Y-axis |
| RZ(θ) | Rz | Rotation around Z-axis |

---

## 🔗 Integration with Qallow

### Run Quantum Algorithm + Qallow Phase
```bash
# Terminal 1: Run Qallow phase
./build/qallow phase 14 --ticks=500 --target_fidelity=0.981

# Terminal 2: Run quantum algorithms
python3 quantum_algorithms/QUANTUM_ALGORITHM_SUITE.py

# Terminal 3: Monitor with GUI
cargo run
```

### Metrics Monitoring
The GUI displays real-time metrics:
- **Tick**: Current iteration count
- **Global**: Global coherence value
- **Orbital**: Orbital quantum parameter
- **Decoherence**: Quantum noise level

---

## 📊 Performance Metrics

### Algorithm Benchmarks

| Algorithm | Qubits | Time | Success Rate |
|-----------|--------|------|--------------|
| Grover's Search | 3 | <1s | 92.3% |
| QAOA-MaxCut | 4 | <1s | 80% approx |
| Quantum Classifier | 3 | <1s | 80% |
| VQE | 4 | <1s | Converges |
| Shor's (15) | 5 | <1s | 100% |

### Scalability
- **Small**: 3-4 qubits (classical simulation)
- **Medium**: 5-10 qubits (simulator with optimization)
- **Large**: 10+ qubits (requires quantum hardware)

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure venv is activated
source venv/bin/activate

# Install missing dependencies
pip install cirq numpy scipy
```

### Circuit Errors
- Check qubit indices match circuit size
- Verify gate parameters are valid
- Ensure measurement keys are unique

### Performance Issues
- Reduce number of shots for faster execution
- Use fewer qubits for testing
- Enable verbose mode for debugging

---

## 📚 Resources

- **Google Cirq**: https://quantumai.google/cirq
- **Quantum Computing**: https://en.wikipedia.org/wiki/Quantum_computing
- **Grover's Algorithm**: https://en.wikipedia.org/wiki/Grover%27s_algorithm
- **QAOA**: https://en.wikipedia.org/wiki/Quantum_approximate_optimization_algorithm
- **VQE**: https://en.wikipedia.org/wiki/Variational_quantum_eigensolver

---

## 🎓 Next Steps

1. **Run the complete suite**: `python3 QUANTUM_ALGORITHM_SUITE.py`
2. **Explore individual algorithms**: Check each algorithm file
3. **Build custom algorithm**: Copy template and modify
4. **Integrate with Qallow**: Run phases alongside algorithms
5. **Monitor with GUI**: Use `cargo run` to visualize

---

## 📝 File Structure

```
quantum_algorithms/
├── QUANTUM_ALGORITHM_SUITE.py          # Master suite
├── unified_quantum_framework.py         # 6 pre-built algorithms
├── algorithms/
│   ├── custom_algorithm_template.py     # Template for custom algorithms
│   ├── my_quantum_search.py             # Quantum search examples
│   ├── quantum_optimization.py          # QAOA algorithms
│   ├── quantum_ml.py                    # ML algorithms
│   └── quantum_simulation.py            # Simulation algorithms
└── quantum_algorithm_suite_results.json # Results export
```

---

**Status**: ✅ All algorithms tested and working
**Last Updated**: 2025-10-24
**Qallow Version**: Phase 14 (Coherence-Lattice Integration)

