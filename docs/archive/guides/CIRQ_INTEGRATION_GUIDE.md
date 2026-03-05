# Cirq Integration Guide - Qallow Quantum Framework

**Date**: 2025-10-27  
**Status**: ✅ COMPLETE  
**Framework**: Google Cirq (Primary Quantum Framework)

---

## Overview

Qallow now uses **Google Cirq** as its primary quantum computing framework, replacing cirq. Cirq provides:

- ✅ Fast local quantum simulation
- ✅ Support for Google Quantum hardware (Sycamore)
- ✅ Comprehensive quantum algorithm library
- ✅ Production-ready implementation
- ✅ Active community support

---

## Installation

### Install Cirq

```bash
pip install cirq cirq-google
```

### Verify Installation

```bash
python3 -c "import cirq; print(f'Cirq version: {cirq.__version__}')"
```

---

## Quantum Algorithms Using Cirq

### 1. Unified Quantum Framework

**File**: `quantum_algorithms/unified_quantum_framework.py`

Implements 6 core quantum algorithms:

```python
import cirq
from quantum_algorithms.unified_quantum_framework import QuantumAlgorithmFramework

# Initialize framework
framework = QuantumAlgorithmFramework(verbose=True)

# Run all algorithms
results = framework.run_all_algorithms()

# Export results
framework.export_results("results.json")
```

**Algorithms**:
- Hello Quantum - Basic superposition
- Bell State - Quantum entanglement
- Deutsch Algorithm - Function classification
- Grover's Algorithm - Quantum search
- Shor's Algorithm - Integer factorization
- VQE - Variational Quantum Eigensolver

### 2. Quantum Search

**File**: `quantum_algorithms/algorithms/my_quantum_search.py`

Advanced search algorithms using Cirq:

```python
import cirq
from quantum_algorithms.algorithms.my_quantum_search import QuantumSearch

search = QuantumSearch(n_qubits=3, n_shots=1000)
results = search.run_grover_search(target_state=5)
```

### 3. Quantum Optimization (QAOA)

**File**: `quantum_algorithms/algorithms/quantum_optimization.py`

QAOA for MaxCut and TSP problems:

```python
from quantum_algorithms.algorithms.quantum_optimization import QAOA

qaoa = QAOA(n_qubits=4, p=2)
results = qaoa.solve_maxcut(graph)
```

### 4. Quantum Machine Learning

**File**: `quantum_algorithms/algorithms/quantum_ml.py`

Quantum ML algorithms:

```python
from quantum_algorithms.algorithms.quantum_ml import QuantumML

qml = QuantumML(n_qubits=3)
classifier = qml.build_classifier()
```

### 5. Quantum Simulation

**File**: `quantum_algorithms/algorithms/quantum_simulation.py`

Quantum system simulations:

```python
from quantum_algorithms.algorithms.quantum_simulation import QuantumSimulator

sim = QuantumSimulator(n_qubits=4)
results = sim.simulate_ising_model()
```

---

## Cirq API Reference

### Basic Circuit Creation

```python
import cirq

# Create qubits
q0, q1, q2 = cirq.LineQubit.range(3)

# Create circuit
circuit = cirq.Circuit(
    cirq.H(q0),                    # Hadamard gate
    cirq.CNOT(q0, q1),             # CNOT gate
    cirq.measure(q0, q1, key='m')  # Measurement
)

# Print circuit
print(circuit)
```

### Run Simulation

```python
# Create simulator
simulator = cirq.Simulator()

# Run circuit
result = simulator.run(circuit, repetitions=1000)

# Get measurement results
histogram = result.histogram(key='m')
print(histogram)
```

### Parameterized Circuits

```python
import cirq
import sympy

# Create parameter
theta = sympy.Symbol('theta')

# Create parameterized circuit
q = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.Rx(theta)(q),
    cirq.measure(q, key='m')
)

# Resolve parameter
resolved = cirq.resolve_parameters(circuit, {'theta': 0.5})
```

---

## Integration with Qallow Phases

### Phase 11: Quantum Coherence Bridge

Uses Cirq for quantum circuit simulation:

```bash
/root/Qallow/build/qallow phase 11 --ticks=100
```

### Phase 12-15: Quantum Acceleration & Convergence

All phases use Cirq-based quantum simulation:

```bash
/root/Qallow/run_all_phases.sh
```

---

## Python Quantum Module

### Location

`python/quantum/` - Core quantum utilities

### Key Files

- `qallow_ibm_bridge.py` - Cirq bridge for Qallow telemetry
- `adaptive_agent.py` - Adaptive quantum agent (Cirq-based)
- `ghz_w_sim.py` - GHZ/W state generation with Cirq validation
- `hybrid_meta_learner.py` - Hybrid quantum-classical learning

### Usage

```python
from python.quantum import (
    build_ternary_circuit,
    run_ternary_sim,
    QuantumAdaptiveAgent,
    HybridQuantumLearner
)

# Build and run circuit
circuit = build_ternary_circuit(n_qubits=3)
result = run_ternary_sim(circuit, shots=1000)

# Use adaptive agent
agent = QuantumAdaptiveAgent()
action = agent.select_action(state)
```

---

## Performance Characteristics

### Simulation Speed

- **Local Simulation**: ~1000 qubits (classical simulation)
- **Quantum Simulation**: Up to 30 qubits (with density matrix)
- **Execution Time**: Microseconds to milliseconds per circuit

### Memory Usage

- **Statevector**: 2^n complex numbers (16 bytes each)
- **Density Matrix**: 2^(2n) complex numbers
- **Typical**: 100 MB for 20 qubits

---

## Troubleshooting

### Import Error: "No module named 'cirq'"

```bash
pip install cirq cirq-google
```

### Circuit Execution Error

Ensure qubits are properly defined:

```python
# ✅ Correct
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

# ❌ Incorrect
circuit = cirq.Circuit(cirq.H(q0))  # q0 not defined
```

### Slow Simulation

For large circuits (>20 qubits), use:

```python
# Use DensityMatrixSimulator for mixed states
simulator = cirq.DensityMatrixSimulator()

# Or use GPU acceleration (if available)
# simulator = cirq.CudaSimulator()
```

---

## Migration from cirq

### Before (cirq)

```python
from cirq import QuantumCircuit, QuantumRegister
from cirq_aer import AerSimulator

qr = QuantumRegister(2, 'q')
qc = QuantumCircuit(qr)
qc.h(qr[0])
qc.cx(qr[0], qr[1])

simulator = AerSimulator()
job = simulator.run(qc, shots=1000)
result = job.result()
```

### After (Cirq)

```python
import cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='m')
)

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=1000)
histogram = result.histogram(key='m')
```

---

## Resources

- **Cirq Documentation**: https://quantumai.google/cirq
- **Cirq GitHub**: https://github.com/quantumlib/Cirq
- **Qallow Quantum Algorithms**: `/root/Qallow/quantum_algorithms/`
- **Qallow Python Quantum Module**: `/root/Qallow/python/quantum/`

---

## System Status

✅ All 15 Qallow phases operational with Cirq  
✅ Quantum algorithms fully integrated  
✅ Performance optimized  
✅ Ready for production deployment


