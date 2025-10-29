# 🔬 Quantum Algorithms - Complete Guide

**Duration**: 45 minutes | **Difficulty**: Intermediate | **Prerequisites**: 01_getting_started.md

## Overview

Qallow includes 6 quantum algorithms implemented in Python using Cirq:

1. **Hello Quantum** - Basic circuit operations
2. **Bell State** - Quantum entanglement
3. **Deutsch Algorithm** - Function classification
4. **Grover's Algorithm** - Database search
5. **VQE** - Variational quantum eigensolver
6. **QAOA** - Quantum approximate optimization

## Setup

### Activate Python Environment

```bash
cd /root/Qallow
source venv/bin/activate

# Verify Cirq is installed
python3 -c "import cirq; print(cirq.__version__)"
```

### Install Additional Dependencies (if needed)

```bash
pip install cirq numpy scipy
```

## Algorithm 1: Hello Quantum

### Purpose
Demonstrates basic quantum circuit operations and measurements.

### Run

```bash
python3 quantum_algorithms/algorithms/hello_quantum.py
```

### Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║              Hello Quantum - Cirq Example                      ║
╚════════════════════════════════════════════════════════════════╝

✅ Created 3 qubits: q(0), q(1), q(2)

Circuit:
0: ───H───@───M('result')───
          │   │
1: ───────X───M─────────────
              │
2: ───X───────M─────────────

Measurement result: 111
```

### What It Does

1. Creates 3 qubits
2. Applies Hadamard gate (superposition)
3. Applies CNOT gates (entanglement)
4. Measures all qubits
5. Displays results

## Algorithm 2: Bell State

### Purpose
Demonstrates quantum entanglement using Bell states.

### Run

```bash
python3 quantum_algorithms/algorithms/bell_state.py
```

### Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║                    Bell State (Entanglement)                   ║
╚════════════════════════════════════════════════════════════════╝

Bell State Circuit:
0: ───H───@───M('result')───
          │   │
1: ───────X───M─────────────

Running 1000 shots:
Counter({0: 505, 3: 495})

Note: You should see only |00⟩ and |11⟩ states (qubits are entangled)
```

### What It Does

1. Creates 2 qubits
2. Applies Hadamard to first qubit
3. Applies CNOT (creates entanglement)
4. Measures 1000 times
5. Shows only |00⟩ and |11⟩ states (perfect correlation)

## Algorithm 3: Deutsch Algorithm

### Purpose
Determines if a function is constant or balanced.

### Run

```bash
python3 quantum_algorithms/algorithms/deutsch_algorithm.py
```

### Expected Output

```
╔════════════════════════════════════════════════════════════════╗
║                    Deutsch Algorithm                          ║
╚════════════════════════════════════════════════════════════════╝

Testing with Identity function (constant):
Result: Counter({0: 100})
Expected: All 0s (constant function)
```

### What It Does

1. Tests if a function is constant or balanced
2. Uses quantum interference
3. Requires only 1 function evaluation (vs 2 classically)
4. Demonstrates quantum advantage

## Algorithm 4: Grover's Algorithm

### Purpose
Searches unsorted database quadratically faster than classical.

### Run

```bash
python3 quantum_algorithms/algorithms/grovers_algorithm.py
```

### Expected Output

```
Grover's Algorithm - Database Search
Searching for marked element in 8-element database
Marked element: 5

Iterations: 2
Success probability: 0.98
```

### What It Does

1. Creates superposition of all states
2. Marks target element
3. Applies Grover operator
4. Amplifies marked state
5. Measures result

## Algorithm 5: VQE (Variational Quantum Eigensolver)

### Purpose
Finds ground state energy of quantum systems.

### Run

```bash
python3 quantum_algorithms/algorithms/vqe_algorithm.py
```

### Expected Output

```
VQE - Ground State Energy Calculation
Hamiltonian: H = Z0 + Z1 + Z0*Z1

Iteration 1: Energy = -2.234
Iteration 2: Energy = -2.456
...
Final Energy: -2.828 (theoretical: -2.828)
```

### What It Does

1. Defines quantum Hamiltonian
2. Creates parameterized circuit
3. Optimizes parameters classically
4. Evaluates energy quantum mechanically
5. Converges to ground state

## Algorithm 6: QAOA (Quantum Approximate Optimization)

### Purpose
Solves combinatorial optimization problems.

### Run

```bash
python3 alg/main.py run --quick
```

### Expected Output

```
[QAOA] Loading config: /var/qallow/ising_spec.json
[QAOA] System size: N=8
[QAOA] QAOA depth: p=2
[QAOA] Starting SPSA optimization (50 iterations)...
[QAOA] Iteration  10: Energy = -3.990000
[QAOA] Iteration  20: Energy = -3.990000
[QAOA] Iteration  30: Energy = -3.990000
[QAOA] Iteration  40: Energy = -4.286000
[QAOA] Iteration  50: Energy = -4.334000

[QAOA] Optimization complete
[QAOA] Best energy: -4.334000
```

### What It Does

1. Loads Ising problem specification
2. Creates QAOA circuit
3. Uses SPSA for parameter optimization
4. Evaluates energy at each iteration
5. Returns best solution found

## Running All Algorithms

### Quick Run (All Algorithms)

```bash
python3 alg/main.py run --quick
```

### Full Run (More Iterations)

```bash
python3 alg/main.py run
```

### Custom Configuration

Edit `alg/main.py` to customize:
- Number of qubits
- QAOA depth
- Optimization iterations
- Problem specification

## Interpreting Results

### Success Metrics

| Metric | Good | Excellent |
|--------|------|-----------|
| Bell State Correlation | >90% | >99% |
| Deutsch Accuracy | >80% | 100% |
| Grover Success | >90% | >98% |
| VQE Convergence | <1% error | <0.1% error |
| QAOA Energy | Near optimal | Optimal |

### Common Issues

**Low Bell State Correlation**
```bash
# Increase shots
python3 quantum_algorithms/algorithms/bell_state.py --shots=10000
```

**Deutsch Algorithm Fails**
```bash
# Check circuit depth
python3 quantum_algorithms/algorithms/deutsch_algorithm.py --verbose
```

**QAOA Doesn't Converge**
```bash
# Increase iterations
python3 alg/main.py run --iterations=200
```

## Analyzing Results

### View Report

```bash
cat /var/qallow/quantum_report.md
```

### View JSON Results

```bash
cat /var/qallow/quantum_report.json | python3 -m json.tool
```

### Extract Metrics

```python
import json

with open('/var/qallow/quantum_report.json') as f:
    report = json.load(f)
    
print(f"Success Rate: {report['success_rate']}")
print(f"QAOA Energy: {report['qaoa_energy']}")
print(f"Algorithms: {report['algorithms']}")
```

## Advanced: Custom Algorithms

### Create New Algorithm

```python
# quantum_algorithms/algorithms/my_algorithm.py
import cirq

def my_algorithm():
    # Create qubits
    q0, q1 = cirq.LineQubit.range(2)

    # Create circuit
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.measure(q0, q1, key='result')
    )

    # Simulate
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=1000)
    counts = result.measurements['result']

    print(f"Results: {counts}")

if __name__ == '__main__':
    my_algorithm()
```

### Run Custom Algorithm

```bash
python3 quantum_algorithms/algorithms/my_algorithm.py
```

## Integration with Phases

### Run Quantum Algorithms Before Phase 13

```bash
#!/bin/bash
# Run quantum algorithms first
python3 alg/main.py run --quick

# Then run phases
./build/qallow phase 13 --ticks=400
./build/qallow phase 14 --ticks=600
./build/qallow phase 15 --ticks=800
```

### Use Quantum Results in Phases

```bash
# Extract QAOA energy
ENERGY=$(python3 -c "
import json
with open('/var/qallow/quantum_report.json') as f:
    print(json.load(f)['qaoa_energy'])
")

echo "Using QAOA energy: $ENERGY"
```

## 📚 Next Steps

- **Telemetry Analysis**: `04_telemetry_analysis.md`
- **Advanced Quantum**: `docs/QUANTUM_WORKLOAD_GUIDE.md`
- **Custom Algorithms**: `quantum_algorithms/README.md`

---

**Pro Tip**: Combine quantum algorithms with phases for hybrid quantum-classical optimization!

