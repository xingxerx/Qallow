# IBM Quantum Platform Workload with CUDA Acceleration and Error Correction

## Overview

This guide provides comprehensive instructions for setting up and running quantum workloads on IBM Quantum Platform with:
- **CUDA Acceleration**: GPU-accelerated quantum state simulation
- **Quantum Error Correction**: Surface code and error mitigation strategies
- **Adaptive Learning**: Automatic parameter optimization based on results
- **Integrated Monitoring**: Real-time performance tracking and analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Quantum Workload System                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ IBM Quantum      │  │ CUDA Simulator   │                 │
│  │ Platform         │  │ (GPU-Accelerated)│                 │
│  └────────┬─────────┘  └────────┬─────────┘                 │
│           │                     │                            │
│           └──────────┬──────────┘                            │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │ Quantum Circuits    │                           │
│           │ - Bell State        │                           │
│           │ - GHZ State         │                           │
│           │ - VQE Ansatz        │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │ Error Correction    │                           │
│           │ - Surface Code      │                           │
│           │ - Syndrome Measure  │                           │
│           │ - Error Mitigation  │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │ Learning System     │                           │
│           │ - Result Analysis   │                           │
│           │ - Adaptive Updates  │                           │
│           │ - Performance Track │                           │
│           └──────────┬──────────┘                           │
│                      │                                       │
│           ┌──────────▼──────────┐                           │
│           │ Output & Monitoring │                           │
│           │ - JSON Results      │                           │
│           │ - Logs              │                           │
│           │ - Metrics           │                           │
│           └─────────────────────┘                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Setup Environment

```bash
cd /root/Qallow
bash scripts/setup_quantum_workload.sh
```

This will:
- Create Python virtual environment
- Install Qiskit and dependencies
- Install CUDA support (if available)
- Create necessary directories

### 2. Configure IBM Quantum (Optional)

```bash
source qiskit-env/bin/activate
python3 << 'EOF'
from qiskit_ibm_runtime import QiskitRuntimeService

# Save your IBM Quantum credentials
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token="YOUR_API_TOKEN",
    instance="YOUR_CRN"
)
EOF
```

Get credentials from: https://quantum.cloud.ibm.com

### 3. Run Quantum Workload

```bash
bash scripts/run_quantum_workload.sh
```

This executes:
1. CUDA benchmark (if available)
2. IBM Quantum workload (Bell state, GHZ state)
3. Learning system analysis
4. Generates results and logs

## Components

### quantum_ibm_workload.py

Main quantum workload executor with:
- **QuantumErrorCorrectionManager**: Surface code encoding and syndrome measurement
- **IBMQuantumWorkload**: Circuit creation, transpilation, and execution
- Support for Bell states, GHZ states, and VQE ansatz

**Key Methods:**
```python
workload = IBMQuantumWorkload(use_simulator=True, cuda_enabled=True)
workload.initialize_backend()

# Create circuits
bell_qc = workload.create_bell_state_circuit()
ghz_qc = workload.create_ghz_state_circuit(n_qubits=10)

# Define observables
observables, labels = workload.define_observables(n_qubits=2)

# Execute
result = workload.execute_workload(bell_qc, observables, shots=1000)

# Analyze
analysis = workload.analyze_results(result, labels)
```

### quantum_cuda_bridge.py

CUDA-accelerated quantum simulator with:
- **CUDAQuantumSimulator**: GPU-accelerated state vector simulation
- **QuantumErrorCorrectionSimulator**: Surface code error correction
- Benchmark utilities for performance testing

**Key Methods:**
```python
sim = CUDAQuantumSimulator(n_qubits=10, use_cuda=True)
sim.apply_hadamard(0)
sim.apply_cnot(0, 1)
measurements = sim.measure_all()
probs = sim.get_probabilities()
```

### quantum_learning_system.py

Adaptive learning system with:
- **QuantumLearningSystem**: Process results and extract learning signals
- **QuantumErrorCorrectionLearner**: Learn optimal error correction strategies
- Automatic parameter optimization

**Key Methods:**
```python
learner = QuantumLearningSystem()
analysis = learner.process_quantum_results(results)
report = learner.get_performance_report()
learner.save_learning_history()
```

## Output Files

### Logs
- `logs/quantum_workload.log` - Main workload execution log
- `logs/cuda_benchmark.log` - CUDA performance benchmark
- `logs/learning_system.log` - Learning system analysis

### Data
- `data/quantum_results/` - Individual quantum execution results (JSON)
- `data/cuda_benchmark.json` - CUDA benchmark metrics
- `data/quantum_learning_history_*.json` - Learning history and performance

### State
- `adapt_state.json` - Current adaptive learning state

## Performance Metrics

### Quantum Metrics
- **Expectation Values**: Mean, std, min, max
- **Measurement Errors**: Error rates and standard deviations
- **Entanglement Score**: Detected entanglement level (0-1)
- **Circuit Depth**: Transpiled circuit depth

### CUDA Metrics
- **State Size**: 2^n_qubits
- **Basis States**: Number of non-zero amplitudes
- **Execution Time**: Gate application and measurement time
- **Memory Usage**: State vector memory requirements

### Learning Metrics
- **Learning Rate**: Adaptive learning rate (0.001-0.1)
- **Human Score**: Feedback score (0-1)
- **Iterations**: Number of completed iterations
- **Trend**: Improving or degrading performance

## Error Correction

### Surface Code
- Code distance: 3 (default)
- Physical qubits: 2*d^2 - 1 = 17
- Logical qubits: 1
- Error threshold: ~1%

### Error Mitigation
- Resilience level: 1 (basic error mitigation)
- Readout error correction
- Dynamical decoupling (optional)

## Advanced Usage

### Custom Circuit

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

# Create custom circuit
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Define observables
obs = [SparsePauliOp("ZZI"), SparsePauliOp("IZZ")]

# Execute
workload = IBMQuantumWorkload()
workload.initialize_backend()
result = workload.execute_workload(qc, obs)
```

### Real QPU Execution

```python
# Use real backend instead of simulator
workload = IBMQuantumWorkload(use_simulator=False)
workload.initialize_backend()  # Connects to least busy QPU
```

### Batch Execution

```python
circuits = [
    workload.create_bell_state_circuit(),
    workload.create_ghz_state_circuit(5),
    workload.create_ghz_state_circuit(10)
]

for qc in circuits:
    result = workload.execute_workload(qc, observables)
    analysis = workload.analyze_results(result, labels)
```

## Troubleshooting

### CUDA Not Found
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- Install cupy: `pip install cupy-cuda11x`

### IBM Quantum Connection Failed
- Check internet connection
- Verify API token: https://quantum.cloud.ibm.com/account
- Use simulator mode: `use_simulator=True`

### Low Entanglement Score
- Increase circuit depth
- Add more CNOT gates
- Use different ansatz

### High Error Rates
- Enable error mitigation: `resilience_level=1`
- Increase shot count
- Use error correction codes

## References

- **IBM Quantum**: https://quantum.cloud.ibm.com
- **Qiskit**: https://qiskit.org
- **CUDA**: https://developer.nvidia.com/cuda-toolkit
- **Surface Codes**: https://arxiv.org/abs/quant-ph/9707002
- **Error Mitigation**: https://arxiv.org/abs/2210.08763

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review output in `data/quantum_results/`
3. Check `adapt_state.json` for learning progress
4. Consult IBM Quantum documentation

