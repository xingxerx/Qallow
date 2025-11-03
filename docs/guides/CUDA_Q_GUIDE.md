# CUDA-Q Integration Guide for Qallow

## Overview

CUDA-Q is a comprehensive framework for hybrid quantum-classical computing developed by NVIDIA. It's already integrated into your Qallow project as a third-party library at `/root/Qallow/third_party/cuda-quantum`.

## Current Status

✅ **CUDA-Q is available in your codebase** but needs to be built and installed.

### What's Included:
- Full CUDA-Q source code (C++ and Python)
- CMake build system integration
- Examples and documentation
- Multiple quantum backends (CPU, GPU, simulators)
- Support for physical QPUs (IonQ, Quantinuum, etc.)

## Installation & Setup

### Option 1: Build from Source (Recommended)

```bash
cd /root/Qallow
mkdir -p build
cd build

# Configure with CUDA-Q support
cmake .. -DQALLOW_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# Build CUDA-Q
cmake --build . --target cudaq --parallel
```

### Option 2: Install via pip (Simpler)

```bash
# Activate your environment
source /root/Qallow/qiskit-env/bin/activate

# Install CUDA-Q Python package
pip install cuda-quantum
```

## Using CUDA-Q with Python

### Basic Example: Bell State

```python
import cudaq

# Define a quantum kernel
@cudaq.kernel
def bell_state():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cx(qubits[0], qubits[1])
    mz(qubits)

# Run on default simulator
result = cudaq.sample(bell_state, shots=1000)
print(result)
```

### Available Targets

```python
import cudaq

# List all available targets
targets = cudaq.get_targets()
print("Available targets:", targets)

# Set target (default is 'qasm-sim')
cudaq.set_target("qasm-sim")  # CPU simulator
# cudaq.set_target("nvidia-mqpu")  # GPU simulator (if CUDA available)
# cudaq.set_target("ionq")  # Physical IonQ QPU
# cudaq.set_target("quantinuum")  # Physical Quantinuum QPU
```

### VQE (Variational Quantum Eigensolver) Example

```python
import cudaq
from cudaq import spin

cudaq.set_target("qasm-sim")

@cudaq.kernel
def ansatz(theta: float):
    q = cudaq.qvector(2)
    ry(theta, q[0])
    cx(q[0], q[1])

# Define Hamiltonian
hamiltonian = 0.5 * spin.z(0) + 0.5 * spin.z(1)

# Run VQE
result = cudaq.vqe(
    ansatz,
    hamiltonian,
    optimizer=cudaq.optimizers.cobyla(),
    parameter_count=1
)

print(f"Minimum energy: {result.energy}")
```

## Using CUDA-Q with C++

### Basic Example

```cpp
#include <cudaq.h>

int main() {
    // Define a quantum kernel
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(2);
    
    kernel.h(q[0]);
    kernel.cx(q[0], q[1]);
    kernel.mz(q);
    
    // Sample the kernel
    auto result = cudaq::sample(kernel);
    
    for (auto& [bitstring, count] : result) {
        std::cout << bitstring << " : " << count << std::endl;
    }
    
    return 0;
}
```

### Compile with nvq++

```bash
nvq++ -o bell_state bell_state.cpp
./bell_state
```

## Integration with Qallow Phases

### Phase 14: Photonic Integration

CUDA-Q can simulate photonic quantum systems:

```python
import cudaq

# Set photonic target
cudaq.set_target("photonics")

@cudaq.kernel
def photonic_circuit():
    # Photonic operations
    qubits = cudaq.qvector(4)
    # ... photonic gates ...
    mz(qubits)

result = cudaq.sample(photonic_circuit)
```

### Phase 15: AGI Synthesis

Use CUDA-Q for hybrid quantum-classical algorithms:

```python
import cudaq
import numpy as np

@cudaq.kernel
def hybrid_kernel(params: list):
    q = cudaq.qvector(len(params))
    for i, param in enumerate(params):
        ry(param, q[i])
    # Entangle
    for i in range(len(params)-1):
        cx(q[i], q[i+1])
    mz(q)

# Classical optimization loop
for iteration in range(100):
    # Evaluate quantum circuit
    result = cudaq.sample(hybrid_kernel, params)
    # Update params classically
    params = optimize(params, result)
```

## Key CUDA-Q Functions

| Function | Purpose |
|----------|---------|
| `@cudaq.kernel` | Decorator for quantum kernels |
| `cudaq.sample()` | Sample quantum circuit |
| `cudaq.observe()` | Measure expectation values |
| `cudaq.vqe()` | Variational Quantum Eigensolver |
| `cudaq.set_target()` | Select quantum backend |
| `cudaq.get_targets()` | List available backends |
| `cudaq.qvector(n)` | Allocate n qubits |
| `cudaq.make_kernel()` | Create kernel programmatically |

## Available Quantum Gates

```python
# Single-qubit gates
h(qubit)           # Hadamard
x(qubit)           # Pauli-X
y(qubit)           # Pauli-Y
z(qubit)           # Pauli-Z
rx(angle, qubit)   # Rotation X
ry(angle, qubit)   # Rotation Y
rz(angle, qubit)   # Rotation Z
s(qubit)           # S gate
t(qubit)           # T gate

# Multi-qubit gates
cx(control, target)        # CNOT
cy(control, target)        # Controlled-Y
cz(control, target)        # Controlled-Z
swap(qubit1, qubit2)       # SWAP

# Measurement
mz(qubits)         # Measure in Z basis
```

## Performance Tips

1. **Use GPU backends** for larger circuits (>20 qubits)
2. **Batch operations** for multiple circuit evaluations
3. **Set random seed** for reproducibility: `cudaq.set_random_seed(42)`
4. **Use async execution** for parallel runs: `cudaq.sample_async()`

## Troubleshooting

### CUDA-Q not found
```bash
# Ensure it's in your Python path
export PYTHONPATH=/root/Qallow/third_party/cuda-quantum/python:$PYTHONPATH
```

### GPU backend not available
```bash
# Check available targets
python -c "import cudaq; print(cudaq.get_targets())"

# Install cuQuantum for GPU support
pip install cuquantum
```

### Build errors
```bash
# Clean and rebuild
cd /root/Qallow/build
cmake --build . --target clean
cmake --build . --target cudaq --parallel
```

## Resources

- **Official Docs**: https://nvidia.github.io/cuda-quantum/
- **Examples**: `/root/Qallow/third_party/cuda-quantum/examples/`
- **Python API**: `/root/Qallow/third_party/cuda-quantum/python/cudaq/`
- **C++ API**: `/root/Qallow/third_party/cuda-quantum/include/cudaq/`

## Next Steps

1. ✅ Install CUDA-Q (pip or build from source)
2. ✅ Run example quantum circuits
3. ✅ Integrate with Qallow phases
4. ✅ Optimize for your use case

---

**Ready to use CUDA-Q with Qallow!** 🚀

