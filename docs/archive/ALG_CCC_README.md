# CCC Algorithm Module - Constraint-Coherence-Cognition

## Overview

The **CCC (Constraint-Coherence-Cognition)** algorithm module is a hybrid quantum-classical optimization framework integrated into Qallow. It combines:

- **Constraint Handling**: Gray code encoding for discrete optimization
- **Coherence Tracking**: Lyapunov exponent estimation via Koopman operators
- **Cognition**: Ethics-aware QAOA with multi-stakeholder governance

## Architecture

### Directory Structure

```
alg_ccc/
├── CMakeLists.txt              # CUDA/C++ build configuration
├── include/
│   └── ccc.hpp                 # Header with GPU entrypoints
├── hamiltonian.cu              # Hamiltonian assembly kernels
├── koopman_cuda.cu             # Koopman operator & Lyapunov estimation
├── qaoa_constraint.py          # QAOA circuit generator (Qiskit)
└── tests/
    └── test_gray.cpp           # Gray code unit tests
```

### Key Components

#### 1. **Header: `include/ccc.hpp`**
- `CCCParams` struct: Configuration (M_modes, b_ctrl, H_horizon, ethics_tau, etc.)
- GPU entrypoints:
  - `fit_koopman_batched()` - Batched Koopman operator fitting
  - `lyap_jacobian_norms()` - Lyapunov exponent estimation
  - `ethics_score_forward()` - Ethics scoring MLP
  - `gray2int_batch()` - Gray code decoding
  - `reward_grad_bits()` - Gradient computation

#### 2. **CUDA Kernels: `hamiltonian.cu`**
- `gray2int()` - CPU/GPU Gray code decoder
- `k_build_cost_coeffs()` - Assemble Hamiltonian σ^z coefficients
- Supports batched execution for multiple problem instances

#### 3. **Koopman Operators: `koopman_cuda.cu`**
- Placeholder implementations (ready for full kernels)
- Batched least-squares for Koopman fitting
- Lyapunov exponent estimation from Jacobian norms
- Ethics scoring via neural network

#### 4. **QAOA Circuit: `qaoa_constraint.py`**
- Generates Qiskit quantum circuits
- Layers: initialization → ethics projector → cost → mixer → measurement
- Exports circuit and parameters to JSON
- Integrates with Qallow via Python bridge

## Building

### Prerequisites
- CUDA 13.0+
- CMake 3.22+
- C++20 compiler
- Python 3.10+ with Qiskit

### Build Steps

```bash
# Configure with CUDA support
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON

# Build alg_ccc library
cmake --build build --target alg_ccc -j$(nproc)

# Build full Qallow with CCC integration
make ACCELERATOR=CPU -j$(nproc)
```

### Verification

```bash
# Test Gray code decoder
./build/alg_ccc/tests/test_gray

# Generate QAOA circuit
python3 alg_ccc/qaoa_constraint.py --alg=ccc --dump-circuit

# Check exported plan
cat data/logs/ccc_plan.json
```

## Usage

### Python QAOA Generator

```bash
# Basic usage
python3 alg_ccc/qaoa_constraint.py --alg=ccc

# With custom config
python3 alg_ccc/qaoa_constraint.py \
  --config=my_config.json \
  --export=output/plan.json \
  --dump-circuit

# Config file format (JSON)
{
  "M": 8,           # Number of mode qubits
  "b": 6,           # Number of control bits
  "H": 4,           # Horizon length
  "alpha": 1.0,     # Lyapunov weight
  "beta": 1.0,      # Reward weight
  "rho": 0.1,       # Constraint penalty
  "gamma": 5.0,     # Cost phase angle
  "eta": 1.0,       # Mixer strength
  "kappa": 0.1,     # Ethics coupling
  "xi": 0.1,        # Decoherence rate
  "ethics_tau": 0.94, # Ethics threshold
  "layers": 2,      # QAOA layers
  "shots": 2048     # Measurement shots
}
```

### C++ Integration

```cpp
#include "ccc.hpp"
using namespace qallow::ccc;

// Prepare data on GPU
float* X_t, *X_tp, *K_out;
// ... allocate and copy data ...

// Fit Koopman operator
fit_koopman_batched(X_t, X_tp, K_out, B, T, d);

// Estimate Lyapunov exponents
lyap_jacobian_norms(J_t, lambda, B, T, d, M);

// Compute ethics score
ethics_score_forward(feats, w, E_out, B, T, F);
```

## Algorithm Details

### Gray Code Encoding

Maps discrete control bits to continuous optimization space:
- `gray2int(0b000) = 0`
- `gray2int(0b001) = 1`
- `gray2int(0b011) = 2`
- `gray2int(0b010) = 3`

Minimizes Hamming distance between adjacent codes → smoother gradients.

### QAOA Circuit Structure

```
1. Initialization: H gates on all qubits
2. Ethics Projector: Flip eth[0] if any ctrl bit = 1
3. Cost Layer: RZ rotations on modes + ctrl
4. Mixer Layer: Conditional RX (ethics-safe)
5. Measurement: Collapse to classical bits
```

### Hamiltonian

```
H = α·Σ_m λ_m σ^z_m + ρ·Σ_b c_b σ^z_b + κ·E(c)
```

Where:
- `λ_m`: Lyapunov exponents (stability)
- `c_b`: Control bits (discrete choices)
- `E(c)`: Ethics score (multi-stakeholder)

## Integration with Qallow

### Quantum Bridge

The CCC module connects to Qallow's quantum bridge:

```
Qallow VM
  ↓
Quantum Bridge (src/mind/quantum_bridge.c)
  ↓
CCC QAOA Generator (Python)
  ↓
Qiskit Simulator / Real Hardware
  ↓
Results → Qallow Ethics Module
```

### Execution Flow

1. **Qallow** calls CCC via Python subprocess
2. **CCC** generates QAOA circuit with current parameters
3. **Qiskit** simulates or runs on real quantum hardware
4. **Results** fed back to Qallow for ethics evaluation
5. **Feedback** updates parameters for next iteration

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | GPU Memory |
|-----------|-----------|-----------|
| Gray decode | O(b log b) | O(B·b) |
| Koopman fit | O(T·d²) | O(B·T·d²) |
| Lyapunov est. | O(T·d·M) | O(B·T·d·M) |
| Ethics score | O(T·F) | O(B·T·F) |

### Typical Parameters

- **Modes (M)**: 8-16 qubits
- **Control bits (b)**: 4-8 bits
- **Horizon (H)**: 2-4 steps
- **Batch size (B)**: 32-256
- **Time steps (T)**: 10-100

## Roadmap

### Phase 2 (Current)
- ✅ Scaffold and build system
- ✅ Gray code decoder
- ✅ QAOA circuit generator
- ⏳ Full Koopman kernels
- ⏳ Lyapunov estimation
- ⏳ Ethics MLP

### Phase 3
- Qallow integration
- Real quantum hardware support
- Distributed execution
- Performance benchmarks

### Phase 4
- Hybrid quantum-classical ML
- Reinforcement learning loop
- Multi-objective optimization
- Hardware-specific compilation

## Testing

### Unit Tests

```bash
# Gray code tests
./build/alg_ccc/tests/test_gray

# Expected output:
# gray2int(0b000) = 0 ✓
# gray2int(0b001) = 1 ✓
# gray2int(0b011) = 2 ✓
# gray2int(0b010) = 3 ✓
```

### Integration Tests

```bash
# Generate circuit
python3 alg_ccc/qaoa_constraint.py --dump-circuit

# Verify JSON export
python3 -m json.tool data/logs/ccc_plan.json

# Check Qiskit integration
python3 -c "from qiskit import QuantumCircuit; print('Qiskit OK')"
```

## Troubleshooting

### Build Issues

**Error**: `CUDA compiler not found`
```bash
# Solution: Install CUDA 13.0+
# Or disable CUDA:
cmake -S . -B build -DQALLOW_ENABLE_CUDA=OFF
```

**Error**: `ccc.hpp: No such file or directory`
```bash
# Solution: Ensure CMakeLists.txt includes alg_ccc
grep "add_subdirectory(alg_ccc)" CMakeLists.txt
```

### Runtime Issues

**Error**: `ModuleNotFoundError: No module named 'qiskit'`
```bash
# Solution: Install Qiskit
source venv/bin/activate
pip install qiskit qiskit-aer
```

**Error**: `Gray code test fails`
```bash
# Solution: Verify Gray code implementation
python3 -c "
def gray2int(g):
    x = g
    while g:
        g >>= 1
        x ^= g
    return x
print(gray2int(0b011))  # Should be 2
"
```

## References

- **Gray Codes**: https://en.wikipedia.org/wiki/Gray_code
- **QAOA**: https://arxiv.org/abs/1411.4028
- **Koopman Operators**: https://arxiv.org/abs/1710.00564
- **Qiskit**: https://qiskit.org/
- **Qallow**: https://github.com/xingxerx/Qallow

## Contributing

To extend CCC:

1. **Add new kernels** in `hamiltonian.cu` or `koopman_cuda.cu`
2. **Update header** in `include/ccc.hpp` with new entrypoints
3. **Add tests** in `tests/`
4. **Update CMakeLists.txt** if adding new files
5. **Document** changes in this README

## License

Same as Qallow project.

---

**Status**: Phase 1 Complete ✅
**Last Updated**: 2025-10-23
**Next**: Generate full CUDA kernels for Phase 2

