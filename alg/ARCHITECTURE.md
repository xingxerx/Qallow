# ALG Architecture & Design

## Overview

**ALG** (Quantum Algorithm Optimizer) is a unified command-line tool that orchestrates quantum optimization for Qallow's coherence-lattice integration system.

It implements **QAOA (Quantum Approximate Optimization Algorithm)** with **SPSA (Simultaneous Perturbation Stochastic Approximation)** to automatically tune the control gain parameter (α_eff) used by Phases 14 and 15.

---

## System Architecture

### High-Level Flow

```
User Command
    ↓
main.py (CLI Router)
    ↓
    ├─→ build.py (Dependency Check)
    ├─→ run.py (Optimizer Execution)
    ├─→ test.py (Validation)
    └─→ verify.py (Results Check)
    ↓
qaoa_spsa.py (Quantum Algorithm)
    ↓
Qiskit/AerSimulator
    ↓
/var/qallow/qaoa_gain.json (Output)
```

### Module Responsibilities

#### `main.py`
- **Role**: CLI entry point and command router
- **Responsibilities**:
  - Parse command-line arguments
  - Route to appropriate subcommand
  - Handle errors and exit codes
  - Print usage information

#### `core/build.py`
- **Role**: Dependency management
- **Responsibilities**:
  - Check Python version (3.8+)
  - Verify installed packages (NumPy, SciPy, Qiskit)
  - Install missing dependencies via pip
  - Create output directories

#### `core/run.py`
- **Role**: Optimizer orchestration
- **Responsibilities**:
  - Load or create configuration
  - Invoke QAOA optimizer
  - Handle results export
  - Provide progress feedback

#### `core/test.py`
- **Role**: Internal validation
- **Responsibilities**:
  - Run small test cases (8-node ring)
  - Verify results file exists
  - Check value ranges
  - Report test status

#### `core/verify.py`
- **Role**: Results verification
- **Responsibilities**:
  - Validate JSON structure
  - Check value ranges
  - Verify config consistency
  - Report validation status

#### `qaoa_spsa.py`
- **Role**: Quantum algorithm implementation
- **Responsibilities**:
  - Load Ising model from config
  - Implement QAOA circuit
  - Run SPSA optimizer
  - Map energy to control gain

---

## Quantum Algorithm Details

### QAOA (Quantum Approximate Optimization Algorithm)

**Purpose**: Find low-energy configurations of an Ising Hamiltonian

**Circuit Structure**:
```
|0⟩ ──H── [Cost Layer] ──[Mixer Layer]── Measure
|0⟩ ──H── [Cost Layer] ──[Mixer Layer]── Measure
...
```

**Cost Layer** (depth p):
- Applies phase rotations based on Ising couplings
- Encodes problem structure into quantum state

**Mixer Layer** (depth p):
- Applies X-rotations to explore solution space
- Balances exploration and exploitation

**Parameters**:
- γ (gamma): Cost layer rotation angles
- β (beta): Mixer layer rotation angles

### SPSA (Simultaneous Perturbation Stochastic Approximation)

**Purpose**: Optimize QAOA parameters without computing explicit gradients

**Algorithm**:
```
1. Initialize parameters θ randomly
2. For each iteration:
   a. Generate random perturbation Δ
   b. Evaluate energy at θ + c·Δ and θ - c·Δ
   c. Estimate gradient: g ≈ (E+ - E-) / (2c·Δ)
   d. Update: θ ← θ - a·g
   e. Decrease step sizes: a, c → a/(k+1)^0.602
3. Return best parameters found
```

**Advantages**:
- Requires only 2 function evaluations per iteration
- Robust to noise
- Converges to local minimum

### Energy-to-Gain Mapping

```
Ising Energy (E)
    ↓
Normalize: E_norm = max(0, min(1, -E/100))
    ↓
Map to gain range: α_eff = α_min + E_norm · (α_max - α_min)
    ↓
Control Gain (α_eff)
```

**Interpretation**:
- Lower energy → higher gain (more aggressive tuning)
- Higher energy → lower gain (conservative tuning)

---

## Configuration Format

### JSON Schema

```json
{
  "N": 8,                          // Number of qubits
  "p": 2,                          // QAOA depth (layers)
  "csv_j": "/path/to/topology.csv", // Coupling matrix CSV
  "alpha_min": 0.001,              // Minimum gain
  "alpha_max": 0.01,               // Maximum gain
  "spsa_iterations": 50,           // SPSA iterations
  "spsa_a": 0.1,                   // SPSA step size (a)
  "spsa_c": 0.1                    // SPSA perturbation (c)
}
```

### Topology CSV Format

```csv
# node_i,node_j,coupling_J
0,1,1.0
1,2,1.0
2,3,0.8
3,0,1.2
```

---

## Output Format

### Results JSON

```json
{
  "energy": -9.456789,
  "alpha_eff": 0.006421,
  "iterations": 50,
  "system_size": 8,
  "qaoa_depth": 2,
  "timestamp": "2025-10-23T15:30:45.123456",
  "config_path": "/var/qallow/ising_spec.json"
}
```

---

## Integration Points

### Phase 14 Integration

Phase 14 uses α_eff to control coherence-lattice integration:

```c
// In qallow_phase14.c
double alpha_eff = load_alpha_from_json("/var/qallow/qaoa_gain.json");
for (int tick = 0; tick < ticks; tick++) {
    fidelity += alpha_eff * coherence_update();
}
```

### Phase 15 Integration

Phase 15 uses α_eff for convergence tuning:

```c
// In qallow_phase15.c
double alpha_eff = load_alpha_from_json("/var/qallow/qaoa_gain.json");
score = apply_convergence_filter(score, alpha_eff);
```

---

## Performance Characteristics

### Time Complexity
- **QAOA circuit**: O(p·N²) gates
- **SPSA iterations**: O(iterations·2) circuit evaluations
- **Total**: O(iterations·p·N²)

### Space Complexity
- **Quantum state**: O(2^N) amplitudes
- **Classical storage**: O(N²) for coupling matrix

### Typical Performance
- **System size**: 8-16 qubits
- **Execution time**: 1-5 minutes
- **Memory usage**: ~100 MB
- **Convergence**: 50-100 iterations

---

## Error Handling

### Build Errors
- Missing Python 3.8+
- Failed package installation
- Missing output directory

### Run Errors
- Invalid configuration file
- Missing topology CSV
- Qiskit import failure
- Optimizer divergence

### Verify Errors
- Missing results file
- Invalid JSON format
- Out-of-range values
- Config mismatch

---

## Extension Points

### Adding New Algorithms

1. Create new module in `core/`
2. Implement `exec(args)` function
3. Register in `main.py`

### Custom Topologies

1. Create CSV file with couplings
2. Reference in config: `"csv_j": "/path/to/custom.csv"`
3. Run optimizer

### Custom Gain Mapping

Modify `map_energy_to_gain()` in `qaoa_spsa.py`:

```python
def map_energy_to_gain(energy, alpha_min, alpha_max):
    # Custom mapping logic
    normalized = custom_normalization(energy)
    return alpha_min + normalized * (alpha_max - alpha_min)
```

---

## Testing Strategy

### Unit Tests
- Module imports
- Configuration loading
- Ising energy calculation
- Verify functions

### Integration Tests
- CLI command execution
- End-to-end workflow
- Results validation

### Performance Tests
- Execution time
- Memory usage
- Convergence rate

---

## Deployment

### Installation Methods

1. **From source**:
   ```bash
   cd alg && pip install -e .
   ```

2. **Via CMake**:
   ```bash
   cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
   make install
   ```

3. **Manual**:
   ```bash
   cp alg/main.py /usr/local/bin/alg
   chmod +x /usr/local/bin/alg
   ```

### System Requirements
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- Qiskit 0.39+
- 100 MB disk space
- 1 GB RAM

---

## Future Enhancements

1. **GPU Acceleration**: CUDA support for circuit simulation
2. **Distributed SPSA**: Parallel gradient estimation
3. **Adaptive Depth**: Automatic QAOA depth selection
4. **Hardware Compilation**: Direct quantum hardware execution
5. **Noise Models**: Realistic decoherence simulation

---

## References

- **QAOA**: Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- **SPSA**: Spall, "Multivariate Stochastic Approximation Using Simultaneous Perturbation" (1992)
- **Qiskit**: https://qiskit.org/
- **Ising Model**: https://en.wikipedia.org/wiki/Ising_model

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-23  
**Status**: Production Ready ✓

