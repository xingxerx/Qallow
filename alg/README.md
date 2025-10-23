# ALG - Quantum Algorithm Optimizer for Qallow

**ALG** is a unified command-line tool that orchestrates quantum optimization for Qallow's coherence-lattice integration system.

It implements **QAOA (Quantum Approximate Optimization Algorithm)** with **SPSA (Simultaneous Perturbation Stochastic Approximation)** to automatically tune the control gain parameter used by Phases 14 and 15.

---

## Features

- **Single executable** with four subcommands
- **QAOA + SPSA** quantum optimizer for Ising Hamiltonian minimization
- **Automatic dependency management** (Qiskit, NumPy, SciPy)
- **JSON-based configuration** and results export
- **Built-in testing and validation**
- **Production-ready** with comprehensive error handling

---

## Installation

### From Source

```bash
cd /root/Qallow/alg
pip install -e .
```

This installs the `alg` command globally.

### Manual Setup

```bash
cd /root/Qallow/alg
python3 main.py build
```

---

## Usage

### Commands

#### `alg build`
Checks and installs dependencies (Qiskit, NumPy, SciPy).

```bash
alg build
```

Output:
```
[ALG BUILD] Python 3.10 OK
[ALG BUILD] ✓ numpy 1.23.5
[ALG BUILD] ✓ scipy 1.9.3
[ALG BUILD] ✓ qiskit 0.39.5
[ALG BUILD] ✓ qiskit-aer 0.11.2
[ALG BUILD] ✓ Output directory: /var/qallow
```

#### `alg run`
Executes the quantum optimizer (QAOA + SPSA).

```bash
alg run
alg run --config=/var/qallow/ising_spec.json
```

Output:
```
[ALG RUN] Using config: /var/qallow/ising_spec.json
[ALG RUN] Loaded config: N=8, p=2
[ALG RUN] Starting quantum optimization...
[QAOA] Iteration  10: Energy = -8.234567
[QAOA] Iteration  20: Energy = -9.123456
...
[ALG RUN] ✓ Optimization complete
[ALG RUN] Energy: -9.456789
[ALG RUN] Alpha_eff: 0.006421
[ALG RUN] Output: /var/qallow/qaoa_gain.json
```

#### `alg test`
Runs internal validation on 8-node ring model.

```bash
alg test
alg test --quick
```

Output:
```
[ALG TEST] Running 8-node ring test...
[ALG TEST] ✓ Test passed
[ALG TEST]   Energy: -8.123456
[ALG TEST]   Alpha_eff: 0.005234
```

#### `alg verify`
Validates results JSON and checks value ranges.

```bash
alg verify
```

Output:
```
[ALG VERIFY] Checking: /var/qallow/qaoa_gain.json
[ALG VERIFY] ✓ JSON is valid
[ALG VERIFY] ✓ All required fields present
[ALG VERIFY] ✓ All values within expected ranges
[ALG VERIFY] ✓ Results consistent with config

[ALG VERIFY] Results Summary:
  Energy:      -9.456789
  Alpha_eff:   0.006421
  Iterations:  50
  Timestamp:   2025-10-23T15:30:45.123456
```

---

## Configuration

### Default Configuration

If no config is provided, ALG creates a default 8-node ring topology:

```json
{
  "N": 8,
  "p": 2,
  "csv_j": "/var/qallow/ring8.csv",
  "alpha_min": 0.001,
  "alpha_max": 0.01,
  "spsa_iterations": 50,
  "spsa_a": 0.1,
  "spsa_c": 0.1
}
```

### Custom Configuration

Create `/var/qallow/ising_spec.json`:

```json
{
  "N": 16,
  "p": 3,
  "csv_j": "/var/qallow/custom_topology.csv",
  "alpha_min": 0.001,
  "alpha_max": 0.015,
  "spsa_iterations": 100,
  "spsa_a": 0.15,
  "spsa_c": 0.12
}
```

### Topology CSV Format

`/var/qallow/custom_topology.csv`:

```csv
# node_i,node_j,coupling_J
0,1,1.0
1,2,1.0
2,3,0.8
3,0,1.2
...
```

---

## Output

### Results File

`/var/qallow/qaoa_gain.json`:

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

## Integration with Qallow

### Phase 14 Integration

Use the optimized gain in Phase 14:

```bash
./build/qallow phase 14 \
  --ticks=600 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --gain_alpha=$(jq .alpha_eff /var/qallow/qaoa_gain.json)
```

### Phase 15 Integration

The gain is automatically used by Phase 15 for convergence tuning.

---

## Workflow Example

```bash
# 1. Build and check dependencies
alg build

# 2. Run quantum optimizer
alg run

# 3. Test results
alg test

# 4. Verify output
alg verify

# 5. Use in Qallow
./build/qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```

---

## Architecture

### Module Structure

```
alg/
├── main.py              # CLI entry point
├── core/
│   ├── __init__.py
│   ├── build.py         # Dependency management
│   ├── run.py           # Optimizer execution
│   ├── test.py          # Internal validation
│   └── verify.py        # Results verification
├── qaoa_spsa.py         # QAOA + SPSA implementation
├── setup.py             # Installation script
└── README.md            # This file
```

### Algorithm Flow

```
Config → Load Ising Model → SPSA Optimizer → QAOA Circuit
                                    ↓
                            Measure Energy
                                    ↓
                            Update Parameters
                                    ↓
                            Map Energy → Gain
                                    ↓
                            Output JSON
```

---

## Performance

- **Execution time**: 1-5 minutes (depends on system size and iterations)
- **Memory usage**: ~100 MB (for 8-16 qubits)
- **Accuracy**: Converges to local minimum within 50-100 iterations

---

## Troubleshooting

### Missing Dependencies

```bash
alg build
```

### Invalid Configuration

Check `/var/qallow/ising_spec.json` format and ensure CSV file exists.

### Verification Failures

```bash
alg verify
```

Check that `/var/qallow/qaoa_gain.json` exists and is valid JSON.

---

## References

- **QAOA**: Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
- **SPSA**: Spall, "Multivariate Stochastic Approximation Using Simultaneous Perturbation" (1992)
- **Qiskit**: https://qiskit.org/
- **Qallow**: https://github.com/xingxerx/Qallow

---

## License

MIT License - See LICENSE file for details

---

## Support

For issues or questions:
- GitHub: https://github.com/xingxerx/Qallow/issues
- Email: dev@qallow.io

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-23  
**Status**: Production Ready ✓

