# Quantum Echoes Algorithm - Quick Start Guide

## What is Quantum Echoes?

Quantum Echoes is an OTOC (Out-of-Time-Order Correlator) based algorithm that demonstrates verifiable quantum advantage. It simulates quantum information scrambling by:

1. Evolving a quantum system forward in time
2. Perturbing a single qubit (butterfly effect)
3. Reversing the evolution
4. Measuring the overlap (OTOC fidelity)

This reveals how quantum systems scramble information - a key metric for quantum advantage.

## Quick Start

### 1. Install Dependencies

```bash
pip install --break-system-packages numpy qiskit qiskit-aer
```

### 2. Run the Demo

```bash
cd /root/Qallow
python3 examples/quantum_echoes_demo.py --n-qubits=3 --t-steps=2 --shots=256
```

**Output:**
```json
{
  "otoc_fidelity": 0.7234,
  "n_qubits": 3,
  "t_steps": 2,
  "perturb_qubit": 0,
  "shots": 256,
  "phase14_ready": false
}
```

### 3. Run Unit Tests

```bash
python3 -m unittest tests.unit.test_quantum_echoes -v
```

**Expected Output:**
```
Ran 18 tests in ~2 seconds
OK
```

## Demo Options

```bash
python3 examples/quantum_echoes_demo.py \
  --n-qubits=5 \           # Number of qubits (default: 5)
  --t-steps=4 \            # Evolution time steps (default: 4)
  --shots=2048 \           # Measurement shots (default: 2048)
  --log=telemetry.csv      # Optional: log telemetry to CSV
```

## Understanding the Output

| Field | Meaning |
|-------|---------|
| `otoc_fidelity` | OTOC value (0-1). Higher = less scrambling |
| `n_qubits` | Number of qubits in system |
| `t_steps` | Evolution time steps |
| `perturb_qubit` | Which qubit was perturbed |
| `shots` | Number of measurement shots |
| `phase14_ready` | Ready for Phase 14 if fidelity ≥ 0.981 |

## Phase 14 Integration

When `phase14_ready = true`, the result is ready for Phase 14 coherence-lattice integration:

```bash
# Phase 14 will use the OTOC fidelity as a gain source
qallow phase 14 --ticks=600 --target_fidelity=0.981
```

## Test Suite Overview

### 18 Comprehensive Tests

**Engine Tests (9 tests)**
- Initialization and basic execution
- Metadata validation
- Error handling (invalid parameters)
- Echo decay curves
- Different qubit counts

**Demo Tests (4 tests)**
- Basic execution
- Phase 14 readiness logic
- Telemetry CSV logging
- Append mode for multiple runs

**Integration Tests (2 tests)**
- Full pipeline execution
- Statistical properties

**Edge Cases (3 tests)**
- Single qubit systems
- Single time step
- Large shot counts

### Run Specific Tests

```bash
# Run only engine tests
python3 -m unittest tests.unit.test_quantum_echoes.TestQuantumEchoesEngine -v

# Run only demo tests
python3 -m unittest tests.unit.test_quantum_echoes.TestQuantumEchoesDemo -v

# Run a single test
python3 -m unittest tests.unit.test_quantum_echoes.TestQuantumEchoesEngine.test_run_quantum_echoes_basic -v
```

## Telemetry Logging

Log results to CSV for Phase 13 ethics monitoring:

```bash
python3 examples/quantum_echoes_demo.py \
  --n-qubits=5 \
  --t-steps=4 \
  --log=data/logs/quantum_echoes.csv
```

**CSV Format:**
```
phase,otoc_fidelity,ticks,ethics_delta
11,0.7234,2048,0.2234
11,0.8156,2048,0.3156
```

## Performance Benchmarks

| Configuration | Runtime | OTOC Fidelity |
|---------------|---------|---------------|
| 3 qubits, 2 steps, 256 shots | ~50ms | 0.1-0.9 |
| 5 qubits, 4 steps, 2048 shots | ~200ms | 0.1-0.8 |
| Full test suite (18 tests) | ~2s | N/A |

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'qiskit'`

**Solution:**
```bash
pip install --break-system-packages qiskit qiskit-aer numpy
```

### Issue: Tests fail with Qiskit errors

**Solution:** Ensure Python 3.10+
```bash
python3 --version
```

### Issue: Low OTOC fidelity (< 0.1)

This is normal! OTOC values depend on:
- Number of qubits (more qubits = lower fidelity)
- Evolution time steps (more steps = lower fidelity)
- Random unitary complexity

To get higher fidelity:
- Use fewer qubits: `--n-qubits=2`
- Use fewer time steps: `--t-steps=1`

## Algorithm Details

### OTOC Formula

```
OTOC = |⟨ψ_initial | ψ_final⟩|²
```

### Echo Decay Pattern

```
OTOC(t) ≈ exp(-λt)
```

Where λ is the chaos exponent (information scrambling rate).

### Phase 14 Readiness Threshold

```
phase14_ready = (otoc_fidelity >= 0.981)
```

## Integration with Qallow Phases

```
Phase 11 (Quantum Echoes)
    ↓
    Computes OTOC fidelity
    ↓
Phase 13 (Ethics Monitoring)
    ↓
    Logs telemetry
    ↓
Phase 14 (Coherence Lattice)
    ↓
    Uses OTOC as gain source
```

## Next Steps

1. **Run the demo**: `python3 examples/quantum_echoes_demo.py`
2. **Run the tests**: `python3 -m unittest tests.unit.test_quantum_echoes -v`
3. **Explore parameters**: Try different `--n-qubits` and `--t-steps` values
4. **Check telemetry**: Look at generated CSV files in `data/logs/`
5. **Integrate with Phase 14**: Use high-fidelity results for coherence lattice tuning

## References

- **Implementation**: `/root/Qallow/examples/quantum_echoes_demo.py`
- **Tests**: `/root/Qallow/tests/unit/test_quantum_echoes.py`
- **Documentation**: `/root/Qallow/tests/unit/README_QUANTUM_ECHOES.md`
- **Willow Demo**: https://youtu.be/mEBCQidaNTQ
- **OTOC Theory**: https://arxiv.org/abs/1405.3289

## Support

For issues or questions:
1. Check the test output: `python3 -m unittest tests.unit.test_quantum_echoes -v`
2. Review the README: `/root/Qallow/tests/unit/README_QUANTUM_ECHOES.md`
3. Check implementation: `/root/Qallow/examples/quantum_echoes_demo.py`

