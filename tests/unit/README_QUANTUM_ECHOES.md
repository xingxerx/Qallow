# Quantum Echoes Algorithm - Unit Tests

## Overview

This directory contains comprehensive unit tests for the **Quantum Echoes Algorithm** (Phase 11: Quantum-Coherence Pipeline), which implements OTOC (Out-of-Time-Order Correlator) based verifiable quantum advantage from Google's Willow chip demo (October 2025).

## Files

- **`test_quantum_echoes.py`** - Complete unit test suite with 18 tests
- **`../examples/quantum_echoes_demo.py`** - Implementation of the quantum echoes algorithm

## Test Coverage

### 1. Core Engine Tests (`TestQuantumEchoesEngine`)

Tests the `QuantumEchoesEngine` class functionality:

- **`test_engine_initialization`** - Verify engine initializes with correct parameters
- **`test_run_quantum_echoes_basic`** - Test basic quantum echoes execution
- **`test_run_quantum_echoes_metadata`** - Verify correct metadata in results
- **`test_invalid_t_steps`** - Validate error handling for invalid time steps
- **`test_invalid_perturb_qubit`** - Validate error handling for invalid qubit indices
- **`test_echo_decay_curve`** - Test OTOC decay computation over multiple time steps
- **`test_echo_decay_monotonic_trend`** - Verify statistical decay trend
- **`test_different_qubit_counts`** - Test with 2, 3, and 4 qubit systems
- **`test_reproducibility_with_seed`** - Verify engine initialization consistency

### 2. Demo Function Tests (`TestQuantumEchoesDemo`)

Tests the `run_quantum_echoes_demo` function:

- **`test_demo_basic_execution`** - Test basic demo execution
- **`test_demo_phase14_readiness_threshold`** - Verify Phase 14 readiness logic (≥0.981 fidelity)
- **`test_demo_telemetry_logging`** - Test CSV telemetry logging
- **`test_demo_telemetry_append_mode`** - Verify telemetry appends to existing files

### 3. Integration Tests (`TestQuantumEchoesIntegration`)

Tests full pipeline integration:

- **`test_full_pipeline_execution`** - Test complete quantum echoes pipeline
- **`test_echo_decay_statistics`** - Verify statistical properties of echo decay

### 4. Edge Case Tests (`TestQuantumEchoesEdgeCases`)

Tests boundary conditions:

- **`test_single_qubit`** - Test with 1-qubit system
- **`test_single_time_step`** - Test with t_steps=1
- **`test_large_shot_count`** - Test with 4096 measurement shots

## Running the Tests

### Run All Tests

```bash
cd /root/Qallow
python3 -m unittest tests.unit.test_quantum_echoes -v
```

### Run Specific Test Class

```bash
python3 -m unittest tests.unit.test_quantum_echoes.TestQuantumEchoesEngine -v
```

### Run Specific Test

```bash
python3 -m unittest tests.unit.test_quantum_echoes.TestQuantumEchoesEngine.test_run_quantum_echoes_basic -v
```

## Test Results

All 18 tests pass successfully:

```
Ran 18 tests in ~2 seconds

OK
```

## Key Test Assertions

### Fidelity Validation
- All OTOC fidelity values are in range [0.0, 1.0]
- Fidelity represents quantum state overlap after echo protocol

### Phase 14 Readiness
- Fidelity ≥ 0.981 → `phase14_ready = True`
- Fidelity < 0.981 → `phase14_ready = False`

### Telemetry Logging
- CSV files created with headers: `phase`, `otoc_fidelity`, `ticks`, `ethics_delta`
- Multiple runs append to existing files
- Ethics delta = `otoc_fidelity - 0.5` (bias toward harmony)

### Error Handling
- `ValueError` raised for `t_steps < 1`
- `ValueError` raised for `perturb_qubit >= n_qubits`

## Algorithm Details

### Quantum Echoes Protocol

1. **Forward Evolution**: Apply random unitaries U(t) to initial state |ψ⟩
2. **Perturbation**: Flip butterfly qubit with Pauli-X
3. **Backward Evolution**: Apply U†(-t) to reverse dynamics
4. **Measurement**: Calculate OTOC as fidelity overlap

### OTOC Calculation

```
OTOC = |⟨ψ_initial | ψ_final⟩|²
```

Where:
- `ψ_initial` = |00...0⟩ (initial state)
- `ψ_final` = final state after echo protocol

### Echo Decay

OTOC typically decays exponentially with time steps:
```
OTOC(t) ≈ exp(-λt)
```

Where λ is the chaos exponent (information scrambling rate).

## Dependencies

- `numpy` - Numerical computations
- `qiskit` - Quantum circuit framework
- `qiskit-aer` - Quantum simulator backend

Install with:
```bash
pip install numpy qiskit qiskit-aer
```

## Integration with Qallow

### Phase 11 Integration
The quantum echoes algorithm is part of Phase 11: Quantum-Coherence Pipeline.

### Phase 14 Handoff
Results with fidelity ≥ 0.981 are ready for Phase 14 coherence-lattice integration:
```python
if otoc_echo >= 0.981:
    # Ready for Phase 14 QAOA tuner
    phase14_ready = True
```

### Telemetry
Results are logged to CSV for Phase 13 closed-loop ethics monitoring:
```
phase,otoc_fidelity,ticks,ethics_delta
11,0.7234,2048,0.2234
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Test Suite Runtime | ~2 seconds |
| Single Test Runtime | 50-200 ms |
| Typical OTOC Fidelity | 0.1-0.9 |
| Phase 14 Threshold | ≥ 0.981 |

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError: No module named 'qiskit'`:
```bash
pip install --break-system-packages qiskit qiskit-aer numpy
```

### Test Failures
If tests fail with Qiskit errors, ensure you're using Python 3.10+:
```bash
python3 --version
```

## Future Enhancements

- [ ] Add noise models for realistic quantum hardware simulation
- [ ] Implement SWAP test for direct OTOC measurement
- [ ] Add benchmarks comparing classical vs quantum performance
- [ ] Integrate with IBM Quantum hardware backend
- [ ] Add visualization of echo decay curves

## References

- Google Willow Chip Demo: https://youtu.be/mEBCQidaNTQ
- OTOC Theory: https://arxiv.org/abs/1405.3289
- Qallow Phase 11 Docs: `/root/Qallow/docs/`

