# Quick Start: Quantum Optimizer

**5-Minute Guide to Using the Quantum Optimizer**

---

## Installation (Optional)

```bash
# Install Bayesian optimization (recommended but not required)
bash scripts/install_quantum_optimizer_deps.sh

# Or manually
pip install bayesian-optimization>=1.4.0
```

**Note**: Works without `bayesian-optimization` using grid search fallback.

---

## Basic Usage

### 1. Import

```python
from python.quantum import QuantumCircuit, QuantumOptimizer
```

### 2. Create Circuit

```python
circuit = QuantumCircuit(num_qubits=16, noise_level=0.01)
```

### 3. Create Optimizer

```python
optimizer = QuantumOptimizer(circuit)
```

### 4. Optimize

```python
result = optimizer.optimize_parameters(init_points=5, n_iter=25)
print(f"Best parameters: {result['best_params']}")
print(f"Best fidelity: {result['best_value']:.6f}")
```

---

## Complete Example

```python
from python.quantum import QuantumCircuit, QuantumOptimizer

# Setup
circuit = QuantumCircuit(num_qubits=16)
optimizer = QuantumOptimizer(circuit)

# Baseline
baseline = circuit.run()
print(f"Baseline error: {baseline['error_rate']:.6f}")

# Optimize
result = optimizer.optimize_parameters(init_points=5, n_iter=25)

# Test optimized
optimized = circuit.run(
    num_layers=result['best_params']['num_layers'],
    entanglement_strength=result['best_params']['entanglement']
)
print(f"Optimized error: {optimized['error_rate']:.6f}")
print(f"Improvement: {(baseline['error_rate'] - optimized['error_rate']) / baseline['error_rate'] * 100:.1f}%")
```

---

## Adaptive Training

```python
# After initial optimization
initial_params = result['best_params']

# Refine parameters
refined = optimizer.adaptive_training(initial_params, n_iter=15)

print(f"Refined parameters: {refined['refined_params']}")
print(f"Improvement: {refined['improvement']:.6f}")
```

---

## Run Demo

```bash
python3 examples/quantum_optimizer_demo.py
```

---

## Run Tests

```bash
python3 tests/test_quantum_optimizer.py
```

---

## API Reference

### QuantumCircuit

```python
circuit = QuantumCircuit(num_qubits=16, noise_level=0.01)

# Run circuit
result = circuit.run(num_layers=5, entanglement_strength=0.5)
# Returns: {'error_rate', 'fidelity', 'qubits_used', 'time_taken'}

# Evaluate performance
performance = circuit.evaluate_performance(5, 0.5)
# Returns: float (0-1, higher is better)

# Reset to defaults
circuit.reset()

# Get optimal parameters
optimal = circuit.get_optimal_params()
# Returns: {'num_layers', 'entanglement_strength'}
```

### QuantumOptimizer

```python
optimizer = QuantumOptimizer(circuit, use_bayesian=True)

# Optimize parameters
result = optimizer.optimize_parameters(
    init_points=5,  # Random initialization points
    n_iter=25,      # Optimization iterations
    acq='ei'        # Acquisition function: 'ei', 'ucb', 'poi'
)
# Returns: {
#   'best_params': {'num_layers', 'entanglement'},
#   'best_value': float,
#   'best_error_rate': float,
#   'iterations': int,
#   'history': list
# }

# Adaptive training
refined = optimizer.adaptive_training(
    initial_params={'num_layers': 5, 'entanglement': 0.5},
    n_iter=15
)
# Returns: {
#   'refined_params': {'num_layers', 'entanglement'},
#   'best_value': float,
#   'best_error_rate': float,
#   'improvement': float
# }

# Get summary
summary = optimizer.get_optimization_summary()
# Returns: str (formatted summary)
```

---

## Parameters

### Circuit Parameters
- **num_qubits**: Number of qubits (default: 16)
- **noise_level**: Base noise level (default: 0.01)
- **num_layers**: Circuit layers (range: 2-10)
- **entanglement_strength**: Entanglement (range: 0.1-0.9)

### Optimizer Parameters
- **init_points**: Random initialization (default: 5)
- **n_iter**: Optimization iterations (default: 25)
- **acq**: Acquisition function (default: 'ei')
  - 'ei': Expected Improvement
  - 'ucb': Upper Confidence Bound
  - 'poi': Probability of Improvement

---

## Troubleshooting

### Import Error: bayes_opt
**Solution**: Install or use grid search fallback (automatic)
```bash
pip install bayesian-optimization
```

### Slow Optimization
**Solution**: Reduce iterations
```python
result = optimizer.optimize_parameters(init_points=3, n_iter=10)
```

### Poor Convergence
**Solution**: Increase iterations or try different acquisition
```python
result = optimizer.optimize_parameters(init_points=10, n_iter=50, acq='ucb')
```

---

## Performance Tips

1. **Start Small**: Use fewer iterations for testing
2. **Use Bayesian**: Install `bayesian-optimization` for best results
3. **Adaptive Training**: Refine after initial optimization
4. **Monitor History**: Check `optimization_history` for insights
5. **Adjust Bounds**: Modify parameter ranges if needed

---

## Example Output

```
Baseline error: 0.156234
Optimized error: 0.067891
Improvement: 56.5%

Best parameters:
  - Num Layers: 7
  - Entanglement: 0.523

Optimization Summary:
  Total Iterations: 30
  Best Fidelity: 0.932109
  Average Time: 0.023s
```

---

## Documentation

- **Full Guide**: `python/quantum/QUANTUM_OPTIMIZER_README.md`
- **Implementation**: `QUANTUM_OPTIMIZER_IMPLEMENTATION.md`
- **Tests**: `tests/test_quantum_optimizer.py`
- **Demo**: `examples/quantum_optimizer_demo.py`

---

## Support

For issues or questions:
1. Check `python/quantum/QUANTUM_OPTIMIZER_README.md`
2. Review `QUANTUM_OPTIMIZER_IMPLEMENTATION.md`
3. Run tests: `python3 tests/test_quantum_optimizer.py`
4. See Feature 004 docs: `specs/004-agi-evolution/`

---

**Quick Start Complete!** 🎉

You're ready to optimize quantum circuits with Bayesian optimization.

