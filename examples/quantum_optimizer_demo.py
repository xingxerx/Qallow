#!/usr/bin/env python3
"""
Quantum Optimizer Demo for AGI Evolution Feature 004
Demonstrates Bayesian optimization of quantum circuit parameters

Usage:
    python examples/quantum_optimizer_demo.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.quantum.quantum_circuit import QuantumCircuit
from python.quantum.quantum_optimizer import QuantumOptimizer


def main():
    """Main demonstration function."""
    print("=" * 70)
    print("Quantum Optimizer Demo - AGI Evolution Feature 004")
    print("=" * 70)
    print()
    
    # Step 1: Create quantum circuit
    print("Step 1: Initializing Quantum Circuit")
    print("-" * 70)
    quantum_circuit = QuantumCircuit(num_qubits=16, noise_level=0.01)
    print(f"Created: {quantum_circuit}")
    print()
    
    # Step 2: Test baseline performance
    print("Step 2: Testing Baseline Performance")
    print("-" * 70)
    baseline_result = quantum_circuit.run()
    print(f"Baseline Error Rate: {baseline_result['error_rate']:.6f}")
    print(f"Baseline Fidelity: {baseline_result['fidelity']:.6f}")
    print(f"Execution Time: {baseline_result['time_taken']:.4f}s")
    print()
    
    # Step 3: Create optimizer
    print("Step 3: Creating Quantum Optimizer")
    print("-" * 70)
    optimizer = QuantumOptimizer(quantum_circuit, use_bayesian=True)
    print("Optimizer initialized with Bayesian optimization")
    print()
    
    # Step 4: Run initial optimization
    print("Step 4: Running Initial Optimization")
    print("-" * 70)
    print("Performing Bayesian optimization with 5 init points and 25 iterations...")
    print()
    
    best_params = optimizer.optimize_parameters(init_points=5, n_iter=25, acq='ei')
    
    print("Optimization Results:")
    print(f"  Best Parameters:")
    print(f"    - Num Layers: {best_params['best_params']['num_layers']}")
    print(f"    - Entanglement: {best_params['best_params']['entanglement']:.4f}")
    print(f"  Best Fidelity: {best_params['best_value']:.6f}")
    print(f"  Best Error Rate: {best_params['best_error_rate']:.6f}")
    print(f"  Total Iterations: {best_params['iterations']}")
    print()
    
    # Step 5: Adaptive training (refinement)
    print("Step 5: Adaptive Training (Parameter Refinement)")
    print("-" * 70)
    print("Refining parameters with adaptive training...")
    print()
    
    refined_params = optimizer.adaptive_training(
        initial_params=best_params['best_params'],
        n_iter=15
    )
    
    print("Refinement Results:")
    print(f"  Refined Parameters:")
    print(f"    - Num Layers: {refined_params['refined_params']['num_layers']}")
    print(f"    - Entanglement: {refined_params['refined_params']['entanglement']:.4f}")
    print(f"  Refined Fidelity: {refined_params['best_value']:.6f}")
    print(f"  Refined Error Rate: {refined_params['best_error_rate']:.6f}")
    print(f"  Improvement: {refined_params['improvement']:.6f}")
    print()
    
    # Step 6: Test optimized circuit
    print("Step 6: Testing Optimized Circuit")
    print("-" * 70)
    final_result = quantum_circuit.run(
        num_layers=refined_params['refined_params']['num_layers'],
        entanglement_strength=refined_params['refined_params']['entanglement']
    )
    
    print(f"Final Performance:")
    print(f"  Error Rate: {final_result['error_rate']:.6f}")
    print(f"  Fidelity: {final_result['fidelity']:.6f}")
    print(f"  Execution Time: {final_result['time_taken']:.4f}s")
    print()
    
    # Step 7: Performance comparison
    print("Step 7: Performance Comparison")
    print("-" * 70)
    improvement = (baseline_result['error_rate'] - final_result['error_rate']) / baseline_result['error_rate'] * 100
    print(f"Baseline Error Rate: {baseline_result['error_rate']:.6f}")
    print(f"Optimized Error Rate: {final_result['error_rate']:.6f}")
    print(f"Error Reduction: {improvement:.2f}%")
    print()
    
    # Step 8: Optimization summary
    print("Step 8: Optimization Summary")
    print("-" * 70)
    print(optimizer.get_optimization_summary())
    
    # Step 9: Key implementation details
    print("=" * 70)
    print("Key Implementation Details")
    print("=" * 70)
    print("""
1. Bayesian Optimization Framework:
   - Uses BayesianOptimization library with Gaussian Process regression
   - Hyperparameters optimized: num_layers (2-10) and entanglement_strength (0.1-0.9)
   - Expected Improvement (EI) acquisition function for exploration/exploitation

2. Quantum Circuit Interface:
   - run() method accepts optimized parameters
   - _quantum_simulation() implements simplified error model
   - Considers layer count, entanglement, and coherence decay

3. Adaptive Training:
   - adaptive_training() allows incremental parameter refinement
   - Maintains constraints on parameter space during optimization
   - Tighter bounds around initial parameters for fine-tuning

4. Fallback Mechanisms:
   - Grid search if Bayesian optimization unavailable
   - Local search for adaptive training fallback
   - Graceful degradation without external dependencies

5. Performance Metrics:
   - Tracks error rate, fidelity, and execution time
   - Maintains optimization history for analysis
   - Provides summary statistics and improvement metrics
""")
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

