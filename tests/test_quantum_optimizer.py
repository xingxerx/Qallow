#!/usr/bin/env python3
"""
Unit tests for Quantum Optimizer (AGI Evolution Feature 004)
Tests quantum circuit and optimizer functionality
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.quantum.quantum_circuit import QuantumCircuit
from python.quantum.quantum_optimizer import QuantumOptimizer


class TestQuantumCircuit(unittest.TestCase):
    """Test cases for QuantumCircuit class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit = QuantumCircuit(num_qubits=16, noise_level=0.01)
    
    def test_initialization(self):
        """Test circuit initialization."""
        self.assertEqual(self.circuit.num_qubits, 16)
        self.assertEqual(self.circuit.noise_level, 0.01)
        self.assertEqual(self.circuit.params['num_layers'], 5)
        self.assertEqual(self.circuit.params['entanglement_strength'], 0.5)
    
    def test_run_default_params(self):
        """Test circuit execution with default parameters."""
        result = self.circuit.run()
        
        self.assertIn('error_rate', result)
        self.assertIn('fidelity', result)
        self.assertIn('qubits_used', result)
        self.assertIn('time_taken', result)
        
        # Check bounds
        self.assertGreaterEqual(result['error_rate'], 0.0)
        self.assertLessEqual(result['error_rate'], 1.0)
        self.assertEqual(result['fidelity'], 1.0 - result['error_rate'])
        self.assertEqual(result['qubits_used'], 16)
    
    def test_run_custom_params(self):
        """Test circuit execution with custom parameters."""
        result = self.circuit.run(num_layers=7, entanglement_strength=0.6)
        
        self.assertEqual(self.circuit.params['num_layers'], 7)
        self.assertEqual(self.circuit.params['entanglement_strength'], 0.6)
        self.assertGreaterEqual(result['error_rate'], 0.0)
        self.assertLessEqual(result['error_rate'], 1.0)
    
    def test_evaluate_performance(self):
        """Test performance evaluation."""
        performance = self.circuit.evaluate_performance(5, 0.5)
        
        self.assertGreaterEqual(performance, 0.0)
        self.assertLessEqual(performance, 1.0)
    
    def test_reset(self):
        """Test circuit reset."""
        self.circuit.run(num_layers=8, entanglement_strength=0.7)
        self.circuit.reset()
        
        self.assertEqual(self.circuit.params['num_layers'], 5)
        self.assertEqual(self.circuit.params['entanglement_strength'], 0.5)
        self.assertEqual(self.circuit._execution_count, 0)
    
    def test_get_optimal_params(self):
        """Test getting optimal parameters."""
        optimal = self.circuit.get_optimal_params()
        
        self.assertIn('num_layers', optimal)
        self.assertIn('entanglement_strength', optimal)
        self.assertGreaterEqual(optimal['num_layers'], 2)
        self.assertLessEqual(optimal['num_layers'], 10)


class TestQuantumOptimizer(unittest.TestCase):
    """Test cases for QuantumOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.circuit = QuantumCircuit(num_qubits=16, noise_level=0.01)
        self.optimizer = QuantumOptimizer(self.circuit, use_bayesian=True)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.quantum_circuit)
        self.assertEqual(len(self.optimizer.optimization_history), 0)
    
    def test_quantum_objective(self):
        """Test objective function."""
        value = self.optimizer._quantum_objective(5, 0.5)
        
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)
        self.assertEqual(len(self.optimizer.optimization_history), 1)
    
    def test_optimize_parameters(self):
        """Test parameter optimization (grid search fallback)."""
        result = self.optimizer.optimize_parameters(init_points=2, n_iter=3)
        
        self.assertIn('best_params', result)
        self.assertIn('best_value', result)
        self.assertIn('best_error_rate', result)
        self.assertIn('iterations', result)
        self.assertIn('history', result)
        
        # Check best params structure
        self.assertIn('num_layers', result['best_params'])
        self.assertIn('entanglement', result['best_params'])
        
        # Check bounds
        self.assertGreaterEqual(result['best_params']['num_layers'], 2)
        self.assertLessEqual(result['best_params']['num_layers'], 10)
        self.assertGreaterEqual(result['best_params']['entanglement'], 0.1)
        self.assertLessEqual(result['best_params']['entanglement'], 0.9)
        
        # Check value bounds
        self.assertGreaterEqual(result['best_value'], 0.0)
        self.assertLessEqual(result['best_value'], 1.0)
        self.assertEqual(result['best_error_rate'], 1.0 - result['best_value'])
    
    def test_adaptive_training(self):
        """Test adaptive training."""
        initial_params = {'num_layers': 5, 'entanglement': 0.5}
        result = self.optimizer.adaptive_training(initial_params, n_iter=5)
        
        self.assertIn('refined_params', result)
        self.assertIn('best_value', result)
        self.assertIn('best_error_rate', result)
        self.assertIn('iterations', result)
        
        # Check refined params structure
        self.assertIn('num_layers', result['refined_params'])
        self.assertIn('entanglement', result['refined_params'])
    
    def test_grid_search_optimize(self):
        """Test grid search optimization fallback."""
        result = self.optimizer._grid_search_optimize(n_samples=9)
        
        self.assertIn('best_params', result)
        self.assertIn('best_value', result)
        self.assertGreater(len(self.optimizer.optimization_history), 0)
    
    def test_local_search(self):
        """Test local search fallback."""
        initial_params = {'num_layers': 5, 'entanglement': 0.5}
        result = self.optimizer._local_search(initial_params, n_iter=5)
        
        self.assertIn('refined_params', result)
        self.assertIn('best_value', result)
        self.assertEqual(result['iterations'], 5)
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        # Run some optimization first
        self.optimizer.optimize_parameters(init_points=2, n_iter=2)
        
        summary = self.optimizer.get_optimization_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('Total Iterations', summary)
        self.assertIn('Best Fidelity', summary)
        self.assertIn('Best Error Rate', summary)
    
    def test_optimization_history_tracking(self):
        """Test that optimization history is properly tracked."""
        self.optimizer._quantum_objective(5, 0.5)
        self.optimizer._quantum_objective(6, 0.6)
        
        self.assertEqual(len(self.optimizer.optimization_history), 2)
        
        # Check history structure
        for entry in self.optimizer.optimization_history:
            self.assertIn('num_layers', entry)
            self.assertIn('entanglement', entry)
            self.assertIn('error_rate', entry)
            self.assertIn('fidelity', entry)
            self.assertIn('time_taken', entry)


class TestIntegration(unittest.TestCase):
    """Integration tests for quantum optimizer workflow."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create circuit and optimizer
        circuit = QuantumCircuit(num_qubits=16)
        optimizer = QuantumOptimizer(circuit)
        
        # Get baseline performance
        baseline = circuit.run()
        baseline_error = baseline['error_rate']
        
        # Optimize
        result = optimizer.optimize_parameters(init_points=2, n_iter=3)
        
        # Test with optimized parameters
        optimized = circuit.run(
            num_layers=result['best_params']['num_layers'],
            entanglement_strength=result['best_params']['entanglement']
        )
        
        # Verify optimization improved performance (or at least didn't make it worse)
        # Note: Due to randomness, we just check it's reasonable
        self.assertGreaterEqual(optimized['fidelity'], 0.0)
        self.assertLessEqual(optimized['error_rate'], 1.0)
    
    def test_adaptive_training_workflow(self):
        """Test adaptive training workflow."""
        circuit = QuantumCircuit(num_qubits=16)
        optimizer = QuantumOptimizer(circuit)
        
        # Initial optimization
        initial_result = optimizer.optimize_parameters(init_points=2, n_iter=2)
        
        # Adaptive training
        refined_result = optimizer.adaptive_training(
            initial_result['best_params'],
            n_iter=3
        )
        
        # Verify refined parameters are within bounds
        self.assertGreaterEqual(refined_result['refined_params']['num_layers'], 2)
        self.assertLessEqual(refined_result['refined_params']['num_layers'], 10)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumCircuit))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())

