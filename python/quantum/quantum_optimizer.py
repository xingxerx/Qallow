"""
Quantum Optimizer for AGI Evolution Feature 004
Implements Bayesian optimization with Gaussian Process for quantum circuit parameter tuning
"""

import numpy as np
from typing import Dict, Any, Callable, Optional, Tuple
import warnings

# Try to import required libraries with graceful fallback
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

try:
    from bayes_opt import BayesianOptimization
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False
    warnings.warn("bayesian-optimization not available. Install with: pip install bayesian-optimization")


class QuantumOptimizer:
    """
    Quantum-enhanced Bayesian optimizer for circuit parameter tuning.
    
    This class implements Bayesian optimization using Gaussian Process regression
    to efficiently explore the parameter space of quantum circuits. It uses
    Expected Improvement (EI) acquisition function to balance exploration and
    exploitation.
    
    Attributes:
        quantum_circuit: The quantum circuit to optimize
        kernel: Gaussian Process kernel (RBF)
        gpr: Gaussian Process Regressor for surrogate modeling
        bayes_optimizer: Bayesian optimization engine
        optimization_history: History of optimization iterations
    """
    
    def __init__(self, quantum_circuit, use_bayesian: bool = True):
        """
        Initialize quantum optimizer.
        
        Args:
            quantum_circuit: QuantumCircuit instance to optimize
            use_bayesian: Whether to use Bayesian optimization (requires bayes_opt)
        """
        self.quantum_circuit = quantum_circuit
        self.use_bayesian = use_bayesian and BAYESOPT_AVAILABLE
        self.optimization_history = []
        
        # Initialize Gaussian Process components
        if SKLEARN_AVAILABLE:
            # RBF kernel with automatic length scale tuning
            self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5))
            self.gpr = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=1e-2,  # Noise level
                n_restarts_optimizer=10,
                normalize_y=True
            )
        else:
            self.kernel = None
            self.gpr = None
        
        # Initialize Bayesian optimizer if available
        if self.use_bayesian:
            self.bayes_optimizer = BayesianOptimization(
                f=self._quantum_objective,
                pbounds={
                    'num_layers': (2, 10),
                    'entanglement': (0.1, 0.9)
                },
                random_state=42,
                verbose=0
            )
        else:
            self.bayes_optimizer = None
    
    def _quantum_objective(self, num_layers: float, entanglement: float) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            num_layers: Number of circuit layers (will be converted to int)
            entanglement: Entanglement strength parameter
            
        Returns:
            Objective score (higher is better): 1.0 - error_rate
        """
        # Convert to appropriate types
        num_layers_int = int(round(num_layers))
        entanglement_float = float(entanglement)
        
        # Execute quantum circuit with these parameters
        result = self.quantum_circuit.run(
            num_layers=num_layers_int,
            entanglement_strength=entanglement_float
        )
        
        # Objective: minimize error rate (maximize fidelity)
        objective_value = 1.0 - result['error_rate']
        
        # Store in history
        self.optimization_history.append({
            'num_layers': num_layers_int,
            'entanglement': entanglement_float,
            'error_rate': result['error_rate'],
            'fidelity': objective_value,
            'time_taken': result['time_taken']
        })
        
        return objective_value
    
    def optimize_parameters(self, init_points: int = 5, n_iter: int = 25, 
                          acq: str = 'ei') -> Dict[str, Any]:
        """
        Optimize quantum circuit parameters using Bayesian optimization.
        
        Args:
            init_points: Number of random initialization points
            n_iter: Number of optimization iterations
            acq: Acquisition function ('ei' = Expected Improvement, 'ucb' = Upper Confidence Bound)
            
        Returns:
            Dictionary containing:
                - best_params: Optimal parameters found
                - best_value: Best objective value achieved
                - iterations: Number of iterations performed
                - history: Optimization history
        """
        if not self.use_bayesian:
            # Fallback to grid search if Bayesian optimization not available
            return self._grid_search_optimize(n_samples=init_points + n_iter)
        
        # Clear history
        self.optimization_history = []
        
        # Run Bayesian optimization
        self.bayes_optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
            acq=acq
        )
        
        # Extract best parameters
        best_params = self.bayes_optimizer.max['params']
        best_value = self.bayes_optimizer.max['target']
        
        return {
            'best_params': {
                'num_layers': int(round(best_params['num_layers'])),
                'entanglement': best_params['entanglement']
            },
            'best_value': best_value,
            'best_error_rate': 1.0 - best_value,
            'iterations': len(self.optimization_history),
            'history': self.optimization_history
        }
    
    def adaptive_training(self, initial_params: Dict[str, float], 
                         n_iter: int = 15) -> Dict[str, Any]:
        """
        Perform adaptive training starting from initial parameters.
        
        This method refines parameters incrementally using Bayesian optimization
        with tighter bounds around the initial values.
        
        Args:
            initial_params: Starting parameters (num_layers, entanglement)
            n_iter: Number of refinement iterations
            
        Returns:
            Dictionary with refined parameters and performance metrics
        """
        if not self.use_bayesian:
            # Fallback to local search
            return self._local_search(initial_params, n_iter)
        
        # Define tighter bounds around initial parameters
        num_layers_init = initial_params.get('num_layers', 5)
        entanglement_init = initial_params.get('entanglement', 0.5)
        
        # Create new optimizer with tighter bounds
        adaptive_optimizer = BayesianOptimization(
            f=self._quantum_objective,
            pbounds={
                'num_layers': (max(2, num_layers_init - 2), min(10, num_layers_init + 2)),
                'entanglement': (max(0.1, entanglement_init - 0.2), min(0.9, entanglement_init + 0.2))
            },
            random_state=42,
            verbose=0
        )
        
        # Clear history
        self.optimization_history = []
        
        # Run adaptive optimization
        adaptive_optimizer.maximize(
            init_points=3,
            n_iter=n_iter,
            acq='ei'
        )
        
        best_params = adaptive_optimizer.max['params']
        best_value = adaptive_optimizer.max['target']
        
        return {
            'refined_params': {
                'num_layers': int(round(best_params['num_layers'])),
                'entanglement': best_params['entanglement']
            },
            'best_value': best_value,
            'best_error_rate': 1.0 - best_value,
            'iterations': len(self.optimization_history),
            'improvement': best_value - self._quantum_objective(
                initial_params.get('num_layers', 5),
                initial_params.get('entanglement', 0.5)
            )
        }
    
    def _grid_search_optimize(self, n_samples: int = 30) -> Dict[str, Any]:
        """
        Fallback grid search optimization when Bayesian optimization unavailable.
        
        Args:
            n_samples: Number of grid points to sample
            
        Returns:
            Dictionary with best parameters found
        """
        print("Using grid search optimization (Bayesian optimization not available)")
        
        best_value = -np.inf
        best_params = None
        self.optimization_history = []
        
        # Grid search over parameter space
        for num_layers in np.linspace(2, 10, int(np.sqrt(n_samples))):
            for entanglement in np.linspace(0.1, 0.9, int(np.sqrt(n_samples))):
                value = self._quantum_objective(num_layers, entanglement)
                if value > best_value:
                    best_value = value
                    best_params = {
                        'num_layers': int(round(num_layers)),
                        'entanglement': entanglement
                    }
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'best_error_rate': 1.0 - best_value,
            'iterations': len(self.optimization_history),
            'history': self.optimization_history
        }
    
    def _local_search(self, initial_params: Dict[str, float], 
                     n_iter: int) -> Dict[str, Any]:
        """
        Local search around initial parameters (fallback for adaptive training).
        
        Args:
            initial_params: Starting parameters
            n_iter: Number of iterations
            
        Returns:
            Dictionary with refined parameters
        """
        best_value = self._quantum_objective(
            initial_params.get('num_layers', 5),
            initial_params.get('entanglement', 0.5)
        )
        best_params = initial_params.copy()
        
        for _ in range(n_iter):
            # Random perturbation
            num_layers = best_params['num_layers'] + np.random.randint(-1, 2)
            num_layers = max(2, min(10, num_layers))
            entanglement = best_params['entanglement'] + np.random.uniform(-0.1, 0.1)
            entanglement = max(0.1, min(0.9, entanglement))
            
            value = self._quantum_objective(num_layers, entanglement)
            if value > best_value:
                best_value = value
                best_params = {'num_layers': num_layers, 'entanglement': entanglement}
        
        return {
            'refined_params': best_params,
            'best_value': best_value,
            'best_error_rate': 1.0 - best_value,
            'iterations': n_iter
        }
    
    def get_optimization_summary(self) -> str:
        """
        Get a summary of the optimization process.
        
        Returns:
            Formatted string with optimization statistics
        """
        if not self.optimization_history:
            return "No optimization performed yet."
        
        best_iter = max(self.optimization_history, key=lambda x: x['fidelity'])
        
        summary = f"""
Quantum Optimizer Summary
========================
Total Iterations: {len(self.optimization_history)}
Best Fidelity: {best_iter['fidelity']:.6f}
Best Error Rate: {best_iter['error_rate']:.6f}
Best Parameters:
  - Num Layers: {best_iter['num_layers']}
  - Entanglement: {best_iter['entanglement']:.4f}
Average Time per Iteration: {np.mean([h['time_taken'] for h in self.optimization_history]):.4f}s
"""
        return summary

