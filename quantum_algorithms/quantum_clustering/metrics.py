"""Evaluation metrics and profiling for quantum clustering."""

import numpy as np
import logging
import time
from typing import Dict, Any, Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .dataset import SparseDataset

logger = logging.getLogger(__name__)


def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute Adjusted Rand Index.
    
    Args:
        true_labels: Ground truth cluster assignments (n,)
        pred_labels: Predicted cluster assignments (n,)
        
    Returns:
        ARI score in [-1, 1]. 1 = perfect agreement, 0 = random, -1 = disagreement
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("Label arrays must have same length")
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    logger.info(f"ARI: {ari:.4f}")
    return ari


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Compute Normalized Mutual Information.
    
    Args:
        true_labels: Ground truth cluster assignments (n,)
        pred_labels: Predicted cluster assignments (n,)
        
    Returns:
        NMI score in [0, 1]. 1 = perfect agreement, 0 = independent
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("Label arrays must have same length")
    
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    logger.info(f"NMI: {nmi:.4f}")
    return nmi


class PerformanceProfiler:
    """Profile quantum state preparation and measurement costs."""
    
    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, list] = {}
        self.measurements: Dict[str, int] = {}
        self.state_preps: Dict[str, int] = {}
    
    def record_state_prep(self, name: str, duration: float, num_qubits: int, depth: int):
        """Record state preparation timing.
        
        Args:
            name: Operation name
            duration: Execution time in seconds
            num_qubits: Number of qubits used
            depth: Circuit depth
        """
        if name not in self.timings:
            self.timings[name] = []
            self.state_preps[name] = 0
        
        self.timings[name].append(duration)
        self.state_preps[name] += 1
        logger.debug(f"State prep '{name}': {duration*1e6:.2f} µs, {num_qubits} qubits, depth {depth}")
    
    def record_measurement(self, name: str, num_shots: int):
        """Record measurement.
        
        Args:
            name: Operation name
            num_shots: Number of measurement shots
        """
        if name not in self.measurements:
            self.measurements[name] = 0
        
        self.measurements[name] += num_shots
        logger.debug(f"Measurement '{name}': {num_shots} shots")
    
    def summary(self) -> Dict[str, Any]:
        """Get profiling summary.
        
        Returns:
            Dictionary with timing and measurement statistics
        """
        summary = {
            "total_state_preps": sum(self.state_preps.values()),
            "total_measurements": sum(self.measurements.values()),
            "operations": {}
        }
        
        for name in self.timings:
            times = self.timings[name]
            summary["operations"][name] = {
                "count": len(times),
                "total_time": sum(times),
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
            }
        
        return summary
    
    def report(self) -> str:
        """Generate profiling report."""
        summary = self.summary()
        lines = [
            "=" * 60,
            "PERFORMANCE PROFILING REPORT",
            "=" * 60,
            f"Total state preparations: {summary['total_state_preps']}",
            f"Total measurements: {summary['total_measurements']}",
            "",
            "Per-operation statistics:",
        ]
        
        for name, stats in summary["operations"].items():
            lines.extend([
                f"  {name}:",
                f"    Count: {stats['count']}",
                f"    Total: {stats['total_time']*1e3:.2f} ms",
                f"    Mean: {stats['mean_time']*1e6:.2f} µs",
                f"    Std: {stats['std_time']*1e6:.2f} µs",
            ])
        
        lines.append("=" * 60)
        return "\n".join(lines)


class ClusteringMetrics:
    """Comprehensive clustering evaluation metrics."""
    
    def __init__(self, dataset: SparseDataset):
        """Initialize metrics tracker.
        
        Args:
            dataset: SparseDataset with ground truth labels
        """
        self.dataset = dataset
        self.true_labels = dataset.ground_truth_labels
        self.pred_labels = None
        self.profiler = PerformanceProfiler()
    
    def evaluate(self, pred_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality.
        
        Args:
            pred_labels: Predicted cluster assignments (n,)
            
        Returns:
            Dictionary with ARI, NMI, and other metrics
        """
        self.pred_labels = pred_labels
        
        metrics = {
            "ari": compute_ari(self.true_labels, pred_labels),
            "nmi": compute_nmi(self.true_labels, pred_labels),
        }
        
        logger.info(f"Clustering metrics: {metrics}")
        return metrics
    
    def report(self) -> str:
        """Generate comprehensive evaluation report."""
        if self.pred_labels is None:
            return "No predictions yet"
        
        metrics = self.evaluate(self.pred_labels)
        profiling = self.profiler.summary()
        
        lines = [
            "=" * 60,
            "CLUSTERING EVALUATION REPORT",
            "=" * 60,
            f"Dataset: {self.dataset.summary()}",
            "",
            "Clustering Quality:",
            f"  ARI: {metrics['ari']:.4f}",
            f"  NMI: {metrics['nmi']:.4f}",
            "",
            "Performance:",
            f"  State preparations: {profiling['total_state_preps']}",
            f"  Total measurements: {profiling['total_measurements']}",
            "=" * 60,
        ]
        
        return "\n".join(lines)

