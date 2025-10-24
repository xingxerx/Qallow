"""
Quantum Clustering Module - NISQ-optimized sparse vector clustering.

This module implements quantum random projection (QRP) for dimensionality reduction
and similarity estimation on NISQ devices (20-60 qubits, mid-gate fidelity).

Components:
    - dataset: Sparse vector generation and ground truth clustering
    - sparse_encoder: Quantum state preparation with fidelity validation
    - config: Configuration management
    - metrics: Evaluation metrics (ARI, runtime profiling)
    - backend: Quantum backend abstraction (Qiskit/Cirq)

Usage:
    from quantum_algorithms.quantum_clustering import (
        ClusteringConfig,
        SparseDataset,
        SparseEncoder,
    )
    
    config = ClusteringConfig(n=50, d=256, s=8, k=5, m=6)
    dataset = SparseDataset.generate(config)
    encoder = SparseEncoder(config)
    
    # Prepare quantum states
    for vector in dataset.vectors:
        state_ref = encoder.prepare_state(vector)
"""

from .config import ClusteringConfig
from .dataset import SparseDataset, SparseVector
from .sparse_encoder import SparseEncoder
from .metrics import compute_ari, profile_state_prep

__all__ = [
    "ClusteringConfig",
    "SparseDataset",
    "SparseVector",
    "SparseEncoder",
    "compute_ari",
    "profile_state_prep",
]

__version__ = "0.1.0"

