# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] Quantum Clustering Module - NISQ-optimized sparse vector clustering.
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] This module implements quantum random projection (QRP) for dimensionality reduction
# [REVIEWED] # [REVIEWED] # [REVIEWED] and similarity estimation on NISQ devices (20-60 qubits, mid-gate fidelity).
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] Components:
# [REVIEWED] # [REVIEWED] # [REVIEWED]     - dataset: Sparse vector generation and ground truth clustering
# [REVIEWED] # [REVIEWED] # [REVIEWED]     - sparse_encoder: Quantum state preparation with fidelity validation
# [REVIEWED] # [REVIEWED] # [REVIEWED]     - config: Configuration management
# [REVIEWED] # [REVIEWED] # [REVIEWED]     - metrics: Evaluation metrics (ARI, runtime profiling)
# [REVIEWED] # [REVIEWED] # [REVIEWED]     - backend: Quantum backend abstraction (Qiskit/Cirq)
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] Usage:
# [REVIEWED] # [REVIEWED] # [REVIEWED]     from quantum_algorithms.quantum_clustering import (
# [REVIEWED] # [REVIEWED] # [REVIEWED]         ClusteringConfig,
# [REVIEWED] # [REVIEWED] # [REVIEWED]         SparseDataset,
# [REVIEWED] # [REVIEWED] # [REVIEWED]         SparseEncoder,
# [REVIEWED] # [REVIEWED] # [REVIEWED]     )
# [REVIEWED] # [REVIEWED] # [REVIEWED]     
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

