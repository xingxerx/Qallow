"""Sparse vector dataset generation and management."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import logging
from scipy.sparse import csr_matrix, coo_matrix
from .config import ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class SparseVector:
    """Sparse vector representation.
    
    Attributes:
        indices: Indices of nonzero elements
        values: Values of nonzero elements
        dimension: Total dimension
        cluster_id: Ground truth cluster assignment (if known)
    """
    indices: np.ndarray  # Shape (s,)
    values: np.ndarray   # Shape (s,)
    dimension: int
    cluster_id: Optional[int] = None
    
    def to_dense(self) -> np.ndarray:
        """Convert to dense vector."""
        dense = np.zeros(self.dimension)
        dense[self.indices] = self.values
        return dense
    
    def norm(self) -> float:
        """Compute L2 norm."""
        return float(np.linalg.norm(self.values))
    
    def normalize(self) -> "SparseVector":
        """Return normalized copy."""
        norm = self.norm()
        if norm < 1e-10:
            logger.warning("Vector has near-zero norm")
            return SparseVector(self.indices, self.values, self.dimension, self.cluster_id)
        return SparseVector(
            self.indices,
            self.values / norm,
            self.dimension,
            self.cluster_id
        )


@dataclass
class SparseDataset:
    """Sparse vector dataset with ground truth clustering.
    
    Attributes:
        vectors: List of SparseVector objects
        config: ClusteringConfig
        ground_truth_labels: Cluster assignments (n,)
        cluster_centers: Dense cluster centers (k, d)
    """
    vectors: List[SparseVector]
    config: ClusteringConfig
    ground_truth_labels: np.ndarray = field(default_factory=lambda: np.array([]))
    cluster_centers: np.ndarray = field(default_factory=lambda: np.array([]))
    
    @staticmethod
    def generate(config: ClusteringConfig) -> "SparseDataset":
        """Generate synthetic sparse dataset.
        
        Args:
            config: ClusteringConfig with n, d, s, k, seed, distribution
            
        Returns:
            SparseDataset with ground truth labels and cluster centers
        """
        np.random.seed(config.seed)
        logger.info(f"Generating dataset: {config}")
        
        # Generate cluster centers
        cluster_centers = np.random.randn(config.k, config.d)
        if config.normalize:
            cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        vectors = []
        labels = np.zeros(config.n, dtype=int)
        
        # Generate vectors around cluster centers
        for i in range(config.n):
            cluster_id = i % config.k  # Balanced clusters
            labels[i] = cluster_id
            
            # Start from cluster center
            center = cluster_centers[cluster_id]
            
            # Add noise
            noise = np.random.randn(config.d)
            if config.distribution == "gaussian":
                vector = center + 0.1 * noise
            elif config.distribution == "exponential":
                vector = center + 0.1 * np.abs(np.random.exponential(1.0, config.d))
            else:  # uniform
                vector = center + 0.1 * np.random.uniform(-1, 1, config.d)
            
            # Make nonnegative
            vector = np.abs(vector)
            
            # Normalize
            if config.normalize:
                vector = vector / (np.linalg.norm(vector) + 1e-10)
            
            # Sparsify: keep only s largest elements
            if config.s < config.d:
                indices = np.argsort(np.abs(vector))[-config.s:]
                indices = np.sort(indices)
                values = vector[indices]
            else:
                indices = np.arange(config.d)
                values = vector
            
            sparse_vec = SparseVector(
                indices=indices,
                values=values,
                dimension=config.d,
                cluster_id=cluster_id
            )
            vectors.append(sparse_vec)
        
        logger.info(f"Generated {len(vectors)} vectors, {config.k} clusters")
        
        return SparseDataset(
            vectors=vectors,
            config=config,
            ground_truth_labels=labels,
            cluster_centers=cluster_centers
        )
    
    def to_sparse_matrix(self) -> csr_matrix:
        """Convert to scipy sparse matrix (n, d)."""
        rows, cols, data = [], [], []
        for i, vec in enumerate(self.vectors):
            rows.extend([i] * len(vec.indices))
            cols.extend(vec.indices)
            data.extend(vec.values)
        return csr_matrix((data, (rows, cols)), shape=(len(self.vectors), self.config.d))
    
    def to_dense_matrix(self) -> np.ndarray:
        """Convert to dense matrix (n, d)."""
        return np.array([v.to_dense() for v in self.vectors])
    
    def summary(self) -> str:
        """Return dataset summary."""
        sparsity = sum(len(v.indices) for v in self.vectors) / (len(self.vectors) * self.config.d)
        return (
            f"SparseDataset(n={len(self.vectors)}, d={self.config.d}, "
            f"s={self.config.s}, k={self.config.k}, sparsity={sparsity:.3f})"
        )

