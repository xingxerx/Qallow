"""Unit tests for dataset generation and sparse vectors."""

import pytest
import numpy as np
from quantum_algorithms.quantum_clustering import (
    ClusteringConfig,
    SparseDataset,
    SparseVector,
)


class TestSparseVector:
    """Test SparseVector class."""
    
    def test_creation(self):
        """Test sparse vector creation."""
        indices = np.array([0, 2, 5])
        values = np.array([0.5, 0.3, 0.2])
        vec = SparseVector(indices, values, dimension=10)
        
        assert len(vec.indices) == 3
        assert vec.dimension == 10
        assert vec.cluster_id is None
    
    def test_to_dense(self):
        """Test conversion to dense."""
        indices = np.array([0, 2, 5])
        values = np.array([0.5, 0.3, 0.2])
        vec = SparseVector(indices, values, dimension=10)
        
        dense = vec.to_dense()
        assert dense.shape == (10,)
        assert dense[0] == 0.5
        assert dense[2] == 0.3
        assert dense[5] == 0.2
        assert dense[1] == 0.0
    
    def test_norm(self):
        """Test L2 norm computation."""
        indices = np.array([0, 1])
        values = np.array([3.0, 4.0])
        vec = SparseVector(indices, values, dimension=10)
        
        assert np.isclose(vec.norm(), 5.0)
    
    def test_normalize(self):
        """Test normalization."""
        indices = np.array([0, 1])
        values = np.array([3.0, 4.0])
        vec = SparseVector(indices, values, dimension=10)
        
        normalized = vec.normalize()
        assert np.isclose(normalized.norm(), 1.0)
        assert np.isclose(normalized.values[0], 0.6)
        assert np.isclose(normalized.values[1], 0.8)


class TestClusteringConfig:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, m=6)
        assert config.n == 50
        assert config.d == 256
        assert config.s == 8
        assert config.k == 5
        assert config.m == 6
    
    def test_invalid_n(self):
        """Test invalid n."""
        with pytest.raises(ValueError):
            ClusteringConfig(n=2000, d=256, s=8, k=5)
    
    def test_invalid_d(self):
        """Test invalid d."""
        with pytest.raises(ValueError):
            ClusteringConfig(n=50, d=2000, s=8, k=5)
    
    def test_invalid_s(self):
        """Test invalid s."""
        with pytest.raises(ValueError):
            ClusteringConfig(n=50, d=256, s=300, k=5)
    
    def test_invalid_k(self):
        """Test invalid k."""
        with pytest.raises(ValueError):
            ClusteringConfig(n=50, d=256, s=8, k=100)
    
    def test_total_qubits(self):
        """Test qubit count calculation."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, m=6)
        # log2(256) = 8 address qubits + 6 feature + 2 ancilla = 16
        assert config.total_qubits == 16
    
    def test_to_dict(self):
        """Test configuration export."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, m=6)
        cfg_dict = config.to_dict()
        
        assert cfg_dict["n"] == 50
        assert cfg_dict["d"] == 256
        assert cfg_dict["total_qubits"] == 16


class TestSparseDataset:
    """Test dataset generation."""
    
    def test_generate_basic(self):
        """Test basic dataset generation."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        assert len(dataset.vectors) == 50
        assert dataset.config == config
        assert len(dataset.ground_truth_labels) == 50
        assert dataset.cluster_centers.shape == (5, 256)
    
    def test_sparsity(self):
        """Test that vectors are sparse."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        for vec in dataset.vectors:
            assert len(vec.indices) <= config.s
            assert len(vec.values) == len(vec.indices)
    
    def test_normalization(self):
        """Test vector normalization."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, normalize=True, seed=42)
        dataset = SparseDataset.generate(config)
        
        for vec in dataset.vectors:
            norm = vec.norm()
            assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_cluster_balance(self):
        """Test balanced cluster assignment."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        # With n=50, k=5, should have 10 vectors per cluster
        unique, counts = np.unique(dataset.ground_truth_labels, return_counts=True)
        assert len(unique) == 5
        assert all(c == 10 for c in counts)
    
    def test_reproducibility(self):
        """Test reproducibility with fixed seed."""
        config1 = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        config2 = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        
        dataset1 = SparseDataset.generate(config1)
        dataset2 = SparseDataset.generate(config2)
        
        # Check that datasets are identical
        for v1, v2 in zip(dataset1.vectors, dataset2.vectors):
            assert np.allclose(v1.values, v2.values)
            assert np.array_equal(v1.indices, v2.indices)
    
    def test_to_sparse_matrix(self):
        """Test conversion to sparse matrix."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        sparse_mat = dataset.to_sparse_matrix()
        assert sparse_mat.shape == (50, 256)
        assert sparse_mat.nnz <= 50 * 8  # At most 50 * 8 nonzeros
    
    def test_to_dense_matrix(self):
        """Test conversion to dense matrix."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        dense_mat = dataset.to_dense_matrix()
        assert dense_mat.shape == (50, 256)
    
    def test_summary(self):
        """Test dataset summary."""
        config = ClusteringConfig(n=50, d=256, s=8, k=5, seed=42)
        dataset = SparseDataset.generate(config)
        
        summary = dataset.summary()
        assert "n=50" in summary
        assert "d=256" in summary
        assert "k=5" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

