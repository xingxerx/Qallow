"""Unit tests for sparse encoder."""

import pytest
import numpy as np
from quantum_algorithms.quantum_clustering import (
    ClusteringConfig,
    SparseDataset,
    SparseEncoder,
    SparseVector,
)


class TestSparseEncoderQiskit:
    """Test sparse encoder with Qiskit backend."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ClusteringConfig(
            n=10, d=64, s=4, k=2, m=4, backend="qiskit", seed=42
        )
    
    @pytest.fixture
    def encoder(self, config):
        """Create encoder instance."""
        return SparseEncoder(config)
    
    @pytest.fixture
    def sparse_vector(self):
        """Create test sparse vector."""
        indices = np.array([0, 5, 10, 15])
        values = np.array([0.5, 0.3, 0.2, 0.1])
        return SparseVector(indices, values, dimension=64)
    
    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.config.backend == "qiskit"
        assert encoder.backend == "qiskit"
    
    def test_prepare_state_structure(self, encoder, sparse_vector):
        """Test state preparation returns correct structure."""
        state_dict = encoder.prepare_state(sparse_vector)
        
        assert "circuit" in state_dict
        assert "qubits" in state_dict
        assert "depth" in state_dict
        assert "vector_norm" in state_dict
        assert "backend" in state_dict
        assert state_dict["backend"] == "qiskit"
    
    def test_prepare_state_qubits(self, encoder, sparse_vector):
        """Test qubit count is correct."""
        state_dict = encoder.prepare_state(sparse_vector)
        
        # d=64 -> 6 address qubits, m=4 feature, 1 ancilla = 11 total
        assert state_dict["qubits"] == 11
        assert state_dict["address_qubits"] == 6
        assert state_dict["feature_qubits"] == 4
    
    def test_prepare_state_depth(self, encoder, sparse_vector):
        """Test circuit depth is reasonable."""
        state_dict = encoder.prepare_state(sparse_vector)
        
        # Depth should be positive and not too large
        assert state_dict["depth"] > 0
        assert state_dict["depth"] < 100
    
    def test_vector_norm_preserved(self, encoder, sparse_vector):
        """Test vector norm is preserved."""
        state_dict = encoder.prepare_state(sparse_vector)
        
        expected_norm = np.linalg.norm(sparse_vector.values)
        assert np.isclose(state_dict["vector_norm"], expected_norm)
    
    def test_prepare_multiple_vectors(self, encoder):
        """Test preparing multiple vectors."""
        config = ClusteringConfig(n=5, d=64, s=4, k=2, m=4, backend="qiskit", seed=42)
        dataset = SparseDataset.generate(config)
        
        for vec in dataset.vectors:
            state_dict = encoder.prepare_state(vec)
            assert state_dict["qubits"] > 0
            assert state_dict["depth"] > 0


class TestSparseEncoderCirq:
    """Test sparse encoder with Cirq backend."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ClusteringConfig(
            n=10, d=64, s=4, k=2, m=4, backend="cirq", seed=42
        )
    
    @pytest.fixture
    def encoder(self, config):
        """Create encoder instance."""
        try:
            return SparseEncoder(config)
        except ImportError:
            pytest.skip("Cirq not available")
    
    @pytest.fixture
    def sparse_vector(self):
        """Create test sparse vector."""
        indices = np.array([0, 5, 10, 15])
        values = np.array([0.5, 0.3, 0.2, 0.1])
        return SparseVector(indices, values, dimension=64)
    
    def test_prepare_state_cirq(self, encoder, sparse_vector):
        """Test state preparation with Cirq."""
        state_dict = encoder.prepare_state(sparse_vector)
        
        assert "circuit" in state_dict
        assert state_dict["backend"] == "cirq"
        assert state_dict["qubits"] == 11


class TestSparseEncoderEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_vector(self):
        """Test encoding zero vector."""
        config = ClusteringConfig(n=10, d=64, s=4, k=2, m=4, backend="qiskit")
        encoder = SparseEncoder(config)
        
        indices = np.array([0])
        values = np.array([0.0])
        vec = SparseVector(indices, values, dimension=64)
        
        state_dict = encoder.prepare_state(vec)
        assert state_dict["vector_norm"] == 0.0
    
    def test_single_nonzero(self):
        """Test vector with single nonzero."""
        config = ClusteringConfig(n=10, d=64, s=1, k=2, m=4, backend="qiskit")
        encoder = SparseEncoder(config)
        
        indices = np.array([10])
        values = np.array([1.0])
        vec = SparseVector(indices, values, dimension=64)
        
        state_dict = encoder.prepare_state(vec)
        assert state_dict["qubits"] > 0
    
    def test_full_vector(self):
        """Test dense vector (all nonzeros)."""
        config = ClusteringConfig(n=10, d=16, s=16, k=2, m=4, backend="qiskit")
        encoder = SparseEncoder(config)
        
        indices = np.arange(16)
        values = np.ones(16) / 4.0
        vec = SparseVector(indices, values, dimension=16)
        
        state_dict = encoder.prepare_state(vec)
        assert state_dict["qubits"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

