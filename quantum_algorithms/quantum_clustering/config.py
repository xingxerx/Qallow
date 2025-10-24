"""Configuration management for quantum clustering."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusteringConfig:
    """Configuration for quantum clustering pipeline.
    
    Attributes:
        n: Number of vectors (up to 1000)
        d: Dimension of each vector (up to 1000)
        s: Number of nonzeros per vector (sparse)
        k: Number of clusters
        m: Projection dimension (feature qubits). Default 6.
        seed: Random seed for reproducibility. Default 42.
        backend: Quantum backend ('qiskit' or 'cirq'). Default 'qiskit'.
        simulator: Simulator type ('aer' for Qiskit, 'clifford' for Cirq). Default 'aer'.
        shots: Number of measurement shots. Default 1000.
        max_circuit_depth: Maximum circuit depth for NISQ devices. Default 20.
        normalize: Whether to normalize vectors. Default True.
        distribution: Data distribution ('uniform', 'gaussian', 'exponential'). Default 'uniform'.
    """
    
    # Dataset parameters
    n: int = 50  # Number of vectors
    d: int = 256  # Dimension
    s: int = 8  # Sparsity (nonzeros per vector)
    k: int = 5  # Number of clusters
    
    # Quantum parameters
    m: int = 6  # Projection dimension (feature qubits)
    seed: int = 42  # Random seed
    backend: Literal["qiskit", "cirq"] = "qiskit"
    simulator: str = "aer"  # 'aer' for Qiskit, 'clifford' for Cirq
    shots: int = 1000  # Measurement shots
    max_circuit_depth: int = 20  # NISQ device limit
    
    # Data parameters
    normalize: bool = True
    distribution: Literal["uniform", "gaussian", "exponential"] = "uniform"
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.n <= 1000:
            raise ValueError(f"n must be in [1, 1000], got {self.n}")
        if not 1 <= self.d <= 1000:
            raise ValueError(f"d must be in [1, 1000], got {self.d}")
        if not 1 <= self.s <= self.d:
            raise ValueError(f"s must be in [1, {self.d}], got {self.s}")
        if not 1 <= self.k <= self.n:
            raise ValueError(f"k must be in [1, {self.n}], got {self.k}")
        if not 1 <= self.m <= 20:
            raise ValueError(f"m must be in [1, 20] (qubit limit), got {self.m}")
        if self.shots < 100:
            logger.warning(f"shots={self.shots} is low; recommend >= 500 for accuracy")
        if self.backend not in ["qiskit", "cirq"]:
            raise ValueError(f"backend must be 'qiskit' or 'cirq', got {self.backend}")
        
        logger.setLevel(self.log_level)
    
    @property
    def total_qubits(self) -> int:
        """Total qubits needed: log2(d) for address + m for features + 2 ancilla."""
        import math
        address_qubits = math.ceil(math.log2(self.d))
        return address_qubits + self.m + 2
    
    @property
    def state_prep_depth_estimate(self) -> int:
        """Estimate state preparation depth: ~2*s rotation layers."""
        return 2 * self.s
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "n": self.n,
            "d": self.d,
            "s": self.s,
            "k": self.k,
            "m": self.m,
            "seed": self.seed,
            "backend": self.backend,
            "shots": self.shots,
            "total_qubits": self.total_qubits,
            "state_prep_depth_estimate": self.state_prep_depth_estimate,
        }
    
    def __str__(self) -> str:
        """Human-readable configuration summary."""
        return (
            f"ClusteringConfig(n={self.n}, d={self.d}, s={self.s}, k={self.k}, "
            f"m={self.m}, qubits={self.total_qubits}, backend={self.backend})"
        )

