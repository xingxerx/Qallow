"""Real hardware/backends integration for Qallow

Backends:
 - GPU: uses CuPy (preferred) or PyCUDA (if CuPy unavailable)
 - Quantum: uses Cirq
 - Neuromorphic: reuses existing Python neuromorphic simulator in virtual_computer

This package provides a small, high-level API the AgentLightning Runner can call to run
real workloads when available and fall back or skip when not.
"""

from .gpu_backend import GPUBackend, get_gpu_backend
from .quantum_backend import QuantumBackend
from .neuromorphic_backend import NeuromorphicBackend

__all__ = ["GPUBackend", "get_gpu_backend", "QuantumBackend", "NeuromorphicBackend"]
