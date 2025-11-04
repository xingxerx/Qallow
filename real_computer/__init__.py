# [REVIEWED] # [REVIEWED] # [REVIEWED] """Real hardware/backends integration for Qallow
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] Backends:
# [REVIEWED] # [REVIEWED] # [REVIEWED]  - GPU: uses CuPy (preferred) or PyCUDA (if CuPy unavailable)
# [REVIEWED] # [REVIEWED] # [REVIEWED]  - Quantum: uses Cirq
# [REVIEWED] # [REVIEWED] # [REVIEWED]  - Neuromorphic: reuses existing Python neuromorphic simulator in virtual_computer
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] This package provides a small, high-level API the AgentLightning Runner can call to run
# [REVIEWED] # [REVIEWED] # [REVIEWED] real workloads when available and fall back or skip when not.
# [REVIEWED] # [REVIEWED] # [REVIEWED] """
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] from .gpu_backend import GPUBackend, get_gpu_backend
# [REVIEWED] # [REVIEWED] # [REVIEWED] from .quantum_backend import QuantumBackend
# [REVIEWED] # [REVIEWED] # [REVIEWED] from .neuromorphic_backend import NeuromorphicBackend
# [REVIEWED] # [REVIEWED] # [REVIEWED] 
# [REVIEWED] # [REVIEWED] # [REVIEWED] __all__ = ["GPUBackend", "get_gpu_backend", "QuantumBackend", "NeuromorphicBackend"]
