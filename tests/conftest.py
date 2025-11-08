"""
Pytest configuration and fixtures for Qallow testing
Handles CUDA-Q mocking for CI/CD environments while supporting GPU tests
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List


# ============================================================================
# CUDA-Q Mocking Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def cuda_q_mock():
    """
    Module-level cudaq mock for CI/CD environments
    
    Use when: Running tests in CI without GPU access
    This mocks the entire cudaq module with sensible defaults
    """
    mock_cudaq = MagicMock()
    
    # Mock version
    mock_cudaq.__version__ = "0.8.0-mock"
    
    # Mock kernel decorator
    def kernel_decorator(func):
        """Decorator that makes a kernel callable"""
        return func
    
    mock_cudaq.kernel = kernel_decorator
    
    # Mock target system
    mock_target = MagicMock()
    mock_target.Nvidia = MagicMock(return_value=mock_target)
    mock_target.Quantinuum = MagicMock(return_value=mock_target)
    mock_target.IonQ = MagicMock(return_value=mock_target)
    mock_target.IQM = MagicMock(return_value=mock_target)
    mock_cudaq.target = mock_target
    
    # Mock set_target
    mock_cudaq.set_target = MagicMock()
    
    # Mock set_random_seed
    mock_cudaq.set_random_seed = MagicMock()
    
    # Mock gates
    mock_cudaq.ry = MagicMock()
    mock_cudaq.cx = MagicMock()
    mock_cudaq.mz = MagicMock()
    
    # Mock qvector
    mock_qvector = MagicMock()
    mock_qvector.__getitem__ = MagicMock(return_value=MagicMock())
    mock_cudaq.qvector = MagicMock(return_value=mock_qvector)
    
    # Mock sample function - returns dict-like object
    def mock_sample(kernel, shots_count=1024):
        """Mock sampling returns dict of bitstrings -> counts"""
        result = MagicMock()
        
        # Simulated measurement results
        counts = {
            "0" * 5: 400,
            "1" * 5: 350,
            "01010": 150,
            "10101": 124,
        }
        
        # Make counts accessible as dict or via counts() method
        result.__getitem__ = lambda k: counts[k]
        result.__iter__ = lambda: iter(counts)
        result.counts = MagicMock(return_value=counts)
        result.items = lambda: counts.items()
        result.values = lambda: counts.values()
        
        return result
    
    mock_cudaq.sample = MagicMock(side_effect=mock_sample)
    
    return mock_cudaq


@pytest.fixture
def cuda_q_mocked_import(cuda_q_mock, monkeypatch):
    """
    Inject cudaq mock into sys.modules
    
    Use: @pytest.mark.cuda_q_mock
    This makes 'import cudaq' return the mock in your test
    """
    monkeypatch.setitem(sys.modules, 'cudaq', cuda_q_mock)
    return cuda_q_mock


# ============================================================================
# Marker Registration
# ============================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers",
        "cuda_q_gpu: Tests requiring actual CUDA-Q with GPU (skip in CI)"
    )
    config.addinivalue_line(
        "markers",
        "cuda_q_mock: Tests using mocked CUDA-Q (safe for CI)"
    )
    config.addinivalue_line(
        "markers",
        "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers",
        "unit: Unit tests"
    )


# ============================================================================
# Conditional Test Collection
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on environment
    - Skip GPU tests in CI
    - Only run mock tests in CI
    """
    import os
    
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    
    if is_ci:
        # In CI: skip GPU tests, run mock tests
        for item in items:
            if "cuda_q_gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping GPU tests in CI"))


# ============================================================================
# Quantum Bridge Test Fixtures
# ============================================================================

@pytest.fixture
def cuda_q_config():
    """Fixture for CudaQConfig"""
    from dataclasses import dataclass
    
    @dataclass
    class CudaQConfig:
        backend: str = "nvidia"
        shots: int = 1024
        seed: int = None
        optimization_level: int = 2
    
    return CudaQConfig()


@pytest.fixture
def quantum_sample():
    """Fixture for QuantumSample dataclass"""
    from dataclasses import dataclass
    
    @dataclass
    class QuantumSample:
        bitstring: str
        probability: float
        energy: float = None
        timestamp: float = 0.0
    
    return QuantumSample


@pytest.fixture
def quantum_sampling_result():
    """Fixture for QuantumSamplingResult dataclass"""
    from dataclasses import dataclass
    from typing import List, Dict
    
    @dataclass
    class QuantumSamplingResult:
        n_samples: int
        n_qubits: int
        backend: str
        samples: List[Any]
        execution_time_ms: float
        circuit_depth: int
        circuit_gates: int
        metrics: Dict[str, Any]
    
    return QuantumSamplingResult


# ============================================================================
# Mock Quantum Results Fixtures
# ============================================================================

@pytest.fixture
def mock_sampling_result():
    """Fixture providing realistic mock sampling results"""
    result_dict = {
        "n_samples": 4,
        "n_qubits": 5,
        "backend": "mock",
        "samples": [
            {"bitstring": "00000", "probability": 0.35},
            {"bitstring": "11111", "probability": 0.32},
            {"bitstring": "01010", "probability": 0.18},
            {"bitstring": "10101", "probability": 0.15},
        ],
        "execution_time_ms": 5.2,
        "circuit_depth": 5,
        "circuit_gates": 50,
        "metrics": {
            "shot_efficiency": 0.98,
            "unique_bitstrings": 4,
            "device": "mock_gpu",
        }
    }
    return result_dict


@pytest.fixture
def realistic_circuit_params():
    """Fixture for realistic circuit parameters"""
    params = {}
    for d in range(5):  # depth=5
        for q in range(5):  # n_qubits=5
            params[f"theta_{d}_{q}"] = 0.3 * (d + 1) + 0.1 * q
    return params


# ============================================================================
# Environment Detection Fixtures
# ============================================================================

@pytest.fixture
def is_ci_environment():
    """Detect if running in CI"""
    import os
    return os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture
def cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture
def cudaq_installed():
    """Check if CUDA-Q is installed"""
    try:
        import cudaq
        return True
    except ImportError:
        return False


# ============================================================================
# Parametrization Fixtures
# ============================================================================

@pytest.fixture(
    params=[
        ("nvidia", 5, 10, 1024),
        ("nvidia", 10, 15, 2048),
        ("mock", 3, 5, 512),
    ]
)
def circuit_config_variations(request):
    """
    Parametrized fixture providing various circuit configurations
    Returns: (backend, n_qubits, circuit_depth, n_shots)
    """
    return request.param


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically clean up temporary files after each test"""
    yield tmp_path
    # Cleanup happens automatically when tmp_path scope ends


# ============================================================================
# Logging Fixtures
# ============================================================================

@pytest.fixture
def caplog_verbose(caplog):
    """Fixture for verbose logging in tests"""
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================================================
# Test Data Factories
# ============================================================================

class QuantumSampleFactory:
    """Factory for creating test QuantumSample objects"""
    
    @staticmethod
    def create(
        bitstring: str = "00000",
        probability: float = 0.5,
        energy: float = None
    ) -> Dict[str, Any]:
        """Create a QuantumSample dict"""
        return {
            "bitstring": bitstring,
            "probability": probability,
            "energy": energy,
            "timestamp": 0.0
        }
    
    @staticmethod
    def create_batch(n_samples: int = 10, n_qubits: int = 5) -> List[Dict[str, Any]]:
        """Create a batch of random QuantumSample dicts"""
        import numpy as np
        
        samples = []
        probabilities = np.random.dirichlet(np.ones(n_samples))
        
        for i, prob in enumerate(probabilities):
            bitstring = format(i, f'0{n_qubits}b')
            samples.append(QuantumSampleFactory.create(
                bitstring=bitstring,
                probability=float(prob)
            ))
        
        return samples


@pytest.fixture
def sample_factory():
    """Fixture providing QuantumSampleFactory"""
    return QuantumSampleFactory


# ============================================================================
# CUDA-Q Integration Testing Helpers
# ============================================================================

@pytest.fixture
def cuda_q_test_harness(cuda_q_mock):
    """
    Complete test harness for CUDA-Q testing
    Provides mocking setup + helper methods
    """
    class TestHarness:
        def __init__(self, mock_cudaq):
            self.mock_cudaq = mock_cudaq
            self.calls = []
        
        def get_sample_calls(self):
            """Get list of cudaq.sample() calls"""
            return self.mock_cudaq.sample.call_args_list
        
        def get_set_target_calls(self):
            """Get list of cudaq.set_target() calls"""
            return self.mock_cudaq.set_target.call_args_list
        
        def verify_kernel_called(self):
            """Verify a kernel was decorated"""
            return self.mock_cudaq.kernel.call_count > 0
        
        def verify_gates_used(self):
            """Verify gates were called"""
            ry_called = self.mock_cudaq.ry.call_count > 0
            cx_called = self.mock_cudaq.cx.call_count > 0
            return ry_called and cx_called
    
    return TestHarness(cuda_q_mock)


# ============================================================================
# pytest.ini Configuration Documentation
# ============================================================================
# For complete pytest configuration, create/update pytest.ini:
#
# [pytest]
# testpaths = tests
# python_files = test_*.py
# python_classes = Test*
# python_functions = test_*
# markers =
#     cuda_q_gpu: Tests requiring actual CUDA-Q with GPU
#     cuda_q_mock: Tests using mocked CUDA-Q
#     integration: Integration tests
#     unit: Unit tests
# filterwarnings =
#     ignore::DeprecationWarning
#     ignore::PendingDeprecationWarning
#
# Command-line usage:
#   pytest -m "not cuda_q_gpu"           # Skip GPU tests
#   pytest -m "cuda_q_mock"              # Run only mock tests
#   pytest -m "cuda_q_gpu"               # Run only GPU tests
#   pytest -m "integration"              # Run integration tests
#   pytest --verbose --capture=no        # Verbose output
#   pytest --tb=short                    # Short traceback format
