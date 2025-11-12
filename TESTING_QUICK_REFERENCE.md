# Quick Reference: CUDA-Q Mandatory Testing

## Test Markers

### GPU Tests (Requires NVIDIA GPU + CUDA-Q)
```bash
pytest -m "cuda_q_gpu"
```
**Use when**: Running on machine with GPU + CUDA-Q installed  
**Example tests**:
- `test_cuda_q_gpu_nvidia_backend()`
- `test_cuda_q_gpu_real_sampling()`
- Performance benchmarks

### Mock Tests (CI/CD Safe)
```bash
pytest -m "cuda_q_mock"
```
**Use when**: Running in CI/CD or without GPU  
**Example tests**:
- `test_orchestrator_backend_selection()`
- `test_cuda_q_bridge_mocking()`
- Logic verification (no GPU access needed)

### All Tests
```bash
pytest tests/
```

## Environment Variables

```bash
# Enable CI/CD mode (auto-skip GPU tests)
export CI=true
pytest tests/

# Set backend preference
export QALLOW_BACKEND=cirq
./qallow run unified

# Enable verbose logging
export QALLOW_DEBUG=1
pytest -vv tests/
```

## Common Test Patterns

### Test Requiring GPU
```python
@pytest.mark.cuda_q_gpu
def test_real_quantum_sampling(cuda_q_mocked_import):
    from python.quantum.cuda_q_bridge import CudaQBridge
    bridge = CudaQBridge()
    result = bridge.quantum_sample({...}, n_qubits=5)
    assert len(result.samples) > 0
```

### Test Using Mock (CI Safe)
```python
@pytest.mark.cuda_q_mock
def test_orchestrator_with_mock(cuda_q_mocked_import):
    from python.quantum.orchestrator import QuantumOrchestrator
    orch = QuantumOrchestrator(backend="cuda_q")  # Uses mock
    status = orch.get_backend_status()
    assert status["cuda_q"]["available"] is True
```

### Test Data Factory
```python
@pytest.mark.cuda_q_mock
def test_sample_generation(sample_factory):
    samples = sample_factory.create_batch(n_samples=10, n_qubits=5)
    assert len(samples) == 10
    assert all(len(s["bitstring"]) == 5 for s in samples)
```

## CI/CD Integration

### GitHub Actions Example
```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install pytest pytest-mock
          # Don't install CUDA-Q in CI
      
      - name: Run mock tests
        run: pytest -m "cuda_q_mock"  # Skips GPU tests
```

## Troubleshooting

### CUDA-Q ImportError in tests
```
ERROR: CUDA-Q is REQUIRED but not available.

Solution:
  # Option 1: Install CUDA-Q
  pip install cuda-quantum>=0.8.0
  
  # Option 2: Use mock tests in CI
  pytest -m "cuda_q_mock"
  
  # Option 3: Use conftest.py fixtures to mock
  pytest -m "cuda_q_mock" tests/
```

### GPU not detected
```
ERROR: Failed to initialize CUDA-Q backend: NVIDIA GPU not detected

Solution:
  # Check GPU availability
  nvidia-smi
  
  # Verify CUDA installation
  nvcc --version
  
  # Reinstall CUDA-Q
  pip install --upgrade cuda-quantum
```

### Mock not working in tests
```
ERROR: cudaq module not found

Solution:
  # Ensure conftest.py is in tests/ directory
  ls tests/conftest.py
  
  # Use cuda_q_mocked_import fixture
  def test_example(cuda_q_mocked_import):
      # Now 'import cudaq' will use mock
      pass
```

## Performance Benchmarks

### Expected Times (per 1024 shots)

| Test | Backend | Time | Notes |
|------|---------|------|-------|
| `test_cuda_q_gpu_*` | Real GPU | 2-5ms | 25-50x speedup |
| `test_cuda_q_mock_*` | Mock | <1ms | Instant, no GPU |
| `test_cirq_cpu_*` | Cirq | 50-200ms | CPU simulator |

### Running Benchmarks
```bash
# Run with timing
pytest -v --durations=10 tests/

# Profile GPU tests
pytest -m "cuda_q_gpu" -v --profile tests/
```

## Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
markers =
    cuda_q_gpu: Tests requiring GPU
    cuda_q_mock: Tests using mock
    integration: Integration tests
filterwarnings =
    ignore::DeprecationWarning
```

### pyproject.toml
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
markers = [
    "cuda_q_gpu: Tests requiring CUDA-Q GPU",
    "cuda_q_mock: Tests using mocked CUDA-Q",
]
```

## See Also

- `tests/conftest.py` - Pytest fixtures
- `tests/meta_learning/integration/test_orchestrator.py` - Integration tests
- `CUDA_Q_MANDATORY_REFACTORING.md` - Architecture overview
