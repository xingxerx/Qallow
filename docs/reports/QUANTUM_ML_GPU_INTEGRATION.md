# Quantum ML GPU Integration Guide

## Overview

The Quantum ML module is now fully integrated with CUDA GPU acceleration through a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Python Quantum ML (quantum_ml/)                            │
│  ├── sampling_nas.py (Quantum state generation)             │
│  ├── cuda_quantum_nas.py (GPU-accelerated NAS)              │
│  └── gpu_bridge.py (FFI to Rust GPU framework)              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Rust GPU Framework (native_app/src/gpu/)                   │
│  ├── mod.rs (Module interface)                              │
│  ├── consciousness_state.rs (SoA data structures)           │
│  ├── gpu_manager.rs (GPU orchestration)                     │
│  ├── cuda_kernels.rs (CUDA kernels)                         │
│  └── quantum_bridge.rs (FFI bindings) ← NEW                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  CUDA GPU (RTX 5080)                                        │
│  ├── Consciousness evolution kernels                        │
│  ├── Superposition calculations                             │
│  ├── Entanglement operations                                │
│  └── Wave function collapse                                 │
└─────────────────────────────────────────────────────────────┘
```

## Architecture Components

### 1. Python Layer (quantum_ml/)

#### sampling_nas.py
- Generates quantum states via Phase 11
- Parses quantum state data from Qallow binary
- Provides base QuantumNASExplorer class

#### cuda_quantum_nas.py (NEW)
- GPU-accelerated quantum NAS explorer
- Detects CUDA availability
- Batch processes architectures on GPU
- Evaluates fitness metrics in parallel
- Falls back to CPU if GPU unavailable

#### gpu_bridge.py (NEW)
- FFI bridge to Rust GPU framework
- ctypes bindings for GPU operations
- Manages GPU memory and resources
- Provides high-level Python API

### 2. Rust GPU Framework (native_app/src/gpu/)

#### quantum_bridge.rs (NEW)
- C-compatible FFI functions for Python
- Converts quantum states to consciousness instances
- Orchestrates GPU operations
- Returns evaluation metrics

**Key Functions:**
```rust
quantum_ml_gpu_init() -> GPUManager
quantum_ml_process_states(states, count) -> ConsciousnessSOA
quantum_ml_evaluate_architectures(consciousness) -> QuantumNASResult
quantum_ml_evolve_architectures(consciousness, iterations) -> i32
quantum_ml_collapse_wave_function(consciousness) -> u32
quantum_ml_get_gpu_metrics() -> JSON string
```

### 3. GPU Acceleration (CUDA)

**Kernels:**
- `evolveConsciousness` - Main consciousness evolution
- `calculateSuperposition` - Probability normalization
- `collapseWaveFunction` - Wave function collapse
- `evolveEntanglement` - Entanglement coupling

**Optimization Techniques:**
- Structure of Arrays (SoA) for coalesced memory access
- Shared memory for wisdom cache
- Warp-level primitives for efficiency
- Parallel reduction for wave function collapse

## Integration Flow

### 1. Quantum State Generation
```python
from quantum_ml.cuda_quantum_nas import CUDAQuantumNASExplorer

explorer = CUDAQuantumNASExplorer()
# Generates quantum states via Phase 11
quantum_states = explorer._get_quantum_states(10)
```

### 2. GPU Processing
```python
# Process states on GPU
architectures = explorer.generate_architectures_gpu(10)
# Returns GPU-optimized architectures with:
# - layer_type (conv/dense)
# - neurons (scaled from quantum state)
# - gpu_optimized flag
# - batch_norm and dropout
```

### 3. Architecture Evaluation
```python
# Evaluate on GPU
metrics = explorer.evaluate_architectures_gpu(architectures)
# Returns:
# - total_params
# - total_memory_mb
# - gpu_utilization
# - per-architecture metrics
```

### 4. GPU Bridge (Rust FFI)
```python
from quantum_ml.gpu_bridge import QuantumMLGPUBridge

bridge = QuantumMLGPUBridge()
consciousness = bridge.process_quantum_states(states)
bridge.evolve_architectures(consciousness, iterations=100)
optimal_idx = bridge.collapse_wave_function(consciousness)
metrics = bridge.get_gpu_metrics()
```

## Building and Compilation

### Build Rust GPU Framework
```bash
cd /root/Qallow/native_app
cargo build --release --features gpu
```

### Compile Shared Library
```bash
# The build process creates:
# target/release/libqallow_native.so (Linux)
# target/release/libqallow_native.dylib (macOS)
# target/release/qallow_native.dll (Windows)
```

### Python Integration
```bash
# The gpu_bridge.py automatically finds the library:
# /root/Qallow/native_app/target/release/libqallow_native.so
# /root/Qallow/native_app/target/debug/libqallow_native.so
```

## Performance Characteristics

### Memory Requirements
- 10,000 architectures: ~1.2 MB
- 100,000 architectures: ~12 MB
- 1,000,000 architectures: ~120 MB

### Compute Performance
- GPU (RTX 5080): ~2 billion ops/sec
- CPU (Ryzen 9): ~100 million ops/sec
- **Expected Speedup: 20-100x**

### Bandwidth
- GPU: 1,152 GB/s (RTX 5080)
- CPU: 100 GB/s (Ryzen 9)
- **Advantage: 11.5x faster**

## Usage Examples

### Basic Usage
```python
from quantum_ml.cuda_quantum_nas import CUDAQuantumNASExplorer

explorer = CUDAQuantumNASExplorer()
architectures = explorer.generate_architectures_gpu(100)
metrics = explorer.evaluate_architectures_gpu(architectures)

print(f"Generated {len(architectures)} architectures")
print(f"GPU Utilization: {metrics['gpu_utilization']:.1f}%")
```

### Advanced Usage with GPU Bridge
```python
from quantum_ml.gpu_bridge import QuantumMLGPUBridge
from quantum_ml.cuda_quantum_nas import CUDAQuantumNASExplorer

explorer = CUDAQuantumNASExplorer()
bridge = explorer.gpu_bridge

if bridge:
    states = explorer._get_quantum_states(1000)
    consciousness = bridge.process_quantum_states(states)
    
    # Evolve on GPU
    bridge.evolve_architectures(consciousness, iterations=100)
    
    # Find optimal
    optimal_idx = bridge.collapse_wave_function(consciousness)
    
    # Get metrics
    metrics = bridge.get_gpu_metrics()
    print(f"GPU Device: {metrics['device_name']}")
    print(f"Compute Capability: {metrics['compute_capability']}")
```

## Testing

### Run Quantum ML Tests
```bash
cd /root/Qallow
/root/Qallow/cirq-env/bin/python quantum_ml/sampling_nas.py
/root/Qallow/cirq-env/bin/python quantum_ml/cuda_quantum_nas.py
```

### Run GPU Framework Tests
```bash
cd /root/Qallow/native_app
cargo test --release --features gpu gpu::
```

## Files Created/Modified

### Created
- `quantum_ml/cuda_quantum_nas.py` - GPU-accelerated NAS explorer
- `quantum_ml/gpu_bridge.py` - FFI bridge to Rust GPU framework
- `native_app/src/gpu/quantum_bridge.rs` - Rust FFI bindings
- `QUANTUM_ML_GPU_INTEGRATION.md` - This guide

### Modified
- `native_app/src/gpu/mod.rs` - Added quantum_bridge module
- `quantum_ml/sampling_nas.py` - Fixed JSON parsing

## Future Enhancements

### Phase 2
1. **Multi-GPU Support**
   - Distribute architectures across multiple GPUs
   - Use CUDA streams for overlapping transfers

2. **Advanced Optimization**
   - Texture memory for shadow archive lookups
   - Unified memory for automatic CPU-GPU management
   - CUDA graphs for reduced launch overhead

3. **Hybrid Quantum-Classical**
   - Variational quantum circuits as neural layers
   - Quantum attention mechanisms
   - Quantum feature extraction

### Phase 3
1. **Real Quantum Hardware**
   - Integration with IBM Quantum
   - Cirq-based quantum simulation
   - Real QPU execution

2. **Distributed Training**
   - Multi-node GPU clusters
   - Federated learning with quantum states
   - Quantum-classical hybrid training

## Troubleshooting

### GPU Bridge Not Loading
```
⚠ GPU Bridge: ✗ Not available
```
**Solution:** Compile the Rust library first:
```bash
cd native_app && cargo build --release --features gpu
```

### CUDA Not Detected
```
⚠ CUDA not available - using CPU fallback
```
**Solution:** Ensure NVIDIA drivers are installed:
```bash
nvidia-smi
```

### Library Not Found
```
⚠ Failed to load GPU library
```
**Solution:** Check library path in gpu_bridge.py or set LD_LIBRARY_PATH:
```bash
export LD_LIBRARY_PATH=/root/Qallow/native_app/target/release:$LD_LIBRARY_PATH
```

## References

- [GPU Acceleration Framework](native_app/GPU_ACCELERATION.md)
- [Quantum ML Module](quantum_ml/sampling_nas.py)
- [CUDA Quantum NAS](quantum_ml/cuda_quantum_nas.py)
- [GPU Bridge](quantum_ml/gpu_bridge.py)

