# Simple CUDA-Q Installation Guide

## Status

❌ **CUDA-Q is NOT installed yet** - but it's available in your codebase

✅ **CUDA-Q source code**: 1.3 GB at `/root/Qallow/third_party/cuda-quantum/`

## Why Build Failed

The CMake build script tried to build CUDA-Q from source, but:
- CUDA-Q requires LLVM, cuQuantum, and many dependencies
- Building from source takes 30+ minutes
- Requires significant disk space and compilation resources

## Recommended Solution: Use Pre-built Wheels

### Option 1: Install from PyPI (Easiest)

```bash
# Activate your environment
source /root/Qallow/cirq-env/bin/activate

# Install CUDA-Q
pip install cuda-quantum

# Verify installation
python3 -c "import cudaq; print(f'CUDA-Q {cudaq.__version__} installed!')"
```

**Pros**: Fast, simple, works immediately
**Cons**: May not have latest features

### Option 2: Use Docker Image (Recommended for Full Features)

```bash
# Pull NVIDIA's CUDA-Q Docker image
docker pull nvcr.io/nvidia/cuda-quantum:latest

# Run container
docker run -it nvcr.io/nvidia/cuda-quantum:latest

# Inside container, CUDA-Q is already installed
python3 -c "import cudaq; print(cudaq.get_targets())"
```

**Pros**: Complete environment, all dependencies included
**Cons**: Requires Docker

### Option 3: Build from Source (Advanced)

Only if you need the latest development version:

```bash
cd /root/Qallow/third_party/cuda-quantum
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/root/Qallow/install

cmake --build . --parallel $(nproc)
cmake --install .

# Add to Python path
export PYTHONPATH=/root/Qallow/install/lib/python:$PYTHONPATH
```

**Pros**: Latest features, customizable
**Cons**: Takes 30+ minutes, requires dependencies

## Quick Test

After installation, test with:

```bash
python3 << 'EOF'
import cudaq

@cudaq.kernel
def bell_state():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cx(qubits[0], qubits[1])
    mz(qubits)

result = cudaq.sample(bell_state, shots=100)
print("✅ CUDA-Q is working!")
print(result)
EOF
```

## Available Targets After Installation

```bash
python3 -c "import cudaq; print('Available targets:', cudaq.get_targets())"
```

Expected output:
```
Available targets: ['qasm-sim', 'stim', 'nvidia-mqpu', ...]
```

## Troubleshooting

### "No module named 'cudaq'"
```bash
# Install CUDA-Q
pip install cuda-quantum
```

### "nvidia-mqpu not available"
```bash
# Install GPU support
pip install cuquantum
```

### "ImportError: libcustatevec.so"
```bash
# Install CUDA libraries
pip install cuquantum-cu11  # For CUDA 11
# or
pip install cuquantum-cu12  # For CUDA 12
```

## Next Steps

1. **Install CUDA-Q**:
   ```bash
   pip install cuda-quantum
   ```

2. **Run examples**:
   ```bash
   python3 /root/Qallow/examples_cudaq_quickstart.py
   ```

3. **Read guides**:
   ```bash
   cat /root/Qallow/CUDA_Q_GUIDE.md
   cat /root/Qallow/CUDAQ_QALLOW_INTEGRATION.md
   ```

4. **Integrate with Qallow**:
   - Add CUDA-Q to Phase 13, 14, 15
   - Build hybrid quantum-classical workflows

## Resources

- **Official Installation**: https://nvidia.github.io/cuda-quantum/install.html
- **Docker Hub**: https://hub.docker.com/r/nvcr.io/nvidia/cuda-quantum
- **PyPI Package**: https://pypi.org/project/cuda-quantum/

---

**Ready to use CUDA-Q!** 🚀

