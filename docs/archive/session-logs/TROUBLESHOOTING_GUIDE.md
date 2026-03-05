# Troubleshooting Guide: CUDA + Cirq + Kimi-K2 + CUDA-Q

## Quick Diagnostics

Run this to check all components:

```bash
python3 << 'EOF'
import sys
print("=" * 70)
print("SYSTEM DIAGNOSTICS")
print("=" * 70)

# CUDA
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"✗ PyTorch: {e}")

# Cirq
try:
    import cirq
    print(f"✓ Cirq: {cirq.__version__}")
except Exception as e:
    print(f"✗ Cirq: {e}")

# CUDA-Q
try:
    import cudaq
    print(f"✓ CUDA-Q: {cudaq.__version__}")
except Exception as e:
    print(f"✗ CUDA-Q: {e}")

# Kimi-K2
try:
    from python.agents.kimi_k2_agent import create_kimi_k2_agent
    print(f"✓ Kimi-K2 Agent: Available")
except Exception as e:
    print(f"✗ Kimi-K2 Agent: {e}")

# vLLM Server
try:
    import requests
    response = requests.get("http://localhost:8000/v1/models", timeout=2)
    print(f"✓ vLLM Server: Running")
except:
    print(f"✗ vLLM Server: Not running")

print("=" * 70)
EOF
```

---

## Common Issues & Solutions

### 1. CUDA Not Available

**Error**: `CUDA not available` or `torch.cuda.is_available() = False`

**Causes**:
- NVIDIA driver not installed
- CUDA toolkit not installed
- PyTorch CPU version installed

**Solutions**:

```bash
# Check NVIDIA driver
nvidia-smi

# If not found, install driver
# Ubuntu/Debian:
sudo apt-get install nvidia-driver-XXX

# Reinstall PyTorch with CUDA
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Verify**:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

### 2. Cirq Installation Issues

**Error**: `ModuleNotFoundError: No module named 'cirq'`

**Solution**:
```bash
pip install cirq
```

**Verify**:
```bash
python3 -c "import cirq; print(cirq.__version__)"
```

---

### 3. CUDA-Q Installation Issues

**Error**: `ModuleNotFoundError: No module named 'cudaq'`

**Solution**:
```bash
# Install CUDA-Q (not cuda-quantum)
pip install cudaq

# If that fails, try:
pip install cudaq --upgrade
```

**Verify**:
```bash
python3 -c "import cudaq; print(cudaq.__version__)"
```

---

### 4. Kimi-K2 Server Not Running

**Error**: `Connection error` or `Failed to connect to http://localhost:8000/v1`

**Causes**:
- vLLM server not started
- Server crashed
- Port 8000 already in use

**Solutions**:

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# If not running, start it
bash scripts/setup_kimi_k2_vllm.sh

# If port 8000 is in use, use different port
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8001

# Update environment variable
export KIMI_K2_BASE_URL=http://localhost:8001/v1
```

**Check logs**:
```bash
tail -f data/logs/kimi_k2.log
```

---

### 5. Out of Memory (OOM) Errors

**Error**: `CUDA out of memory` or `RuntimeError: CUDA out of memory`

**Causes**:
- GPU memory insufficient for model
- Other processes using GPU memory
- GPU memory utilization too high

**Solutions**:

```bash
# Check GPU memory usage
nvidia-smi

# Kill other GPU processes
pkill -f vllm
pkill -f python

# Reduce GPU memory utilization
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 1 0.7

# Or use smaller model
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Base 8000 1 0.8
```

---

### 6. Slow Inference

**Problem**: Kimi-K2 responses are very slow (>10 seconds)

**Causes**:
- Using CPU instead of GPU
- GPU memory swapping
- Model not optimized
- Network latency

**Solutions**:

```bash
# Verify GPU is being used
nvidia-smi  # Should show vllm process

# Check GPU utilization
watch -n 1 nvidia-smi

# Increase batch size
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 1 0.9

# Use flash attention
export VLLM_ATTENTION_BACKEND=flash_attn
bash scripts/setup_kimi_k2_vllm.sh
```

---

### 7. CUDA-Q Kernel Errors

**Error**: `error: unhandled function call - xxx`

**Causes**:
- Using wrong gate names
- Gate not available in CUDA-Q

**Solutions**:

```python
# Use correct CUDA-Q gates:
# h() - Hadamard
# x() - Pauli X
# y() - Pauli Y
# z() - Pauli Z
# cx() - CNOT (not x.ctrl())
# cz() - Controlled Z
# rx(), ry(), rz() - Rotation gates

# Correct example:
@cudaq.kernel
def correct_kernel():
    q0 = cudaq.qubit()
    q1 = cudaq.qubit()
    h(q0)
    cx(q0, q1)  # Not x.ctrl(q0, q1)
```

---

### 8. Cirq Circuit Issues

**Error**: `ValueError: Unsupported operation` or circuit won't simulate

**Causes**:
- Unsupported gate
- Invalid qubit references
- Measurement issues

**Solutions**:

```python
# Use standard Cirq gates
import cirq

q0, q1 = cirq.LineQubit.range(2)

# Correct circuit
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# Simulate
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
```

---

### 9. Port Already in Use

**Error**: `Address already in use` or `Port 8000 already in use`

**Solution**:

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8001
```

---

### 10. Model Download Issues

**Error**: `Connection error` or `Failed to download model`

**Causes**:
- Network issues
- HuggingFace API down
- Insufficient disk space

**Solutions**:

```bash
# Check disk space
df -h

# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Download model manually
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "moonshotai/Kimi-K2-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
EOF

# Then start vLLM
bash scripts/setup_kimi_k2_vllm.sh
```

---

## Performance Optimization

### Enable Flash Attention

```bash
export VLLM_ATTENTION_BACKEND=flash_attn
bash scripts/setup_kimi_k2_vllm.sh
```

### Multi-GPU Setup

```bash
# 4 GPUs with tensor parallelism
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 4 0.85

# Specify GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/setup_kimi_k2_vllm.sh
```

### Increase Batch Size

```bash
# Edit config/kimi_k2.yaml
# Increase max_num_seqs and max_num_batched_tokens
```

---

## Testing & Verification

### Test CUDA
```bash
python3 -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = torch.matmul(x, y)
print('✓ CUDA working')
"
```

### Test Cirq
```bash
python3 -c "
import cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
simulator = cirq.Simulator()
result = simulator.simulate(circuit)
print('✓ Cirq working')
"
```

### Test CUDA-Q
```bash
python3 -c "
import cudaq
@cudaq.kernel
def bell():
    q0 = cudaq.qubit()
    q1 = cudaq.qubit()
    h(q0)
    cx(q0, q1)
result = cudaq.sample(bell, shots_count=100)
print('✓ CUDA-Q working')
"
```

### Test Kimi-K2
```bash
python3 -c "
from python.agents.kimi_k2_agent import create_kimi_k2_agent
agent = create_kimi_k2_agent()
response = agent.chat('Hello!')
print('✓ Kimi-K2 working')
"
```

### Run Full Integration Tests
```bash
python3 tests/test_integration_cuda_cirq_kimi_cudaq.py
python3 tests/test_qaoa_with_kimi_k2.py
```

---

## Getting Help

1. **Check logs**:
   ```bash
   tail -f data/logs/kimi_k2.log
   ```

2. **Run diagnostics**:
   ```bash
   python3 tests/test_integration_cuda_cirq_kimi_cudaq.py
   ```

3. **Check documentation**:
   - `docs/KIMI_K2_INTEGRATION.md`
   - `docs/KIMI_K2_QUICK_REFERENCE.md`
   - `INTEGRATION_TEST_RESULTS.md`

4. **Check GitHub issues**:
   - [Kimi-K2 Issues](https://github.com/MoonshotAI/Kimi-K2/issues)
   - [vLLM Issues](https://github.com/vllm-project/vllm/issues)
   - [CUDA-Q Issues](https://github.com/NVIDIA/cuda-quantum/issues)

---

## Quick Reference

| Component | Check | Fix |
|-----------|-------|-----|
| CUDA | `nvidia-smi` | Install NVIDIA driver |
| PyTorch | `python3 -c "import torch; print(torch.cuda.is_available())"` | `pip install torch` |
| Cirq | `python3 -c "import cirq"` | `pip install cirq` |
| CUDA-Q | `python3 -c "import cudaq"` | `pip install cudaq` |
| Kimi-K2 | `curl http://localhost:8000/v1/models` | `bash scripts/setup_kimi_k2_vllm.sh` |

---

## Emergency Reset

If everything is broken:

```bash
# Kill all processes
pkill -f vllm
pkill -f python

# Clear cache
rm -rf ~/.cache/huggingface
rm -rf data/kimi_k2/*

# Reinstall dependencies
pip install --upgrade torch cirq cudaq openai vllm

# Start fresh
bash scripts/setup_kimi_k2_quick_start.sh
bash scripts/setup_kimi_k2_vllm.sh
```

---

## Success Indicators

✓ All tests pass:
```bash
python3 tests/test_integration_cuda_cirq_kimi_cudaq.py
```

✓ vLLM server running:
```bash
curl http://localhost:8000/v1/models
```

✓ GPU being used:
```bash
nvidia-smi  # Shows vllm process
```

✓ Kimi-K2 responding:
```bash
python3 -c "from python.agents.kimi_k2_agent import create_kimi_k2_agent; print(create_kimi_k2_agent().chat('Hi'))"
```

