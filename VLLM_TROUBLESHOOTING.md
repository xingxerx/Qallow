# vLLM Kimi-K2 Troubleshooting Guide

## Issue: "Engine core initialization failed" or GPU Memory Error

### Root Cause
The Kimi-K2 model is very large (~50GB) and requires significant GPU memory. With 16GB VRAM, you need to use quantization and reduce context length.

### Solution 1: Use Optimized Setup Script (RECOMMENDED)

```bash
bash scripts/setup_kimi_k2_vllm_optimized.sh
```

This script uses:
- **fp8 quantization**: Reduces model size by ~75%
- **max-model-len: 4096**: Reduced from 131K for memory efficiency
- **gpu-memory-utilization: 0.65**: Conservative to avoid OOM
- **max-num-batched-tokens: 4096**: Reduced for stability

### Solution 2: Manual Configuration

If you want to use the standard script with custom settings:

```bash
# Set environment variables before running
export KIMI_K2_GPU_MEMORY_UTIL=0.65
export KIMI_K2_QUANTIZATION=fp8
export KIMI_K2_MAX_MODEL_LEN=4096

bash scripts/setup_kimi_k2_vllm.sh
```

### Solution 3: Use Smaller Model Alternative

If Kimi-K2 still doesn't fit, use a smaller model:

```bash
# Option A: Use Qwen2.5 (smaller, faster)
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --quantization fp8 \
    --gpu-memory-utilization 0.7

# Option B: Use Llama 2 (well-tested)
vllm serve meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --quantization fp8 \
    --gpu-memory-utilization 0.7
```

## Common Issues and Solutions

### Issue 1: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XXGiB
```

**Solutions:**
1. Reduce `gpu-memory-utilization` from 0.7 to 0.5
2. Reduce `max-model-len` from 4096 to 2048
3. Reduce `max-num-batched-tokens` from 4096 to 2048
4. Use a smaller model

**Example:**
```bash
vllm serve moonshotai/Kimi-K2-Instruct \
    --port 8000 \
    --quantization fp8 \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.5
```

### Issue 2: "Connection refused" on localhost:8000

**Symptoms:**
```
ConnectionError: Failed to connect to http://localhost:8000
```

**Solutions:**
1. Check if vLLM is still starting (first run downloads ~50GB model)
2. Check GPU memory: `nvidia-smi`
3. Check vLLM logs for errors
4. Restart the server

**Verification:**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Check GPU memory
nvidia-smi

# Check if port is in use
lsof -i :8000
```

### Issue 3: "Model not found" or Download Issues

**Symptoms:**
```
FileNotFoundError: Model not found
```

**Solutions:**
1. Ensure HuggingFace token is set (for gated models):
   ```bash
   huggingface-cli login
   ```

2. Pre-download the model:
   ```bash
   python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
   AutoTokenizer.from_pretrained('moonshotai/Kimi-K2-Instruct'); \
   AutoModelForCausalLM.from_pretrained('moonshotai/Kimi-K2-Instruct', trust_remote_code=True)"
   ```

3. Use a local model path:
   ```bash
   bash scripts/setup_kimi_k2_vllm.sh /path/to/local/model
   ```

### Issue 4: "Slow inference" or "High latency"

**Symptoms:**
- Responses take >10 seconds
- GPU utilization is low

**Solutions:**
1. Increase `gpu-memory-utilization` to 0.8
2. Increase `max-num-batched-tokens` to 8192
3. Increase `max-num-seqs` to 256
4. Check if other processes are using GPU: `nvidia-smi`

### Issue 5: "Tool calling not working"

**Symptoms:**
```
Tool calls not recognized or parsed incorrectly
```

**Solutions:**
1. Ensure `--tool-call-parser kimi_k2` is set
2. Ensure `--enable-auto-tool-choice` is set
3. Check vLLM version: `pip show vllm` (should be >=0.10.0)
4. Verify tool format in request

## Performance Tuning

### For 16GB VRAM (RTX 5080)

**Conservative (Stable):**
```bash
--gpu-memory-utilization 0.65
--max-model-len 4096
--max-num-batched-tokens 4096
--quantization fp8
```

**Balanced (Recommended):**
```bash
--gpu-memory-utilization 0.70
--max-model-len 8192
--max-num-batched-tokens 8192
--quantization fp8
```

**Aggressive (Fast, may OOM):**
```bash
--gpu-memory-utilization 0.80
--max-model-len 16384
--max-num-batched-tokens 16384
--quantization fp8
```

### For 24GB VRAM (RTX 6000)

```bash
--gpu-memory-utilization 0.80
--max-model-len 32768
--max-num-batched-tokens 16384
--quantization fp8
```

### For 40GB+ VRAM (A100, H100)

```bash
--gpu-memory-utilization 0.90
--max-model-len 131072
--max-num-batched-tokens 32768
--quantization fp8
```

## Monitoring

### Check Server Health

```bash
# Health check
curl http://localhost:8000/v1/models

# Get model info
curl http://localhost:8000/v1/models | jq

# Monitor GPU
watch -n 1 nvidia-smi
```

### Check Logs

```bash
# View vLLM logs (if running in background)
tail -f /tmp/vllm.log

# Check for errors
grep -i error /tmp/vllm.log
```

## Quick Start Commands

### Optimized Setup (Recommended)
```bash
bash scripts/setup_kimi_k2_vllm_optimized.sh
```

### Standard Setup with Custom Memory
```bash
export KIMI_K2_GPU_MEMORY_UTIL=0.65
bash scripts/setup_kimi_k2_vllm.sh
```

### Manual vLLM Command
```bash
vllm serve moonshotai/Kimi-K2-Instruct \
    --port 8000 \
    --quantization fp8 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.65 \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2
```

## Next Steps

1. **Try optimized setup:**
   ```bash
   bash scripts/setup_kimi_k2_vllm_optimized.sh
   ```

2. **Wait for model download** (first run only, ~10-30 minutes)

3. **Test the server:**
   ```bash
   curl http://localhost:8000/v1/models
   ```

4. **Use the communication methods:**
   - REST API: `python3 demo_rest_api_client.py`
   - Python SDK: `python3 demo_interactive.py`

## Support

- **vLLM Docs**: https://docs.vllm.ai/
- **Kimi-K2 Docs**: https://platform.moonshot.ai/docs/overview
- **GPU Memory Calculator**: https://huggingface.co/spaces/hf-accelerate/model-memory-usage

