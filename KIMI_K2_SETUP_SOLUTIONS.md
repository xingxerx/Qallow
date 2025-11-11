# Kimi-K2 Setup Solutions

## Problem Summary

The Kimi-K2 model has compatibility issues with vLLM 0.11.0 on RTX 5080 (16GB VRAM):
- Model is based on DeepSeek V3 architecture
- vLLM doesn't have full support for this architecture yet
- Even with fp8 quantization, the model fails to load

## Solution Options

### Option 1: Use Alternative LLM (RECOMMENDED - Fastest)

Use a smaller, fully compatible model instead:

```bash
# Qwen2.5-7B (Recommended - best quality/speed)
bash scripts/setup_alternative_llm.sh qwen

# Llama 2 7B (Well-tested, stable)
bash scripts/setup_alternative_llm.sh llama

# Mistral 7B (Fast, good quality)
bash scripts/setup_alternative_llm.sh mistral
```

**Advantages:**
- ✅ Works immediately (no compatibility issues)
- ✅ Fits in 16GB VRAM
- ✅ Fast inference (7B models)
- ✅ Good quality responses
- ✅ Full tool calling support

**Disadvantages:**
- ❌ Smaller models (7B vs 1T)
- ❌ Shorter context (4K vs 128K)

**Recommended:** Start with Qwen2.5-7B

### Option 2: Wait for vLLM Update

vLLM team is working on better DeepSeek V3 support. Check:
- https://github.com/vllm-project/vllm/issues (search "deepseek")
- Update vLLM when new version is released

```bash
pip install --upgrade vllm
```

### Option 3: Use SGLang Instead of vLLM

SGLang has better DeepSeek support:

```bash
pip install sglang[all]
python3 -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-Instruct \
    --port 8000 \
    --quantization fp8 \
    --max-model-len 4096
```

### Option 4: Use Ollama (Simplest)

Ollama handles model management automatically:

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:7b
ollama serve
```

Then use the REST API on `http://localhost:11434`

## Quick Start (Recommended Path)

### Step 1: Use Alternative LLM (2 minutes)

```bash
bash scripts/setup_alternative_llm.sh qwen
```

Wait for the model to download and load (~5-15 minutes on first run).

### Step 2: Verify Server is Running

```bash
curl http://localhost:8000/v1/models
```

You should see:
```json
{
  "object": "list",
  "data": [
    {
      "id": "llm",
      "object": "model",
      "owned_by": "vllm"
    }
  ]
}
```

### Step 3: Test with Communication Methods

```bash
# Option A: REST API
python3 demo_rest_api_client.py

# Option B: Python SDK (requires adapter)
python3 demo_interactive.py
```

## Model Comparison

| Model | Size | VRAM | Speed | Quality | Context | Tool Call |
|-------|------|------|-------|---------|---------|-----------|
| Qwen2.5-7B | 7B | 12GB | Fast | Good | 128K | ✅ |
| Llama 2 7B | 7B | 12GB | Fast | Good | 4K | ✅ |
| Mistral 7B | 7B | 12GB | Fast | Good | 32K | ✅ |
| Kimi-K2 | 1T | 16GB+ | Slow | Excellent | 128K | ✅ |

## Adapter for Kimi-K2 Agent

If you want to keep using the Kimi-K2 agent code with alternative models:

Create `python/agents/llm_adapter.py`:

```python
from python.agents.kimi_k2_agent import KimiK2Agent, KimiK2Config

class LLMAdapter(KimiK2Agent):
    """Adapter to use alternative LLMs with Kimi-K2 agent interface"""
    
    def __init__(self, model_name: str = "llm", base_url: str = "http://localhost:8000/v1"):
        config = KimiK2Config(
            base_url=base_url,
            model_name=model_name,
            temperature=0.6,
            max_tokens=4096
        )
        super().__init__(config)

# Usage
agent = LLMAdapter()
response = agent.chat("Hello!")
print(response)
```

## Troubleshooting

### Issue: "Connection refused" on localhost:8000

**Solution:**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# If not running, start it
bash scripts/setup_alternative_llm.sh qwen
```

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Reduce memory usage
bash scripts/setup_alternative_llm.sh qwen
# Then modify the script to use:
# --gpu-memory-utilization 0.6
# --max-model-len 2048
```

### Issue: "Model not found"

**Solution:**
```bash
# Pre-download the model
python3 -c "from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

## Next Steps

1. **Choose your approach:**
   - ✅ **Recommended:** Use alternative LLM (Option 1)
   - ⏳ **Wait:** For vLLM update (Option 2)
   - 🔧 **Advanced:** Use SGLang (Option 3)
   - 🎯 **Simplest:** Use Ollama (Option 4)

2. **Start the server:**
   ```bash
   bash scripts/setup_alternative_llm.sh qwen
   ```

3. **Test communication:**
   ```bash
   python3 demo_rest_api_client.py
   ```

4. **Integrate with your app:**
   - Use REST API endpoints
   - Or use Python SDK with adapter

## Resources

- **Qwen2.5 Docs:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **vLLM Docs:** https://docs.vllm.ai/
- **Ollama:** https://ollama.ai/
- **SGLang:** https://github.com/hiyouga/LLaMA-Factory

## Support

If you want to use Kimi-K2 specifically:
1. Monitor vLLM GitHub for DeepSeek V3 support updates
2. Try SGLang (better DeepSeek support)
3. Use a machine with 24GB+ VRAM
4. Contact Moonshot AI for deployment guidance

