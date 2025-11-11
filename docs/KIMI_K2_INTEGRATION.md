# Kimi-K2 Integration Guide

## Overview

Kimi-K2 is a state-of-the-art open-source LLM developed by Moonshot AI with:
- **1 Trillion total parameters** (32B activated)
- **Mixture-of-Experts (MoE)** architecture
- **128K context length**
- **Strong tool-calling capabilities**
- **No API key required** for local inference

This guide shows how to integrate Kimi-K2 into Qallow for local inference without API keys.

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
bash scripts/setup_kimi_k2_quick_start.sh
```

This installs:
- OpenAI SDK (for OpenAI-compatible API)
- vLLM (recommended inference engine)
- Required Python packages

### 2. Start Inference Server

**Option A: vLLM (Recommended)**
```bash
bash scripts/setup_kimi_k2_vllm.sh
```

**Option B: SGLang**
```bash
bash scripts/setup_kimi_k2_sglang.sh
```

The server will start on `http://localhost:8000/v1`

### 3. Start Chat Server

In a new terminal:
```bash
export QALLOW_CHAT_BACKEND=kimi_k2
export KIMI_K2_BASE_URL=http://localhost:8000/v1
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

### 4. Test It

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what is quantum computing?"}'
```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                    Qallow Application                   │
├─────────────────────────────────────────────────────────┤
│  python/agents/kimi_k2_agent.py (KimiK2Agent)          │
│  - Chat interface                                       │
│  - Tool calling support                                 │
│  - Streaming responses                                  │
├─────────────────────────────────────────────────────────┤
│  python/chat_server/main.py (FastAPI)                  │
│  - /chat endpoint                                       │
│  - /chat/tools endpoint                                 │
│  - /health endpoint                                     │
├─────────────────────────────────────────────────────────┤
│  OpenAI-Compatible API (localhost:8000/v1)             │
├─────────────────────────────────────────────────────────┤
│  Inference Engine (vLLM / SGLang / KTransformers)      │
├─────────────────────────────────────────────────────────┤
│  Kimi-K2 Model (Local or HuggingFace)                  │
└─────────────────────────────────────────────────────────┘
```

## Configuration

### Environment Variables

```bash
# Backend selection
export QALLOW_CHAT_BACKEND=kimi_k2

# Kimi-K2 server
export KIMI_K2_BASE_URL=http://localhost:8000/v1

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs

# vLLM specific
export VLLM_ATTENTION_BACKEND=flash_attn
```

### Config File

Edit `config/kimi_k2.yaml`:

```yaml
kimi_k2:
  model_name: "moonshotai/Kimi-K2-Instruct"
  engine: "vllm"
  base_url: "http://localhost:8000/v1"
  temperature: 0.6  # Recommended for Kimi-K2
  max_tokens: 4096
  enable_tools: true
```

## Usage Examples

### Basic Chat

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()
response = agent.chat("What is quantum computing?")
print(response)
```

### Tool Calling

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import json

agent = create_kimi_k2_agent()

# Define tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string", "description": "City name"}
            }
        }
    }
}]

# Tool implementation
def get_weather(city: str):
    return {"weather": "Sunny", "temp": 25}

tool_map = {"get_weather": get_weather}

# Chat with tools
response = agent.chat_with_tools(
    "What's the weather in Beijing?",
    tools=tools,
    tool_map=tool_map
)
print(response)
```

### REST API

```bash
# Simple chat
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum entanglement",
    "session_id": "user123"
  }'

# With tool calling
curl -X POST http://localhost:8008/chat/tools \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is the weather?",
    "tools": [...]
  }'

# Health check
curl http://localhost:8008/health
```

## Deployment Options

### Single GPU

```bash
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 1 0.9
```

### Multi-GPU (Tensor Parallelism)

```bash
# 4 GPUs
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 4 0.85
```

### Multi-Node (Advanced)

See `config/kimi_k2.yaml` for SGLang multi-node configuration.

## Performance Tips

1. **GPU Memory**: Use `--gpu-memory-utilization 0.9` for vLLM
2. **Batch Size**: Increase `max-num-seqs` for higher throughput
3. **Quantization**: Model uses block-fp8 format (already optimized)
4. **Temperature**: Use 0.6 (recommended for Kimi-K2)

## Troubleshooting

### Connection Error

```
Error: Failed to connect to Kimi-K2 at http://localhost:8000/v1
```

**Solution**: Ensure vLLM/SGLang server is running:
```bash
bash scripts/setup_kimi_k2_vllm.sh
```

### Out of Memory

```
CUDA out of memory
```

**Solution**: Reduce GPU memory utilization:
```bash
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 1 0.7
```

### Slow Inference

**Solution**: 
- Use GPU (not CPU)
- Increase batch size
- Use vLLM instead of SGLang
- Enable flash attention

## Integration with Qallow

### Phase 14 (QAOA Optimization)

```bash
./build/qallow phase 14 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --agent-kimi-k2
```

### Chat Server

```bash
export QALLOW_CHAT_BACKEND=kimi_k2
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

## Files Added

- `python/agents/kimi_k2_agent.py` - Main integration module
- `config/kimi_k2.yaml` - Configuration file
- `scripts/setup_kimi_k2_vllm.sh` - vLLM setup script
- `scripts/setup_kimi_k2_sglang.sh` - SGLang setup script
- `scripts/setup_kimi_k2_quick_start.sh` - Quick start script
- `docs/KIMI_K2_INTEGRATION.md` - This file

## References

- [Kimi-K2 GitHub](https://github.com/MoonshotAI/Kimi-K2)
- [vLLM Documentation](https://docs.vllm.ai/)
- [SGLang Documentation](https://sglang.ai/)
- [Moonshot AI Platform](https://platform.moonshot.ai/)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review `config/kimi_k2.yaml` for configuration options
3. Check logs in `data/logs/kimi_k2.log`
4. Visit [Kimi-K2 GitHub Issues](https://github.com/MoonshotAI/Kimi-K2/issues)

