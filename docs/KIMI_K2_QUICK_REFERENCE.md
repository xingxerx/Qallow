# Kimi-K2 Quick Reference

## Installation (One-time)

```bash
# Quick setup
bash scripts/setup_kimi_k2_quick_start.sh

# Or manual
pip install openai vllm transformers
```

## Running Kimi-K2

### Terminal 1: Start Inference Server

```bash
# vLLM (recommended)
bash scripts/setup_kimi_k2_vllm.sh

# Or SGLang
bash scripts/setup_kimi_k2_sglang.sh
```

Server runs on: `http://localhost:8000/v1`

### Terminal 2: Start Chat Server (Optional)

```bash
export QALLOW_CHAT_BACKEND=kimi_k2
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

Chat API runs on: `http://localhost:8008`

## Python Usage

### Simple Chat

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()
response = agent.chat("Hello!")
print(response)
```

### With Tools

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()

tools = [{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "My tool description",
        "parameters": {
            "type": "object",
            "required": ["param"],
            "properties": {"param": {"type": "string"}}
        }
    }
}]

def my_tool(param: str):
    return {"result": f"Processed {param}"}

response = agent.chat_with_tools(
    "Use my_tool with 'test'",
    tools=tools,
    tool_map={"my_tool": my_tool}
)
print(response)
```

## REST API

### Chat

```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Health Check

```bash
curl http://localhost:8008/health
```

## Configuration

### Environment Variables

```bash
export QALLOW_CHAT_BACKEND=kimi_k2
export KIMI_K2_BASE_URL=http://localhost:8000/v1
export CUDA_VISIBLE_DEVICES=0  # GPU selection
```

### Config File

Edit `config/kimi_k2.yaml`:
- `temperature`: 0.6 (recommended)
- `max_tokens`: 4096
- `enable_tools`: true/false

## Common Commands

```bash
# Setup
bash scripts/setup_kimi_k2_quick_start.sh

# Start vLLM
bash scripts/setup_kimi_k2_vllm.sh

# Start SGLang
bash scripts/setup_kimi_k2_sglang.sh

# Start chat server
export QALLOW_CHAT_BACKEND=kimi_k2
cd python/chat_server && uvicorn main:app --port 8008

# Test connection
python3 -c "from python.agents.kimi_k2_agent import create_kimi_k2_agent; print(create_kimi_k2_agent().chat('Hi'))"
```

## Multi-GPU Setup

```bash
# 4 GPUs with tensor parallelism
bash scripts/setup_kimi_k2_vllm.sh moonshotai/Kimi-K2-Instruct 8000 4 0.85

# Specify GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/setup_kimi_k2_vllm.sh
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Start vLLM: `bash scripts/setup_kimi_k2_vllm.sh` |
| Out of memory | Reduce GPU util: `bash scripts/setup_kimi_k2_vllm.sh ... 0.7` |
| Slow inference | Use GPU, increase batch size, use vLLM |
| Model not found | Will auto-download from HuggingFace |

## Files

| File | Purpose |
|------|---------|
| `python/agents/kimi_k2_agent.py` | Main agent class |
| `config/kimi_k2.yaml` | Configuration |
| `scripts/setup_kimi_k2_vllm.sh` | vLLM setup |
| `scripts/setup_kimi_k2_sglang.sh` | SGLang setup |
| `docs/KIMI_K2_INTEGRATION.md` | Full documentation |

## Model Info

- **Name**: Kimi-K2-Instruct
- **Parameters**: 1T total (32B activated)
- **Context**: 128K tokens
- **Format**: block-fp8
- **License**: Modified MIT
- **Source**: [GitHub](https://github.com/MoonshotAI/Kimi-K2)

## Performance

- **Recommended Temperature**: 0.6
- **Max Tokens**: 4096
- **GPU Memory**: ~40GB for single GPU
- **Throughput**: Depends on GPU and batch size

## Next Steps

1. Run quick start: `bash scripts/setup_kimi_k2_quick_start.sh`
2. Start inference: `bash scripts/setup_kimi_k2_vllm.sh`
3. Test: `python3 -c "from python.agents.kimi_k2_agent import create_kimi_k2_agent; print(create_kimi_k2_agent().chat('Hi'))"`
4. Read full docs: `docs/KIMI_K2_INTEGRATION.md`

