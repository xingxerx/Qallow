# Qallow AI Agents

Autonomous AI agents for quantum optimization, ethics validation, and system tuning.

## Overview

The Qallow agents module provides intelligent, autonomous agents that integrate with the Qallow quantum computing framework. These agents use large language models (LLMs) to optimize quantum algorithms, validate ethics constraints, and tune system parameters.

## Features

- **🤖 Autonomous QAOA Optimization**: LLM-guided parameter tuning for Phase 14
- **🛡️ Ethics Validation**: Automatic Phase 13 ethics checks before inference
- **🚀 Multi-GPU Support**: Distributed inference with Ray/MPI
- **🔒 Local & Private**: No cloud dependencies, runs entirely on your hardware
- **📊 Full Telemetry**: Comprehensive logging and metrics

## Quick Start

### 1. Install Ollama

```bash
# Run quick start script
./scripts/quick_start_ollama.sh
```

Or manually:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start service
ollama serve &

# Pull a model
ollama pull llama2:13b  # For testing
ollama pull llama2:70b  # For production
```

### 2. Test the Agent

```bash
# Direct agent test
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --nodes 256 \
  --target 0.981

# With Phase 14
./build/qallow phase 14 --agent-ollama
```

## Architecture

```
python/agents/
├── __init__.py                  # Module exports
├── qallow_agent_ollama.py       # Main Ollama agent
└── README.md                    # This file

Integration Points:
├── Phase 13 (Ethics)            # backend/cpu/phase13_harmonic.c
├── Phase 14 (QAOA)              # backend/cpu/phase14_coherence.c
├── Chat Server                  # python/chat_server/main.py
└── Native App                   # native_app/src/backend/api_client.rs
```

## Agent Types

### OllamaAgent

**Purpose**: Autonomous quantum optimization using local LLMs

**Capabilities**:
- QAOA parameter optimization
- Phase 13 ethics validation
- Multi-GPU distributed inference
- Telemetry and logging

**Usage**:
```python
from python.agents.qallow_agent_ollama import OllamaAgent, OllamaConfig

# Create agent
config = OllamaConfig(
    model="llama2:70b",
    num_gpu=8,
    qaoa_nodes=256,
    qaoa_target_fidelity=0.981
)
agent = OllamaAgent(config)

# Run optimization
result = agent.optimize_qaoa()
print(result)
# {
#   "p": 3,
#   "gamma": 0.42,
#   "beta": 0.19,
#   "alpha_eff": 0.0048,
#   "reasoning": "..."
# }
```

## Configuration

### OllamaConfig

```python
@dataclass
class OllamaConfig:
    # Model configuration
    model: str = "llama2:70b"
    host: str = "http://localhost:11434"
    temperature: float = 0.3
    num_gpu: int = 8
    num_experts: int = 8  # For MoE models
    max_tokens: int = 4096
    timeout: int = 300
    
    # Qallow paths
    data_dir: Path = Path("data/quantum")
    log_dir: Path = Path("data/logs")
    output_file: Path = Path("data/quantum/agent_output.jsonl")
    gain_json: Path = Path("data/quantum/ollama_gain.json")
    
    # Ethics
    ethics_enabled: bool = True
    ethics_threshold: float = 0.85
    
    # QAOA
    qaoa_nodes: int = 256
    qaoa_target_fidelity: float = 0.981
```

### Environment Variables

```bash
# Backend selection
export QALLOW_CHAT_BACKEND=ollama

# Ollama configuration
export OLLAMA_MODEL=llama2:70b
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_NUM_GPU=8

# Agent configuration
export QALLOW_AGENT_ETHICS=1
export QALLOW_AGENT_THRESHOLD=0.85
```

## CLI Usage

### Direct Agent Invocation

```bash
# QAOA optimization
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:70b \
  --nodes 256 \
  --target 0.981 \
  --num-gpu 8

# Get status
python3 -m python.agents.qallow_agent_ollama --task status

# Disable ethics gate (for testing)
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --no-ethics
```

### Via Qallow Binary

```bash
# Phase 14 with Ollama agent
./build/qallow phase 14 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --agent-ollama

# With custom model
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=deepseek-v3:70b
```

### Via Chat Server

```bash
# Start server
export QALLOW_CHAT_BACKEND=ollama
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008

# Use API
curl -X POST http://localhost:8008/quantum/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "qaoa_optimize",
    "nodes": 256,
    "target_fidelity": 0.981
  }'
```

## Output Files

### 1. Agent Output Log
**Path**: `data/quantum/agent_output.jsonl`

JSONL format with one entry per task:
```json
{
  "timestamp": 1234567890.123,
  "session_id": "agent_1234567890",
  "task": "qaoa_optimize",
  "model": "llama2:70b",
  "duration_ms": 5432,
  "input": {"nodes": 256, "target_fidelity": 0.981},
  "output": {
    "p": 3,
    "gamma": 0.42,
    "beta": 0.19,
    "alpha_eff": 0.0048,
    "reasoning": "MoE routing stabilizes coherence"
  },
  "ethics": {"pass": true, "score": 0.92}
}
```

### 2. Gain JSON
**Path**: `data/quantum/ollama_gain.json`

Consumed by Phase 14:
```json
{
  "alpha_eff": 0.0048,
  "timestamp": 1234567890.123,
  "session_id": "agent_1234567890",
  "source": "ollama_agent"
}
```

## Testing

```bash
# Run all tests
pytest tests/test_ollama_agent.py -v

# Run specific test
pytest tests/test_ollama_agent.py::TestOllamaAgent::test_qaoa_optimization -v

# Run integration tests
pytest tests/test_ollama_agent.py -v -m integration
```

## Supported Models

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| `llama2:7b` | 7B | 8GB | Fast | Good |
| `llama2:13b` | 13B | 16GB | Medium | Better |
| `llama2:70b` | 70B | 80GB | Slow | Best |
| `deepseek-v3:70b` | 70B MoE | 80GB | Medium | Best |
| `codellama:70b` | 70B | 80GB | Slow | Best (code) |

## Performance Tips

### Multi-GPU Setup

```bash
# Set GPU count
export OLLAMA_NUM_GPU=8

# Verify usage
watch -n 1 nvidia-smi
```

### Distributed Inference

```bash
# Head node
ray start --head --num-gpus=8

# Worker nodes
ray start --address=<HEAD_IP>:6379 --num-gpus=8
```

### Model Quantization

```bash
# 4-bit quantization (less VRAM)
ollama pull llama2:70b-q4

# 8-bit quantization (balanced)
ollama pull llama2:70b-q8
```

## Troubleshooting

### Ollama Not Running
```bash
curl http://localhost:11434/api/tags
ollama serve &
```

### Model Not Found
```bash
ollama list
ollama pull llama2:70b
```

### Out of Memory
```bash
# Use smaller model
ollama pull llama2:13b

# Or reduce context
export OLLAMA_NUM_CTX=2048
```

### Ethics Gate Failing
```bash
# Test Phase 13
./build/qallow phase 13 --ticks=10

# Disable for testing
python3 -m python.agents.qallow_agent_ollama --no-ethics
```

## Documentation

- **Full Guide**: [docs/OLLAMA_AGENT_GUIDE.md](../../docs/OLLAMA_AGENT_GUIDE.md)
- **DeepSeek Integration**: [DEEPSEEK_INTEGRATION.md](../../DEEPSEEK_INTEGRATION.md)
- **Phase 13 Ethics**: [docs/guides/PHASE13_ETHICS_GUIDE.md](../../docs/guides/PHASE13_ETHICS_GUIDE.md)
- **Phase 14 QAOA**: [docs/guides/PHASE14_QAOA_GUIDE.md](../../docs/guides/PHASE14_QAOA_GUIDE.md)

## Contributing

When adding new agents:

1. Create agent class in `python/agents/`
2. Add to `__init__.py` exports
3. Add CLI integration in `interface/main.c`
4. Add tests in `tests/test_*.py`
5. Update documentation

## License

See [LICENSE](../../LICENSE) for details.

