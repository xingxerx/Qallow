# Qallow Ollama Agent Guide

Complete guide for setting up and using the Ollama-powered AI agent for autonomous quantum optimization.

## Overview

The Qallow Ollama Agent provides:
- **Autonomous QAOA optimization** for Phase 14
- **Phase 13 ethics validation** before LLM inference
- **Multi-GPU distributed inference** via Ray/MPI
- **Local, private LLM hosting** (no cloud dependencies)
- **Support for large models**: Llama2-70B, DeepSeek-V3, etc.

## Quick Start (5 Minutes)

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve &

# Pull a model (choose one)
ollama pull llama2:70b          # 70B parameter model (recommended)
ollama pull deepseek-v3:70b     # DeepSeek V3 (if available)
ollama pull llama2:13b          # Smaller, faster (for testing)
```

### 2. Test the Agent

```bash
# Test agent directly
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --nodes 256 \
  --target 0.981

# Expected output: JSON with optimized QAOA parameters
```

### 3. Run with Phase 14

```bash
# Build Qallow (if not already built)
./scripts/build_all.sh

# Run Phase 14 with Ollama agent
./build/qallow phase 14 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --agent-ollama

# Or specify a different model
./build/qallow phase 14 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --agent-ollama \
  --ollama-model=deepseek-v3:70b
```

## Supercomputer Setup (Multi-GPU)

For systems with multiple GPUs (8+ GPUs, 80GB+ VRAM):

```bash
# Run automated setup script
./scripts/setup_ollama_supercomputer.sh \
  --model llama2:70b \
  --num-gpu 8

# For distributed setup with Ray
./scripts/setup_ollama_supercomputer.sh \
  --model llama2:70b \
  --num-gpu 8 \
  --distributed \
  --head-node

# On worker nodes
./scripts/setup_ollama_supercomputer.sh \
  --worker-node <HEAD_NODE_IP>:6379 \
  --num-gpu 8
```

## Chat Server Integration

The Ollama agent integrates with the existing chat server:

### Start Chat Server with Ollama

```bash
# Set environment variables
export QALLOW_CHAT_BACKEND=ollama
export OLLAMA_MODEL=llama2:70b

# Start server
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

### API Endpoints

#### Chat Endpoint
```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain QAOA optimization",
    "session_id": "test",
    "backend": "ollama"
  }'
```

#### Quantum Task Endpoint
```bash
curl -X POST http://localhost:8008/quantum/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "qaoa_optimize",
    "nodes": 256,
    "target_fidelity": 0.981
  }'
```

## Configuration

### Environment Variables

```bash
# Backend selection
export QALLOW_CHAT_BACKEND=ollama    # ollama, mock, deepseek

# Ollama configuration
export OLLAMA_MODEL=llama2:70b       # Model to use
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_NUM_GPU=8              # Number of GPUs

# Agent configuration
export QALLOW_AGENT_ETHICS=1         # Enable Phase 13 ethics gate
export QALLOW_AGENT_THRESHOLD=0.85   # Ethics threshold
```

### Python Configuration

```python
from python.agents.qallow_agent_ollama import OllamaAgent, OllamaConfig

# Create custom config
config = OllamaConfig(
    model="llama2:70b",
    num_gpu=8,
    temperature=0.3,
    qaoa_nodes=256,
    qaoa_target_fidelity=0.981,
    ethics_enabled=True,
    ethics_threshold=0.85
)

# Initialize agent
agent = OllamaAgent(config)

# Run optimization
result = agent.optimize_qaoa(nodes=256, target_fidelity=0.981)
print(result)
```

## Output Files

The agent generates several output files:

### 1. Agent Output Log
**Location**: `data/quantum/agent_output.jsonl`

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
    "reasoning": "MoE routing stabilizes long-range coherence"
  },
  "ethics": {"pass": true, "score": 0.92, "reason": "Phase 13 validation"}
}
```

### 2. Gain JSON (for Phase 14)
**Location**: `data/quantum/ollama_gain.json`

```json
{
  "alpha_eff": 0.0048,
  "timestamp": 1234567890.123,
  "session_id": "agent_1234567890",
  "source": "ollama_agent"
}
```

This file is automatically consumed by Phase 14 when using `--agent-ollama`.

## Model Selection

### Recommended Models

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| `llama2:13b` | 13B | 16GB | Testing, development |
| `llama2:70b` | 70B | 80GB | Production, supercomputers |
| `deepseek-v3:70b` | 70B MoE | 80GB | Advanced reasoning |
| `codellama:70b` | 70B | 80GB | Code-focused tasks |

### Pull Models

```bash
# List available models
ollama list

# Pull a model
ollama pull llama2:70b

# Remove a model
ollama rm llama2:13b
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/test_ollama_agent.py -v

# Run specific test
pytest tests/test_ollama_agent.py::TestOllamaAgent::test_qaoa_optimization -v

# Run integration tests only
pytest tests/test_ollama_agent.py -v -m integration
```

### Manual Testing

```bash
# Test agent directly
python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize

# Test with custom parameters
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:70b \
  --nodes 512 \
  --target 0.99 \
  --num-gpu 8

# Get agent status
python3 -m python.agents.qallow_agent_ollama --task status
```

## Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve &

# Or as systemd service
sudo systemctl start ollama
```

### Model Not Found

```bash
# List available models
ollama list

# Pull missing model
ollama pull llama2:70b
```

### Out of Memory

```bash
# Use smaller model
ollama pull llama2:13b

# Or reduce context size
export OLLAMA_NUM_CTX=2048
```

### Ethics Gate Failing

```bash
# Disable ethics gate for testing
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --no-ethics

# Or check Phase 13 binary
./build/qallow phase 13 --ticks=10
```

## Performance Optimization

### Multi-GPU Setup

```bash
# Set number of GPUs
export OLLAMA_NUM_GPU=8

# Verify GPU usage
watch -n 1 nvidia-smi
```

### Distributed Inference (Ray)

```bash
# On head node
ray start --head --num-gpus=8

# On worker nodes
ray start --address=<HEAD_IP>:6379 --num-gpus=8

# Check cluster status
ray status
```

### Model Quantization

For systems with limited VRAM:

```bash
# Pull quantized model (4-bit)
ollama pull llama2:70b-q4

# Or 8-bit
ollama pull llama2:70b-q8
```

## Integration with Existing Systems

### With ALG (QAOA Optimizer)

The Ollama agent complements the existing ALG system:

```bash
# Use ALG for deterministic optimization
./alg/alg.sh run --nodes 256

# Use Ollama agent for LLM-guided optimization
./build/qallow phase 14 --agent-ollama
```

### With Phase 13 (Ethics)

The agent automatically validates prompts via Phase 13:

```python
# Ethics validation is automatic
agent = OllamaAgent(config)
result = agent.optimize_qaoa()  # Phase 13 runs before LLM
```

### With Native App (Rust)

The native app connects to the chat server:

```rust
// In native_app/src/backend/api_client.rs
let response = api_client.chat("Optimize QAOA").await?;
```

## Next Steps

1. **Scale to larger models**: Try DeepSeek-V3 or Llama3-70B
2. **Distributed setup**: Use Ray for multi-node inference
3. **Custom prompts**: Modify agent prompts for specific tasks
4. **Integration**: Connect to your own systems via the API

## References

- [Ollama Documentation](https://ollama.ai/docs)
- [Qallow Architecture](./architecture/QALLOW_SYSTEM_ARCHITECTURE.md)
- [DeepSeek Integration](../DEEPSEEK_INTEGRATION.md)
- [Phase 13 Ethics](./guides/PHASE13_ETHICS_GUIDE.md)
- [Phase 14 QAOA](./guides/PHASE14_QAOA_GUIDE.md)

