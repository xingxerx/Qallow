# Qallow Ollama Agent - Quick Reference Card

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Run setup script
./scripts/quick_start_ollama.sh

# 2. Test agent
python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --nodes 16 --target 0.95 --no-ethics

# 3. Run with Phase 14
./build/qallow phase 14 --agent-ollama --nodes=256 --target_fidelity=0.981
```

## 📋 Common Commands

### Agent Direct Usage
```bash
# QAOA optimization
python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --nodes 256 --target 0.981

# With custom model
python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --model llama2:70b --num-gpu 8

# Get status
python3 -m python.agents.qallow_agent_ollama --task status

# Disable ethics (testing only)
python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --no-ethics
```

### Phase 14 Integration
```bash
# Basic usage
./build/qallow phase 14 --agent-ollama

# With parameters
./build/qallow phase 14 --agent-ollama --nodes=512 --target_fidelity=0.99

# Custom model
./build/qallow phase 14 --agent-ollama --ollama-model=deepseek-v3:70b
```

### Chat Server
```bash
# Start server
export QALLOW_CHAT_BACKEND=ollama
cd python/chat_server && uvicorn main:app --host 0.0.0.0 --port 8008

# Chat endpoint
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "backend": "ollama"}'

# Quantum task endpoint
curl -X POST http://localhost:8008/quantum/task \
  -H "Content-Type: application/json" \
  -d '{"task": "qaoa_optimize", "nodes": 256, "target_fidelity": 0.981}'
```

### Ollama Management
```bash
# Start service
ollama serve &

# List models
ollama list

# Pull model
ollama pull llama2:70b

# Remove model
ollama rm llama2:13b

# Check status
curl http://localhost:11434/api/tags
```

## 🔧 Configuration

### Environment Variables
```bash
export QALLOW_CHAT_BACKEND=ollama
export OLLAMA_MODEL=llama2:70b
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_NUM_GPU=8
export QALLOW_AGENT_ETHICS=1
export QALLOW_AGENT_THRESHOLD=0.85
```

### Python Config
```python
from python.agents.qallow_agent_ollama import OllamaAgent, OllamaConfig

config = OllamaConfig(
    model="llama2:70b",
    num_gpu=8,
    qaoa_nodes=256,
    qaoa_target_fidelity=0.981
)
agent = OllamaAgent(config)
result = agent.optimize_qaoa()
```

## 📁 Output Files

| File | Description |
|------|-------------|
| `data/quantum/agent_output.jsonl` | Agent task log (JSONL) |
| `data/quantum/ollama_gain.json` | Gain for Phase 14 |
| `data/logs/phase14_*.csv` | Phase 14 results |

## 🧪 Testing

```bash
# All tests
pytest tests/test_ollama_agent.py -v

# Specific test
pytest tests/test_ollama_agent.py::TestOllamaAgent::test_qaoa_optimization -v

# Integration tests
pytest tests/test_ollama_agent.py -v -m integration
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Ollama not running | `ollama serve &` |
| Model not found | `ollama pull llama2:70b` |
| Out of memory | Use smaller model: `ollama pull llama2:13b` |
| Ethics gate failing | Test Phase 13: `./build/qallow phase 13 --ticks=10` |
| Import error | Check Python path: `export PYTHONPATH=$PWD` |

## 📊 Model Selection

| Model | Size | VRAM | Speed | Use Case |
|-------|------|------|-------|----------|
| `llama2:7b` | 7B | 8GB | Fast | Testing |
| `llama2:13b` | 13B | 16GB | Medium | Development |
| `llama2:70b` | 70B | 80GB | Slow | Production |
| `deepseek-v3:70b` | 70B MoE | 80GB | Medium | Advanced |

## 🚀 Multi-GPU Setup

```bash
# Supercomputer setup
./scripts/setup_ollama_supercomputer.sh --model llama2:70b --num-gpu 8

# Distributed (Ray)
./scripts/setup_ollama_supercomputer.sh --distributed --head-node

# Worker node
./scripts/setup_ollama_supercomputer.sh --worker-node <HEAD_IP>:6379 --num-gpu 8

# Monitor GPUs
watch -n 1 nvidia-smi
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md) | Full guide |
| [python/agents/README.md](python/agents/README.md) | Agent module docs |
| [OLLAMA_INTEGRATION_COMPLETE.md](OLLAMA_INTEGRATION_COMPLETE.md) | Integration summary |
| http://localhost:8008/docs | API docs (when server running) |

## 🎯 Key Features

- ✅ Autonomous QAOA optimization
- ✅ Phase 13 ethics validation
- ✅ Multi-GPU support (8+ GPUs)
- ✅ Distributed inference (Ray/MPI)
- ✅ Local & private (no cloud)
- ✅ Full telemetry
- ✅ Chat server integration
- ✅ Native app support

## 💡 Tips

1. **Start small**: Use `llama2:13b` for testing, then scale to `70b`
2. **Monitor GPUs**: Use `nvidia-smi` to check utilization
3. **Ethics gate**: Disable with `--no-ethics` for testing only
4. **Quantization**: Use `q4` or `q8` models for less VRAM
5. **Distributed**: Use Ray for multi-node clusters

## 🔗 Quick Links

- **Setup**: `./scripts/quick_start_ollama.sh`
- **Test**: `pytest tests/test_ollama_agent.py -v`
- **Run**: `./build/qallow phase 14 --agent-ollama`
- **Docs**: `docs/OLLAMA_AGENT_GUIDE.md`

---

**Need help?** See [docs/OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md) for detailed documentation.

