# How to Run Qallow with Ollama Agent

Complete guide for running the Qallow quantum computing system with the Ollama AI agent.

## 🚀 Quick Start (5 Minutes)

### Automated Setup

```bash
# Run the quick start script
./scripts/quick_start_ollama.sh

# This will:
# 1. Install Ollama
# 2. Download llama2:13b model
# 3. Test the agent
# 4. Show next steps
```

### Manual Setup

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve &

# 3. Pull a model
ollama pull llama2:13b

# 4. Build Qallow
./scripts/build_all.sh

# 5. Run Phase 14 with agent
./build/qallow phase 14 --agent-ollama
```

---

## 📋 Running Different Components

### 1. Ollama Agent (Python)

**Direct agent execution for QAOA optimization:**

```bash
# Basic usage
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize

# With custom parameters
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:70b \
  --nodes 256 \
  --target 0.981 \
  --num-gpu 8

# Get agent status
python3 -m python.agents.qallow_agent_ollama --task status

# Disable ethics gate (testing only)
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --no-ethics
```

**Output:**
```json
{
  "p": 3,
  "gamma": 0.42,
  "beta": 0.19,
  "alpha_eff": 0.0048,
  "reasoning": "MoE routing stabilizes long-range coherence"
}
```

**Output files:**
- `data/quantum/agent_output.jsonl` - Task log
- `data/quantum/ollama_gain.json` - Gain for Phase 14

---

### 2. Phase 14 with Agent (C Binary)

**Run Phase 14 coherence simulation with AI-optimized parameters:**

```bash
# Basic usage
./build/qallow phase 14 --agent-ollama

# With custom parameters
./build/qallow phase 14 \
  --agent-ollama \
  --nodes=512 \
  --target_fidelity=0.99 \
  --ticks=1000

# With specific Ollama model
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=deepseek-v3:70b

# Without agent (traditional mode)
./build/qallow phase 14 \
  --nodes=256 \
  --target_fidelity=0.981 \
  --alpha=0.005
```

**What happens:**
1. Phase 14 calls the Ollama agent
2. Agent validates prompt via Phase 13 (ethics)
3. Agent queries Ollama LLM for optimal parameters
4. Agent exports gain to `data/quantum/ollama_gain.json`
5. Phase 14 reads gain and runs simulation
6. Results saved to `data/logs/phase14_*.csv`

---

### 3. Chat Server (FastAPI)

**Start the chat server for API access:**

```bash
# Set environment variables
export QALLOW_CHAT_BACKEND=ollama
export OLLAMA_MODEL=llama2:70b

# Start server
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008

# Server will be available at:
# - API: http://localhost:8008
# - Docs: http://localhost:8008/docs
# - Health: http://localhost:8008/health
```

**API Endpoints:**

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

#### Health Check
```bash
curl http://localhost:8008/health
```

**Interactive API Docs:**
Open http://localhost:8008/docs in your browser for interactive Swagger UI.

---

### 4. Native App (Rust GUI)

**Run the native desktop application:**

```bash
# Make sure chat server is running first!
# (See section 3 above)

# Build and run native app
cd native_app
cargo run --release

# Or build first, then run
cargo build --release
./target/release/qallow-native-app
```

**Features:**
- Real-time metrics dashboard
- Chat interface with Ollama agent
- Phase execution controls
- GPU monitoring
- Audit logs viewer

---

## 🔄 Complete Workflow Example

### Scenario: Optimize QAOA for 512 nodes

```bash
# Terminal 1: Start Ollama (if not running)
ollama serve

# Terminal 2: Start chat server (optional, for API access)
export QALLOW_CHAT_BACKEND=ollama
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008

# Terminal 3: Run Phase 14 with agent
cd /path/to/Qallow
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:70b \
  --nodes=512 \
  --target_fidelity=0.99 \
  --ticks=1000

# Terminal 4: Monitor GPU usage (optional)
watch -n 1 nvidia-smi

# Terminal 5: Tail agent logs (optional)
tail -f data/quantum/agent_output.jsonl
```

**Expected output:**
```
[PHASE14] Coherence-lattice integration
[PHASE14] nodes=512 ticks=1000 target_fidelity=0.990
[PHASE14] agent-ollama enabled (model=llama2:70b)
[PHASE14] Running Ollama agent for QAOA optimization...
[Agent] Querying Ollama: llama2:70b
[Agent] ✓ Response received (8432ms)
[Agent] Success → data/quantum/ollama_gain.json
[PHASE14] ✓ Ollama agent completed successfully
[PHASE14] alpha from gain_json = 0.004823
[PHASE14] Running simulation...
[PHASE14] Final fidelity: 0.9912
[PHASE14] ✓ Target achieved
```

---

## 🎯 Common Use Cases

### Use Case 1: Quick Test

```bash
# Fast test with small model
ollama pull llama2:7b
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics
```

### Use Case 2: Production Run

```bash
# High-quality optimization with 70B model
ollama pull llama2:70b
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:70b \
  --nodes=512 \
  --target_fidelity=0.99
```

### Use Case 3: Distributed Multi-GPU

```bash
# Setup for 8 GPUs
./scripts/setup_ollama_supercomputer.sh \
  --model llama2:70b \
  --num-gpu 8

# Run with multi-GPU
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:70b \
  --num-gpu 8 \
  --nodes 1024 \
  --target 0.995
```

### Use Case 4: API Integration

```bash
# Start server
export QALLOW_CHAT_BACKEND=ollama
cd python/chat_server && uvicorn main:app --port 8008 &

# Use from Python
import requests

response = requests.post(
    "http://localhost:8008/quantum/task",
    json={
        "task": "qaoa_optimize",
        "nodes": 256,
        "target_fidelity": 0.981
    }
)
print(response.json())
```

---

## 🛠️ Troubleshooting

### Ollama Not Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
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

### Build Errors

```bash
# Clean and rebuild
./scripts/build_clean.sh
./scripts/build_all.sh

# Or just rebuild specific components
cd build
make clean
cmake ..
make -j$(nproc)
```

### Import Errors

```bash
# Set Python path
export PYTHONPATH=$PWD

# Or install in development mode
pip install -e .
```

### Out of Memory

```bash
# Use smaller model
ollama pull llama2:13b

# Or use quantized model
ollama pull llama2:70b-q4

# Reduce context size
export OLLAMA_NUM_CTX=2048
```

---

## 📊 Monitoring

### GPU Usage

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop (if installed)
nvtop
```

### Agent Logs

```bash
# Tail agent output
tail -f data/quantum/agent_output.jsonl

# Pretty print with jq
tail -f data/quantum/agent_output.jsonl | jq .
```

### Phase 14 Results

```bash
# View latest results
ls -lt data/logs/phase14_*.csv | head -1 | xargs cat

# Plot results (if you have Python plotting tools)
python3 scripts/plot_phase14.py data/logs/phase14_latest.csv
```

---

## 🔗 Next Steps

1. **Read the full guide**: [docs/OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md)
2. **Try different models**: `ollama pull deepseek-v3:70b`
3. **Scale to multiple GPUs**: `./scripts/setup_ollama_supercomputer.sh`
4. **Integrate with your code**: See API examples above
5. **Run tests**: `pytest tests/test_ollama_agent.py -v`

---

## 📚 Documentation

- **Quick Reference**: [OLLAMA_QUICK_REFERENCE.md](OLLAMA_QUICK_REFERENCE.md)
- **Full Guide**: [docs/OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md)
- **Integration Summary**: [OLLAMA_INTEGRATION_COMPLETE.md](OLLAMA_INTEGRATION_COMPLETE.md)
- **API Docs**: http://localhost:8008/docs (when server running)

---

## ✅ Verification Checklist

Before running, make sure:

- [ ] Ollama is installed: `ollama --version`
- [ ] Ollama is running: `curl http://localhost:11434/api/tags`
- [ ] Model is downloaded: `ollama list | grep llama2`
- [ ] Qallow is built: `ls build/qallow`
- [ ] Python dependencies: `pip install -r config/requirements.txt`

---

**Ready to run!** Start with `./scripts/quick_start_ollama.sh` for the easiest experience.

