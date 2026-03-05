# 🚀 Qallow Ollama Agent - Quick Commands

Copy and paste these commands to get started immediately!

---

## ✅ Verify Setup (30 seconds)

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check models are available
ollama list

# Check agent imports
python3 -c "from python.agents.qallow_agent_ollama import OllamaAgent; print('✓ Agent ready')"
```

---

## 🧪 Test the Agent (30 seconds)

```bash
cd ~/Qallow

# Quick test with llama2:7b
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics
```

**Expected Output:**
```
✓ Model llama2:7b is available
✓ Initialized OllamaAgent
✓ QAOA optimization complete: p=6, alpha_eff=0.0100
```

---

## 🎯 Run Different Configurations

### Small (Fast, 17 seconds)
```bash
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics
```

### Medium (Better Quality, 30 seconds)
```bash
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:13b \
  --nodes 64 \
  --target 0.97 \
  --no-ethics
```

### Large (Best Quality, 60 seconds)
```bash
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:13b \
  --nodes 256 \
  --target 0.981 \
  --no-ethics
```

### With Ethics Validation
```bash
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95
```

---

## 🔧 Phase 14 Integration

### Build First (if needed)
```bash
cd ~/Qallow
./scripts/build_all.sh
```

### Run Phase 14 with Agent
```bash
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:7b \
  --nodes=256 \
  --target_fidelity=0.981
```

### Run Phase 14 with Larger Model
```bash
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:13b \
  --nodes=512 \
  --target_fidelity=0.99
```

---

## 💬 Chat Server

### Start Server
```bash
export QALLOW_CHAT_BACKEND=ollama
export OLLAMA_MODEL=llama2:7b
cd ~/Qallow/python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

### Test API (in another terminal)
```bash
curl -X POST http://localhost:8008/quantum/task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "qaoa_optimize",
    "nodes": 256,
    "target_fidelity": 0.981
  }'
```

### Check Server Health
```bash
curl http://localhost:8008/health
```

---

## 📊 Check Results

### View Latest Output
```bash
cd ~/Qallow
cat data/quantum/agent_output.jsonl | tail -1 | python3 -m json.tool
```

### View Gain File
```bash
cat data/quantum/ollama_gain.json
```

### Count Completed Tasks
```bash
wc -l data/quantum/agent_output.jsonl
```

---

## 🎓 Download More Models

### Llama2 Models
```bash
# 7B model (3.8GB, fast)
ollama pull llama2:7b

# 13B model (7.4GB, better)
ollama pull llama2:13b

# 70B model (40GB, best)
ollama pull llama2:70b
```

### Other Models
```bash
# DeepSeek (MoE)
ollama pull deepseek-v3:70b

# Mistral
ollama pull mistral:7b

# Neural Chat
ollama pull neural-chat:7b
```

---

## 🔍 Troubleshooting

### Ollama Not Running
```bash
# Start Ollama
ollama serve &

# Or in background
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

### Model Not Found
```bash
# List available models
ollama list

# Pull missing model
ollama pull llama2:7b
```

### Out of Memory
```bash
# Use smaller model
ollama pull llama2:7b

# Or quantized version
ollama pull llama2:13b-q4
```

### Check Ollama Logs
```bash
tail -f /tmp/ollama.log
```

---

## 📈 Performance Monitoring

### Check Agent Performance
```bash
# Count tasks
echo "Total tasks: $(wc -l < data/quantum/agent_output.jsonl)"

# Average duration
python3 -c "
import json
durations = []
with open('data/quantum/agent_output.jsonl') as f:
    for line in f:
        durations.append(json.loads(line)['duration_ms'])
print(f'Average duration: {sum(durations)/len(durations):.0f}ms')
"
```

### Monitor Ollama
```bash
# Check running processes
ps aux | grep ollama

# Check memory usage
free -h

# Check GPU usage (if NVIDIA)
nvidia-smi
```

---

## 🚀 Advanced Usage

### Multi-GPU Setup
```bash
export OLLAMA_NUM_GPU=8
ollama serve &
```

### Custom Configuration
```bash
export QALLOW_AGENT_ETHICS=1
export QALLOW_AGENT_THRESHOLD=0.85
export OLLAMA_TEMPERATURE=0.3
export OLLAMA_NUM_PREDICT=256
```

### Run Tests
```bash
cd ~/Qallow
pytest tests/test_ollama_agent.py -v
```

### Run with Logging
```bash
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics \
  2>&1 | tee agent_run.log
```

---

## 📚 Documentation

```bash
# View full guide
cat docs/OLLAMA_AGENT_GUIDE.md

# View setup guide
cat SETUP_COMPLETE.md

# View quick reference
cat OLLAMA_QUICK_REFERENCE.md

# View this file
cat QUICK_COMMANDS.md
```

---

## ✨ One-Liner Examples

```bash
# Quick test
cd ~/Qallow && python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --model llama2:7b --nodes 16 --target 0.95 --no-ethics

# View results
cat ~/Qallow/data/quantum/agent_output.jsonl | tail -1 | python3 -m json.tool

# Count tasks
wc -l ~/Qallow/data/quantum/agent_output.jsonl

# Start chat server
cd ~/Qallow/python/chat_server && uvicorn main:app --port 8008

# Test API
curl -X POST http://localhost:8008/quantum/task -H "Content-Type: application/json" -d '{"task":"qaoa_optimize","nodes":256,"target_fidelity":0.981}'
```

---

## 🎯 Recommended Workflow

1. **Verify Setup** (30 seconds)
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Test Agent** (30 seconds)
   ```bash
   python3 -m python.agents.qallow_agent_ollama --task qaoa_optimize --model llama2:7b --nodes 16 --target 0.95 --no-ethics
   ```

3. **Check Results** (5 seconds)
   ```bash
   cat data/quantum/agent_output.jsonl | tail -1 | python3 -m json.tool
   ```

4. **Run Phase 14** (varies)
   ```bash
   ./build/qallow phase 14 --agent-ollama --ollama-model=llama2:7b
   ```

5. **Start Chat Server** (continuous)
   ```bash
   cd python/chat_server && uvicorn main:app --port 8008
   ```

---

## 🎉 You're Ready!

Pick a command above and run it now! 🚀

**Questions?** See [SETUP_COMPLETE.md](SETUP_COMPLETE.md) or [docs/OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md)

