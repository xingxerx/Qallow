# Manual Setup Guide - Qallow Ollama Agent

Since the automated script requires sudo, here's a manual step-by-step guide.

## Step 1: Install Ollama

Run these commands **from the Qallow root directory** (not from `native_app`):

```bash
# Go to Qallow root
cd ~/Qallow

# Install Ollama (requires sudo password)
curl -fsSL https://ollama.com/install.sh | sh
```

**Alternative (if you don't have sudo):**
Download the binary manually:
```bash
# Download Ollama binary
curl -L https://ollama.com/download/ollama-linux-amd64 -o ollama
chmod +x ollama
sudo mv ollama /usr/local/bin/
```

## Step 2: Start Ollama Service

```bash
# Start Ollama in background
ollama serve > /tmp/ollama.log 2>&1 &

# Wait a few seconds
sleep 3

# Verify it's running
curl http://localhost:11434/api/tags
```

**Expected output:**
```json
{"models":[]}
```

## Step 3: Pull a Model

```bash
# For testing (faster, smaller)
ollama pull llama2:7b

# OR for better quality (slower, larger)
ollama pull llama2:13b

# OR for production (best quality, requires 80GB VRAM)
ollama pull llama2:70b
```

**This will take 5-15 minutes depending on your internet speed.**

## Step 4: Test the Agent

```bash
# Make sure you're in Qallow root
cd ~/Qallow

# Test the agent (without ethics gate for quick test)
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics
```

**Expected output:**
```json
{
  "p": 3,
  "gamma": 0.42,
  "beta": 0.19,
  "alpha_eff": 0.0048,
  "reasoning": "..."
}
```

## Step 5: Build Qallow (if not already built)

```bash
# Check if already built
ls build/qallow

# If not, build it
./scripts/build_all.sh
```

## Step 6: Run Phase 14 with Agent

```bash
# Run Phase 14 with Ollama agent
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:7b \
  --nodes=256 \
  --target_fidelity=0.981
```

---

## Quick Commands (Copy-Paste)

**From Qallow root directory:**

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start Ollama
ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# 3. Pull model
ollama pull llama2:7b

# 4. Test agent
python3 -m python.agents.qallow_agent_ollama \
  --task qaoa_optimize \
  --model llama2:7b \
  --nodes 16 \
  --target 0.95 \
  --no-ethics

# 5. Run Phase 14
./build/qallow phase 14 \
  --agent-ollama \
  --ollama-model=llama2:7b
```

---

## Troubleshooting

### "No such file or directory"
**Problem:** You're in the wrong directory (e.g., `native_app`)

**Solution:**
```bash
cd ~/Qallow
pwd  # Should show: /home/xing/Qallow
```

### "Ollama not running"
**Problem:** Ollama service not started

**Solution:**
```bash
# Check if running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve > /tmp/ollama.log 2>&1 &
```

### "Model not found"
**Problem:** Model not downloaded

**Solution:**
```bash
# List models
ollama list

# Pull missing model
ollama pull llama2:7b
```

### "Module not found: python.agents"
**Problem:** Python can't find the module

**Solution:**
```bash
# Set Python path
export PYTHONPATH=/home/xing/Qallow

# Or run from Qallow root
cd ~/Qallow
python3 -m python.agents.qallow_agent_ollama --task status
```

### "build/qallow not found"
**Problem:** Qallow not built yet

**Solution:**
```bash
cd ~/Qallow
./scripts/build_all.sh
```

---

## Next Steps

Once everything is working:

1. **Try larger models:**
   ```bash
   ollama pull llama2:13b
   ./build/qallow phase 14 --agent-ollama --ollama-model=llama2:13b
   ```

2. **Start chat server:**
   ```bash
   export QALLOW_CHAT_BACKEND=ollama
   cd python/chat_server
   uvicorn main:app --host 0.0.0.0 --port 8008
   ```

3. **Run tests:**
   ```bash
   pytest tests/test_ollama_agent.py -v
   ```

4. **Read full docs:**
   - [HOW_TO_RUN.md](HOW_TO_RUN.md)
   - [docs/OLLAMA_AGENT_GUIDE.md](docs/OLLAMA_AGENT_GUIDE.md)

