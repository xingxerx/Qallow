# DeepSeek-1 AI Baseline Setup Guide

**Purpose**: Install DeepSeek-1 as the foundational AI baseline for Qallow AGI Evolution (Feature 004)

**Created**: 2025-11-07  
**Status**: ✅ Ready for use

---

## Quick Start (3 minutes)

### Option 1: Mock Backend (Testing Only)

```bash
# No additional setup required - uses mock reasoning
cd /home/xing/Qallow
source .venv/bin/activate
python python/deepseek_baseline.py
```

### Option 2: Local Inference via Ollama

#### Step 1: Install Ollama

```bash
# Download from https://ollama.com/download
# Or use package manager:
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
# In another terminal: ollama pull deepseek-chat
```

#### Step 2: Test DeepSeek Baseline

```bash
cd /home/xing/Qallow
source .venv/bin/activate
export DEEPSEEK_BACKEND=ollama
python python/deepseek_baseline.py
```

### Option 3: DeepSeek Cloud API

#### Step 1: Get API Key

1. Visit https://platform.deepseek.com
2. Create account and generate API key
3. Set environment variable:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

#### Step 2: Test with API

```bash
cd /home/xing/Qallow
source .venv/bin/activate
export DEEPSEEK_BACKEND=api
python python/deepseek_baseline.py
```

---

## Environment Variables

```bash
# Backend selection
export DEEPSEEK_BACKEND=ollama          # or: api, mock
export DEEPSEEK_MODEL=deepseek-chat     # or: deepseek-reasoner

# Ollama configuration
export OLLAMA_HOST=http://localhost:11434

# DeepSeek API configuration
export DEEPSEEK_API_KEY=sk_...
export DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```

---

## Integration with Feature 004

### 1. Cognitive State Reasoning

Meta-learning uses DeepSeek to reason about optimization progress:

```python
from python.deepseek_baseline import DeepSeekClient, DeepSeekConfig

config = DeepSeekConfig(backend="ollama")
client = DeepSeekClient(config)

result = client.reason_cognitive_state(
    iteration=25,
    current_loss=0.145,
    best_loss=0.089,
    ethics_score=0.94
)
# Returns: {"analysis": "...", "status": "converging", "action": "..."}
```

### 2. Constitution Ethics Audit

Verify compliance with §1.2 (Self-Improvement), §2.4 (Quantum), §3.1 (Ethics):

```python
audit = client.audit_ethics(
    action="update_parameters",
    loss_improvement=0.056,
    iteration=25
)
# Returns: {"safety": 0.95, "control": 0.98, "honesty": 0.92, "passed": True}
```

### 3. Telemetry Export

Results integrated into Qallow telemetry:

```bash
# CSV format includes reasoning scores
tail data/logs/metalearn_deepseek.csv
# iteration,loss,deepseek_status,safety,control,honesty
# 0,1.234,converging,0.95,0.98,0.92
```

---

## Architecture

### DeepSeekClient Class

**Backends**:
- **Mock**: In-memory reasoning (testing, no inference)
- **Ollama**: Local inference (requires ollama service)
- **API**: Cloud inference (requires API key)

**Methods**:
- `reason_cognitive_state()`: Analyze optimization progress
- `audit_ethics()`: Verify Constitution compliance
- `get_status()`: Check backend availability

### Integration Points

1. **Meta-Learning**: Use DeepSeek reasoning to guide next sampling
2. **Ethics Audit**: Validate each optimization step
3. **Telemetry**: Log reasoning scores alongside metrics

---

## Performance Notes

| Backend | Latency | Cost | Setup |
|---------|---------|------|-------|
| Mock | <1ms | Free | None |
| Ollama (GPU) | 50-200ms | Free | Ollama + Model |
| Ollama (CPU) | 500ms-2s | Free | Ollama + Model |
| API | 200-500ms | $$ | API Key |

**Recommendation for Feature 004**:
- **Development**: Use Mock backend (instant, no setup)
- **Testing**: Use Ollama (local, repeatable)
- **Production**: Use API (scalable, maintained)

---

## Troubleshooting

### Ollama Not Available

```bash
# Verify Ollama is running
curl http://localhost:11434/api/version

# If not running, start it:
ollama serve

# Pull deepseek model:
ollama pull deepseek-chat
```

### API Key Not Working

```bash
# Verify key is set
echo $DEEPSEEK_API_KEY

# Test with curl
curl https://api.deepseek.com/v1/models \
  -H "Authorization: Bearer $DEEPSEEK_API_KEY"
```

### Module Import Error

```bash
# Ensure venv is activated
source /home/xing/Qallow/.venv/bin/activate

# Check ollama installation
python -c "from ollama import Client; print('✓ ollama installed')"
```

---

## Next Steps

1. **Build Meta-Learning Engine**: `make build-meta-learning`
2. **Run Tests**: `ctest -R meta_learning`
3. **Integrate with Phase 004**: See `specs/004-agi-evolution/quickstart.md`
4. **Compare Baselines**: Add different AI models (Claude, GPT, Gemini, etc.)

---

## Files Generated

- `python/deepseek_baseline.py` - DeepSeek client and integration
- `config/deepseek.env` - Environment template (create as needed)
- This guide - `docs/DEEPSEEK_SETUP.md`

---

## References

- DeepSeek Docs: https://platform.deepseek.com/docs
- Ollama: https://ollama.com
- Feature 004: `specs/004-agi-evolution/spec.md`
- Data Model: `specs/004-agi-evolution/data-model.md`

---

**Version**: 1.0.0  
**Status**: ✅ Ready for Feature 004 implementation
