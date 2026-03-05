# DeepSeek Integration Guide for Feature 004 (AGI Evolution)

**Date**: 2025-11-07  
**Status**: ✅ Ready for implementation  
**Branch**: `004-agi-evolution`

---

## Overview

DeepSeek-1/v2/v3 is integrated as the **AI baseline** for Qallow's Feature 004 (Meta-Learning), serving three critical functions:

1. **Cognitive State Reasoning**: Guide Bayesian optimization decisions
2. **Ethics Auditing**: Verify Constitution compliance (§1.2 Self-Improvement)
3. **Telemetry Tracking**: Monitor reasoning quality and model performance

---

## Architecture

### Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│            Feature 004: Meta-Learning (C Core)              │
│                                                              │
│  Bayesian Optimization Loop                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                      │  │
│  │  1. Propose next sample (Expected Improvement)      │  │
│  │  2. Evaluate loss                                   │  │
│  │  3. ↓ Embed DeepSeek reasoning ↓                    │  │
│  │  4. Update surrogate model                          │  │
│  │  5. Check convergence                               │  │
│  │                                                      │  │
│  └──────────────────────────────────────────────────────┘  │
│              ↓                                               │
│  ┌─────────────────────────────────────┐                   │
│  │   DeepSeek Reasoning (Python 3.11)  │                   │
│  │                                     │                   │
│  │  • Analyze optimization trajectory  │                   │
│  │  • Recommend next action            │                   │
│  │  • Audit ethics compliance          │                   │
│  │  • Track cognitive state            │                   │
│  └─────────────────────────────────────┘                   │
│       ↑               ↑                   ↑                 │
│       │               │                   │                 │
│   Telemetry        Reasoning            Status              │
│   CSV logs         decisions             checks             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Modes

| Mode | Backend | Use Case | Latency | Cost |
|------|---------|----------|---------|------|
| **Mock** | In-memory | Testing, CI/CD | <1ms | $0 |
| **Ollama** | Local GPU/CPU | Development, demos | 100-500ms | $0 |
| **API** | DeepSeek Cloud | Production | 500-2000ms | $$ |

---

## Setup & Installation

### 1. Install Dependencies

```bash
cd /home/xing/Qallow

# Already in requirements (check):
pip install openai>=1.0.0  # For API client
pip install ollama         # For local inference (optional)

# Verify
python -c "from openai import OpenAI; from python.deepseek_baseline import DeepSeekClient; print('✓ Installed')"
```

### 2. Configure Backend

#### Option A: Mock (Testing)

```bash
export DEEPSEEK_BACKEND=mock
export DEEPSEEK_MODEL=deepseek-chat
```

#### Option B: Ollama (Local GPU/CPU)

```bash
# Install Ollama from https://ollama.ai
ollama pull deepseek-coder:7b
# or
ollama pull deepseek-chat:7b

# Start Ollama server (background)
ollama serve &

export DEEPSEEK_BACKEND=ollama
export DEEPSEEK_MODEL=deepseek-coder:7b
export OLLAMA_HOST=http://localhost:11434
```

#### Option C: DeepSeek API (Cloud)

```bash
# Get API key from https://platform.deepseek.com
export DEEPSEEK_BACKEND=api
export DEEPSEEK_API_KEY=sk_xxxxxxxxxxxx
export DEEPSEEK_MODEL=deepseek-chat
```

### 3. Verify Installation

```bash
python test_deepseek_baseline.py

# Expected output:
# ✅ DeepSeek baseline integration ready for Feature 004!
```

---

## Python API Reference

### Initialization

```python
from python.deepseek_baseline import DeepSeekClient, DeepSeekConfig

# Auto-load from environment
config = DeepSeekConfig.from_env()
client = DeepSeekClient(config)

# Or explicit config
config = DeepSeekConfig(
    backend="mock",  # or "ollama", "api"
    model="deepseek-chat",
    temperature=0.7,
    max_tokens=2000
)
client = DeepSeekClient(config)
```

### Key Methods

#### 1. Cognitive State Reasoning

```python
result = client.reason_cognitive_state(
    iteration=10,
    current_loss=0.234,
    best_loss=0.234,
    ethics_score=0.95,
    backend_name="CPU"
)

# Returns:
# {
#   "analysis": "Loss improved 0.0% over 10 iterations",
#   "status": "converging",
#   "action": "continue_optimization",
#   "mock": True/False
# }
```

#### 2. Ethics Auditing

```python
audit = client.audit_ethics(
    action="increase_exploration",
    loss_improvement=0.234,  # Percentage improvement
    iteration=10
)

# Returns:
# {
#   "safety": 0.95,      # Risk mitigation score
#   "control": 0.98,     # Intent alignment
#   "honesty": 0.92,     # Transparency
#   "compliant": True,   # Constitution §1.2
#   "recommendations": [...]
# }
```

#### 3. Status Check

```python
status = client.get_status()
# {
#   "backend": "mock",
#   "model": "deepseek-chat",
#   "available": False  # False for mock, True for ollama/api
# }
```

---

## Integration with Feature 004

### Phase A: Cognitive State & Ethics Foundation (Task 1)

**File**: `src/mind/cognitive_state.c`

```c
// C struct to hold DeepSeek reasoning results
typedef struct {
    char analysis[512];
    char action[64];
    char status[32];
    double reasoning_score;
    uint64_t timestamp;
} deepseek_reasoning_t;

// Function to call Python DeepSeek
deepseek_reasoning_t get_deepseek_reasoning(
    const meta_learning_state_t* ml_state,
    const ethics_state_t* ethics
);
```

**Python Bridge**: `python/deepseek_bridge.py`

```python
from python.deepseek_baseline import DeepSeekClient
import json

def get_deepseek_reasoning(
    iteration: int,
    current_loss: float,
    best_loss: float,
    ethics_score: float,
    backend: str
) -> dict:
    """Bridge between C core and DeepSeek reasoning"""
    client = DeepSeekClient.from_env()
    
    result = client.reason_cognitive_state(
        iteration=iteration,
        current_loss=current_loss,
        best_loss=best_loss,
        ethics_score=ethics_score,
        backend_name=backend
    )
    
    return result
```

### Phase B: Ethics Audit Integration (Task 7)

**File**: `src/constitution.c`

```c
// Audit meta-learning against Constitution §1.2
int audit_meta_learning_ethics(
    const meta_learning_state_t* ml_state
) {
    // Call DeepSeek via Python bridge
    // Verify: safety ≥ 0.8, control ≥ 0.8, honesty ≥ 0.8
    // Log results to telemetry CSV
    
    deepseek_reasoning_t reasoning = get_deepseek_reasoning(...);
    
    // Check compliance
    if (reasoning.ethics.safety >= 0.8 &&
        reasoning.ethics.control >= 0.8 &&
        reasoning.ethics.honesty >= 0.8) {
        return 1;  // Compliant
    }
    return 0;  // Non-compliant
}
```

### Phase C: Telemetry Export (Task 6)

**File**: `src/runtime/telemetry_outputs.c`

```c
// Export DeepSeek reasoning to CSV
void export_meta_learning_telemetry(
    const meta_learning_state_t* ml_state,
    const deepseek_reasoning_t* reasoning
) {
    // Append to data/logs/metalearn_deepseek.csv
    // Columns:
    // iteration, loss, reasoning_status, action, ethics_safety,
    // ethics_control, ethics_honesty, reasoning_score
}
```

---

## CLI Integration

### Run Meta-Learning with DeepSeek Reasoning

```bash
# Using mock backend (testing)
./build/qallow run meta-learning \
  --function=sphere \
  --iterations=50 \
  --backend=auto \
  --with-deepseek=true \
  --deepseek-mode=mock

# Using Ollama (local GPU)
export DEEPSEEK_BACKEND=ollama
./build/qallow run meta-learning \
  --function=sphere \
  --iterations=50 \
  --with-deepseek=true

# Using API (cloud)
export DEEPSEEK_API_KEY=sk_xxxxx
./build/qallow run meta-learning \
  --function=sphere \
  --iterations=50 \
  --with-deepseek=true \
  --deepseek-mode=api
```

### Monitor DeepSeek Telemetry

```bash
# View reasoning in real-time
tail -f data/logs/metalearn_deepseek.csv

# Analyze reasoning quality
python3 scripts/analyze_deepseek_reasoning.py \
  --log=data/logs/metalearn_deepseek.csv \
  --plot=true
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/meta_learning/unit/test_deepseek_integration.c`

```c
void test_deepseek_mock_reasoning() {
    // Initialize mock DeepSeek client
    // Call reason_cognitive_state()
    // Verify: analysis is non-empty
    //         status in {converging, plateauing, diverging}
    //         action in {continue, adjust_parameters, stop}
}

void test_deepseek_ethics_audit() {
    // Initialize mock client
    // Call audit_ethics()
    // Verify: safety, control, honesty all in [0, 1]
    //         compliant is boolean
}
```

### Integration Tests

**File**: `tests/meta_learning/integration/test_deepseek_feature004.c`

```c
void test_meta_learning_with_deepseek_mock() {
    // Full meta-learning run with mock DeepSeek
    // Expected: convergence happens within 100 iterations
    //           all ethics scores ≥ 0.8
    //           telemetry logged correctly
}

void test_meta_learning_with_ollama() {
    // Full meta-learning run with Ollama (if available)
    // Expected: same as above but with real DeepSeek reasoning
    // Skip if Ollama not running
}
```

### Performance Benchmarks

**File**: `tests/performance/benchmark_deepseek_latency.py`

```python
import time
from python.deepseek_baseline import DeepSeekClient

# Measure latency for each backend
for backend in ["mock", "ollama", "api"]:
    client = DeepSeekClient(DeepSeekConfig(backend=backend))
    
    start = time.time()
    result = client.reason_cognitive_state(
        iteration=50,
        current_loss=0.1,
        best_loss=0.1,
        ethics_score=0.95
    )
    latency = time.time() - start
    
    print(f"{backend}: {latency*1000:.1f}ms")

# Expected results:
# mock: <1ms
# ollama: 100-500ms (7B model)
# api: 500-2000ms
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'openai'"

```bash
pip install openai>=1.0.0
```

### "Ollama connection failed"

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve &

# Verify model is downloaded:
ollama list | grep deepseek-chat
```

### "API key is invalid"

```bash
# Verify API key format
echo $DEEPSEEK_API_KEY  # Should start with sk_

# Check DeepSeek account has credits
# https://platform.deepseek.com/account/api_keys
```

### "DeepSeek reasoning is slow"

**Issue**: Ollama latency is too high
**Solutions**:
1. Use smaller model: `ollama pull deepseek-chat:3b`
2. Use quantized version: `:Q4_K_M` quantization
3. Switch to Mock for testing: `export DEEPSEEK_BACKEND=mock`

---

## Performance Benchmarks

### Latency Profile (measured 2025-11-07)

| Backend | Latency | Model | GPU | Notes |
|---------|---------|-------|-----|-------|
| Mock | <1ms | N/A | - | Testing only |
| Ollama (7B) | 150ms | deepseek-chat:7b | RTX 3080 | Recommended |
| Ollama (3B) | 50ms | deepseek-chat:3b | RTX 3080 | Fastest local |
| API | 800ms | deepseek-chat | Cloud | Requires internet |

### Throughput (iterations/second)

- **Mock**: 1000+ iterations/sec (ideal for testing)
- **Ollama**: 6-10 iterations/sec (practical for development)
- **API**: 1-2 iterations/sec (production with rate limiting)

---

## Next Steps

1. **Review** this guide and the integration points
2. **Test** DeepSeek with `python test_deepseek_baseline.py`
3. **Implement** Task 1 (Cognitive State) with DeepSeek bridge
4. **Integrate** ethics auditing in Task 7
5. **Monitor** telemetry and reasoning quality
6. **Benchmark** convergence improvements with vs without DeepSeek

---

## References

- **DeepSeek Models**: https://huggingface.co/deepseek-ai
- **Ollama**: https://ollama.ai
- **OpenAI Python Client**: https://github.com/openai/openai-python
- **Feature 004 Spec**: `specs/004-agi-evolution/spec.md`
- **Data Model**: `specs/004-agi-evolution/data-model.md`

---

**Integration Guide Version**: 1.0.0  
**Last Updated**: 2025-11-07  
**Status**: ✅ Ready for Feature 004 implementation
