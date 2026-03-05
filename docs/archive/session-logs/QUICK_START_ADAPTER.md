# Quick Start: LLM Adapter

## Overview

The LLM Adapter provides a unified interface for communicating with any OpenAI-compatible LLM server (vLLM, SGLang, etc.).

## ✅ Status: WORKING

The Qwen2.5-7B model is running and ready to use!

```
✓ Connected to LLM server at http://localhost:8000/v1
✓ Available models: ['llm']
```

## 🚀 Quick Start (30 seconds)

### 1. Simple Chat

```python
from python.agents.llm_adapter import create_llm_adapter

adapter = create_llm_adapter()
response = adapter.chat("Hello! What is quantum computing?")
print(response)
```

### 2. Streaming Chat

```python
from python.agents.llm_adapter import create_llm_adapter

adapter = create_llm_adapter()
for chunk in adapter.chat_stream("Tell me about quantum gates"):
    print(chunk, end="", flush=True)
```

### 3. Chat with Tools

```python
from python.agents.llm_adapter import create_llm_adapter

adapter = create_llm_adapter()

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_circuit",
            "description": "Analyze a quantum circuit",
            "parameters": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"}
                }
            }
        }
    }
]

result = adapter.chat_with_tools("Analyze this circuit: H(q0)", tools=tools)
print(result)
```

## 📡 API Reference

### Create Adapter

```python
from python.agents.llm_adapter import create_llm_adapter

# Default (uses http://localhost:8000/v1, model "llm")
adapter = create_llm_adapter()

# Custom configuration
adapter = create_llm_adapter(
    model_name="llm",
    base_url="http://localhost:8000/v1"
)
```

### Methods

#### `chat(message, system_prompt=None)`
Send a message and get a response.

```python
response = adapter.chat("What is QAOA?")
response = adapter.chat(
    "Optimize this problem",
    system_prompt="You are a quantum computing expert"
)
```

#### `chat_stream(message, system_prompt=None)`
Stream a response chunk by chunk.

```python
for chunk in adapter.chat_stream("Explain quantum entanglement"):
    print(chunk, end="", flush=True)
```

#### `chat_with_tools(message, tools, system_prompt=None)`
Chat with tool calling support.

```python
result = adapter.chat_with_tools(
    "Use the analyze_circuit tool",
    tools=[...],
    system_prompt="You are helpful"
)
```

#### `set_temperature(temperature)`
Set temperature for responses (0.0 - 2.0).

```python
adapter.set_temperature(0.7)
```

#### `set_max_tokens(max_tokens)`
Set maximum tokens for responses.

```python
adapter.set_max_tokens(1024)
```

#### `get_config()`
Get current configuration.

```python
config = adapter.get_config()
print(config)
# Output:
# {
#     'model_name': 'llm',
#     'base_url': 'http://localhost:8000/v1',
#     'temperature': 0.6,
#     'max_tokens': 2048,
#     'top_p': 0.9
# }
```

## 🔧 Configuration

### Default Configuration

```python
model_name = "llm"
base_url = "http://localhost:8000/v1"
api_key = "not-needed"
temperature = 0.6
max_tokens = 2048
top_p = 0.9
```

### Custom Configuration

```python
from python.agents.llm_adapter import LLMAdapter, LLMConfig

config = LLMConfig(
    model_name="llm",
    base_url="http://localhost:8000/v1",
    temperature=0.8,
    max_tokens=1024,
    top_p=0.95
)

adapter = LLMAdapter(config)
```

## 📚 Examples

### Example 1: Quantum Circuit Analysis

```python
from python.agents.llm_adapter import create_llm_adapter
import cirq

# Create a quantum circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)

# Analyze with AI
adapter = create_llm_adapter()
analysis = adapter.chat(f"Analyze this quantum circuit: {circuit}")
print(analysis)
```

### Example 2: QAOA Optimization

```python
from python.agents.llm_adapter import create_llm_adapter

adapter = create_llm_adapter()

# Ask for QAOA optimization advice
response = adapter.chat("""
I have a QAOA circuit for MaxCut problem.
What parameters should I use for:
- Mixing angle
- Problem angle
- Number of layers
""")
print(response)
```

### Example 3: Streaming Response

```python
from python.agents.llm_adapter import create_llm_adapter

adapter = create_llm_adapter()

print("Quantum Computing Explanation:")
print("-" * 40)
for chunk in adapter.chat_stream("Explain quantum superposition in detail"):
    print(chunk, end="", flush=True)
print("\n" + "-" * 40)
```

## 🔗 Integration with Quantum Frameworks

### With Cirq

```python
from python.agents.llm_adapter import create_llm_adapter
import cirq

adapter = create_llm_adapter()

# Create circuit
circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))

# Analyze
response = adapter.chat(f"What does this circuit do? {circuit}")
print(response)
```

### With CUDA-Q

```python
from python.agents.llm_adapter import create_llm_adapter
import cudaq

adapter = create_llm_adapter()

@cudaq.kernel
def bell_pair():
    qubits = cudaq.qvector(2)
    h(qubits[0])
    cx(qubits[0], qubits[1])

result = cudaq.sample(bell_pair)
response = adapter.chat(f"Analyze this result: {result}")
print(response)
```

## ⚙️ Troubleshooting

### Issue: Connection refused

**Solution:** Make sure the vLLM server is running:
```bash
bash scripts/setup_alternative_llm.sh qwen
```

### Issue: Model not found

**Solution:** Check available models:
```bash
curl http://localhost:8000/v1/models
```

### Issue: Max tokens too large

**Solution:** Reduce max_tokens:
```python
adapter.set_max_tokens(1024)
```

### Issue: Slow responses

**Solution:** Check GPU memory:
```bash
nvidia-smi
```

## 📖 Documentation

- **LLM Adapter:** `python/agents/llm_adapter.py`
- **Setup Guide:** `SETUP_COMPLETE_SUMMARY.md`
- **Communication Guide:** `COMMUNICATION_GUIDE.md`
- **Troubleshooting:** `VLLM_TROUBLESHOOTING.md`

## 🎯 Next Steps

1. **Try the examples above**
2. **Integrate with your quantum workflows**
3. **Customize configuration as needed**
4. **Deploy to production**

## 📞 Support

For issues or questions:
1. Check `VLLM_TROUBLESHOOTING.md`
2. Check `COMMUNICATION_GUIDE.md`
3. Review example code above
4. Check vLLM documentation: https://docs.vllm.ai/

---

**Status:** ✅ Ready to use
**Model:** Qwen2.5-7B
**Server:** http://localhost:8000/v1

