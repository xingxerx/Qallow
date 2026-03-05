# Communication Guide: How to Interact with the Integrated System

## 🎯 Overview

There are **4 main ways** to communicate with your integrated CUDA + Cirq + Kimi-K2 + CUDA-Q system:

1. **REST API** (FastAPI Chat Server)
2. **Python SDK** (Direct imports)
3. **Command Line** (Scripts)
4. **Interactive Python** (REPL)

---

## 1️⃣ REST API (FastAPI Chat Server)

### Start the Server

```bash
# Terminal 1: Start vLLM server
bash scripts/setup_kimi_k2_vllm.sh

# Terminal 2: Start chat server with Kimi-K2 backend
export QALLOW_CHAT_BACKEND=kimi_k2
cd python/chat_server
uvicorn main:app --host 0.0.0.0 --port 8008
```

### API Endpoints

#### **POST /chat** - Send a message
```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello! Analyze this quantum circuit result: 00: 250, 11: 250",
    "backend": "kimi_k2"
  }'
```

**Response:**
```json
{
  "response": "The quantum circuit shows a perfect Bell pair state...",
  "backend": "kimi_k2",
  "timestamp": "2025-11-11T15:55:00"
}
```

#### **POST /chat/stream** - Streaming responses
```bash
curl -X POST http://localhost:8008/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain QAOA optimization"}' \
  --stream
```

#### **POST /chat/tools** - Tool calling
```bash
curl -X POST http://localhost:8008/chat/tools \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Create a Bell pair circuit",
    "tools": ["create_circuit", "simulate_circuit"]
  }'
```

#### **GET /health** - Server status
```bash
curl http://localhost:8008/health
```

**Response:**
```json
{
  "status": "healthy",
  "backend": "kimi_k2",
  "kimi_k2_available": true,
  "vllm_server": "http://localhost:8000/v1",
  "timestamp": "2025-11-11T15:55:00"
}
```

#### **GET /models** - Available models
```bash
curl http://localhost:8008/models
```

---

## 2️⃣ Python SDK (Direct Imports)

### Basic Chat

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

# Create agent
agent = create_kimi_k2_agent()

# Send message
response = agent.chat("Hello! What is QAOA?")
print(response)
```

### Streaming Chat

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()

# Stream response
for chunk in agent.chat_stream("Explain quantum circuits"):
    print(chunk, end="", flush=True)
```

### Tool Calling

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()

# Use tools
response = agent.chat_with_tools(
    message="Create a QAOA circuit for 3 qubits",
    tools=[
        {
            "name": "create_circuit",
            "description": "Create a quantum circuit",
            "parameters": {"type": "object", "properties": {"qubits": {"type": "integer"}}}
        }
    ]
)
print(response)
```

### Quantum Integration

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq
import cudaq

# Create quantum circuit with Cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

# Simulate with Cirq
simulator = cirq.Simulator()
result = simulator.simulate(circuit)

# Analyze with Kimi-K2
agent = create_kimi_k2_agent()
analysis = agent.chat(f"Analyze this quantum state: {result}")
print(analysis)
```

---

## 3️⃣ Command Line Scripts

### Run Integration Tests

```bash
# Basic integration tests
python3 tests/test_integration_cuda_cirq_kimi_cudaq.py

# Advanced QAOA tests
python3 tests/test_qaoa_with_kimi_k2.py
```

### Deploy Servers

```bash
# Start vLLM server
bash scripts/setup_kimi_k2_vllm.sh

# Start SGLang server (alternative)
bash scripts/setup_kimi_k2_sglang.sh

# Quick setup (one-time)
bash scripts/setup_kimi_k2_quick_start.sh
```

### Custom Python Scripts

```bash
# Create a script: my_quantum_app.py
python3 my_quantum_app.py
```

---

## 4️⃣ Interactive Python (REPL)

### Start Python REPL

```bash
python3
```

### Interactive Session

```python
# Import everything
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq
import cudaq
import torch

# Create agent
agent = create_kimi_k2_agent()

# Chat interactively
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    response = agent.chat(user_input)
    print(f"Agent: {response}\n")
```

---

## 📊 Communication Patterns

### Pattern 1: Simple Chat
```
User → Chat Server → Kimi-K2 → Response
```

### Pattern 2: Quantum Analysis
```
User → Python SDK → Cirq/CUDA-Q → Quantum Result → Kimi-K2 → Analysis
```

### Pattern 3: Tool Calling
```
User → Chat Server → Kimi-K2 (with tools) → Tool Execution → Result
```

### Pattern 4: Streaming
```
User → Chat Server → Kimi-K2 (streaming) → Chunks → User
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Backend selection
export QALLOW_CHAT_BACKEND=kimi_k2  # or: mock, ollama, deepseek

# Kimi-K2 server
export KIMI_K2_BASE_URL=http://localhost:8000/v1

# vLLM server
export VLLM_PORT=8000
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.9

# CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Python Configuration

```python
from python.agents.kimi_k2_agent import KimiK2Config, KimiK2Agent

# Custom configuration
config = KimiK2Config(
    base_url="http://localhost:8000/v1",
    temperature=0.6,
    max_tokens=4096,
    timeout=300
)

agent = KimiK2Agent(config)
```

---

## 📝 Example Workflows

### Workflow 1: Quantum Circuit Analysis

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq

# Create circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1),
    cirq.measure(q0, q1, key='result')
)

# Simulate
simulator = cirq.Simulator()
result = simulator.simulate(circuit)

# Analyze
agent = create_kimi_k2_agent()
analysis = agent.chat(f"What does this quantum state represent? {result}")
print(analysis)
```

### Workflow 2: QAOA Optimization

```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cudaq
import numpy as np

# Define QAOA kernel
@cudaq.kernel
def qaoa():
    qubits = cudaq.qvector(3)
    h(qubits)
    for i in range(2):
        cz(qubits[i], qubits[i+1])
    for qubit in qubits:
        rx(np.pi/4, qubit)

# Execute
result = cudaq.sample(qaoa, shots_count=1000)

# Analyze
agent = create_kimi_k2_agent()
analysis = agent.chat(f"Optimize this QAOA result: {result}")
print(analysis)
```

### Workflow 3: REST API Integration

```bash
#!/bin/bash

# Start servers
bash scripts/setup_kimi_k2_vllm.sh &
sleep 5
cd python/chat_server && uvicorn main:app --port 8008 &
sleep 2

# Send requests
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# Check health
curl http://localhost:8008/health
```

---

## 🚀 Quick Start Examples

### Example 1: Simple Chat (5 seconds)
```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
agent = create_kimi_k2_agent()
print(agent.chat("Hi!"))
```

### Example 2: REST API (10 seconds)
```bash
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Example 3: Quantum Analysis (30 seconds)
```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq

q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
result = cirq.Simulator().simulate(circuit)

agent = create_kimi_k2_agent()
print(agent.chat(f"Analyze: {result}"))
```

---

## 📚 Documentation References

- **API Details**: See `docs/KIMI_K2_INTEGRATION.md`
- **Quick Commands**: See `docs/KIMI_K2_QUICK_REFERENCE.md`
- **Troubleshooting**: See `TROUBLESHOOTING_GUIDE.md`
- **Test Examples**: See `tests/test_integration_cuda_cirq_kimi_cudaq.py`

---

## ✅ Verification

### Check Server Status
```bash
curl http://localhost:8008/health
```

### Check vLLM Server
```bash
curl http://localhost:8000/v1/models
```

### Run Tests
```bash
python3 tests/test_integration_cuda_cirq_kimi_cudaq.py
```

---

## 🎯 Summary

| Method | Speed | Complexity | Use Case |
|--------|-------|-----------|----------|
| **REST API** | Fast | Medium | Web apps, microservices |
| **Python SDK** | Fast | Low | Scripts, notebooks |
| **CLI Scripts** | Medium | Low | Automation, testing |
| **Interactive** | Slow | Low | Development, learning |

**Choose based on your needs!**

