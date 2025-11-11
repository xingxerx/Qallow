# Communication Index: How to Interact with Your App

## 🎯 Quick Navigation

### **I want to...**

- **[Send a simple message](#simple-message)** → 30 seconds
- **[Build a web application](#web-app)** → 2 minutes
- **[Write a Python script](#python-script)** → 1 minute
- **[Analyze quantum circuits](#quantum-analysis)** → 5 minutes
- **[Run QAOA optimization](#qaoa)** → 5 minutes
- **[Learn by experimenting](#interactive)** → 10 minutes
- **[Automate tasks](#automation)** → 5 minutes

---

## 📡 4 Communication Methods

### **1. REST API (HTTP Requests)**
- **Best for:** Web apps, microservices, remote access
- **Port:** 8008
- **Setup:** 2 minutes
- **File:** `COMMUNICATION_GUIDE.md` (REST API section)
- **Demo:** `python3 demo_rest_api_client.py`

### **2. Python SDK (Direct Import)**
- **Best for:** Scripts, notebooks, direct integration
- **Setup:** 30 seconds
- **File:** `COMMUNICATION_GUIDE.md` (Python SDK section)
- **Demo:** `python3 demo_interactive.py`

### **3. CLI Scripts (Shell Commands)**
- **Best for:** Automation, testing, deployment
- **Setup:** 1 minute
- **File:** `COMMUNICATION_GUIDE.md` (CLI section)
- **Demo:** `python3 tests/test_integration_cuda_cirq_kimi_cudaq.py`

### **4. Interactive Python (REPL)**
- **Best for:** Development, learning, experimentation
- **Setup:** 10 seconds
- **File:** `COMMUNICATION_GUIDE.md` (Interactive section)
- **Demo:** `python3 demo_interactive.py` (Option 6)

---

## 🚀 Quick Start Examples

### Simple Message
```bash
# Terminal 1
bash scripts/setup_kimi_k2_vllm.sh

# Terminal 2
python3 << 'PYTHON'
from python.agents.kimi_k2_agent import create_kimi_k2_agent
agent = create_kimi_k2_agent()
print(agent.chat("Hello!"))
PYTHON
```

### Web Application
```bash
# Terminal 1
bash scripts/setup_kimi_k2_vllm.sh

# Terminal 2
cd python/chat_server && uvicorn main:app --port 8008

# Terminal 3 (or from your web app)
curl -X POST http://localhost:8008/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Python Script
```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent

agent = create_kimi_k2_agent()
response = agent.chat("Your message here")
print(response)
```

### Quantum Analysis
```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cirq

# Create circuit
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))

# Simulate
result = cirq.Simulator().simulate(circuit)

# Analyze
agent = create_kimi_k2_agent()
print(agent.chat(f"Analyze: {result}"))
```

### QAOA Optimization
```python
from python.agents.kimi_k2_agent import create_kimi_k2_agent
import cudaq
import numpy as np

@cudaq.kernel
def qaoa():
    qubits = cudaq.qvector(3)
    h(qubits)
    for i in range(2):
        cz(qubits[i], qubits[i+1])
    for qubit in qubits:
        rx(np.pi/4, qubit)

result = cudaq.sample(qaoa, shots_count=1000)
agent = create_kimi_k2_agent()
print(agent.chat(f"Optimize: {result}"))
```

### Interactive Experimentation
```bash
python3 demo_interactive.py
# Choose option 6 for interactive chat loop
```

### Automation
```bash
# Run tests
python3 tests/test_integration_cuda_cirq_kimi_cudaq.py

# Or create a script and run it
python3 my_automation_script.py
```

---

## 📚 Documentation Files

| File | Purpose | Best For |
|------|---------|----------|
| **COMMUNICATION_GUIDE.md** | Complete reference | All methods |
| **demo_interactive.py** | 6 interactive examples | Learning |
| **demo_rest_api_client.py** | 6 API examples | Web apps |
| **docs/KIMI_K2_INTEGRATION.md** | Detailed setup | Setup |
| **docs/KIMI_K2_QUICK_REFERENCE.md** | Quick commands | Reference |
| **TROUBLESHOOTING_GUIDE.md** | Common issues | Debugging |

---

## 🎯 API Endpoints

```
POST   /chat          - Send message
POST   /chat/stream   - Stream response
POST   /chat/tools    - Tool calling
GET    /health        - Server status
GET    /models        - Available models
```

---

## 🐍 Python SDK Methods

```python
agent.chat(message)                    # Simple chat
agent.chat_stream(message)             # Streaming
agent.chat_with_tools(message, tools)  # Tool calling
agent.get_config()                     # Get config
agent.set_temperature(value)           # Set temperature
```

---

## ✅ Verification

```bash
# Check vLLM server
curl http://localhost:8000/v1/models

# Check chat server
curl http://localhost:8008/health

# Check dependencies
pip list | grep -E "openai|vllm|cirq|cudaq"

# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Check agent
python3 -c "from python.agents.kimi_k2_agent import create_kimi_k2_agent"
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Connection refused :8000 | `bash scripts/setup_kimi_k2_vllm.sh` |
| Connection refused :8008 | `cd python/chat_server && uvicorn main:app --port 8008` |
| Module not found | `pip install -r requirements.txt` |
| CUDA out of memory | Reduce batch size or use smaller model |
| No response | Check server logs and restart |

---

## 📊 Performance Comparison

| Method | Latency | Throughput | Complexity | Scalability |
|--------|---------|-----------|-----------|------------|
| REST API | 50-100ms | High | Medium | Excellent |
| Python SDK | 10-50ms | Very High | Low | Good |
| CLI Scripts | 100-500ms | Medium | Low | Good |
| Interactive | 50-100ms | Low | Very Low | Poor |

---

## 🎓 Example Workflows

### Workflow 1: Quantum Circuit Analysis
1. Create circuit with Cirq
2. Simulate with Cirq
3. Analyze with Kimi-K2

### Workflow 2: QAOA Optimization
1. Define QAOA kernel with CUDA-Q
2. Execute kernel
3. Analyze results with Kimi-K2

### Workflow 3: Web Application
1. Start vLLM server
2. Start FastAPI chat server
3. Send HTTP requests from web app

### Workflow 4: Batch Processing
1. Create Python script
2. Import Kimi-K2 agent
3. Process multiple messages in loop

---

## 🚀 Next Steps

1. **Choose your method** - Pick from REST API, Python SDK, CLI, or Interactive
2. **Start servers** - Run `bash scripts/setup_kimi_k2_vllm.sh`
3. **Run demo** - Try `python3 demo_interactive.py` or `python3 demo_rest_api_client.py`
4. **Integrate** - Use in your application
5. **Refer to docs** - Check `COMMUNICATION_GUIDE.md` for details

---

## 📞 Support

- **Setup issues:** See `TROUBLESHOOTING_GUIDE.md`
- **API details:** See `docs/KIMI_K2_INTEGRATION.md`
- **Quick commands:** See `docs/KIMI_K2_QUICK_REFERENCE.md`
- **Examples:** See `demo_interactive.py` or `demo_rest_api_client.py`

---

## ✨ Summary

You have **4 complete ways** to communicate with your integrated system:

1. **REST API** - For web apps and remote access
2. **Python SDK** - For scripts and direct integration
3. **CLI Scripts** - For automation and testing
4. **Interactive Python** - For development and learning

**Start now:** `python3 demo_interactive.py`

