# Getting Started with Qallow Multi-Agent Orchestration

Welcome! This guide will get you up and running in 5 minutes.

## 🚀 Quick Start (5 minutes)

### Step 1: Navigate to the orchestration directory

```bash
cd qallow/agents/orchestration
```

### Step 2: Start the local development environment

```bash
docker compose up --build
```

This starts:
- Redis (message broker)
- Orchestrator API (port 8000)
- 3 Worker replicas
- Flower monitoring UI (port 5555)

### Step 3: In another terminal, run the test suite

```bash
cd qallow/agents/orchestration
python test_harness.py
```

Expected output:
```
✓ Health check passed
✓ Heavy compute test passed
✓ Fetch test passed
✓ Sleep test passed
✓ Parallel execution test passed
✓ Batch processing test passed
```

### Step 4: Submit a job manually

```bash
curl -X POST http://localhost:8000/submit \
  -H "content-type: application/json" \
  -d '{"kind":"sleep_ms","args":{"ms":1000}}'
```

Response:
```json
{"task_id":"abc123...","status":"submitted"}
```

### Step 5: Check the status

```bash
curl http://localhost:8000/status/abc123...
```

Response:
```json
{"task_id":"abc123...","status":"completed","result":{"elapsed_ms":1050}}
```

## 📊 Monitoring

### Flower UI (Task Monitoring)
- URL: http://localhost:5555
- Shows all tasks, workers, and execution history

### Orchestrator API
- URL: http://localhost:8000
- Endpoints:
  - `GET /health` - Health check
  - `POST /submit` - Submit job
  - `GET /status/{task_id}` - Get status
  - `GET /metrics` - Metrics

### Redis CLI
```bash
docker exec qallow-redis redis-cli
> KEYS *
> GET task:abc123
```

## 📚 Documentation

### For Local Development
- **[orchestration/QUICKSTART.md](./qallow/agents/orchestration/QUICKSTART.md)** - Detailed local setup
- **[orchestration/README.md](./qallow/agents/orchestration/README.md)** - Architecture overview

### For Production Deployment
- **[orchestration/k8s/DEPLOYMENT_GUIDE.md](./qallow/agents/orchestration/k8s/DEPLOYMENT_GUIDE.md)** - Step-by-step K8s deployment
- **[orchestration/k8s/README.md](./qallow/agents/orchestration/k8s/README.md)** - K8s manifests overview

### For Integration
- **[qallow/agents/integration.py](./qallow/agents/integration.py)** - Integration with existing agents
- **[qallow/agents/INDEX.md](./qallow/agents/INDEX.md)** - Complete navigation guide

### For Implementation Details
- **[MULTIAGENT_DELIVERY.md](./MULTIAGENT_DELIVERY.md)** - What was delivered
- **[qallow/agents/IMPLEMENTATION_SUMMARY.md](./qallow/agents/IMPLEMENTATION_SUMMARY.md)** - Technical details

## 🔧 Common Tasks

### Submit a Heavy Compute Job

```bash
curl -X POST http://localhost:8000/submit \
  -H "content-type: application/json" \
  -d '{"kind":"heavy_compute","args":{"n":10000000}}'
```

### Submit a Fetch Job

```bash
curl -X POST http://localhost:8000/submit \
  -H "content-type: application/json" \
  -d '{"kind":"fetch","args":{"url":"https://api.github.com"}}'
```

### Submit Multiple Jobs in Parallel

```bash
for i in {1..5}; do
  curl -X POST http://localhost:8000/submit \
    -H "content-type: application/json" \
    -d "{\"kind\":\"sleep_ms\",\"args\":{\"ms\":1000}}" &
done
wait
```

### Check Orchestrator Metrics

```bash
curl http://localhost:8000/metrics
```

### View Worker Logs

```bash
docker compose logs worker-1
docker compose logs worker-2
docker compose logs worker-3
```

### View Orchestrator Logs

```bash
docker compose logs orchestrator
```

## 🐍 Python Integration

### Using the Integration Bridge

```python
from qallow.agents.integration import create_bridge

# Create bridge to orchestrator
bridge = create_bridge("http://localhost:8000")

# Submit inference task
task_id = bridge.submit_inference_task(
    model_path="/models/agent-v1",
    input_data={"query": "What is AI?"}
)

# Wait for result
result = bridge.wait_for_task(task_id, timeout=300)
print(result)
```

### Submitting Batch Tasks

```python
from qallow.agents.integration import create_bridge

bridge = create_bridge("http://localhost:8000")

# Submit batch inference
task_id = bridge.submit_batch_inference(
    model_path="/models/agent-v1",
    input_batch=[
        {"query": "What is AI?"},
        {"query": "What is ML?"},
        {"query": "What is DL?"}
    ]
)

result = bridge.wait_for_task(task_id)
print(result)
```

## 🚀 Production Deployment

### Build Docker Image

```bash
cd qallow/agents/orchestration
docker build -t your-registry/qallow-agents:latest .
docker push your-registry/qallow-agents:latest
```

### Deploy to Kubernetes

```bash
# Update image in k8s manifests
sed -i 's|your-registry/qallow-agents:latest|your-registry/qallow-agents:latest|g' k8s/*.yaml

# Deploy
bash deploy.sh
```

### Verify Deployment

```bash
kubectl -n qallow-agents get pods
kubectl -n qallow-agents get svc
kubectl -n qallow-agents get hpa
```

## 🧪 Testing

### Run Automated Tests

```bash
cd qallow/agents/orchestration
python test_harness.py
```

### Run Specific Test

```python
from test_harness import OrchestratorTestHarness

harness = OrchestratorTestHarness("http://localhost:8000")
harness.test_heavy_compute()
harness.test_parallel_execution()
```

## 🛑 Stopping Services

### Stop Local Development

```bash
docker compose down
```

### Remove All Data

```bash
docker compose down -v
```

## 🆘 Troubleshooting

### Services won't start

```bash
# Check logs
docker compose logs

# Rebuild images
docker compose build --no-cache

# Start fresh
docker compose down -v
docker compose up --build
```

### Can't connect to orchestrator

```bash
# Check if service is running
docker compose ps

# Check logs
docker compose logs orchestrator

# Test connection
curl http://localhost:8000/health
```

### Workers not processing tasks

```bash
# Check worker logs
docker compose logs worker-1

# Check Redis connection
docker exec qallow-redis redis-cli ping

# Check task queue
docker exec qallow-redis redis-cli LLEN celery
```

### High memory usage

```bash
# Check memory usage
docker stats

# Reduce worker concurrency in docker-compose.yml
# Restart services
docker compose restart
```

## 📖 Next Steps

1. **Explore Local Setup** - Follow [orchestration/QUICKSTART.md](./qallow/agents/orchestration/QUICKSTART.md)
2. **Understand Architecture** - Read [orchestration/README.md](./qallow/agents/orchestration/README.md)
3. **Deploy to Production** - Follow [orchestration/k8s/DEPLOYMENT_GUIDE.md](./qallow/agents/orchestration/k8s/DEPLOYMENT_GUIDE.md)
4. **Integrate with Agents** - Use [qallow/agents/integration.py](./qallow/agents/integration.py)
5. **Customize Tasks** - Edit [orchestration/tasks.py](./qallow/agents/orchestration/tasks.py)

## 📞 Support

For issues or questions:
1. Check relevant documentation
2. Review test harness for examples
3. Check logs: `docker compose logs`
4. Check K8s logs: `kubectl -n qallow-agents logs`

## 📄 File Structure

```
qallow/agents/
├── INDEX.md                          # Navigation guide
├── IMPLEMENTATION_SUMMARY.md         # What was built
├── integration.py                    # Integration layer
├── config.yaml                       # Configuration
└── orchestration/
    ├── README.md                     # Architecture
    ├── QUICKSTART.md                 # Local setup
    ├── requirements.txt              # Dependencies
    ├── Dockerfile                    # Container
    ├── docker-compose.yml           # Local dev
    ├── test_harness.py              # Tests
    ├── tasks.py                     # Task definitions
    ├── worker.py                    # Celery worker
    ├── orchestrator.py              # FastAPI API
    └── k8s/
        ├── README.md                # K8s guide
        ├── DEPLOYMENT_GUIDE.md      # Deployment
        ├── namespace.yaml
        ├── redis.yaml
        ├── orchestrator.yaml
        ├── worker.yaml
        ├── monitoring.yaml
        ├── ingress.yaml
        └── deploy.sh
```

---

**Ready to get started?** Run:
```bash
cd qallow/agents/orchestration
docker compose up --build
```

Then in another terminal:
```bash
python test_harness.py
```

Enjoy! 🚀

