# Qallow Unified Cluster Deployment

## Prerequisites
- Kubernetes v1.28 or newer with GPU-capable nodes
- NVIDIA device plugin for Kubernetes (`kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/nvidia-device-plugin.yml`)
- Docker image `qallow/unified:latest` built with `build/qallow_unified_cuda`
- kubectl configured for the target cluster

## Deployment Steps
```bash
# Clone or copy the manifests to your workstation
cd /path/to/Qallow

# Create persistent storage for logs
kubectl apply -f k8s/qallow-logs-pvc.yaml

# Ensure the NVIDIA device plugin is installed (only once per cluster)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/nvidia-device-plugin.yml

# Deploy Qallow core workloads and monitoring stack
./ops/deploy_all.sh
```

## Accessing Monitoring
```bash
# In a separate terminal
./ops/port_forward.sh
```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin / qallow)

Import the dashboard JSON file located at `monitoring/dashboards/qallow-unified-metrics.json` into Grafana via **Dashboards → Import**.

## Service Checks
```bash
./ops/test_smoke.sh
```
This verifies the core deployment, service availability, and returns a sample of the `/metrics` output.

## Service Level Objectives
- Elastic Drift < 0.02 seconds
- Harmonic Resonance Index ≈ 1.0
- Ethics Integrity ≥ 0.98
- Entropy Factor ≤ 2.5
- GPU Utilization < 90% sustained

## Incident Response
- **Auto-halt trigger**: Alert `QallowEthicsBreach` or `QallowHighEntropy` fires for over two minutes. Alertmanager posts to `ops/auto-halt-webhook.sh`, which scales `qallow-core` to zero replicas.
- **Recovery**: Investigate ethics and entropy metrics, review logs under `/logs`, address root cause, then restart the workload:
  ```bash
  kubectl scale deployment/qallow-core --replicas=3
  ```
- Monitor Grafana dashboard to confirm metrics return to healthy ranges before resuming production traffic.
