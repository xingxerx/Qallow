# Qallow Kubernetes Deployment Guide

This guide covers deploying Qallow as a distributed system on Kubernetes with GPU support, service mesh integration, and persistent storage.

---

## Prerequisites

- Kubernetes 1.24+ cluster
- NVIDIA GPU Operator installed
- Helm 3.0+
- kubectl configured
- Docker registry access

---

## 1. Install NVIDIA GPU Operator

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU Operator
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator-system \
  --create-namespace \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

Verify GPU availability:
```bash
kubectl get nodes -L nvidia.com/gpu
kubectl get pods -n gpu-operator-system
```

---

## 2. Install Istio Service Mesh

```bash
# Download Istio
curl -L https://istio.io/downloadIstio | sh -
cd istio-*

# Install Istio
./bin/istioctl install --set profile=production -y

# Enable sidecar injection
kubectl label namespace qallow istio-injection=enabled
```

---

## 3. Deploy Qallow Namespace and RBAC

```bash
kubectl apply -f k8s/qallow-namespace.yaml

# Verify
kubectl get namespace qallow
kubectl get serviceaccount -n qallow
```

---

## 4. Build and Push Docker Images

### Phase 12 Elasticity Service

```bash
# Build image
docker build -f Dockerfile.phase12 -t qallow/phase12:latest .

# Push to registry
docker tag qallow/phase12:latest your-registry/qallow/phase12:latest
docker push your-registry/qallow/phase12:latest
```

### Phase 13 Harmonic Service

```bash
# Build image
docker build -f Dockerfile.phase13 -t qallow/phase13:latest .

# Push to registry
docker tag qallow/phase13:latest your-registry/qallow/phase13:latest
docker push your-registry/qallow/phase13:latest
```

---

## 5. Deploy Phase 12 and Phase 13

```bash
# Deploy Phase 12
kubectl apply -f k8s/qallow-phase12-deployment.yaml

# Deploy Phase 13
kubectl apply -f k8s/qallow-phase13-deployment.yaml

# Verify deployments
kubectl get deployments -n qallow
kubectl get pods -n qallow
kubectl get svc -n qallow
```

---

## 6. Configure Istio VirtualService and DestinationRule

```bash
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: qallow-phase12
  namespace: qallow
spec:
  hosts:
  - qallow-phase12
  http:
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: qallow-phase12
        port:
          number: 50051
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: qallow-phase12
  namespace: qallow
spec:
  host: qallow-phase12
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
EOF
```

---

## 7. Install Prometheus and Grafana

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Port forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

Access Grafana at `http://localhost:3000` (default: admin/prom-operator)

---

## 8. Deploy ClickHouse for Telemetry Storage

```bash
# Add ClickHouse Helm repo
helm repo add clickhouse https://clickhouse-k8s.github.io/helm-charts
helm repo update

# Install ClickHouse
helm install clickhouse clickhouse/clickhouse \
  --namespace qallow \
  --values - <<EOF
replicas: 3
persistence:
  enabled: true
  size: 100Gi
resources:
  requests:
    cpu: 2
    memory: 4Gi
  limits:
    cpu: 4
    memory: 8Gi
EOF
```

---

## 9. Configure OpenTelemetry Collector

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: otel-collector-config
  namespace: qallow
data:
  config.yaml: |
    receivers:
      prometheus:
        config:
          scrape_configs:
          - job_name: 'qallow'
            static_configs:
            - targets: ['localhost:8080', 'localhost:8081']
    
    processors:
      batch:
        send_batch_size: 1024
        timeout: 10s
    
    exporters:
      clickhouse:
        endpoint: "tcp://clickhouse:9000"
        database: "qallow"
        table: "telemetry"
    
    service:
      pipelines:
        metrics:
          receivers: [prometheus]
          processors: [batch]
          exporters: [clickhouse]
EOF
```

---

## 10. Monitoring and Debugging

### Check Pod Status
```bash
kubectl get pods -n qallow -w
kubectl describe pod <pod-name> -n qallow
kubectl logs <pod-name> -n qallow -f
```

### Check Service Connectivity
```bash
kubectl exec -it <pod-name> -n qallow -- grpcurl -plaintext localhost:50051 list
```

### View Metrics
```bash
kubectl port-forward -n qallow svc/qallow-phase12 8080:8080
curl http://localhost:8080/metrics
```

### Check HPA Status
```bash
kubectl get hpa -n qallow
kubectl describe hpa qallow-phase12-hpa -n qallow
```

---

## 11. Scaling and Performance Tuning

### Manual Scaling
```bash
kubectl scale deployment qallow-phase12 --replicas=5 -n qallow
```

### View HPA Metrics
```bash
kubectl get hpa qallow-phase12-hpa -n qallow -w
```

### Adjust Resource Limits
```bash
kubectl set resources deployment qallow-phase12 \
  --requests=cpu=2,memory=4Gi \
  --limits=cpu=4,memory=8Gi \
  -n qallow
```

---

## 12. Troubleshooting

### Pod Stuck in Pending
```bash
kubectl describe pod <pod-name> -n qallow
# Check: GPU availability, resource requests, node selectors
```

### gRPC Connection Errors
```bash
kubectl logs <pod-name> -n qallow | grep -i grpc
# Check: Service DNS, port exposure, firewall
```

### High Memory Usage
```bash
kubectl top pods -n qallow
# Check: Memory leaks, batch sizes, checkpoint intervals
```

---

## 13. Production Checklist

- [ ] GPU Operator installed and verified
- [ ] Istio service mesh deployed
- [ ] Prometheus + Grafana monitoring active
- [ ] ClickHouse telemetry storage configured
- [ ] OpenTelemetry collector running
- [ ] Pod Disruption Budgets configured
- [ ] Horizontal Pod Autoscalers tuned
- [ ] Network policies enforced
- [ ] Resource quotas and limits set
- [ ] Backup strategy for persistent data
- [ ] Disaster recovery plan documented
- [ ] Security scanning enabled

---

## 14. Cleanup

```bash
# Delete Qallow deployments
kubectl delete -f k8s/qallow-phase12-deployment.yaml
kubectl delete -f k8s/qallow-phase13-deployment.yaml

# Delete namespace (cascades to all resources)
kubectl delete namespace qallow
```

---

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
- [Istio Service Mesh](https://istio.io/latest/docs/)
- [ClickHouse Kubernetes](https://clickhouse.com/docs/en/deployment-guides/kubernetes)

