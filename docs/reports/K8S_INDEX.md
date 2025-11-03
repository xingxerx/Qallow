# Kubernetes Deployment Index for Qallow

**Date:** 2025-10-28  
**Status:** ✅ COMPLETE & READY  
**Kubernetes Version:** 1.34.1  

---

## 📚 Documentation Files

### Quick References
- **`K8S_QUICK_START.md`** - 30-second deployment guide (START HERE!)
- **`K8S_INDEX.md`** - This file - navigation guide

### Comprehensive Guides
- **`K8S_DEPLOYMENT_GUIDE.md`** - Complete deployment instructions
- **`K8S_TROUBLESHOOTING.md`** - Common issues & solutions

---

## 🚀 Deployment Scripts

### Automated Deployment
- **`ops/deploy_all.sh`** (7.2K)
  - Fully automated deployment
  - Error handling & validation
  - GPU support toggle
  - Recommended for production

### Manual Deployment
- **`ops/deploy_manual.sh`** (5.2K)
  - Step-by-step deployment
  - Better for troubleshooting
  - Manual control over each step

---

## 📋 YAML Configuration Files

### Namespace & RBAC
- **`k8s/qallow-namespace.yaml`** (2.5K)
  - Namespace creation
  - Service account
  - ClusterRole & ClusterRoleBinding
  - NetworkPolicy
  - ResourceQuota
  - LimitRange

### Storage
- **`k8s/qallow-logs-pvc.yaml`** (185 bytes)
  - PersistentVolumeClaim (50Gi)
  - ReadWriteOnce access mode

### Qallow Core
- **`k8s/qallow-deploy.yaml`** (2.5K)
  - Qallow core deployment (3 replicas)
  - Service (ClusterIP)
  - CronJob for telemetry upload
  - GPU support configuration

### Phase Deployments
- **`k8s/qallow-phase12-deployment.yaml`** (4.8K)
  - Phase 12 specific deployment
- **`k8s/qallow-phase13-deployment.yaml`** (4.9K)
  - Phase 13 specific deployment

---

## 📊 Monitoring Stack

### Prometheus
- **`monitoring/prometheus-config.yaml`** (306 bytes)
  - Prometheus ConfigMap
  - Scrape configuration
- **`monitoring/prometheus-deploy.yaml`** (1.2K)
  - Prometheus deployment
  - Service

### Grafana
- **`monitoring/grafana-deploy.yaml`** (1.1K)
  - Grafana deployment
  - Service
  - Default credentials: admin/qallow

### AlertManager
- **`monitoring/alertmanager/config.yaml`** (421 bytes)
  - AlertManager ConfigMap
  - Routing configuration
- **`monitoring/alertmanager/deploy.yaml`** (1021 bytes)
  - AlertManager deployment

### Alert Rules
- **`monitoring/alerts/prometheus-rules.yaml`** (850 bytes)
  - Prometheus alert rules
  - Requires PrometheusRule CRD

---

## 🎯 Quick Start

### 1. Prerequisites
```bash
kubectl version --short
kubectl cluster-info
kubectl get nodes
```

### 2. Deploy
```bash
# Option A: Automated
/root/Qallow/ops/deploy_all.sh

# Option B: Manual
/root/Qallow/ops/deploy_manual.sh
```

### 3. Verify
```bash
kubectl get pods -n qallow
kubectl get svc -n qallow
```

### 4. Port Forward
```bash
kubectl port-forward -n qallow svc/qallow-service 8080:8080
kubectl port-forward -n qallow svc/grafana-service 3000:3000
```

### 5. Access
- Qallow: http://localhost:8080
- Grafana: http://localhost:3000 (admin/qallow)

---

## 🔍 Useful Commands

```bash
# Check cluster
kubectl cluster-info
kubectl get nodes

# Check namespace
kubectl get namespace qallow

# Check all resources
kubectl get all -n qallow

# Check pods
kubectl get pods -n qallow
kubectl describe pod -n qallow <pod-name>
kubectl logs -n qallow <pod-name>

# Check services
kubectl get svc -n qallow
kubectl describe svc -n qallow <service-name>

# Check deployments
kubectl get deployments -n qallow
kubectl describe deployment -n qallow <deployment-name>

# Port forward
kubectl port-forward -n qallow svc/<service-name> <local>:<remote>

# Delete resources
kubectl delete namespace qallow
```

---

## ⚠️ Common Issues

### Issue: "couldn't get version/kind" Error
**Solution:** Use deployment scripts, not shell script as YAML
```bash
# ❌ Wrong
kubectl apply -f /root/Qallow/ops/deploy_all.sh

# ✅ Correct
/root/Qallow/ops/deploy_all.sh
```

### Issue: PVC Pending
**Solution:** Create storage class
See: `K8S_TROUBLESHOOTING.md`

### Issue: ImagePullBackOff
**Solution:** Build image locally
```bash
cd /root/Qallow
docker build -t qallow/unified:latest .
```

### Issue: Pods not starting
**Solution:** Check logs
```bash
kubectl logs -n qallow deployment/qallow-core
```

---

## 📊 Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│              Kubernetes Cluster                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         qallow Namespace                     │  │
│  ├──────────────────────────────────────────────┤  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐   │  │
│  │  │  Qallow Core Deployment (3 replicas)│   │  │
│  │  │  - Pods running qallow binary       │   │  │
│  │  │  - GPU support (optional)           │   │  │
│  │  │  - Service: qallow-service:8080     │   │  │
│  │  └─────────────────────────────────────┘   │  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐   │  │
│  │  │  Monitoring Stack                   │   │  │
│  │  │  - Prometheus (metrics collection)  │   │  │
│  │  │  - Grafana (visualization)          │   │  │
│  │  │  - AlertManager (alerting)          │   │  │
│  │  └─────────────────────────────────────┘   │  │
│  │                                              │  │
│  │  ┌─────────────────────────────────────┐   │  │
│  │  │  Storage                            │   │  │
│  │  │  - PVC: qallow-logs-pvc (50Gi)      │   │  │
│  │  └─────────────────────────────────────┘   │  │
│  │                                              │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 📈 Resource Requirements

### Qallow Core
- CPU: 6 cores (request/limit)
- Memory: 12Gi (request/limit)
- GPU: 1 (optional)

### Prometheus
- CPU: 500m (typical)
- Memory: 512Mi (typical)

### Grafana
- CPU: 100m (typical)
- Memory: 256Mi (typical)

### Total (Minimum)
- CPU: 8 cores
- Memory: 16Gi
- Storage: 50Gi

---

## 🧹 Cleanup

```bash
# Delete entire namespace
kubectl delete namespace qallow

# Delete specific resources
kubectl delete deployment qallow-core -n qallow
kubectl delete svc qallow-service -n qallow
kubectl delete pvc qallow-logs-pvc -n qallow
```

---

## 📞 Support

1. **Quick Issues:** See `K8S_QUICK_START.md`
2. **Detailed Help:** See `K8S_TROUBLESHOOTING.md`
3. **Full Guide:** See `K8S_DEPLOYMENT_GUIDE.md`
4. **Check Logs:** `kubectl logs -n qallow deployment/qallow-core`

---

## ✅ Verification Checklist

- [ ] Kubernetes cluster running
- [ ] kubectl configured
- [ ] Deployment scripts executable
- [ ] YAML files present
- [ ] Namespace created
- [ ] Pods running
- [ ] Services accessible
- [ ] Port forwarding working
- [ ] Metrics being collected
- [ ] Grafana accessible

---

**Generated:** 2025-10-28  
**System:** Qallow v2.0  
**License:** MIT  
**Status:** ✅ COMPLETE & TESTED

