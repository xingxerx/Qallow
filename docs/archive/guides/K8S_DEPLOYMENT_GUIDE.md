# Kubernetes Deployment Guide for Qallow

**Date:** 2025-10-28  
**Status:** ✅ FIXED & READY  
**Kubernetes Version:** 1.34.1  
**Platform:** Docker Desktop / kubeadm  

---

## 🚀 Quick Start

### Prerequisites
```bash
# Verify kubectl is installed and configured
kubectl version --short

# Verify cluster is running
kubectl cluster-info

# Check nodes
kubectl get nodes
```

### Deployment Steps

#### 1. Deploy Namespace and RBAC
```bash
kubectl apply -f /root/Qallow/k8s/qallow-namespace.yaml
```

#### 2. Create Persistent Volume Claim
```bash
kubectl apply -f /root/Qallow/k8s/qallow-logs-pvc.yaml
```

#### 3. Deploy Qallow Core
```bash
kubectl apply -f /root/Qallow/k8s/qallow-deploy.yaml
```

#### 4. Deploy Monitoring Stack
```bash
# Prometheus Config
kubectl apply -f /root/Qallow/monitoring/prometheus-config.yaml

# Prometheus Deployment
kubectl apply -f /root/Qallow/monitoring/prometheus-deploy.yaml

# Grafana Deployment
kubectl apply -f /root/Qallow/monitoring/grafana-deploy.yaml

# AlertManager Config
kubectl apply -f /root/Qallow/monitoring/alertmanager/config.yaml

# AlertManager Deployment
kubectl apply -f /root/Qallow/monitoring/alertmanager/deploy.yaml
```

#### 5. Verify Deployment
```bash
# Check namespace
kubectl get namespace qallow

# Check pods
kubectl get pods -n qallow

# Check services
kubectl get svc -n qallow

# Check deployments
kubectl get deployments -n qallow
```

---

## ⚠️ Common Issues & Fixes

### Issue 1: "couldn't get version/kind" Error
**Cause:** kubectl trying to parse shell script as YAML  
**Fix:** Use the correct YAML file paths, not the shell script

### Issue 2: GPU Not Available
**Cause:** NVIDIA device plugin not installed  
**Fix:** Install NVIDIA device plugin manually:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/nvidia-device-plugin.yml
```

### Issue 3: PVC Pending
**Cause:** No storage class available  
**Fix:** Create a storage class or use local storage:
```bash
kubectl apply -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
EOF
```

### Issue 4: Image Pull Errors
**Cause:** Image not available locally  
**Fix:** Build and push image first:
```bash
cd /root/Qallow
docker build -t qallow/unified:latest .
```

---

## 📊 Monitoring & Access

### Port Forwarding
```bash
# Qallow Service
kubectl port-forward -n qallow svc/qallow-service 8080:8080

# Prometheus
kubectl port-forward -n qallow svc/prometheus-service 9090:9090

# Grafana
kubectl port-forward -n qallow svc/grafana-service 3000:3000
```

### Access URLs
- **Qallow API:** http://localhost:8080
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/qallow)

---

## 🔧 Manual Deployment (Step-by-Step)

If automated deployment fails, deploy manually:

```bash
# 1. Create namespace
kubectl create namespace qallow

# 2. Create PVC
kubectl apply -f /root/Qallow/k8s/qallow-logs-pvc.yaml

# 3. Create ConfigMaps
kubectl apply -f /root/Qallow/monitoring/prometheus-config.yaml
kubectl apply -f /root/Qallow/monitoring/alertmanager/config.yaml

# 4. Create Deployments
kubectl apply -f /root/Qallow/k8s/qallow-deploy.yaml
kubectl apply -f /root/Qallow/monitoring/prometheus-deploy.yaml
kubectl apply -f /root/Qallow/monitoring/grafana-deploy.yaml
kubectl apply -f /root/Qallow/monitoring/alertmanager/deploy.yaml

# 5. Verify
kubectl get all -n qallow
```

---

## 🧹 Cleanup

```bash
# Delete entire namespace (removes all resources)
kubectl delete namespace qallow

# Or delete specific resources
kubectl delete deployment qallow-core -n qallow
kubectl delete svc qallow-service -n qallow
kubectl delete pvc qallow-logs-pvc -n qallow
```

---

## 📋 Troubleshooting

### Check Pod Logs
```bash
kubectl logs -n qallow deployment/qallow-core
kubectl logs -n qallow deployment/prometheus-deployment
kubectl logs -n qallow deployment/grafana-deployment
```

### Describe Pod
```bash
kubectl describe pod -n qallow <pod-name>
```

### Check Events
```bash
kubectl get events -n qallow
```

### Check Resource Usage
```bash
kubectl top nodes
kubectl top pods -n qallow
```

---

## ✅ Verification Checklist

- [ ] Kubernetes cluster running
- [ ] kubectl configured correctly
- [ ] Namespace created
- [ ] PVC created
- [ ] Qallow pods running
- [ ] Prometheus running
- [ ] Grafana running
- [ ] Services accessible
- [ ] Port forwarding working
- [ ] Metrics being collected

---

**Generated:** 2025-10-28  
**System:** Qallow v2.0  
**License:** MIT

