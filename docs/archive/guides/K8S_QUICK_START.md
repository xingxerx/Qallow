# Kubernetes Quick Start Guide for Qallow

**Date:** 2025-10-28  
**Status:** ✅ READY TO USE  

---

## 🚀 30-Second Deployment

```bash
# Option 1: Automated (Recommended)
/root/Qallow/ops/deploy_all.sh

# Option 2: Manual step-by-step
/root/Qallow/ops/deploy_manual.sh
```

---

## 📋 Prerequisites

```bash
# Check kubectl is installed
kubectl version --short

# Check cluster is running
kubectl cluster-info

# Check nodes
kubectl get nodes
```

---

## 🎯 Deployment Steps

### Step 1: Deploy Namespace & RBAC
```bash
kubectl apply -f /root/Qallow/k8s/qallow-namespace.yaml
```

### Step 2: Create Storage
```bash
kubectl apply -f /root/Qallow/k8s/qallow-logs-pvc.yaml
```

### Step 3: Deploy Qallow Core
```bash
kubectl apply -f /root/Qallow/k8s/qallow-deploy.yaml
```

### Step 4: Deploy Monitoring
```bash
kubectl apply -f /root/Qallow/monitoring/prometheus-config.yaml
kubectl apply -f /root/Qallow/monitoring/prometheus-deploy.yaml
kubectl apply -f /root/Qallow/monitoring/grafana-deploy.yaml
kubectl apply -f /root/Qallow/monitoring/alertmanager/config.yaml
kubectl apply -f /root/Qallow/monitoring/alertmanager/deploy.yaml
```

### Step 5: Verify
```bash
kubectl get pods -n qallow
kubectl get svc -n qallow
```

---

## 🔌 Port Forwarding

```bash
# Terminal 1: Qallow API
kubectl port-forward -n qallow svc/qallow-service 8080:8080

# Terminal 2: Prometheus
kubectl port-forward -n qallow svc/prometheus-service 9090:9090

# Terminal 3: Grafana
kubectl port-forward -n qallow svc/grafana-service 3000:3000
```

---

## 🌐 Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Qallow API | http://localhost:8080 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/qallow |

---

## 🔍 Monitoring

```bash
# Check pod status
kubectl get pods -n qallow

# View pod logs
kubectl logs -n qallow deployment/qallow-core

# Follow logs in real-time
kubectl logs -n qallow deployment/qallow-core -f

# Describe pod
kubectl describe pod -n qallow <pod-name>

# Check resource usage
kubectl top pods -n qallow
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

## ⚠️ Common Issues

### Issue: "couldn't get version/kind" Error
**Solution:** Use YAML files, not shell scripts
```bash
# ❌ Wrong
kubectl apply -f /root/Qallow/ops/deploy_all.sh

# ✅ Correct
/root/Qallow/ops/deploy_all.sh
```

### Issue: Pods not starting
**Solution:** Check logs
```bash
kubectl logs -n qallow deployment/qallow-core
kubectl describe pod -n qallow <pod-name>
```

### Issue: PVC Pending
**Solution:** Create storage class
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

### Issue: ImagePullBackOff
**Solution:** Build image locally
```bash
cd /root/Qallow
docker build -t qallow/unified:latest .
```

---

## 📚 Full Documentation

- **Deployment Guide:** `K8S_DEPLOYMENT_GUIDE.md`
- **Troubleshooting:** `K8S_TROUBLESHOOTING.md`
- **This Guide:** `K8S_QUICK_START.md`

---

## 🎯 Next Steps

1. **Deploy:** Run `./ops/deploy_all.sh` or `./ops/deploy_manual.sh`
2. **Monitor:** Set up port forwarding to access services
3. **Verify:** Check pod status with `kubectl get pods -n qallow`
4. **Access:** Open browser to http://localhost:3000 (Grafana)
5. **Troubleshoot:** See `K8S_TROUBLESHOOTING.md` if issues occur

---

## 💡 Tips

- Use `kubectl get all -n qallow` to see all resources
- Use `kubectl logs -f` to follow logs in real-time
- Use `kubectl port-forward` to access services locally
- Use `kubectl describe` to get detailed information
- Use `kubectl exec` to run commands in pods

---

**Generated:** 2025-10-28  
**System:** Qallow v2.0  
**License:** MIT

