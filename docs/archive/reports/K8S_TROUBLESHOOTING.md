# Kubernetes Troubleshooting Guide for Qallow

**Date:** 2025-10-28  
**Status:** ✅ COMPREHENSIVE GUIDE  

---

## 🔴 Error: "couldn't get version/kind" JSON Parse Error

### Symptoms
```
error: error loading config file "/root/Qallow/ops/deploy_all.sh": 
couldn't get version/kind; json parse error: json: cannot unmarshal string 
into Go value of type struct { APIVersion string ...
```

### Root Cause
kubectl is trying to parse a shell script as YAML. This happens when:
1. Wrong file path is passed to kubectl
2. Shell script is passed instead of YAML file
3. File doesn't exist or path is incorrect

### Solution
**Use the correct YAML files, not the shell script:**

```bash
# ❌ WRONG - Don't do this
kubectl apply -f /root/Qallow/ops/deploy_all.sh

# ✅ CORRECT - Use individual YAML files
kubectl apply -f /root/Qallow/k8s/qallow-namespace.yaml
kubectl apply -f /root/Qallow/k8s/qallow-logs-pvc.yaml
kubectl apply -f /root/Qallow/k8s/qallow-deploy.yaml
```

**Or use the deployment scripts:**

```bash
# Automated deployment
/root/Qallow/ops/deploy_all.sh

# Manual step-by-step deployment
/root/Qallow/ops/deploy_manual.sh
```

---

## 🟡 Error: "No GPU nodes found"

### Symptoms
```
[WARN] No GPU nodes found. Continuing with CPU-only deployment.
```

### Root Cause
NVIDIA device plugin not installed or no GPU hardware available

### Solution

**Option 1: Install NVIDIA Device Plugin**
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/nvidia-device-plugin.yml
kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system
```

**Option 2: Disable GPU in deployment**
```bash
ENABLE_GPU=false /root/Qallow/ops/deploy_all.sh
```

**Option 3: Check GPU availability**
```bash
# Check if GPU is available
kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable}{"\n"}{end}'

# Check device plugin status
kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset
```

---

## 🟡 Error: "PersistentVolumeClaim is Pending"

### Symptoms
```
NAME                STATUS    VOLUME   CAPACITY   ACCESS MODES   STORAGECLASS
qallow-logs-pvc     Pending                                       
```

### Root Cause
No storage class available or no persistent volumes

### Solution

**Option 1: Create Local Storage Class**
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

**Option 2: Create Persistent Volume**
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: qallow-logs-pv
spec:
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: /data/qallow-logs
EOF
```

**Option 3: Use emptyDir (temporary storage)**
Edit `/root/Qallow/k8s/qallow-deploy.yaml` and change:
```yaml
volumes:
  - name: qallow-logs
    emptyDir: {}
```

---

## 🟡 Error: "ImagePullBackOff"

### Symptoms
```
NAME                    READY   STATUS             RESTARTS   AGE
qallow-core-xxx         0/1     ImagePullBackOff   0          2m
```

### Root Cause
Docker image not available locally or in registry

### Solution

**Option 1: Build image locally**
```bash
cd /root/Qallow
docker build -t qallow/unified:latest .
```

**Option 2: Use existing image**
Edit `/root/Qallow/k8s/qallow-deploy.yaml`:
```yaml
containers:
  - name: qallow-core
    image: ubuntu:latest  # Use available image
    imagePullPolicy: IfNotPresent
```

**Option 3: Check image availability**
```bash
docker images | grep qallow
docker pull qallow/unified:latest
```

---

## 🟡 Error: "CrashLoopBackOff"

### Symptoms
```
NAME                    READY   STATUS             RESTARTS   AGE
qallow-core-xxx         0/1     CrashLoopBackOff   5          2m
```

### Root Cause
Container is crashing on startup

### Solution

**Check pod logs:**
```bash
kubectl logs -n qallow deployment/qallow-core
kubectl logs -n qallow deployment/qallow-core --previous
```

**Describe pod for events:**
```bash
kubectl describe pod -n qallow <pod-name>
```

**Common causes:**
- Missing dependencies
- Wrong command/args
- Permission issues
- Resource limits too low

---

## 🟡 Error: "Insufficient Memory"

### Symptoms
```
Warning  FailedScheduling  pod/qallow-core-xxx  
Insufficient memory
```

### Solution

**Option 1: Reduce resource requests**
Edit `/root/Qallow/k8s/qallow-deploy.yaml`:
```yaml
resources:
  limits:
    cpu: "2"
    memory: 4Gi
  requests:
    cpu: "2"
    memory: 4Gi
```

**Option 2: Check node resources**
```bash
kubectl top nodes
kubectl describe nodes
```

**Option 3: Add more nodes**
```bash
# For Docker Desktop, increase memory in settings
# For kubeadm, add more worker nodes
```

---

## 🟡 Error: "Connection Refused"

### Symptoms
```
curl: (7) Failed to connect to localhost port 8080: Connection refused
```

### Solution

**Step 1: Verify pod is running**
```bash
kubectl get pods -n qallow
```

**Step 2: Check port forwarding**
```bash
# Start port forwarding
kubectl port-forward -n qallow svc/qallow-service 8080:8080

# In another terminal, test
curl http://localhost:8080
```

**Step 3: Check service**
```bash
kubectl get svc -n qallow
kubectl describe svc qallow-service -n qallow
```

**Step 4: Check pod logs**
```bash
kubectl logs -n qallow deployment/qallow-core
```

---

## ✅ Verification Checklist

```bash
# 1. Check cluster
kubectl cluster-info
kubectl get nodes

# 2. Check namespace
kubectl get namespace qallow

# 3. Check pods
kubectl get pods -n qallow

# 4. Check services
kubectl get svc -n qallow

# 5. Check deployments
kubectl get deployments -n qallow

# 6. Check PVC
kubectl get pvc -n qallow

# 7. Check events
kubectl get events -n qallow

# 8. Check resource usage
kubectl top nodes
kubectl top pods -n qallow
```

---

## 🔧 Useful Commands

```bash
# Get all resources in namespace
kubectl get all -n qallow

# Describe resource
kubectl describe pod -n qallow <pod-name>

# View logs
kubectl logs -n qallow <pod-name>
kubectl logs -n qallow <pod-name> -f  # Follow logs

# Execute command in pod
kubectl exec -it -n qallow <pod-name> -- /bin/bash

# Port forward
kubectl port-forward -n qallow svc/<service-name> <local-port>:<remote-port>

# Delete resource
kubectl delete pod -n qallow <pod-name>
kubectl delete deployment -n qallow <deployment-name>

# Edit resource
kubectl edit deployment -n qallow qallow-core

# Scale deployment
kubectl scale deployment qallow-core -n qallow --replicas=5

# Rollout status
kubectl rollout status deployment/qallow-core -n qallow

# View resource YAML
kubectl get deployment -n qallow qallow-core -o yaml
```

---

## 📞 Getting Help

1. **Check logs first:**
   ```bash
   kubectl logs -n qallow deployment/qallow-core
   ```

2. **Describe resources:**
   ```bash
   kubectl describe pod -n qallow <pod-name>
   ```

3. **Check events:**
   ```bash
   kubectl get events -n qallow
   ```

4. **Review documentation:**
   - `K8S_DEPLOYMENT_GUIDE.md` - Deployment guide
   - `K8S_TROUBLESHOOTING.md` - This file

---

**Generated:** 2025-10-28  
**System:** Qallow v2.0  
**License:** MIT

