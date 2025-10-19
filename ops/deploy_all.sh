#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVIDIA_PLUGIN_VERSION=${NVIDIA_PLUGIN_VERSION:-v0.16.0}
NVIDIA_PLUGIN_MANIFEST=${NVIDIA_PLUGIN_MANIFEST:-"https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${NVIDIA_PLUGIN_VERSION}/nvidia-device-plugin.yml"}
NVIDIA_PLUGIN_ROLLOUT_TIMEOUT=${NVIDIA_PLUGIN_ROLLOUT_TIMEOUT:-180s}

echo "[INFO] Applying core namespace and persistent volumes"
kubectl apply -f "$ROOT_DIR/k8s/qallow-namespace.yaml"

set +e
kubectl apply -f "$ROOT_DIR/k8s/qallow-logs-pvc.yaml"
set -e

ensure_nvidia_device_plugin() {
  echo "[INFO] Ensuring NVIDIA device plugin ($NVIDIA_PLUGIN_VERSION) is installed"
  if ! kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset >/dev/null 2>&1; then
    echo "[INFO] Installing NVIDIA device plugin for CUDA scheduling"
  else
    echo "[INFO] NVIDIA device plugin detected, reapplying manifest to keep configuration aligned"
  fi
  kubectl apply -f "$NVIDIA_PLUGIN_MANIFEST"
  kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout="$NVIDIA_PLUGIN_ROLLOUT_TIMEOUT"
}

validate_gpu_nodes() {
  echo "[INFO] Validating GPU-capable nodes advertise CUDA capacity"
  local gpu_nodes gpu_counts total_gpus
  gpu_nodes=$(kubectl get nodes -o jsonpath='{range .items[?(@.status.allocatable["nvidia.com/gpu"])]}{.metadata.name}{"\n"}{end}')
  if [[ -z "$gpu_nodes" ]]; then
    echo "[ERROR] No nodes in the cluster advertise the nvidia.com/gpu resource. CUDA workloads cannot be scheduled." >&2
    exit 1
  fi

  gpu_counts=$(kubectl get nodes -o jsonpath='{range .items[?(@.status.allocatable["nvidia.com/gpu"])]}{.status.allocatable["nvidia.com/gpu"]}{"\n"}{end}')
  total_gpus=$(awk '{sum+=$1} END {print sum+0}' <<<"$gpu_counts")
  echo "[INFO] GPU nodes ready for CUDA workloads:"
  echo "$gpu_nodes"
  echo "[INFO] Total allocatable GPUs: $total_gpus"
}

ensure_nvidia_device_plugin
validate_gpu_nodes

echo "[INFO] Deploying Qallow workloads and monitoring with GPU configuration"
kubectl apply -f "$ROOT_DIR/k8s/qallow-deploy.yaml"
kubectl apply -f "$ROOT_DIR/monitoring/prometheus-config.yaml"
kubectl apply -f "$ROOT_DIR/monitoring/prometheus-deploy.yaml"
kubectl apply -f "$ROOT_DIR/monitoring/grafana-deploy.yaml"

if kubectl get crd prometheusrules.monitoring.coreos.com >/dev/null 2>&1; then
  kubectl apply -f "$ROOT_DIR/monitoring/alerts/prometheus-rules.yaml"
else
  echo "[WARN] PrometheusRule CRD not found. Install kube-prometheus-stack or prometheus-operator before applying alert rules." >&2
fi

kubectl apply -f "$ROOT_DIR/monitoring/alertmanager/config.yaml"
kubectl apply -f "$ROOT_DIR/monitoring/alertmanager/deploy.yaml"

echo "[INFO] Waiting for qallow-core deployment to become available"
kubectl rollout status deployment/qallow-core -n qallow --timeout=300s

echo "[INFO] Qallow cluster components deployed with CUDA acceleration verified"
