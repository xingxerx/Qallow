#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

kubectl apply -f "$ROOT_DIR/k8s/qallow-namespace.yaml"

set +e
kubectl apply -f "$ROOT_DIR/k8s/qallow-logs-pvc.yaml"
set -e
echo "[NOTE] Ensure NVIDIA device plugin is installed: kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.16.0/nvidia-device-plugin.yml"

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

echo "[INFO] Qallow cluster components deployed"
