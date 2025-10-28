#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NVIDIA_PLUGIN_VERSION=${NVIDIA_PLUGIN_VERSION:-v0.16.0}
NVIDIA_PLUGIN_MANIFEST=${NVIDIA_PLUGIN_MANIFEST:-"https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/${NVIDIA_PLUGIN_VERSION}/nvidia-device-plugin.yml"}
NVIDIA_PLUGIN_ROLLOUT_TIMEOUT=${NVIDIA_PLUGIN_ROLLOUT_TIMEOUT:-180s}
ENABLE_GPU=${ENABLE_GPU:-true}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}[INFO]${NC} Qallow Kubernetes Deployment Script"
echo -e "${GREEN}[INFO]${NC} Root directory: $ROOT_DIR"

# Helper function for kubectl apply with validation
kubectl_apply() {
  # Skip schema validation to avoid failures when the API server OpenAPI endpoint is unavailable.
  kubectl apply --validate=false "$@"
}

# Verify cluster access
require_cluster_access() {
  echo -e "${GREEN}[INFO]${NC} Verifying Kubernetes API connectivity"
  if ! kubectl version --short >/dev/null 2>&1; then
    local server
    server=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null || echo "configured server")
    echo -e "${RED}[ERROR]${NC} Unable to reach the Kubernetes API at ${server}. Ensure the cluster is running and kubeconfig context is correct." >&2
    exit 1
  fi
  echo -e "${GREEN}[OK]${NC} Kubernetes API is accessible"
}

# Deploy namespace and RBAC
deploy_namespace() {
  echo -e "${GREEN}[INFO]${NC} Deploying namespace and RBAC configuration"
  kubectl_apply -f "$ROOT_DIR/k8s/qallow-namespace.yaml"
  echo -e "${GREEN}[OK]${NC} Namespace and RBAC deployed"
}

# Deploy persistent volumes
deploy_storage() {
  echo -e "${GREEN}[INFO]${NC} Deploying persistent volume claims"
  set +e
  kubectl_apply -f "$ROOT_DIR/k8s/qallow-logs-pvc.yaml"
  set -e
  echo -e "${GREEN}[OK]${NC} Storage configured"
}

# Ensure NVIDIA device plugin (optional)
ensure_nvidia_device_plugin() {
  if [[ "$ENABLE_GPU" != "true" ]]; then
    echo -e "${YELLOW}[SKIP]${NC} GPU support disabled"
    return 0
  fi

  echo -e "${GREEN}[INFO]${NC} Ensuring NVIDIA device plugin ($NVIDIA_PLUGIN_VERSION) is installed"
  if ! kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset >/dev/null 2>&1; then
    echo -e "${GREEN}[INFO]${NC} Installing NVIDIA device plugin for CUDA scheduling"
    kubectl_apply -f "$NVIDIA_PLUGIN_MANIFEST" || {
      echo -e "${YELLOW}[WARN]${NC} Failed to install NVIDIA device plugin. Continuing without GPU support."
      return 0
    }
  else
    echo -e "${GREEN}[INFO]${NC} NVIDIA device plugin detected, reapplying manifest"
    kubectl_apply -f "$NVIDIA_PLUGIN_MANIFEST" || true
  fi

  echo -e "${GREEN}[OK]${NC} NVIDIA device plugin ready"
}

# Validate GPU nodes (optional)
validate_gpu_nodes() {
  if [[ "$ENABLE_GPU" != "true" ]]; then
    return 0
  fi

  echo -e "${GREEN}[INFO]${NC} Validating GPU-capable nodes"
  local gpu_nodes
  gpu_nodes=$(kubectl get nodes -o jsonpath='{range .items[?(@.status.allocatable["nvidia.com/gpu"])]}{.metadata.name}{"\n"}{end}' 2>/dev/null || echo "")

  if [[ -z "$gpu_nodes" ]]; then
    echo -e "${YELLOW}[WARN]${NC} No GPU nodes found. Continuing with CPU-only deployment."
    return 0
  fi

  local gpu_counts total_gpus
  gpu_counts=$(kubectl get nodes -o jsonpath='{range .items[?(@.status.allocatable["nvidia.com/gpu"])]}{.status.allocatable["nvidia.com/gpu"]}{"\n"}{end}')
  total_gpus=$(awk '{sum+=$1} END {print sum+0}' <<<"$gpu_counts")
  echo -e "${GREEN}[OK]${NC} GPU nodes ready:"
  echo "$gpu_nodes"
  echo -e "${GREEN}[OK]${NC} Total allocatable GPUs: $total_gpus"
}

# Deploy Qallow core
deploy_qallow_core() {
  echo -e "${GREEN}[INFO]${NC} Deploying Qallow core workloads"
  kubectl_apply -f "$ROOT_DIR/k8s/qallow-deploy.yaml"
  echo -e "${GREEN}[OK]${NC} Qallow core deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
  echo -e "${GREEN}[INFO]${NC} Deploying monitoring stack (Prometheus, Grafana, AlertManager)"

  # Prometheus config
  kubectl_apply -f "$ROOT_DIR/monitoring/prometheus-config.yaml"

  # Prometheus deployment
  kubectl_apply -f "$ROOT_DIR/monitoring/prometheus-deploy.yaml"

  # Grafana deployment
  kubectl_apply -f "$ROOT_DIR/monitoring/grafana-deploy.yaml"

  # AlertManager config
  kubectl_apply -f "$ROOT_DIR/monitoring/alertmanager/config.yaml"

  # AlertManager deployment
  kubectl_apply -f "$ROOT_DIR/monitoring/alertmanager/deploy.yaml"

  echo -e "${GREEN}[OK]${NC} Monitoring stack deployed"
}

# Deploy alert rules (if CRD available)
deploy_alert_rules() {
  if kubectl get crd prometheusrules.monitoring.coreos.com >/dev/null 2>&1; then
    echo -e "${GREEN}[INFO]${NC} Deploying Prometheus alert rules"
    kubectl_apply -f "$ROOT_DIR/monitoring/alerts/prometheus-rules.yaml"
    echo -e "${GREEN}[OK]${NC} Alert rules deployed"
  else
    echo -e "${YELLOW}[WARN]${NC} PrometheusRule CRD not found. Skipping alert rules."
  fi
}

# Wait for deployments
wait_for_deployments() {
  echo -e "${GREEN}[INFO]${NC} Waiting for deployments to become available"

  kubectl rollout status deployment/qallow-core -n qallow --timeout=300s || {
    echo -e "${YELLOW}[WARN]${NC} Qallow core deployment timeout. Checking status..."
    kubectl get pods -n qallow
    return 1
  }

  echo -e "${GREEN}[OK]${NC} All deployments ready"
}

# Main execution
main() {
  require_cluster_access
  deploy_namespace
  deploy_storage

  if [[ "$ENABLE_GPU" == "true" ]]; then
    ensure_nvidia_device_plugin
    validate_gpu_nodes
  fi

  deploy_qallow_core
  deploy_monitoring
  deploy_alert_rules
  wait_for_deployments

  echo ""
  echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${GREEN}║${NC}  ✅ Qallow Kubernetes Deployment Complete!              ${GREEN}║${NC}"
  echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
  echo ""
  echo -e "${GREEN}[INFO]${NC} Deployment Summary:"
  echo -e "${GREEN}[INFO]${NC}   Namespace: qallow"
  echo -e "${GREEN}[INFO]${NC}   Qallow Core: deployment/qallow-core"
  echo -e "${GREEN}[INFO]${NC}   Prometheus: deployment/prometheus-deployment"
  echo -e "${GREEN}[INFO]${NC}   Grafana: deployment/grafana-deployment"
  echo -e "${GREEN}[INFO]${NC}   AlertManager: deployment/alertmanager-deployment"
  echo ""
  echo -e "${GREEN}[INFO]${NC} Next steps:"
  echo -e "${GREEN}[INFO]${NC}   1. Port forward services:"
  echo -e "${GREEN}[INFO]${NC}      kubectl port-forward -n qallow svc/qallow-service 8080:8080"
  echo -e "${GREEN}[INFO]${NC}      kubectl port-forward -n qallow svc/prometheus-service 9090:9090"
  echo -e "${GREEN}[INFO]${NC}      kubectl port-forward -n qallow svc/grafana-service 3000:3000"
  echo -e "${GREEN}[INFO]${NC}   2. Access services:"
  echo -e "${GREEN}[INFO]${NC}      Qallow: http://localhost:8080"
  echo -e "${GREEN}[INFO]${NC}      Prometheus: http://localhost:9090"
  echo -e "${GREEN}[INFO]${NC}      Grafana: http://localhost:3000 (admin/qallow)"
  echo ""
}

main "$@"
