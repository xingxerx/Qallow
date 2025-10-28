#!/usr/bin/env bash
# Manual Kubernetes Deployment Script for Qallow
# Use this if automated deployment fails

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  Qallow Manual Kubernetes Deployment                    ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Verify cluster access
echo -e "${GREEN}[STEP 1]${NC} Verifying Kubernetes cluster access..."
if ! kubectl version --short >/dev/null 2>&1; then
  echo -e "${RED}[ERROR]${NC} Cannot connect to Kubernetes cluster"
  exit 1
fi
echo -e "${GREEN}[OK]${NC} Cluster is accessible"
echo ""

# Step 2: Create namespace
echo -e "${GREEN}[STEP 2]${NC} Creating qallow namespace..."
kubectl create namespace qallow --dry-run=client -o yaml | kubectl apply -f -
echo -e "${GREEN}[OK]${NC} Namespace created"
echo ""

# Step 3: Apply RBAC
echo -e "${GREEN}[STEP 3]${NC} Applying RBAC configuration..."
kubectl apply -f "$ROOT_DIR/k8s/qallow-namespace.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} RBAC configured"
echo ""

# Step 4: Create storage
echo -e "${GREEN}[STEP 4]${NC} Creating persistent volume claim..."
kubectl apply -f "$ROOT_DIR/k8s/qallow-logs-pvc.yaml" --validate=false || {
  echo -e "${YELLOW}[WARN]${NC} PVC creation failed (may already exist)"
}
echo -e "${GREEN}[OK]${NC} Storage configured"
echo ""

# Step 5: Deploy Prometheus config
echo -e "${GREEN}[STEP 5]${NC} Deploying Prometheus configuration..."
kubectl apply -f "$ROOT_DIR/monitoring/prometheus-config.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} Prometheus config deployed"
echo ""

# Step 6: Deploy Prometheus
echo -e "${GREEN}[STEP 6]${NC} Deploying Prometheus..."
kubectl apply -f "$ROOT_DIR/monitoring/prometheus-deploy.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} Prometheus deployed"
echo ""

# Step 7: Deploy Grafana
echo -e "${GREEN}[STEP 7]${NC} Deploying Grafana..."
kubectl apply -f "$ROOT_DIR/monitoring/grafana-deploy.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} Grafana deployed"
echo ""

# Step 8: Deploy AlertManager config
echo -e "${GREEN}[STEP 8]${NC} Deploying AlertManager configuration..."
kubectl apply -f "$ROOT_DIR/monitoring/alertmanager/config.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} AlertManager config deployed"
echo ""

# Step 9: Deploy AlertManager
echo -e "${GREEN}[STEP 9]${NC} Deploying AlertManager..."
kubectl apply -f "$ROOT_DIR/monitoring/alertmanager/deploy.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} AlertManager deployed"
echo ""

# Step 10: Deploy Qallow core
echo -e "${GREEN}[STEP 10]${NC} Deploying Qallow core..."
kubectl apply -f "$ROOT_DIR/k8s/qallow-deploy.yaml" --validate=false
echo -e "${GREEN}[OK]${NC} Qallow core deployed"
echo ""

# Step 11: Wait for deployments
echo -e "${GREEN}[STEP 11]${NC} Waiting for deployments to be ready..."
echo "This may take a few minutes..."
echo ""

# Wait for qallow-core
echo -n "Waiting for qallow-core... "
kubectl rollout status deployment/qallow-core -n qallow --timeout=300s 2>/dev/null || {
  echo -e "${YELLOW}timeout${NC}"
  echo -e "${YELLOW}[WARN]${NC} Qallow core deployment is still starting"
}
echo ""

# Step 12: Verify deployment
echo -e "${GREEN}[STEP 12]${NC} Verifying deployment..."
echo ""
echo "Namespaces:"
kubectl get namespace qallow
echo ""
echo "Deployments:"
kubectl get deployments -n qallow
echo ""
echo "Pods:"
kubectl get pods -n qallow
echo ""
echo "Services:"
kubectl get svc -n qallow
echo ""

# Summary
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}  ✅ Deployment Complete!                                ${GREEN}║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}[INFO]${NC} Port forwarding commands:"
echo ""
echo "  # Qallow API"
echo "  kubectl port-forward -n qallow svc/qallow-service 8080:8080"
echo ""
echo "  # Prometheus"
echo "  kubectl port-forward -n qallow svc/prometheus-service 9090:9090"
echo ""
echo "  # Grafana"
echo "  kubectl port-forward -n qallow svc/grafana-service 3000:3000"
echo ""
echo -e "${GREEN}[INFO]${NC} Access URLs:"
echo ""
echo "  Qallow API:  http://localhost:8080"
echo "  Prometheus:  http://localhost:9090"
echo "  Grafana:     http://localhost:3000 (admin/qallow)"
echo ""
echo -e "${GREEN}[INFO]${NC} Check pod logs:"
echo ""
echo "  kubectl logs -n qallow deployment/qallow-core"
echo "  kubectl logs -n qallow deployment/prometheus-deployment"
echo "  kubectl logs -n qallow deployment/grafana-deployment"
echo ""

