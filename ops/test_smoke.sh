#!/usr/bin/env bash
set -euo pipefail

NAMESPACE=${NAMESPACE:-default}

kubectl get deployment -n "$NAMESPACE" qallow-core
kubectl get svc -n "$NAMESPACE" qallow-service

echo "[INFO] Sampling /metrics endpoint"
kubectl run qallow-metrics-check \
  --rm -i --restart=Never -n "$NAMESPACE" \
  --image=curlimages/curl:7.88.1 \
  -- curl -sf qallow-service:8080/metrics | head -n 20

echo "[INFO] Service endpoints verified"
