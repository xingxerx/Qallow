#!/usr/bin/env bash
set -euo pipefail

PROM_NAMESPACE=${PROM_NAMESPACE:-default}
GRAFANA_NAMESPACE=${GRAFANA_NAMESPACE:-default}

kubectl port-forward -n "$PROM_NAMESPACE" svc/prometheus-service 9090:9090 &
PROM_PID=$!

kubectl port-forward -n "$GRAFANA_NAMESPACE" svc/grafana-service 3000:3000 &
GRAF_PID=$!

cleanup() {
  kill "$PROM_PID" "$GRAF_PID" >/dev/null 2>&1 || true
}

trap cleanup INT TERM EXIT

wait "$PROM_PID" "$GRAF_PID"
