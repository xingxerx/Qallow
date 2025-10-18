#!/bin/bash
# Unified runner for Qallow-Bend
set -e

MODE=${1:-auto}
LOG=${2:-qallow_full.csv}
TICKS=${3:-1000}
EPS=${4:-0.0001}
K=${5:-0.001}

echo "[RUN] Launching Qallow-Bend mode=$MODE"
bend run-cu bend/qallow.bend --mode=$MODE --ticks=$TICKS --eps=$EPS --k=$K --log=$LOG
