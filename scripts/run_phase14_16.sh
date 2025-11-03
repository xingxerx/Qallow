#!/usr/bin/env bash
set -euo pipefail

QALLOW_BIN="./build/qallow_unified"
LOG_DIR="logs/phase16"

mkdir -p "$LOG_DIR"

QUANTUM_BIN=${QALLOW_QUANTUM_BIN:-qallow_quantum}
QUANTUM_OUT_DIR="data/quantum"
PHASE14_METRICS="$QUANTUM_OUT_DIR/phase14_metrics.json"
PHASE15_METRICS="$QUANTUM_OUT_DIR/phase15_metrics.json"
PIPELINE_SUMMARY="$QUANTUM_OUT_DIR/pipeline_summary.json"

if command -v "$QUANTUM_BIN" >/dev/null 2>&1; then
  mkdir -p "$QUANTUM_OUT_DIR"
  echo "[RUN] Priming quantum optimizer pipeline via $QUANTUM_BIN"
  if "$QUANTUM_BIN" pipeline \
    --phase14-ticks=600 \
    --nodes=256 \
    --target-fidelity=0.981 \
    --phase15-ticks=800 \
    --phase15-eps=0.000005 \
    --export-phase14 "$PHASE14_METRICS" \
    --export-phase15 "$PHASE15_METRICS" \
    --export-pipeline "$PIPELINE_SUMMARY"; then
    export QALLOW_PHASE14_METRICS="$PHASE14_METRICS"
    export QALLOW_PHASE15_METRICS="$PHASE15_METRICS"
    echo "[RUN] Phase14 metrics: $PHASE14_METRICS"
    echo "[RUN] Phase15 metrics: $PHASE15_METRICS"
  else
    echo "[RUN] Warning: quantum pipeline seed failed (continuing with defaults)" >&2
  fi
else
  echo "[RUN] qallow_quantum binary not found; skipping quantum priming" >&2
fi

"$QALLOW_BIN" run \
  --integrate phase14 phase15 \
  --no-split \
  --self-audit \
  --self-audit-path "$LOG_DIR"
