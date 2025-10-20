#!/usr/bin/env bash
set -euo pipefail

QALLOW_BIN="./build/qallow_unified"
LOG_DIR="logs/phase16"

mkdir -p "$LOG_DIR"

"$QALLOW_BIN" run \
  --integrate phase14 phase15 \
  --no-split \
  --self-audit \
  --self-audit-path "$LOG_DIR"
