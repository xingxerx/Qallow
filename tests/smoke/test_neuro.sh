#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BINARY="${ROOT_DIR}/build/qallow"

if [ ! -x "$BINARY" ]; then
    echo "[TEST] qallow binary not found at $BINARY" >&2
    exit 1
fi

echo "[TEST] Running neuromorphic spike demo"
"$BINARY" phase neuro-demo --nodes=64 --target=0.95
