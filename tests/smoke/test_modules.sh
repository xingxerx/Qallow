#!/usr/bin/env bash
# Qallow smoke-tests for ethics, governance, and phase runners.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_LOG="${ROOT_DIR}/build/test_modules.log"
BIN_CPU="${ROOT_DIR}/build/CPU/qallow_unified_cpu"

log() {
    printf '[test] %s\n' "$*" | tee -a "$BUILD_LOG"
}

run_and_assert() {
    local label=$1
    local expected=$2
    shift 2
    log "Running ${label}: $*"
    if ! output=$("$@" 2>&1); then
        log "FAIL ${label}: command exited with non-zero status"
        printf '%s\n' "$output" | tee -a "$BUILD_LOG"
        return 1
    fi
    printf '%s\n' "$output" >>"$BUILD_LOG"
    if [[ -n "$expected" ]] && ! grep -Fq "$expected" <<<"$output"; then
        log "FAIL ${label}: did not find expected marker '${expected}'"
        return 1
    fi
    log "PASS ${label}"
}

mkdir -p "$(dirname "$BUILD_LOG")"
rm -f "$BUILD_LOG"
touch "$BUILD_LOG"

log "Building CPU binary..."
make -C "$ROOT_DIR" ACCELERATOR=CPU >/dev/null

if [[ ! -x "$BIN_CPU" ]]; then
    log "FAIL binary missing at $BIN_CPU"
    exit 1
fi

run_and_assert "phase12 elasticity" "[PHASE12] Elastic run complete" \
    "$BIN_CPU" run --phase=12 --ticks=8

run_and_assert "phase13 harmonic" "[PHASE13] Harmonic propagation complete" \
    "$BIN_CPU" run --phase=13 --ticks=8 --nodes=4

run_and_assert "governance audit" "[GOVERN] Autonomous governance loop completed successfully" \
    "$BIN_CPU" govern --ticks=4

log "All smoke tests completed"
