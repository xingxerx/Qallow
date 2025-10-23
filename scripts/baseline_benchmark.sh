#!/usr/bin/env bash
set -euo pipefail

printf '╔════════════════════════════════════════════════════════════╗\n'
printf '║       QALLOW BASELINE BENCHMARK - 4-QUBIT ENTANGLEMENT     ║\n'
printf '╚════════════════════════════════════════════════════════════╝\n'

LOG_PREFIX="baseline_$(date +%Y%m%d_%H%M%S)"
VM_LOG="${LOG_PREFIX}_vm_status.log"
THROUGHPUT_LOG="${LOG_PREFIX}_throughput.log"
SMOKE_LOG="${LOG_PREFIX}_smoke.log"

run_and_log() {
  local cmd="$1"
  local outfile="$2"
  printf '[BASELINE] %s\n' "$cmd"
  eval "$cmd" | tee "$outfile"
}

run_and_log "./build/CPU/qallow_unified_cpu run vm --dashboard=1" "$VM_LOG"
run_and_log "./build/CPU/qallow_throughput_bench --qubits=4 --iterations=1000" "$THROUGHPUT_LOG"
run_and_log "./build/CPU/qallow_integration_smoke --pockets=4 --ticks=100" "$SMOKE_LOG"

THROUGHPUT=$(grep -m1 'states/sec' "$THROUGHPUT_LOG" | awk '{print $3}')
COHERENCE=$(grep -m1 'Coherence' "$SMOKE_LOG" | awk '{print $3}')
DECOHERENCE=$(grep -m1 'decoherence' "$SMOKE_LOG" | awk '{print $3}')

printf '\nSummary\n'
printf '  Throughput:   %s states/sec\n' "${THROUGHPUT:-n/a}"
printf '  Coherence:    %s\n' "${COHERENCE:-n/a}"
printf '  Decoherence:  %s\n\n' "${DECOHERENCE:-n/a}"

ARCHIVE="${LOG_PREFIX}_bundle.tar.gz"
tar -czf "$ARCHIVE" "$VM_LOG" "$THROUGHPUT_LOG" "$SMOKE_LOG"
printf 'Logs archived to %s\n' "$ARCHIVE"
