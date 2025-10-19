#!/usr/bin/env bash
set -e
LOG_DIR=/var/log/qallow
OUT_DIR=/var/qallow/benchmarks
mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "═══════════════════════════════════════════"
echo "   Qallow Unified Diagnostic Suite (v13)"
echo "═══════════════════════════════════════════"

echo "[1/4] System verification..."
./build/qallow_unified_cuda verify | tee "$OUT_DIR/system_verify.log"

echo "[2/4] Benchmark run..."
# quiet mode: log full output, show only last 20 lines
./build/qallow_unified_cuda run --bench > "$OUT_DIR/benchmark.log" 2>&1 &
pid=$!
wait $pid
tail -n 20 "$OUT_DIR/benchmark.log"

echo "[3/4] Governance integrity check..."
./build/qallow_unified_cuda govern | tee "$OUT_DIR/governance.log"

echo "[4/4] Accelerator load test..."
tmux new -d -s qallow_test "CUDA_VISIBLE_DEVICES=0 ./build/qallow_unified_cuda run --accelerator --threads=auto > $OUT_DIR/accelerator.log 2>&1"
sleep 10
tmux kill-session -t qallow_test || true

echo "═══════════════════════════════════════════"
echo "   Test complete. Results saved to:"
echo "   $OUT_DIR"
echo "═══════════════════════════════════════════"

