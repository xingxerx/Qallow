#!/bin/bash
# test_closed_loop.sh - Verify the closed-loop ethics system
# Part of Qallow Phase 13

set -e

echo "=========================================="
echo "Qallow Closed-Loop Ethics System Test"
echo "=========================================="
echo ""

# 1. Create necessary directories
echo "[1/6] Creating data directories..."
mkdir -p /root/Qallow/data/telemetry
mkdir -p /root/Qallow/data/snapshots

# 2. Set initial human feedback
echo "[2/6] Setting neutral human feedback..."
echo "0.75" > /root/Qallow/data/human_feedback.txt

# 3. Collect signals
echo "[3/6] Collecting hardware signals..."
python3 /root/Qallow/python/collect_signals.py

# 4. Verify signal file
echo "[4/6] Verifying signal file..."
if [ -f "/root/Qallow/data/telemetry/current_signals.txt" ]; then
    echo "✓ Signal file created:"
    cat /root/Qallow/data/telemetry/current_signals.txt
else
    echo "✗ Signal file missing!"
    exit 1
fi

# 5. Build ethics system with feed module
echo "[5/6] Building ethics system with feed integration..."
cd /root/Qallow/algorithms
make clean
make

# 6. Run ethics test with real data
echo "[6/6] Running ethics computation with hardware signals..."
./ethics_test

echo ""
echo "=========================================="
echo "Test complete! Check output above."
echo "=========================================="
echo ""
echo "To run continuously:"
echo "  python3 /root/Qallow/python/collect_signals.py --loop &"
echo ""
echo "To adjust human feedback:"
echo "  echo '0.9' > /root/Qallow/data/human_feedback.txt"
echo ""
