#!/bin/bash
# demo_continuous.sh - Demonstrate continuous closed-loop monitoring

echo "================================================"
echo "Qallow Closed-Loop Ethics - Continuous Demo"
echo "================================================"
echo ""
echo "This demo will:"
echo "  1. Start continuous signal collection (5s interval)"
echo "  2. Run 3 ethics evaluations with different feedback"
echo "  3. Show real-time adaptation"
echo ""
echo "Press Ctrl+C to stop"
echo ""
sleep 2

# Start collector in background
echo "[*] Starting signal collector..."
python3 /root/Qallow/python/collect_signals.py --loop > /tmp/collector.log 2>&1 &
COLLECTOR_PID=$!
echo "    Collector running (PID: $COLLECTOR_PID)"
sleep 2

# Test 1: Excellent feedback
echo ""
echo "=== TEST 1: Excellent Human Feedback (0.95) ==="
echo "0.95" > /root/Qallow/data/human_feedback.txt
sleep 6  # Wait for fresh collection
cd /root/Qallow/algorithms && ./ethics_test_feed
sleep 3

# Test 2: Marginal feedback
echo ""
echo "=== TEST 2: Marginal Human Feedback (0.65) ==="
echo "0.65" > /root/Qallow/data/human_feedback.txt
sleep 6  # Wait for fresh collection
cd /root/Qallow/algorithms && ./ethics_test_feed
sleep 3

# Test 3: Poor feedback
echo ""
echo "=== TEST 3: Poor Human Feedback (0.40) ==="
echo "0.40" > /root/Qallow/data/human_feedback.txt
sleep 6  # Wait for fresh collection
cd /root/Qallow/algorithms && ./ethics_test_feed
sleep 3

# Test 4: Recovered feedback
echo ""
echo "=== TEST 4: Recovered Human Feedback (0.85) ==="
echo "0.85" > /root/Qallow/data/human_feedback.txt
sleep 6  # Wait for fresh collection
cd /root/Qallow/algorithms && ./ethics_test_feed

# Stop collector
echo ""
echo "[*] Stopping collector..."
kill $COLLECTOR_PID 2>/dev/null

echo ""
echo "================================================"
echo "Demo Complete"
echo "================================================"
echo ""
echo "Signal history:"
tail -n 10 /root/Qallow/data/telemetry/collection.log
echo ""
echo "Latest signals:"
cat /root/Qallow/data/telemetry/current_signals.txt
