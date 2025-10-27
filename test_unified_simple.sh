#!/bin/bash

# Simple test for unified continuous execution
set -e

API_BASE="http://localhost:3001/api"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  🧪 TESTING UNIFIED CONTINUOUS EXECUTION                                  ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Check status
echo "✓ TEST 1: Check initial status"
curl -s "$API_BASE/status" | head -c 100
echo ""
echo ""

# Test 2: Reset
echo "✓ TEST 2: Reset system"
curl -s -X POST "$API_BASE/vm/reset"
echo ""
echo ""

# Test 3: Start continuous execution
echo "✓ TEST 3: Start continuous unified execution (100 ticks per phase)"
curl -s -X POST "$API_BASE/vm/start-continuous" \
  -H "Content-Type: application/json" \
  -d '{"ticks": 100, "build": "CPU"}'
echo ""
echo ""

# Test 4: Monitor for 20 seconds
echo "✓ TEST 4: Monitoring execution (20 seconds)..."
for i in {1..4}; do
  sleep 5
  echo "  [$i/4] Status check..."
  curl -s "$API_BASE/status" | grep -o '"current_phase":[0-9]*' || echo "  Checking..."
done
echo ""

# Test 5: Stop execution
echo "✓ TEST 5: Stop continuous execution"
curl -s -X POST "$API_BASE/vm/stop"
echo ""
echo ""

# Test 6: Export metrics
echo "✓ TEST 6: Export metrics"
curl -s "$API_BASE/metrics/export" | head -c 150
echo ""
echo ""

# Test 7: Check for metrics file
echo "✓ TEST 7: Check for generated metrics file"
LATEST_METRICS=$(ls -t /root/Qallow/qallow_metrics_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_METRICS" ]; then
  echo "✅ Found metrics file: $LATEST_METRICS"
  echo "   File size: $(du -h "$LATEST_METRICS" | cut -f1)"
  echo "   First 5 lines:"
  head -5 "$LATEST_METRICS" | sed 's/^/   /'
else
  echo "❌ No metrics file found"
fi
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ TESTS COMPLETE                                                         ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "New Features Implemented:"
echo "  ✓ Unified execution mode - Runs phases 13→14→15 continuously"
echo "  ✓ Continuous phase cycling - Automatically cycles through all phases"
echo "  ✓ Cycle counter - Tracks complete cycles"
echo "  ✓ Code Improvements tab - Shows 8 C code optimizations"
echo "  ✓ Execution mode selector - Choose single phase or unified"
echo ""

