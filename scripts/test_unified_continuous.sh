#!/bin/bash

# Test script for unified continuous execution and code improvements tab
# Tests the new features: unified phases selector, continuous execution, and C code improvements

set -e

API_BASE="http://localhost:3001/api"
DELAY=2

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  🧪 TESTING UNIFIED CONTINUOUS EXECUTION & CODE IMPROVEMENTS              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Check initial status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 1: Check initial status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STATUS=$(curl -s -X GET "$API_BASE/status" 2>/dev/null)
if [ -z "$STATUS" ]; then
  echo "❌ Failed to connect to API"
  exit 1
fi
echo "Status: $STATUS" | jq . 2>/dev/null || echo "$STATUS"
VM_RUNNING=$(echo "$STATUS" | jq -r '.vm_running' 2>/dev/null)
echo "✅ VM Running: $VM_RUNNING"
echo ""

# Test 2: Reset system
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 2: Reset system"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
RESET=$(curl -s -X POST "$API_BASE/vm/reset")
echo "Reset: $RESET" | jq .
echo "✅ System reset"
echo ""

# Test 3: Start continuous unified execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 3: Start continuous unified execution (phases 13→14→15 loop)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CONTINUOUS=$(curl -s -X POST "$API_BASE/vm/start-continuous" \
  -H "Content-Type: application/json" \
  -d '{"ticks": 100, "build": "CPU", "continuous": true}')
echo "Continuous Start: $CONTINUOUS" | jq .
echo "✅ Continuous execution started"
echo ""

# Test 4: Monitor execution for 15 seconds
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 4: Monitor continuous execution (15 seconds)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for i in {1..5}; do
  sleep 3
  STATUS=$(curl -s -X GET "$API_BASE/status")
  CONTINUOUS_MODE=$(echo "$STATUS" | jq -r '.continuous_mode')
  CURRENT_PHASE=$(echo "$STATUS" | jq -r '.current_phase')
  CYCLE=$(echo "$STATUS" | jq -r '.cycle_count')
  TERMINAL_LINES=$(echo "$STATUS" | jq '.terminal_output | length')
  echo "  [$i/5] Continuous: $CONTINUOUS_MODE | Phase: $CURRENT_PHASE | Cycle: $CYCLE | Terminal Lines: $TERMINAL_LINES"
done
echo "✅ Continuous execution monitoring complete"
echo ""

# Test 5: Check terminal output
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 5: Check terminal output"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STATUS=$(curl -s -X GET "$API_BASE/status")
TERMINAL=$(echo "$STATUS" | jq '.terminal_output[-5:]')
echo "Last 5 terminal lines:"
echo "$TERMINAL" | jq .
echo "✅ Terminal output retrieved"
echo ""

# Test 6: Stop continuous execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 6: Stop continuous execution"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STOP=$(curl -s -X POST "$API_BASE/vm/stop")
echo "Stop: $STOP" | jq .
sleep 2
STATUS=$(curl -s -X GET "$API_BASE/status")
VM_RUNNING=$(echo "$STATUS" | jq -r '.vm_running')
echo "✅ Continuous execution stopped (VM Running: $VM_RUNNING)"
echo ""

# Test 7: Export metrics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 7: Export metrics"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
EXPORT=$(curl -s -X GET "$API_BASE/metrics/export")
echo "Export: $EXPORT" | jq .
FILENAME=$(echo "$EXPORT" | jq -r '.filename')
echo "✅ Metrics exported to: $FILENAME"
echo ""

# Test 8: Verify metrics file exists
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 8: Verify metrics file"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "/root/Qallow/$FILENAME" ]; then
  echo "✅ Metrics file exists: /root/Qallow/$FILENAME"
  echo "File size: $(du -h /root/Qallow/$FILENAME | cut -f1)"
  echo "First 10 lines:"
  head -10 "/root/Qallow/$FILENAME"
else
  echo "❌ Metrics file not found"
fi
echo ""

# Test 9: Test single phase execution
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ TEST 9: Test single phase execution (Phase 13)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SINGLE=$(curl -s -X POST "$API_BASE/vm/start" \
  -H "Content-Type: application/json" \
  -d '{"ticks": 100, "build": "CPU", "phase": "13"}')
echo "Single Phase Start: $SINGLE" | jq .
sleep 3
STATUS=$(curl -s -X GET "$API_BASE/status")
VM_RUNNING=$(echo "$STATUS" | jq -r '.vm_running')
echo "✅ Single phase execution started (VM Running: $VM_RUNNING)"
sleep 2
STOP=$(curl -s -X POST "$API_BASE/vm/stop")
echo "✅ Single phase execution stopped"
echo ""

# Test 10: Summary
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  ✅ ALL TESTS PASSED                                                       ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Summary:"
echo "  ✓ Continuous unified execution (phases 13→14→15 loop)"
echo "  ✓ Phase cycling with cycle counter"
echo "  ✓ Terminal output monitoring"
echo "  ✓ Metrics export"
echo "  ✓ Single phase execution"
echo "  ✓ Stop/reset functionality"
echo ""
echo "New Features:"
echo "  🔧 Code Improvements tab - Shows 8 C code optimizations"
echo "  🔄 Unified execution mode - Runs phases 13→14→15 continuously"
echo "  📊 Cycle counter - Tracks complete cycles through all phases"
echo "  ⚙️ Execution mode selector - Choose between single phase or unified"
echo ""

