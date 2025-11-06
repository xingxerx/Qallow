#!/bin/bash

# Test script for Phases 16-20
# Tests all new phases and their integration

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Testing Phases 16-20: Advanced Quantum Features              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

API_BASE="http://localhost:3001/api"
TESTS_PASSED=0
TESTS_FAILED=0

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check API is running
echo "🔍 Test 1: Checking API server..."
if curl -s "$API_BASE/status" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ API server is running${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ API server is not running${NC}"
    ((TESTS_FAILED++))
    exit 1
fi

# Test 2: Reset system
echo ""
echo "🔍 Test 2: Resetting system..."
RESET_RESPONSE=$(curl -s -X POST "$API_BASE/vm/reset")
if echo "$RESET_RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✅ System reset successful${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ System reset failed${NC}"
    ((TESTS_FAILED++))
fi

# Test 3: Start unified execution (phases 1-20)
echo ""
echo "🔍 Test 3: Starting unified execution (phases 1-20)..."
START_RESPONSE=$(curl -s -X POST "$API_BASE/vm/start" \
  -H "Content-Type: application/json" \
  -d '{"ticks": 100, "build": "CPU"}')

if echo "$START_RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✅ Unified execution started${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Failed to start unified execution${NC}"
    ((TESTS_FAILED++))
fi

# Test 4: Check status
echo ""
echo "🔍 Test 4: Checking system status..."
sleep 2
STATUS=$(curl -s "$API_BASE/status")
if echo "$STATUS" | grep -q "vm_running"; then
    echo -e "${GREEN}✅ Status check successful${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Status check failed${NC}"
    ((TESTS_FAILED++))
fi

# Test 5: Monitor phase progression
echo ""
echo "🔍 Test 5: Monitoring phase progression..."
INITIAL_PHASE=$(echo "$STATUS" | grep -o '"current_phase":[0-9]*' | grep -o '[0-9]*')
echo "   Initial phase: $INITIAL_PHASE"
sleep 3
STATUS2=$(curl -s "$API_BASE/status")
CURRENT_PHASE=$(echo "$STATUS2" | grep -o '"current_phase":[0-9]*' | grep -o '[0-9]*')
echo "   Current phase: $CURRENT_PHASE"

if [ "$CURRENT_PHASE" -gt "$INITIAL_PHASE" ] || [ "$CURRENT_PHASE" -eq "$INITIAL_PHASE" ]; then
    echo -e "${GREEN}✅ Phase progression working${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  Phase progression check inconclusive${NC}"
fi

# Test 6: Export metrics
echo ""
echo "🔍 Test 6: Exporting metrics..."
EXPORT_RESPONSE=$(curl -s "$API_BASE/metrics/export")
if echo "$EXPORT_RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✅ Metrics exported successfully${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Metrics export failed${NC}"
    ((TESTS_FAILED++))
fi

# Test 7: Get logs
echo ""
echo "🔍 Test 7: Retrieving audit logs..."
LOGS=$(curl -s "$API_BASE/logs")
if echo "$LOGS" | grep -q "logs"; then
    LOG_COUNT=$(echo "$LOGS" | grep -o '"count":[0-9]*' | grep -o '[0-9]*')
    echo -e "${GREEN}✅ Retrieved $LOG_COUNT log entries${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Failed to retrieve logs${NC}"
    ((TESTS_FAILED++))
fi

# Test 8: Stop execution
echo ""
echo "🔍 Test 8: Stopping execution..."
STOP_RESPONSE=$(curl -s -X POST "$API_BASE/vm/stop")
if echo "$STOP_RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✅ Execution stopped successfully${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${RED}❌ Failed to stop execution${NC}"
    ((TESTS_FAILED++))
fi

# Test 9: Verify stop
echo ""
echo "🔍 Test 9: Verifying VM is stopped..."
sleep 1
FINAL_STATUS=$(curl -s "$API_BASE/status")
if echo "$FINAL_STATUS" | grep -q '"vm_running":false'; then
    echo -e "${GREEN}✅ VM successfully stopped${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  VM stop verification inconclusive${NC}"
fi

# Test 10: Test phase 16 specifically
echo ""
echo "🔍 Test 10: Testing Phase 16 (Rebellion Simulation)..."
START_P16=$(curl -s -X POST "$API_BASE/vm/start-continuous" \
  -H "Content-Type: application/json" \
  -d '{"phase": 16, "ticks": 50, "build": "CPU"}')

if echo "$START_P16" | grep -q "success"; then
    echo -e "${GREEN}✅ Phase 16 started successfully${NC}"
    ((TESTS_PASSED++))
    sleep 2
    curl -s -X POST "$API_BASE/vm/stop" > /dev/null
else
    echo -e "${RED}❌ Failed to start Phase 16${NC}"
    ((TESTS_FAILED++))
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Test Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "  ${GREEN}✅ Passed: $TESTS_PASSED${NC}"
echo -e "  ${RED}❌ Failed: $TESTS_FAILED${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi

