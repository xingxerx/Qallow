#!/bin/bash

# Test script for Qallow Web App Buttons
# Tests all API endpoints that the buttons use

API_BASE="http://localhost:3001/api"
RESULTS_FILE="/tmp/button_test_results.txt"

echo "🧪 QALLOW WEB APP BUTTON TEST SUITE" > $RESULTS_FILE
echo "===================================" >> $RESULTS_FILE
echo "Test Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_count=0
pass_count=0
fail_count=0

# Helper function to test endpoint
test_endpoint() {
  local name=$1
  local method=$2
  local endpoint=$3
  local data=$4
  
  test_count=$((test_count + 1))
  echo -n "Test $test_count: $name ... "
  
  if [ "$method" = "GET" ]; then
    response=$(curl -s "$API_BASE$endpoint")
  else
    response=$(curl -s -X $method "$API_BASE$endpoint" -H "Content-Type: application/json" -d "$data")
  fi
  
  if echo "$response" | grep -q "success\|vm_running\|metrics\|logs\|filename"; then
    echo -e "${GREEN}✓ PASS${NC}"
    echo "✓ Test $test_count: $name - PASS" >> $RESULTS_FILE
    pass_count=$((pass_count + 1))
  else
    echo -e "${RED}✗ FAIL${NC}"
    echo "✗ Test $test_count: $name - FAIL" >> $RESULTS_FILE
    echo "  Response: $response" >> $RESULTS_FILE
    fail_count=$((fail_count + 1))
  fi
}

echo ""
echo "🔍 Testing API Endpoints..."
echo ""

# Test 1: Status endpoint
test_endpoint "GET /api/status" "GET" "/status" ""

# Test 2: Reset endpoint
test_endpoint "POST /api/vm/reset" "POST" "/vm/reset" "{}"

# Test 3: Config save endpoint
test_endpoint "POST /api/config/save" "POST" "/config/save" '{"ticks": 1000, "build": "CPU", "phase": "13"}'

# Test 4: Metrics export endpoint
test_endpoint "GET /api/metrics/export" "GET" "/metrics/export" ""

# Test 5: Logs endpoint
test_endpoint "GET /api/logs" "GET" "/logs" ""

# Test 6: VM start with parameters
test_endpoint "POST /api/vm/start (Phase 13)" "POST" "/vm/start" '{"ticks": 50, "build": "CPU", "phase": "13"}'

# Wait for VM to complete
echo ""
echo "⏳ Waiting for VM to complete..."
sleep 3

# Test 7: Check status after VM run
test_endpoint "GET /api/status (after VM)" "GET" "/status" ""

# Test 8: Export metrics after run
test_endpoint "GET /api/metrics/export (after run)" "GET" "/metrics/export" ""

# Test 9: VM reset
test_endpoint "POST /api/vm/reset (cleanup)" "POST" "/vm/reset" "{}"

# Test 10: VM start Phase 14
test_endpoint "POST /api/vm/start (Phase 14)" "POST" "/vm/start" '{"ticks": 50, "build": "CPU", "phase": "14"}'

# Wait for VM to complete
sleep 3

# Test 11: VM start Phase 15
test_endpoint "POST /api/vm/start (Phase 15)" "POST" "/vm/start" '{"ticks": 50, "build": "CPU", "phase": "15"}'

# Wait for VM to complete
sleep 3

# Test 12: Final status check
test_endpoint "GET /api/status (final)" "GET" "/status" ""

echo ""
echo "📊 TEST RESULTS"
echo "==============="
echo -e "Total Tests: $test_count"
echo -e "${GREEN}Passed: $pass_count${NC}"
echo -e "${RED}Failed: $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
  echo -e "${GREEN}✅ ALL TESTS PASSED!${NC}"
  echo "✅ ALL TESTS PASSED!" >> $RESULTS_FILE
else
  echo -e "${RED}❌ SOME TESTS FAILED${NC}"
  echo "❌ SOME TESTS FAILED" >> $RESULTS_FILE
fi

echo ""
echo "📝 Full results saved to: $RESULTS_FILE"
cat $RESULTS_FILE

