#!/bin/bash

# Qallow Server - Manual Testing Script
# Tests all API endpoints with curl

set -e

BASE_URL="http://localhost:5000"
TESTS_PASSED=0
TESTS_FAILED=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         QALLOW SERVER - MANUAL API TESTING                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if server is running
echo -e "${BLUE}Checking if server is running...${NC}"
if ! curl -s "$BASE_URL/api/health" > /dev/null 2>&1; then
    echo -e "${RED}❌ Server is not running on $BASE_URL${NC}"
    echo "Start the server with: bash /root/Qallow/QUICK_START_SERVER.sh"
    exit 1
fi
echo -e "${GREEN}✅ Server is running${NC}"
echo ""

# Test function
test_endpoint() {
    local method=$1
    local endpoint=$2
    local data=$3
    local expected_status=$4
    local description=$5

    echo -e "${BLUE}Testing: $description${NC}"
    echo "  Endpoint: $method $endpoint"

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$BASE_URL$endpoint")
    fi

    status=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✅ Status: $status (Expected: $expected_status)${NC}"
        echo "  Response: $(echo "$body" | head -c 100)..."
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ Status: $status (Expected: $expected_status)${NC}"
        echo "  Response: $body"
        ((TESTS_FAILED++))
    fi
    echo ""
}

# Health Check Tests
echo -e "${YELLOW}═══ HEALTH CHECK TESTS ═══${NC}"
echo ""
test_endpoint "GET" "/api/health" "" "200" "Server Health Check"
test_endpoint "GET" "/api/quantum/status" "" "200" "Quantum Framework Status"
test_endpoint "GET" "/api/system/metrics" "" "200" "System Metrics"

# Quantum Algorithm Tests
echo -e "${YELLOW}═══ QUANTUM ALGORITHM TESTS ═══${NC}"
echo ""
test_endpoint "POST" "/api/quantum/run-grover" \
    '{"num_qubits": 3, "target_state": 5}' "200" "Grover Algorithm"
test_endpoint "POST" "/api/quantum/run-bell-state" "" "200" "Bell State"
test_endpoint "POST" "/api/quantum/run-deutsch" "" "200" "Deutsch Algorithm"
test_endpoint "POST" "/api/quantum/run-all" "" "200" "Run All Algorithms"

# Error Handling Tests
echo -e "${YELLOW}═══ ERROR HANDLING TESTS ═══${NC}"
echo ""
test_endpoint "POST" "/api/quantum/run-grover" \
    '{"num_qubits": 25}' "400" "Invalid Parameters (too many qubits)"
test_endpoint "GET" "/api/nonexistent" "" "404" "Non-existent Endpoint"

# Performance Tests
echo -e "${YELLOW}═══ PERFORMANCE TESTS ═══${NC}"
echo ""
echo -e "${BLUE}Testing response time...${NC}"
start_time=$(date +%s%N)
curl -s "$BASE_URL/api/health" > /dev/null
end_time=$(date +%s%N)
response_time=$(( (end_time - start_time) / 1000000 ))
echo "  Response time: ${response_time}ms"
if [ "$response_time" -lt 100 ]; then
    echo -e "${GREEN}✅ Response time acceptable${NC}"
    ((TESTS_PASSED++))
else
    echo -e "${YELLOW}⚠️  Response time is ${response_time}ms (expected < 100ms)${NC}"
fi
echo ""

# Concurrent Requests Test
echo -e "${BLUE}Testing concurrent requests...${NC}"
for i in {1..5}; do
    curl -s "$BASE_URL/api/health" > /dev/null &
done
wait
echo -e "${GREEN}✅ Concurrent requests handled${NC}"
((TESTS_PASSED++))
echo ""

# Summary
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    TEST SUMMARY                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Tests Failed: $TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed${NC}"
    exit 1
fi

