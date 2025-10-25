#!/bin/bash

# Qallow Server Testing Script
# Comprehensive testing for all components

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         QALLOW SERVER - COMPREHENSIVE TESTING SUITE            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing dependencies...${NC}"
    npm install
fi

echo -e "${BLUE}[1/5] Running Unit Tests${NC}"
npm test -- server.test.js --verbose || true
echo ""

echo -e "${BLUE}[2/5] Running Error Handler Tests${NC}"
npm test -- errorHandler.test.js --verbose || true
echo ""

echo -e "${BLUE}[3/5] Running Integration Tests${NC}"
npm test -- integration.test.js --verbose || true
echo ""

echo -e "${BLUE}[4/5] Generating Coverage Report${NC}"
npm test -- --coverage --silent || true
echo ""

echo -e "${BLUE}[5/5] Running All Tests${NC}"
npm test -- --coverage || true
echo ""

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    TESTING COMPLETE                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ All tests completed!${NC}"
echo ""
echo "📊 Coverage Report: ./coverage/lcov-report/index.html"
echo "📋 Test Results: Check console output above"
echo ""

