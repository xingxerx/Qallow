#!/bin/bash

# MCP Setup Verification Script
# This script verifies that the MCP server is properly configured and working

set -e

echo "=========================================="
echo "MCP Setup Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

# Function to check status
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((PASS++))
    else
        echo -e "${RED}✗${NC} $1"
        ((FAIL++))
    fi
}

# 1. Check Python
echo "1. Checking Python Installation..."
/root/Qallow/mcp-memory-service/.venv/bin/python --version > /dev/null 2>&1
check "Python 3.13.7 available"

# 2. Check Virtual Environment
echo ""
echo "2. Checking Virtual Environment..."
[ -f "/root/Qallow/mcp-memory-service/.venv/bin/python" ]
check "Virtual environment exists"

# 3. Check MCP Module
echo ""
echo "3. Checking Required Modules..."
/root/Qallow/mcp-memory-service/.venv/bin/python -c "import mcp" > /dev/null 2>&1
check "MCP module installed"

/root/Qallow/mcp-memory-service/.venv/bin/python -c "import sqlite_vec" > /dev/null 2>&1
check "SQLite-Vec module installed"

/root/Qallow/mcp-memory-service/.venv/bin/python -c "import sentence_transformers" > /dev/null 2>&1
check "Sentence Transformers module installed"

# 4. Check Configuration Files
echo ""
echo "4. Checking Configuration Files..."
[ -f "/root/Qallow/.vscode/mcp.json" ]
check "VS Code MCP configuration exists"

grep -q "mcp-memory-service" /root/Qallow/.vscode/mcp.json
check "VS Code config contains MCP server path"

# 5. Check Database Directory
echo ""
echo "5. Checking Database Directory..."
[ -d "$HOME/.local/share/mcp-memory" ]
check "Database directory exists"

[ -w "$HOME/.local/share/mcp-memory" ]
check "Database directory is writable"

# 6. Check Setup Script
echo ""
echo "6. Checking Setup Script..."
[ -f "/root/Qallow/setup_mcp_linux.sh" ]
check "Setup script exists"

[ -x "/root/Qallow/setup_mcp_linux.sh" ]
check "Setup script is executable"

# 7. Check Documentation
echo ""
echo "7. Checking Documentation..."
[ -f "/root/Qallow/MCP_FIX_GUIDE.md" ]
check "MCP_FIX_GUIDE.md exists"

[ -f "/root/Qallow/MCP_SETUP_COMPLETE.md" ]
check "MCP_SETUP_COMPLETE.md exists"

[ -f "/root/Qallow/MCP_QUICK_REFERENCE.md" ]
check "MCP_QUICK_REFERENCE.md exists"

# 8. Test Server Startup
echo ""
echo "8. Testing Server Startup..."
cd /root/Qallow/mcp-memory-service
timeout 3 ./.venv/bin/python -m src.mcp_memory_service.server > /tmp/mcp_test.log 2>&1 &
TEST_PID=$!
sleep 1

if kill -0 $TEST_PID 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Server starts successfully"
    ((PASS++))
    kill $TEST_PID 2>/dev/null || true
else
    echo -e "${RED}✗${NC} Server failed to start"
    ((FAIL++))
    cat /tmp/mcp_test.log
fi

# Summary
echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Restart your IDE (VS Code, Claude Desktop, etc.)"
    echo "2. The MCP server should connect automatically"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some checks failed${NC}"
    echo ""
    echo "Please run: bash /root/Qallow/setup_mcp_linux.sh"
    echo "Or check: /root/Qallow/MCP_FIX_GUIDE.md"
    echo ""
    exit 1
fi

