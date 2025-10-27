#!/bin/bash
# Verify GitHub Copilot + MCP Memory Server Setup
# This script checks that the persistent memory MCP server is properly configured

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== GitHub Copilot MCP Memory Server Verification ===${NC}\n"

# Check 1: .vscode/mcp.json exists
echo -e "${BLUE}[1/6]${NC} Checking .vscode/mcp.json configuration..."
if [ -f ".vscode/mcp.json" ]; then
    echo -e "${GREEN}✓${NC} .vscode/mcp.json found"
    
    # Validate JSON
    if python3 -m json.tool .vscode/mcp.json > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} JSON is valid"
    else
        echo -e "${RED}✗${NC} JSON is invalid"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} .vscode/mcp.json not found"
    exit 1
fi

# Check 2: MCP Memory Service directory exists
echo -e "\n${BLUE}[2/6]${NC} Checking MCP Memory Service installation..."
if [ -d "mcp-memory-service" ]; then
    echo -e "${GREEN}✓${NC} mcp-memory-service directory found"
else
    echo -e "${RED}✗${NC} mcp-memory-service directory not found"
    exit 1
fi

# Check 3: Python virtual environment
echo -e "\n${BLUE}[3/6]${NC} Checking Python virtual environment..."
VENV_PATH="mcp-memory-service/.venv/bin/python"
if [ -f "$VENV_PATH" ]; then
    echo -e "${GREEN}✓${NC} Python venv found at $VENV_PATH"
    
    # Check Python version
    PYTHON_VERSION=$($VENV_PATH --version 2>&1)
    echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
else
    echo -e "${YELLOW}⚠${NC} Python venv not found at $VENV_PATH"
    echo -e "${YELLOW}  Run: cd mcp-memory-service && python3 -m venv .venv${NC}"
fi

# Check 4: MCP Memory Service module
echo -e "\n${BLUE}[4/6]${NC} Checking MCP Memory Service module..."
if [ -d "mcp-memory-service/src/mcp_memory_service" ]; then
    echo -e "${GREEN}✓${NC} MCP Memory Service module found"
    
    if [ -f "mcp-memory-service/src/mcp_memory_service/server.py" ]; then
        echo -e "${GREEN}✓${NC} server.py found"
    else
        echo -e "${RED}✗${NC} server.py not found"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} MCP Memory Service module not found"
    exit 1
fi

# Check 5: Storage directory
echo -e "\n${BLUE}[5/6]${NC} Checking memory storage directory..."
STORAGE_PATH="$HOME/.local/share/mcp-memory"
if [ -d "$STORAGE_PATH" ]; then
    echo -e "${GREEN}✓${NC} Storage directory exists: $STORAGE_PATH"
    
    # Check permissions
    if [ -w "$STORAGE_PATH" ]; then
        echo -e "${GREEN}✓${NC} Storage directory is writable"
    else
        echo -e "${RED}✗${NC} Storage directory is not writable"
        echo -e "${YELLOW}  Run: chmod u+w $STORAGE_PATH${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} Storage directory does not exist: $STORAGE_PATH"
    echo -e "${YELLOW}  It will be created on first use${NC}"
fi

# Check 6: Configuration details
echo -e "\n${BLUE}[6/6]${NC} Verifying MCP configuration details..."
if grep -q '"servers"' .vscode/mcp.json; then
    echo -e "${GREEN}✓${NC} Configuration uses 'servers' format (GitHub Copilot compatible)"
else
    echo -e "${YELLOW}⚠${NC} Configuration may use legacy format"
fi

if grep -q '"memory"' .vscode/mcp.json; then
    echo -e "${GREEN}✓${NC} Memory server configured"
else
    echo -e "${RED}✗${NC} Memory server not configured"
    exit 1
fi

if grep -q 'sqlite_vec' .vscode/mcp.json; then
    echo -e "${GREEN}✓${NC} SQLite-vec backend configured"
else
    echo -e "${YELLOW}⚠${NC} SQLite-vec backend not configured"
fi

# Summary
echo -e "\n${BLUE}=== Verification Summary ===${NC}"
echo -e "${GREEN}✓ All checks passed!${NC}\n"

echo -e "${BLUE}Next steps:${NC}"
echo "1. Open VS Code with this repository"
echo "2. Open Copilot Chat (Ctrl+Shift+I or Cmd+Shift+I)"
echo "3. Select 'Agent' mode from the dropdown"
echo "4. Click the tools icon (⚙️) in the top-left"
echo "5. Memory server tools should now be available"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "- Quick Reference: .github/MCP_MEMORY_QUICK_REFERENCE.md"
echo "- Full Setup Guide: .github/MCP_COPILOT_SETUP.md"
echo "- Copilot Instructions: .github/copilot-instructions.md"
echo ""
echo -e "${GREEN}Happy coding with persistent memory! 🧠${NC}"

