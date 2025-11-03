#!/bin/bash

# MCP Memory Service Setup Script for Linux
# This script sets up and starts the MCP Memory Service on Linux systems

set -e

echo "=========================================="
echo "MCP Memory Service Linux Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MCP_SERVICE_DIR="/root/Qallow/mcp-memory-service"
MCP_VENV="$MCP_SERVICE_DIR/.venv/bin/python"
MCP_DB_PATH="$HOME/.local/share/mcp-memory"
VSCODE_CONFIG_DIR="/root/Qallow/.vscode"
LOG_FILE="/tmp/mcp_memory_service.log"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if Python is installed
echo "Checking Python installation..."
if [ ! -f "$MCP_VENV" ]; then
    print_error "Virtual environment not found at: $MCP_VENV"
    print_warning "Creating virtual environment..."
    python -m venv "$MCP_SERVICE_DIR/.venv"
fi
print_status "Python found: $($MCP_VENV --version)"

# Create database directory
echo "Setting up database directory..."
mkdir -p "$MCP_DB_PATH"
chmod 755 "$MCP_DB_PATH"
print_status "Database directory: $MCP_DB_PATH"

# Check if MCP service directory exists
if [ ! -d "$MCP_SERVICE_DIR" ]; then
    print_error "MCP service directory not found: $MCP_SERVICE_DIR"
    exit 1
fi
print_status "MCP service directory found"

# Install dependencies
echo "Installing MCP Memory Service dependencies..."
cd "$MCP_SERVICE_DIR"

print_status "Installing core MCP dependencies..."
$MCP_VENV -m pip install -q mcp sqlite-vec sentence-transformers 2>&1 | grep -v "already satisfied" || true

if [ -f "pyproject.toml" ]; then
    print_status "Found pyproject.toml, installing with pip..."
    $MCP_VENV -m pip install -q -e . 2>&1 | grep -v "already satisfied" || print_warning "Some dependencies may have failed to install"
elif [ -f "requirements.txt" ]; then
    print_status "Found requirements.txt, installing dependencies..."
    $MCP_VENV -m pip install -q -r requirements.txt 2>&1 | grep -v "already satisfied" || print_warning "Some dependencies may have failed to install"
fi

print_status "Dependencies installed"

# Create VS Code MCP configuration
echo "Configuring VS Code MCP settings..."
mkdir -p "$VSCODE_CONFIG_DIR"

if [ ! -f "$VSCODE_CONFIG_DIR/mcp.json" ]; then
    print_warning "MCP configuration not found, creating..."
    cat > "$VSCODE_CONFIG_DIR/mcp.json" << 'EOF'
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": [
        "-m",
        "src.mcp_memory_service.server"
      ],
      "cwd": "/root/Qallow/mcp-memory-service",
      "env": {
        "MCP_MEMORY_STORAGE_BACKEND": "sqlite_vec",
        "MCP_MEMORY_SQLITE_VEC_PATH": "/root/.local/share/mcp-memory",
        "PYTHONPATH": "/root/Qallow/mcp-memory-service/src",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
EOF
    print_status "MCP configuration created"
else
    print_status "MCP configuration already exists"
fi

# Configure Claude Desktop if it exists
CLAUDE_CONFIG="$HOME/.config/claude/claude_desktop_config.json"
if [ -f "$CLAUDE_CONFIG" ]; then
    echo "Claude Desktop configuration found"
    print_warning "Please manually update $CLAUDE_CONFIG with the MCP server configuration"
    print_warning "See MCP_FIX_GUIDE.md for details"
fi

# Test MCP server startup
echo "Testing MCP server startup..."
cd "$MCP_SERVICE_DIR"

# Start server in background
export MCP_MEMORY_STORAGE_BACKEND="sqlite_vec"
export MCP_MEMORY_SQLITE_VEC_PATH="$MCP_DB_PATH"
export PYTHONPATH="$MCP_SERVICE_DIR/src"
export LOG_LEVEL="INFO"

# Try to start the server
timeout 5 $MCP_VENV -m src.mcp_memory_service.server > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Check if server is running
if kill -0 $SERVER_PID 2>/dev/null; then
    print_status "MCP server started successfully (PID: $SERVER_PID)"
    print_status "Server logs: $LOG_FILE"
    
    # Try to verify server is responding
    if command -v curl &> /dev/null; then
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            print_status "Server is responding to health checks"
        else
            print_warning "Server started but not responding to health checks yet"
        fi
    fi
    
    # Kill the test server
    kill $SERVER_PID 2>/dev/null || true
    print_status "Test server stopped"
else
    print_error "Failed to start MCP server"
    print_error "Check logs: $LOG_FILE"
    cat "$LOG_FILE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart your IDE (VS Code, Claude Desktop, etc.)"
echo "2. The MCP server should now connect successfully"
echo ""
echo "To manually start the server:"
echo "  cd $MCP_SERVICE_DIR"
echo "  ./.venv/bin/python -m src.mcp_memory_service.server"
echo ""
echo "Or use the system Python with proper environment:"
echo "  export PYTHONPATH=$MCP_SERVICE_DIR/src"
echo "  python -m src.mcp_memory_service.server"
echo ""
echo "For more information, see: /root/Qallow/MCP_FIX_GUIDE.md"
echo ""

