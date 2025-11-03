#!/bin/bash

# Qallow Native Desktop App - Startup Script
# Starts both backend server and native GUI application

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║    🖥️  QALLOW NATIVE DESKTOP APPLICATION - STARTUP 🖥️         ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}❌ Rust is not installed${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${BLUE}[1/5] Checking dependencies...${NC}"
echo "  ✅ Node.js: $(node --version)"
echo "  ✅ Rust: $(rustc --version)"
echo "  ✅ Python: $(python3 --version)"
echo ""

# Check if Cirq is installed
echo -e "${BLUE}[2/5] Checking Cirq installation...${NC}"
if python3 -c "import cirq" 2>/dev/null; then
    echo "  ✅ Cirq is installed"
else
    echo -e "${YELLOW}⚠️  Installing Cirq...${NC}"
    pip install cirq -q
fi
echo ""

# Start backend server
echo -e "${BLUE}[3/5] Starting backend server...${NC}"
cd /root/Qallow/server

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing server dependencies...${NC}"
    npm install -q
fi

# Start backend in background
node server-backend-only.js > /tmp/qallow-backend.log 2>&1 &
BACKEND_PID=$!
echo "  ✅ Backend server started (PID: $BACKEND_PID)"
echo "  📝 Logs: /tmp/qallow-backend.log"

# Wait for backend to be ready
echo -e "${BLUE}[4/5] Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo "  ✅ Backend is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend failed to start${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
echo ""

# Build native app if needed
echo -e "${BLUE}[5/5] Building native application...${NC}"
cd /root/Qallow/native_app

if [ ! -f "target/release/qallow_native_app" ]; then
    echo -e "${YELLOW}🔨 Building native app (this may take a minute)...${NC}"
    cargo build --release -q
fi

echo "  ✅ Native app ready"
echo ""

# Display startup info
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    STARTUP COMPLETE                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}✅ Backend Server${NC}"
echo "   URL: http://localhost:5000"
echo "   WebSocket: ws://localhost:5000"
echo "   IPC Socket: /tmp/qallow-backend.sock"
echo "   Logs: /tmp/qallow-backend.log"
echo ""
echo -e "${GREEN}✅ Native Application${NC}"
echo "   Starting in 2 seconds..."
echo ""

# Wait a moment then start native app
sleep 2

# Start native app
cd /root/Qallow/native_app
./target/release/qallow_native_app > /tmp/qallow-native-app.log 2>&1 &
NATIVE_PID=$!

echo -e "${GREEN}✅ Native app started (PID: $NATIVE_PID)${NC}"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $NATIVE_PID 2>/dev/null || true
    echo -e "${GREEN}✅ Shutdown complete${NC}"
}

trap cleanup EXIT

# Wait for processes
wait
