#!/bin/bash

# Qallow Web Application Startup Script
# Starts both the backend API server and React frontend

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"

API_DIR="${REPO_ROOT}/server"
CLIENT_DIR="${REPO_ROOT}/web-app"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 QALLOW WEB APPLICATION STARTUP                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

if [[ ! -d "${API_DIR}" ]]; then
    echo -e "${YELLOW}❌ API directory not found at ${API_DIR}${NC}"
    exit 1
fi

if [[ ! -d "${CLIENT_DIR}" ]]; then
    echo -e "${YELLOW}❌ Web client directory not found at ${CLIENT_DIR}${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}❌ Node.js is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Node.js found: $(node --version)${NC}"
echo ""

# Start backend API server
echo -e "${BLUE}Starting Web API Server (port 3001)...${NC}"
cd "${API_DIR}"
node server-web.js &
API_PID=$!
echo -e "${GREEN}✓ API Server started (PID: $API_PID)${NC}"
sleep 2

# Start React frontend
echo -e "${BLUE}Starting React Frontend (port 3000)...${NC}"
cd "${CLIENT_DIR}"
npm start &
REACT_PID=$!
echo -e "${GREEN}✓ React Frontend started (PID: $REACT_PID)${NC}"
sleep 3

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ QALLOW WEB APPLICATION IS RUNNING                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}🌐 Web App:${NC}     http://localhost:3000"
echo -e "${GREEN}🔌 API Server:${NC}  http://localhost:3001"
echo -e "${GREEN}📡 WebSocket:${NC}   ws://localhost:3001"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for both processes
wait $API_PID $REACT_PID
