#!/bin/bash

################################################################################
# QALLOW UNIFIED SERVER - QUICK START
# One-command setup and launch
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║     🚀 QALLOW UNIFIED SERVER - QUICK START                    ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Install server dependencies
echo -e "${YELLOW}[1/5]${NC} Installing server dependencies..."
cd "$SCRIPT_DIR/server"
npm install --production > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Server dependencies installed"

# Step 2: Build frontend
echo -e "${YELLOW}[2/5]${NC} Building React frontend..."
cd "$SCRIPT_DIR/app"
if [ ! -d "build" ]; then
  npm run build > /dev/null 2>&1
  echo -e "${GREEN}✓${NC} Frontend built"
else
  echo -e "${GREEN}✓${NC} Frontend already built"
fi

# Step 3: Create environment file
echo -e "${YELLOW}[3/5]${NC} Setting up environment..."
cd "$SCRIPT_DIR/server"
if [ ! -f ".env" ]; then
  cat > .env << EOF
NODE_ENV=development
PORT=5000
FRONTEND_PORT=3000
LOG_LEVEL=info
QUANTUM_FRAMEWORK=cirq
EOF
  echo -e "${GREEN}✓${NC} Environment configured"
else
  echo -e "${GREEN}✓${NC} Environment already configured"
fi

# Step 4: Verify dependencies
echo -e "${YELLOW}[4/5]${NC} Verifying dependencies..."
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}✗${NC} Python 3 not found"
  exit 1
fi
if ! python3 -c "import cirq" 2>/dev/null; then
  echo -e "${YELLOW}⚠${NC} Installing Cirq..."
  pip install cirq -q
fi
echo -e "${GREEN}✓${NC} All dependencies verified"

# Step 5: Start server
echo -e "${YELLOW}[5/5]${NC} Starting server..."
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Server is starting!${NC}"
echo ""
echo -e "  🌐 Dashboard:  ${YELLOW}http://localhost:5000${NC}"
echo -e "  📡 API:        ${YELLOW}http://localhost:5000/api${NC}"
echo -e "  🔌 WebSocket:  ${YELLOW}ws://localhost:5000${NC}"
echo ""
echo -e "  📊 Health:     ${YELLOW}curl http://localhost:5000/api/health${NC}"
echo -e "  🧪 Test:       ${YELLOW}curl http://localhost:5000/api/quantum/status${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Start server
cd "$SCRIPT_DIR/server"
node server.js

