#!/bin/bash

################################################################################
# QALLOW UNIFIED SERVER STARTUP SCRIPT
# Comprehensive server management for frontend, backend, and quantum framework
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SERVER_DIR")"
PORT=${PORT:-5000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
NODE_ENV=${NODE_ENV:-development}
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     🚀 QALLOW UNIFIED SERVER - STARTUP SEQUENCE${NC}${BLUE}             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Function to print status
print_status() {
  echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
  echo -e "${RED}[✗]${NC} $1"
}

print_info() {
  echo -e "${BLUE}[i]${NC} $1"
}

print_warn() {
  echo -e "${YELLOW}[!]${NC} $1"
}

# Check Node.js installation
print_info "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
  print_error "Node.js is not installed"
  exit 1
fi
NODE_VERSION=$(node -v)
print_status "Node.js $NODE_VERSION found"

# Check npm installation
print_info "Checking npm installation..."
if ! command -v npm &> /dev/null; then
  print_error "npm is not installed"
  exit 1
fi
NPM_VERSION=$(npm -v)
print_status "npm $NPM_VERSION found"

# Check Python installation
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
  print_error "Python 3 is not installed"
  exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_status "$PYTHON_VERSION found"

# Check Cirq installation
print_info "Checking Cirq installation..."
if ! python3 -c "import cirq" 2>/dev/null; then
  print_warn "Cirq not found, installing..."
  pip install cirq -q
  print_status "Cirq installed"
else
  print_status "Cirq is installed"
fi

# Install server dependencies
print_info "Installing server dependencies..."
cd "$SERVER_DIR"
if [ ! -d "node_modules" ]; then
  npm install --production
  print_status "Server dependencies installed"
else
  print_status "Server dependencies already installed"
fi

# Build React frontend
print_info "Building React frontend..."
cd "$PROJECT_ROOT/app"
if [ ! -d "build" ]; then
  print_warn "Frontend build not found, building..."
  npm run build
  print_status "Frontend built successfully"
else
  print_status "Frontend build found"
fi

# Create .env file if it doesn't exist
print_info "Checking environment configuration..."
if [ ! -f "$SERVER_DIR/.env" ]; then
  cat > "$SERVER_DIR/.env" << EOF
NODE_ENV=$NODE_ENV
PORT=$PORT
FRONTEND_PORT=$FRONTEND_PORT
LOG_LEVEL=info
QUANTUM_FRAMEWORK=cirq
EOF
  print_status "Environment file created"
else
  print_status "Environment file exists"
fi

# Start the server
print_info "Starting Qallow Unified Server..."
cd "$SERVER_DIR"

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Server Configuration:${NC}"
echo -e "  Port: ${YELLOW}$PORT${NC}"
echo -e "  Frontend Port: ${YELLOW}$FRONTEND_PORT${NC}"
echo -e "  Environment: ${YELLOW}$NODE_ENV${NC}"
echo -e "  Log Directory: ${YELLOW}$LOG_DIR${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Start server with logging
if [ "$NODE_ENV" = "production" ]; then
  print_status "Starting in PRODUCTION mode"
  NODE_ENV=production node server.js 2>&1 | tee "$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
else
  print_status "Starting in DEVELOPMENT mode"
  if command -v nodemon &> /dev/null; then
    nodemon server.js 2>&1 | tee "$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
  else
    node server.js 2>&1 | tee "$LOG_DIR/server-$(date +%Y%m%d-%H%M%S).log"
  fi
fi

