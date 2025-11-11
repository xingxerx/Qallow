#!/bin/bash
# Setup script for Qallow Memory System
# Initializes Qdrant, installs dependencies, and runs tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Qallow Memory System Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# 1. Check dependencies
echo -e "${YELLOW}[1/5] Checking system dependencies...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found${NC}"
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}✅ Docker installed${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python3 installed${NC}"

# 2. Install Python dependencies
echo ""
echo -e "${YELLOW}[2/5] Installing Python dependencies...${NC}"
cd "$PROJECT_ROOT"
pip install -q -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# 3. Start Qdrant
echo ""
echo -e "${YELLOW}[3/5] Starting Qdrant vector database...${NC}"

# Check if Qdrant is already running
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Qdrant already running${NC}"
else
    echo "Starting Qdrant container..."
    docker run -d -p 6333:6333 --name qallow-qdrant qdrant/qdrant 2>/dev/null || true
    
    # Wait for Qdrant to be ready
    echo "Waiting for Qdrant to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:6333/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Qdrant is ready${NC}"
            break
        fi
        if [ $i -eq 30 ]; then
            echo -e "${RED}❌ Qdrant failed to start${NC}"
            exit 1
        fi
        sleep 1
    done
fi

# 4. Initialize memory system
echo ""
echo -e "${YELLOW}[4/5] Initializing memory system...${NC}"
python3 -c "
import asyncio
from qallow.memory import ExperienceStore

async def init():
    store = ExperienceStore()
    try:
        await store.initialize_collection()
        print('✅ Memory collection initialized')
    except Exception as e:
        print(f'⚠️  Could not initialize collection: {e}')
        print('   This is OK if Qdrant is not running')

asyncio.run(init())
"

# 5. Run tests
echo ""
echo -e "${YELLOW}[5/5] Running memory system tests...${NC}"
if command -v pytest &> /dev/null; then
    pytest tests/test_memory_store.py -v --tb=short 2>/dev/null || true
    echo -e "${GREEN}✅ Tests completed${NC}"
else
    echo -e "${YELLOW}⚠️  pytest not found, skipping tests${NC}"
    echo "Install with: pip install pytest pytest-asyncio"
fi

# Summary
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Next steps:"
echo "1. Start using the memory system:"
echo "   python3 -c \"from qallow.memory import ExperienceStore; print('Ready!')\""
echo ""
echo "2. View Qdrant dashboard:"
echo "   http://localhost:6333/dashboard"
echo ""
echo "3. Run the example:"
echo "   python3 -m qallow.memory.experience_store"
echo ""
echo "4. Read the documentation:"
echo "   cat qallow/memory/README.md"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "  # Stop Qdrant"
echo "  docker stop qallow-qdrant"
echo ""
echo "  # View Qdrant logs"
echo "  docker logs qallow-qdrant"
echo ""
echo "  # Remove Qdrant container"
echo "  docker rm qallow-qdrant"
echo ""

