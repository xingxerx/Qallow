#!/bin/bash

# Ollama Setup Script for Continue IDE
# This script installs Ollama and sets up a model for Continue IDE

set -e

echo "=========================================="
echo "Ollama Setup for Continue IDE"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Ollama is already installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama is already installed${NC}"
    OLLAMA_INSTALLED=true
else
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    OLLAMA_INSTALLED=false
fi

# Install Ollama if not present
if [ "$OLLAMA_INSTALLED" = false ]; then
    echo ""
    echo "Installing Ollama..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
        echo -e "${GREEN}✓ Ollama installed${NC}"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
            echo -e "${GREEN}✓ Ollama installed${NC}"
        else
            echo -e "${RED}✗ Homebrew not found. Please install Ollama manually from https://ollama.ai${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ Unsupported OS. Please install Ollama manually from https://ollama.ai${NC}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Ollama Installation Complete"
echo "=========================================="
echo ""

# Check if Ollama service is running
echo "Checking if Ollama is running..."
if curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Ollama is running${NC}"
    OLLAMA_RUNNING=true
else
    echo -e "${YELLOW}⚠ Ollama is not running${NC}"
    OLLAMA_RUNNING=false
fi

echo ""
echo "=========================================="
echo "Model Selection"
echo "=========================================="
echo ""
echo "Choose a model to download:"
echo ""
echo "1) llama2 (recommended - balanced)"
echo "2) mistral (fast, good for coding)"
echo "3) neural-chat (good for conversations)"
echo "4) codellama (specialized for code)"
echo "5) dolphin-mixtral (most capable, slower)"
echo "6) Skip model download"
echo ""
read -p "Enter your choice (1-6): " model_choice

case $model_choice in
    1)
        MODEL="llama2"
        ;;
    2)
        MODEL="mistral"
        ;;
    3)
        MODEL="neural-chat"
        ;;
    4)
        MODEL="codellama"
        ;;
    5)
        MODEL="dolphin-mixtral"
        ;;
    6)
        MODEL=""
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

if [ -n "$MODEL" ]; then
    echo ""
    echo "Downloading $MODEL model..."
    echo ""
    
    if [ "$OLLAMA_RUNNING" = false ]; then
        echo -e "${YELLOW}Starting Ollama service...${NC}"
        
        # Start Ollama in background
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            # Linux - start as service
            sudo systemctl start ollama || ollama serve &
            OLLAMA_PID=$!
            sleep 3
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS - start in background
            ollama serve &
            OLLAMA_PID=$!
            sleep 3
        fi
    fi
    
    # Pull the model
    ollama pull $MODEL
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Model $MODEL downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to download model${NC}"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Start Ollama (if not already running):"
echo "   ollama serve"
echo ""
echo "2. Restart Continue IDE"
echo ""
echo "3. Continue IDE will now use Ollama locally"
echo ""
echo "Available models:"
ollama list 2>/dev/null || echo "   (Start Ollama to see models)"
echo ""
echo "To add more models later:"
echo "   ollama pull <model-name>"
echo ""
echo "For more models, visit: https://ollama.ai/library"
echo ""

