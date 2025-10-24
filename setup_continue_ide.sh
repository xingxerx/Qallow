#!/bin/bash

# Continue IDE Setup Script
# This script helps you configure Continue IDE with your API key

set -e

echo "=========================================="
echo "Continue IDE Setup"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check if Continue config exists
if [ ! -f "/root/Qallow/.continue/config.json" ]; then
    echo -e "${YELLOW}⚠ Continue IDE config not found${NC}"
    echo "Creating default configuration..."
    mkdir -p /root/Qallow/.continue
fi

echo "Choose your AI model provider:"
echo ""
echo "1) Google Gemini (recommended - free tier available)"
echo "2) Anthropic Claude (requires API key)"
echo "3) OpenAI (requires API key)"
echo "4) Ollama (local, no API key needed)"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Google Gemini Setup${NC}"
        echo "1. Go to: https://aistudio.google.com/app/apikey"
        echo "2. Click 'Create API Key'"
        echo "3. Copy your API key"
        echo ""
        read -p "Enter your Gemini API key: " api_key
        
        if [ -z "$api_key" ]; then
            echo -e "${YELLOW}No API key provided${NC}"
            exit 1
        fi
        
        # Add to bashrc
        if ! grep -q "GEMINI_API_KEY" ~/.bashrc; then
            echo "export GEMINI_API_KEY='$api_key'" >> ~/.bashrc
            echo -e "${GREEN}✓ Added to ~/.bashrc${NC}"
        else
            sed -i "s/export GEMINI_API_KEY=.*/export GEMINI_API_KEY='$api_key'/" ~/.bashrc
            echo -e "${GREEN}✓ Updated ~/.bashrc${NC}"
        fi
        
        # Also add to zshrc if it exists
        if [ -f ~/.zshrc ]; then
            if ! grep -q "GEMINI_API_KEY" ~/.zshrc; then
                echo "export GEMINI_API_KEY='$api_key'" >> ~/.zshrc
                echo -e "${GREEN}✓ Added to ~/.zshrc${NC}"
            else
                sed -i "s/export GEMINI_API_KEY=.*/export GEMINI_API_KEY='$api_key'/" ~/.zshrc
                echo -e "${GREEN}✓ Updated ~/.zshrc${NC}"
            fi
        fi
        
        export GEMINI_API_KEY="$api_key"
        echo -e "${GREEN}✓ API key configured${NC}"
        ;;
        
    2)
        echo ""
        echo -e "${BLUE}Anthropic Claude Setup${NC}"
        echo "1. Go to: https://console.anthropic.com/"
        echo "2. Create an API key"
        echo "3. Copy your API key"
        echo ""
        read -p "Enter your Anthropic API key: " api_key
        
        if [ -z "$api_key" ]; then
            echo -e "${YELLOW}No API key provided${NC}"
            exit 1
        fi
        
        if ! grep -q "ANTHROPIC_API_KEY" ~/.bashrc; then
            echo "export ANTHROPIC_API_KEY='$api_key'" >> ~/.bashrc
        else
            sed -i "s/export ANTHROPIC_API_KEY=.*/export ANTHROPIC_API_KEY='$api_key'/" ~/.bashrc
        fi
        
        if [ -f ~/.zshrc ]; then
            if ! grep -q "ANTHROPIC_API_KEY" ~/.zshrc; then
                echo "export ANTHROPIC_API_KEY='$api_key'" >> ~/.zshrc
            else
                sed -i "s/export ANTHROPIC_API_KEY=.*/export ANTHROPIC_API_KEY='$api_key'/" ~/.zshrc
            fi
        fi
        
        export ANTHROPIC_API_KEY="$api_key"
        echo -e "${GREEN}✓ API key configured${NC}"
        ;;
        
    3)
        echo ""
        echo -e "${BLUE}OpenAI Setup${NC}"
        echo "1. Go to: https://platform.openai.com/api-keys"
        echo "2. Create an API key"
        echo "3. Copy your API key"
        echo ""
        read -p "Enter your OpenAI API key: " api_key
        
        if [ -z "$api_key" ]; then
            echo -e "${YELLOW}No API key provided${NC}"
            exit 1
        fi
        
        if ! grep -q "OPENAI_API_KEY" ~/.bashrc; then
            echo "export OPENAI_API_KEY='$api_key'" >> ~/.bashrc
        else
            sed -i "s/export OPENAI_API_KEY=.*/export OPENAI_API_KEY='$api_key'/" ~/.bashrc
        fi
        
        if [ -f ~/.zshrc ]; then
            if ! grep -q "OPENAI_API_KEY" ~/.zshrc; then
                echo "export OPENAI_API_KEY='$api_key'" >> ~/.zshrc
            else
                sed -i "s/export OPENAI_API_KEY=.*/export OPENAI_API_KEY='$api_key'/" ~/.zshrc
            fi
        fi
        
        export OPENAI_API_KEY="$api_key"
        echo -e "${GREEN}✓ API key configured${NC}"
        ;;
        
    4)
        echo ""
        echo -e "${BLUE}Ollama Setup${NC}"
        echo "Ollama runs locally - no API key needed!"
        echo ""
        echo "To install Ollama:"
        echo "  1. Visit: https://ollama.ai"
        echo "  2. Download and install"
        echo "  3. Run: ollama pull llama2"
        echo ""
        echo "Continue IDE will automatically connect to Ollama"
        echo -e "${GREEN}✓ No configuration needed${NC}"
        ;;
        
    *)
        echo -e "${YELLOW}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Reload your shell: source ~/.bashrc"
echo "2. Restart Continue IDE"
echo "3. The error should be gone!"
echo ""
echo "For more details, see: CONTINUE_IDE_SETUP.md"
echo ""

