#!/bin/bash

# ============================================================================
# Continue.dev Setup Script for Qallow
# ============================================================================
# This script helps you set up Continue.dev with the Qallow project
# Supports: Gemini, Claude, OpenAI, and Ollama (local)
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================================
# Main Setup
# ============================================================================

main() {
    print_header "Continue.dev Setup for Qallow"
    
    # Check if .env exists
    if [ ! -f .env ]; then
        print_warning ".env file not found"
        print_info "Creating .env from .env.example..."
        cp .env.example .env
        print_success ".env created"
    else
        print_success ".env file found"
    fi
    
    # Show menu
    echo "Choose your AI model provider:"
    echo ""
    echo "1) Ollama (Local - No API key needed) - RECOMMENDED FOR TESTING"
    echo "2) Google Gemini (Free tier available)"
    echo "3) Anthropic Claude (Requires paid account)"
    echo "4) OpenAI GPT-4 (Requires paid account)"
    echo "5) Skip - I'll configure manually"
    echo ""
    read -p "Enter your choice (1-5): " choice
    
    case $choice in
        1)
            setup_ollama
            ;;
        2)
            setup_gemini
            ;;
        3)
            setup_claude
            ;;
        4)
            setup_openai
            ;;
        5)
            print_info "Skipping automatic setup"
            print_manual_instructions
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
    
    # Final instructions
    print_header "Setup Complete!"
    print_success "Continue.dev is configured"
    print_info "Next steps:"
    echo "  1. Load environment: source .env"
    echo "  2. Open VS Code with Continue extension"
    echo "  3. Continue will use the configured model"
    echo ""
    print_info "To verify: echo \$GEMINI_API_KEY (or your chosen provider)"
}

# ============================================================================
# Setup Functions
# ============================================================================

setup_ollama() {
    print_header "Setting up Ollama (Local AI)"
    
    print_info "Ollama runs AI models locally on your machine"
    print_info "No API key needed - completely private"
    echo ""
    
    # Check if ollama is installed
    if ! command -v ollama &> /dev/null; then
        print_warning "Ollama is not installed"
        echo "Install from: https://ollama.ai"
        echo ""
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Ollama is installed"
    fi
    
    print_info "To use Ollama:"
    echo "  1. Start Ollama: ollama serve"
    echo "  2. In another terminal, pull a model: ollama pull llama2"
    echo "  3. Continue will automatically connect to http://localhost:11434"
    echo ""
    
    # Update .env
    sed -i 's/^OLLAMA_API_BASE=.*/OLLAMA_API_BASE=http:\/\/localhost:11434/' .env
    print_success "Updated .env with Ollama configuration"
    
    print_info "Ollama models available:"
    echo "  - llama2 (7B, fast, good for coding)"
    echo "  - mistral (7B, very fast)"
    echo "  - neural-chat (7B, optimized for chat)"
    echo "  - codellama (7B, optimized for code)"
    echo ""
    print_info "Pull a model with: ollama pull <model-name>"
}

setup_gemini() {
    print_header "Setting up Google Gemini"
    
    print_info "Gemini offers a free tier with generous limits"
    echo ""
    
    read -p "Enter your Gemini API key (or press Enter to skip): " api_key
    
    if [ -z "$api_key" ]; then
        print_warning "Skipped Gemini setup"
        print_info "Get a free API key at: https://aistudio.google.com/app/apikey"
        return
    fi
    
    # Update .env
    sed -i "s/^GEMINI_API_KEY=.*/GEMINI_API_KEY=$api_key/" .env
    print_success "Updated .env with Gemini API key"
    
    # Load the env
    source .env
    print_success "Gemini is configured and ready to use"
}

setup_claude() {
    print_header "Setting up Anthropic Claude"
    
    print_info "Claude requires a paid Anthropic account"
    echo ""
    
    read -p "Enter your Anthropic API key (or press Enter to skip): " api_key
    
    if [ -z "$api_key" ]; then
        print_warning "Skipped Claude setup"
        print_info "Get an API key at: https://console.anthropic.com/"
        return
    fi
    
    # Update .env
    sed -i "s/^ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$api_key/" .env
    print_success "Updated .env with Anthropic API key"
    
    # Load the env
    source .env
    print_success "Claude is configured and ready to use"
}

setup_openai() {
    print_header "Setting up OpenAI GPT-4"
    
    print_info "GPT-4 requires a paid OpenAI account"
    echo ""
    
    read -p "Enter your OpenAI API key (or press Enter to skip): " api_key
    
    if [ -z "$api_key" ]; then
        print_warning "Skipped OpenAI setup"
        print_info "Get an API key at: https://platform.openai.com/api-keys"
        return
    fi
    
    # Update .env
    sed -i "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$api_key/" .env
    print_success "Updated .env with OpenAI API key"
    
    # Load the env
    source .env
    print_success "OpenAI is configured and ready to use"
}

print_manual_instructions() {
    print_header "Manual Configuration Instructions"
    
    echo "1. Edit .env file:"
    echo "   nano .env"
    echo ""
    echo "2. Choose ONE provider and add your API key:"
    echo ""
    echo "   For Gemini:"
    echo "     GEMINI_API_KEY=your-key-here"
    echo "     Get key: https://aistudio.google.com/app/apikey"
    echo ""
    echo "   For Claude:"
    echo "     ANTHROPIC_API_KEY=your-key-here"
    echo "     Get key: https://console.anthropic.com/"
    echo ""
    echo "   For OpenAI:"
    echo "     OPENAI_API_KEY=your-key-here"
    echo "     Get key: https://platform.openai.com/api-keys"
    echo ""
    echo "   For Ollama (local, no key needed):"
    echo "     OLLAMA_API_BASE=http://localhost:11434"
    echo ""
    echo "3. Load the environment:"
    echo "   source .env"
    echo ""
    echo "4. Verify:"
    echo "   echo \$GEMINI_API_KEY  # or your chosen provider"
    echo ""
    echo "5. Open VS Code with Continue extension"
}

# ============================================================================
# Run Main
# ============================================================================

if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi

