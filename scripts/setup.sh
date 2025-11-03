#!/bin/bash
# Qallow Project Automated Setup Script
# This script installs all system and Python dependencies

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                  QALLOW PROJECT SETUP SCRIPT                   ║"
    echo "║            Quantum-Photonic Computing Platform                 ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Check OS
check_os() {
    log_info "Detecting operating system..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS_NAME=$NAME
            OS_VERSION=$VERSION_ID
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
        OS_VERSION=$(sw_vers -productVersion)
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        OS="Windows"
    else
        OS="Unknown"
    fi
    
    log_success "Detected: $OS $OS_VERSION"
}

# Install system packages
install_system_packages() {
    log_info "Installing system dependencies..."
    
    case "$OS" in
        Linux)
            if command -v apt-get &> /dev/null; then
                log_info "Using apt-get package manager..."
                sudo apt-get update
                sudo apt-get install -y \
                    build-essential \
                    cmake \
                    pkg-config \
                    git \
                    python3-pip \
                    python3-venv \
                    python3-dev \
                    libssl-dev \
                    libffi-dev \
                    libsdl2-dev \
                    libsdl2-ttf-dev \
                    libglib2.0-dev \
                    libgl-dev \
                    libxrandr-dev \
                    zlib1g-dev
                log_success "System packages installed"
            elif command -v dnf &> /dev/null; then
                log_warn "Using dnf package manager (Fedora/RHEL)"
                sudo dnf install -y \
                    @development-tools \
                    cmake \
                    pkg-config \
                    python3-pip \
                    python3-devel
                log_success "System packages installed"
            else
                log_error "Unsupported package manager"
                exit 1
            fi
            ;;
        macOS)
            if command -v brew &> /dev/null; then
                log_info "Using Homebrew package manager..."
                brew install python3 cmake pkg-config sdl2 sdl2_ttf
                log_success "System packages installed"
            else
                log_error "Homebrew not found. Install from https://brew.sh"
                exit 1
            fi
            ;;
        Windows)
            log_warn "Windows detected. Please install dependencies manually:"
            log_warn "1. Python 3.11+ from https://www.python.org"
            log_warn "2. Visual Studio Build Tools from https://visualstudio.microsoft.com"
            log_warn "3. Git for Windows from https://git-scm.com"
            ;;
        *)
            log_error "Unsupported operating system"
            exit 1
            ;;
    esac
}

# Check Python version
check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_success "Found Python $PYTHON_VERSION"
    
    # Check version is 3.10+
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        log_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

# Check pip
check_pip() {
    log_info "Checking pip installation..."
    
    if ! python3 -m pip --version &> /dev/null; then
        log_warn "pip not found, installing..."
        sudo apt-get install -y python3-pip
    fi
    
    PIP_VERSION=$(python3 -m pip --version | awk '{print $2}')
    log_success "Found pip $PIP_VERSION"
}

# Create virtual environment
setup_venv() {
    log_info "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        log_warn "Virtual environment already exists"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            return
        fi
    fi
    
    python3 -m venv venv
    log_success "Virtual environment created"
    
    # Activate venv
    source venv/bin/activate
    log_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    log_success "pip, setuptools, and wheel upgraded"
}

# Install Python packages
install_python_packages() {
    log_info "Installing Python packages..."
    
    # Install core requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing core requirements..."
        pip install -r requirements.txt
        log_success "Core requirements installed"
    fi
    
    # Ask for optional packages
    read -p "Install development tools? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-dev.txt" ]; then
            pip install -r requirements-dev.txt
            log_success "Development tools installed"
        fi
    fi
    
    read -p "Install web framework dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-web.txt" ]; then
            pip install -r requirements-web.txt
            log_success "Web framework installed"
        fi
    fi
    
    read -p "Install GPU acceleration support? (requires CUDA 12.0+) (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "requirements-gpu.txt" ]; then
            pip install -r requirements-gpu.txt
            log_success "GPU support installed"
        fi
    fi
}

# Create data directories
setup_directories() {
    log_info "Creating project directories..."
    
    mkdir -p data/logs
    mkdir -p data/quantum_results
    mkdir -p data/telemetry
    
    log_success "Directories created"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check Python
    if ! python3 --version &> /dev/null; then
        log_error "Python verification failed"
        return 1
    fi
    log_success "Python: $(python3 --version)"
    
    # Check pip
    if ! pip --version &> /dev/null; then
        log_error "pip verification failed"
        return 1
    fi
    log_success "pip: $(pip --version | awk '{print $2}')"
    
    # Check core packages
    if python3 -c "import numpy, scipy, pandas" 2>/dev/null; then
        log_success "Core packages (numpy, scipy, pandas) installed"
    else
        log_warn "Some core packages not yet installed"
    fi
    
    # Check project structure
    if [ -f "run_qallow.py" ]; then
        log_success "Project structure verified"
    else
        log_error "Project structure incomplete"
        return 1
    fi
}

# Run tests
run_tests() {
    log_info "Running verification tests..."
    
    if [ -f "test_quantum_complete.py" ]; then
        python3 test_quantum_complete.py
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                    SETUP COMPLETED! ✓                          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Next steps:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run project overview: python3 run_qallow.py"
    echo "  3. Run tests: python3 test_quantum_complete.py"
    echo "  4. Read documentation: cat README.md"
    echo ""
    echo "For more information, see SETUP_GUIDE.md"
    echo ""
}

# Main function
main() {
    print_header
    
    check_os
    check_python
    check_pip
    
    install_system_packages
    setup_venv
    install_python_packages
    setup_directories
    
    if verify_installation; then
        log_success "Installation verified!"
    else
        log_warn "Some verifications failed, but setup may still work"
    fi
    
    print_summary
}

# Run main
main
