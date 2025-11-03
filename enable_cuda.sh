#!/bin/bash

################################################################################
#                                                                              #
#                    QALLOW CUDA TOOLKIT BOOTSTRAP SCRIPT                     #
#                    Automated CUDA 12.6 Setup for WSL Ubuntu                #
#                                                                              #
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  $1"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
}

################################################################################
# MAIN INSTALLATION FLOW
################################################################################

print_header "QALLOW CUDA TOOLKIT BOOTSTRAP"

# Check if running in WSL
if ! grep -qi microsoft /proc/version; then
    log_warning "This script is optimized for WSL. Proceeding anyway..."
fi

# Step 1: Update package index
print_header "Step 1: Updating Package Index"

log_info "Running sudo apt-get update..."
sudo apt-get update || {
    log_error "Failed to update package index"
    exit 1
}
log_success "Package index updated"

log_info "Running sudo apt-get upgrade..."
sudo apt-get upgrade -y || {
    log_error "Failed to upgrade packages"
    exit 1
}
log_success "Packages upgraded"

# Step 2: Add NVIDIA CUDA repository
print_header "Step 2: Adding NVIDIA CUDA Repository"

log_info "Downloading CUDA keyring..."
if [ ! -f cuda-keyring_1.1-1_all.deb ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb || {
        log_error "Failed to download CUDA keyring"
        exit 1
    }
    log_success "CUDA keyring downloaded"
else
    log_info "CUDA keyring already downloaded (using cached version)"
fi

log_info "Installing CUDA keyring..."
sudo dpkg -i cuda-keyring_1.1-1_all.deb || {
    log_error "Failed to install CUDA keyring"
    exit 1
}
log_success "CUDA keyring installed"

log_info "Updating package index with NVIDIA repository..."
sudo apt-get update || {
    log_error "Failed to update package index after adding CUDA repo"
    exit 1
}
log_success "NVIDIA repository added to package index"

# Step 3: Install CUDA Toolkit
print_header "Step 3: Installing CUDA Toolkit 12.6"

log_info "Installing cuda-toolkit-12-6 (this may take 5-10 minutes)..."
sudo apt-get install -y cuda-toolkit-12-6 || {
    log_error "Failed to install CUDA toolkit"
    exit 1
}
log_success "CUDA Toolkit 12.6 installed successfully"

# Step 4: Verify CUDA installation
print_header "Step 4: Verifying CUDA Installation"

if ! command -v nvcc &> /dev/null; then
    log_error "nvcc not found in PATH"
    log_info "Setting PATH to include /usr/local/cuda/bin..."
    export PATH=/usr/local/cuda/bin:$PATH
fi

log_info "Checking nvcc version..."
nvcc_version=$(nvcc --version 2>/dev/null | grep release || echo "unknown")
log_success "CUDA Compiler: $nvcc_version"

nvcc_path=$(which nvcc)
log_success "CUDA Compiler Location: $nvcc_path"

# Step 5: Set environment variables
print_header "Step 5: Configuring Environment Variables"

# Check if variables already exist in bashrc
if grep -q "export PATH=/usr/local/cuda/bin" ~/.bashrc; then
    log_info "CUDA PATH already configured in ~/.bashrc"
else
    log_info "Adding CUDA environment variables to ~/.bashrc..."
    cat >> ~/.bashrc << 'EOF'

# CUDA Toolkit Configuration (added by enable_cuda.sh)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
    log_success "Environment variables added to ~/.bashrc"
fi

# Check if variables already exist in zshrc
if [ -f ~/.zshrc ]; then
    if grep -q "export PATH=/usr/local/cuda/bin" ~/.zshrc; then
        log_info "CUDA PATH already configured in ~/.zshrc"
    else
        log_info "Adding CUDA environment variables to ~/.zshrc..."
        cat >> ~/.zshrc << 'EOF'

# CUDA Toolkit Configuration (added by enable_cuda.sh)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF
        log_success "Environment variables added to ~/.zshrc"
    fi
fi

# Load environment variables for current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

log_success "Environment variables set for current session"

# Step 6: Rebuild Qallow with CUDA support
print_header "Step 6: Rebuilding Qallow with CUDA Support"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

log_info "Project directory: $PROJECT_DIR"
log_info "Build directory: $BUILD_DIR"

if [ -d "$BUILD_DIR" ]; then
    log_info "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
    log_success "Build directory cleaned"
fi

log_info "Creating new build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
log_success "Build directory created"

log_info "Running CMake with CUDA support..."
if cmake .. -DWITH_CUDA=ON; then
    log_success "CMake configuration completed"
else
    log_error "CMake configuration failed"
    log_info "Attempting CMake without explicit CUDA flag..."
    cmake .. || {
        log_error "CMake failed even without CUDA flag"
        exit 1
    }
    log_warning "Building without explicit CUDA flag (auto-detection)"
fi

log_info "Compiling (using $(nproc) cores)..."
if make -j$(nproc); then
    log_success "Compilation completed successfully"
else
    log_error "Compilation failed"
    exit 1
fi

# Step 7: Final verification
print_header "Step 7: Final Verification"

log_info "Verifying CUDA setup..."
echo ""
echo "CUDA Compiler Version:"
nvcc --version
echo ""

if [ -f "$BUILD_DIR/qallow_unified" ]; then
    log_success "Qallow executable built successfully"
    echo ""
    echo "Build Summary:"
    ls -lh "$BUILD_DIR/qallow_unified"
else
    log_warning "Qallow executable not found (may depend on build system)"
fi

echo ""
log_success "Environment variables configured:"
echo "  PATH: /usr/local/cuda/bin (prepended)"
echo "  LD_LIBRARY_PATH: /usr/local/cuda/lib64"

################################################################################
# COMPLETION SUMMARY
################################################################################

print_header "✅ CUDA INSTALLATION COMPLETE"

echo "You can now:"
echo ""
echo "1. Run Qallow with CUDA:"
echo "   cd $PROJECT_DIR"
echo "   ./run_with_improvement.sh 10 120 cuda"
echo ""
echo "2. Verify CUDA is being used:"
echo "   ./run_with_improvement.sh 10 120 cuda 2>&1 | grep -i cuda"
echo ""
echo "3. Use the rebuilt binary:"
echo "   $BUILD_DIR/qallow_unified [options]"
echo ""
echo "📝 Note: Environment variables have been added to ~/.bashrc and ~/.zshrc"
echo "         They will be automatically loaded in new terminal sessions."
echo ""
echo "For immediate use in current session, run:"
echo "   source ~/.bashrc"
echo ""

log_success "Bootstrap complete! CUDA is ready to use."

################################################################################
