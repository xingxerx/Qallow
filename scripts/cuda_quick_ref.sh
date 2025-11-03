#!/bin/bash
# CUDA QUICK REFERENCE - Qallow Project
# Copy this file as ~/cuda_quick_ref.sh and source it anytime

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Activate CUDA (do this once per terminal session)
cuda_activate() {
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    echo "✓ CUDA environment activated"
    nvcc --version | grep release
}

# =============================================================================
# VERIFICATION COMMANDS
# =============================================================================

# Check CUDA compiler
cuda_check() {
    echo "CUDA Compiler:"
    nvcc --version
    echo ""
    echo "CUDA Path:"
    which nvcc
    echo ""
    echo "CUDA Library Path:"
    echo $LD_LIBRARY_PATH | grep cuda
}

# Check GPU availability
cuda_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPUs available:"
        nvidia-smi --query-gpu=index,name --format=csv,noheader
    else
        echo "⚠ nvidia-smi not found (may not have GPU drivers)"
        echo "This is normal in WSL - CUDA will use CPU fallback"
    fi
}

# =============================================================================
# BUILD COMMANDS
# =============================================================================

# Rebuild with CUDA
qallow_rebuild_cuda() {
    cd ~/qallow/Qallow
    rm -rf build
    mkdir -p build
    cd build
    cmake .. -DWITH_CUDA=ON
    make -j$(nproc)
    cd ..
    echo "✓ Qallow rebuilt with CUDA support"
}

# Clean build (remove all build artifacts)
qallow_clean() {
    cd ~/qallow/Qallow
    rm -rf build
    echo "✓ Build directory cleaned"
}

# =============================================================================
# EXECUTION COMMANDS
# =============================================================================

# Run with CUDA backend
qallow_run_cuda() {
    cd ~/qallow/Qallow
    ./run_with_improvement.sh 10 120 cuda
}

# Run CUDA tests
qallow_test_cuda() {
    cd ~/qallow/Qallow/build
    echo "Running CUDA parallel tests..."
    ./qallow_unit_cuda_parallel
    echo ""
    echo "Running CUDA kernel tests..."
    ./test_kernels
}

# Run benchmarks
qallow_benchmark() {
    cd ~/qallow/Qallow/build
    echo "Running throughput benchmark..."
    ./qallow_throughput_bench
}

# =============================================================================
# DIAGNOSTICS
# =============================================================================

# Full CUDA diagnostics
cuda_diagnostics() {
    echo "╔═══════════════════════════════════════════════════╗"
    echo "║         CUDA INSTALLATION DIAGNOSTICS            ║"
    echo "╚═══════════════════════════════════════════════════╝"
    echo ""
    
    echo "1. CUDA Compiler:"
    nvcc --version 2>/dev/null || echo "   ✗ nvcc not found"
    echo ""
    
    echo "2. CUDA Libraries:"
    if [ -f /usr/local/cuda/lib64/libcuda.so ]; then
        echo "   ✓ libcuda.so found"
    else
        echo "   ✗ libcuda.so not found"
    fi
    echo ""
    
    echo "3. CUDA Headers:"
    if [ -f /usr/local/cuda/include/cuda.h ]; then
        echo "   ✓ cuda.h found"
    else
        echo "   ✗ cuda.h not found"
    fi
    echo ""
    
    echo "4. GPU Devices:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus || echo "   No GPUs detected (CPU only)"
    else
        echo "   ⚠ nvidia-smi not installed"
    fi
    echo ""
    
    echo "5. Environment Variables:"
    echo "   PATH includes CUDA: $(echo $PATH | grep -q cuda && echo '✓' || echo '✗')"
    echo "   LD_LIBRARY_PATH set: $([ -n "$LD_LIBRARY_PATH" ] && echo '✓' || echo '✗')"
    echo ""
    
    echo "6. Qallow Build:"
    if [ -f ~/qallow/Qallow/build/qallow_unified_cuda ]; then
        echo "   ✓ qallow_unified_cuda (CUDA build)"
    else
        echo "   ✗ CUDA build not found"
    fi
}

# =============================================================================
# SHORTCUTS
# =============================================================================

# Alias definitions (add to ~/.bashrc to make permanent)
alias cuda_on='cuda_activate'
alias cuda_check='cuda_check'
alias cuda_gpu='cuda_gpu'
alias cuda_diag='cuda_diagnostics'
alias qallow_cuda='qallow_run_cuda'
alias qallow_test='qallow_test_cuda'
alias qallow_bench='qallow_benchmark'

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

cuda_help() {
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════════════╗
║                    CUDA QUICK REFERENCE - USAGE GUIDE                    ║
╚═══════════════════════════════════════════════════════════════════════════╝

SETUP & VERIFICATION:
  cuda_activate      - Activate CUDA environment
  cuda_check         - Verify CUDA compiler
  cuda_gpu           - List available GPUs
  cuda_diagnostics   - Full diagnostics report

BUILDING:
  qallow_rebuild_cuda - Rebuild with CUDA support
  qallow_clean       - Clean all build artifacts

EXECUTION:
  qallow_run_cuda    - Run Qallow with CUDA backend
  qallow_test_cuda   - Run CUDA unit tests
  qallow_benchmark   - Run performance benchmarks

EXAMPLES:

  1. Setup and verify:
     $ cuda_activate
     $ cuda_check
     $ cuda_diagnostics

  2. Rebuild project:
     $ qallow_clean
     $ qallow_rebuild_cuda

  3. Run with CUDA:
     $ qallow_run_cuda

  4. Test CUDA:
     $ qallow_test_cuda

  5. Benchmark:
     $ qallow_benchmark

ENVIRONMENT:
  
  Add to ~/.bashrc for permanent setup:
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

TROUBLESHOOTING:

  "nvcc not found":
    $ source ~/.bashrc
    $ cuda_activate

  CUDA compilation errors:
    $ cuda_diagnostics
    $ nvcc --version

  GPU not detected (normal in WSL):
    $ cuda_gpu
    Application will use CPU fallback

For more help, see: CUDA_INSTALLATION_COMPLETE.md
EOF
}

# Show help on source
echo "✓ CUDA Quick Reference Loaded"
echo ""
echo "Available commands:"
echo "  cuda_activate       - Activate CUDA environment"
echo "  cuda_check          - Verify CUDA setup"
echo "  cuda_diagnostics    - Full diagnostics"
echo "  qallow_run_cuda     - Run with CUDA"
echo "  qallow_test_cuda    - Run CUDA tests"
echo "  qallow_benchmark    - Run benchmarks"
echo ""
echo "Type 'cuda_help' for full usage guide"
echo ""
