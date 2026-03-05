# Qallow System Setup Requirements
# This file documents all system-level dependencies needed to run the full project

## Ubuntu/Debian Package Dependencies
# Install with: sudo apt-get install -y <package>

### Build Tools & Compilation
- build-essential      # GCC, G++, Make, etc.
- cmake>=3.20         # Build system
- pkg-config          # Compilation flags helper
- git                 # Version control

### Development Libraries
- libpython3-dev      # Python development headers
- python3-pip         # Python package manager
- python3-venv        # Python virtual environments

### Graphics & UI
- libsdl2-dev         # Simple DirectMedia Layer (graphics)
- libsdl2-ttf-dev     # SDL2 font rendering
- libglib2.0-dev      # GLib development
- libgl-dev           # OpenGL development
- libxrandr-dev       # X11 display support

### System Libraries
- libssl-dev          # OpenSSL cryptography
- libffi-dev          # Foreign Function Interface
- zlib1g-dev          # Compression library
- libc-dev            # C library development

### Optional: GPU Support
- cuda-toolkit-12-0   # NVIDIA CUDA compiler
- cudnn               # CUDA Deep Neural Network library
- nvidia-utils        # NVIDIA GPU utilities

### Optional: Rust Development
- rustup              # Rust package manager
- cargo               # Rust build system

### Optional: Node.js (for web components)
- nodejs>=18.0        # Node.js runtime
- npm                 # Node package manager

## Python Virtual Environment Setup (Recommended)

1. Create virtual environment:
   python3 -m venv venv

2. Activate virtual environment:
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

3. Install Python dependencies:
   pip install -r requirements.txt

## Install All Requirements (Quick Setup)

For CPU-only (recommended for testing):
   pip install -r requirements.txt -r requirements-dev.txt -r requirements-web.txt

For GPU acceleration (requires CUDA 12.0+):
   pip install -r requirements.txt -r requirements-gpu.txt -r requirements-dev.txt

## System Package Installation (Ubuntu/Debian)

Run this to install all system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  pkg-config \
  git \
  python3-pip \
  python3-venv \
  python3-dev \
  libpython3-dev \
  libssl-dev \
  libffi-dev \
  libsdl2-dev \
  libsdl2-ttf-dev \
  libglib2.0-dev \
  libgl-dev \
  libxrandr-dev \
  zlib1g-dev \
  libc-dev
```

## Verification

After installation, verify setup with:

```bash
python3 --version          # Should be 3.10+
pip --version              # Should be 24.0+
gcc --version              # Should be 11+
cmake --version            # Should be 3.20+
```

## Troubleshooting

If numpy/scipy installation fails:
  - Install: sudo apt-get install -y liblapack-dev libblas-dev gfortran
  - Reinstall: pip install --upgrade numpy scipy

If compilation fails:
  - Install: sudo apt-get install -y build-essential
  - Check: gcc --version (should be 11+)

If CUDA support fails:
  - Install CUDA 12.0+ from: https://developer.nvidia.com/cuda-downloads
  - Verify: nvcc --version
