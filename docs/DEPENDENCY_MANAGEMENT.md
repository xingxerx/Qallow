# 📦 Dependency Management Guide

**Last Updated**: 2025-10-23  
**Status**: Production Ready

## Overview

Qallow has been designed with minimal dependencies while supporting optional advanced features:

### Core Dependencies (Required)
- CMake ≥ 3.20
- GCC ≥ 11 or Clang ≥ 13
- Make or Ninja
- POSIX-compliant OS (Linux, macOS)

### Python Dependencies (Optional)
- Python ≥ 3.10
- Qiskit (quantum algorithms)
- Cirq (quantum simulation)
- NumPy, SciPy (numerical computing)
- Pandas (data analysis)
- Flask (web dashboard)

### GPU Dependencies (Optional)
- CUDA Toolkit ≥ 12.0
- cuDNN ≥ 8.0
- Nsight Compute (profiling)

### UI Dependencies (Optional)
- SDL2 (desktop UI)
- SDL2_ttf (text rendering)

## Checking Dependencies

### Automated Check

```bash
cd /root/Qallow

# Check all dependencies
bash scripts/check_dependencies.sh

# Auto-install missing packages
bash scripts/check_dependencies.sh --auto-install
```

### Manual Check

```bash
# CMake
cmake --version

# Compiler
gcc --version

# Python
python3 --version

# CUDA (if available)
nvcc --version

# Make/Ninja
make --version
ninja --version
```

## Installation by Platform

### Ubuntu/Debian

```bash
# Core dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    python3-venv

# Optional: CUDA
sudo apt-get install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev

# Optional: UI
sudo apt-get install -y \
    libsdl2-dev \
    libsdl2-ttf-dev
```

### macOS

```bash
# Core dependencies
brew install cmake gcc python3

# Optional: CUDA (requires NVIDIA GPU)
# Download from: https://developer.nvidia.com/cuda-downloads

# Optional: UI
brew install sdl2 sdl2_ttf
```

### CentOS/RHEL

```bash
# Core dependencies
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake \
    python3 \
    python3-pip

# Optional: CUDA
sudo yum install -y \
    cuda-toolkit-12-0 \
    cuda-cudnn-12-0
```

## Python Environment Setup

### Create Virtual Environment

```bash
cd /root/Qallow

# Create venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Install Python Dependencies

```bash
# Core quantum packages
pip install qiskit qiskit-aer cirq numpy scipy

# Data analysis
pip install pandas matplotlib seaborn

# Web dashboard
pip install flask flask-cors

# Development
pip install pytest black flake8 mypy
```

### Install from Requirements

```bash
# UI requirements
pip install -r ui/requirements.txt

# Python requirements
pip install -r python/requirements-dev.txt

# ALG requirements
pip install -r alg/requirements.txt
```

## Docker Deployment

### Build Docker Image

```bash
# Build with all dependencies
docker build -t qallow:latest .

# Build with specific features
docker build \
    --build-arg ENABLE_CUDA=1 \
    --build-arg ENABLE_UI=1 \
    -t qallow:full .
```

### Run Docker Container

```bash
# Basic run
docker run -it qallow:latest

# With GPU support
docker run --gpus all -it qallow:latest

# With volume mounts
docker run -v /root/Qallow/data:/root/Qallow/data \
    -it qallow:latest

# With port forwarding (dashboard)
docker run -p 5000:5000 -it qallow:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Dependency Troubleshooting

### CMake Not Found

```bash
# Install CMake
sudo apt-get install cmake

# Or download from: https://cmake.org/download/
```

### Python Packages Missing

```bash
# Activate venv
source venv/bin/activate

# Install missing package
pip install <package-name>

# Verify installation
python3 -c "import <package>; print(<package>.__version__)"
```

### CUDA Not Available

```bash
# Check CUDA installation
nvcc --version

# If not found, install from:
# https://developer.nvidia.com/cuda-downloads

# Or use CPU-only build
bash scripts/build_all.sh --cpu
```

### SDL2 Not Found

```bash
# Install SDL2
sudo apt-get install libsdl2-dev libsdl2-ttf-dev

# Or skip UI build
cmake -S . -B build -DENABLE_UI=OFF
```

## Dependency Versions

### Tested Versions

| Dependency | Tested Version | Minimum | Maximum |
|------------|----------------|---------|---------|
| CMake | 3.20+ | 3.20 | Latest |
| GCC | 11.0+ | 11.0 | Latest |
| Python | 3.10+ | 3.10 | 3.12 |
| Qiskit | 0.43+ | 0.40 | Latest |
| CUDA | 12.0+ | 12.0 | Latest |
| SDL2 | 2.24+ | 2.20 | Latest |

## Updating Dependencies

### Update Python Packages

```bash
# Update all packages
pip install --upgrade pip

# Update specific package
pip install --upgrade qiskit

# Update from requirements
pip install --upgrade -r requirements.txt
```

### Update System Packages

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get upgrade

# macOS
brew update
brew upgrade

# CentOS/RHEL
sudo yum update
```

## Minimal Installation

For minimal footprint (CPU-only, no UI):

```bash
# Install only core dependencies
sudo apt-get install -y build-essential cmake python3

# Build CPU-only
bash scripts/build_all.sh --cpu

# No Python packages needed for basic phases
```

## Full Installation

For complete feature set:

```bash
# Install all dependencies
bash scripts/check_dependencies.sh --auto-install

# Build with all features
bash scripts/build_all.sh --cuda

# Install all Python packages
pip install -r ui/requirements.txt
pip install -r python/requirements-dev.txt
```

## Dependency Conflicts

### Python Version Conflict

```bash
# If Python 3.10 not available
python3.10 -m venv venv

# Or use pyenv
pyenv install 3.10.0
pyenv local 3.10.0
```

### CUDA Version Conflict

```bash
# Check CUDA version
nvcc --version

# If wrong version, install correct one
# https://developer.nvidia.com/cuda-downloads

# Or use CPU-only build
bash scripts/build_all.sh --cpu
```

### Package Version Conflict

```bash
# Check installed versions
pip list

# Downgrade package
pip install qiskit==0.40.0

# Or use requirements file
pip install -r requirements-pinned.txt
```

## Security

### Verify Package Integrity

```bash
# Check package signatures
pip install --require-hashes -r requirements.txt

# Or use pip-audit
pip install pip-audit
pip-audit
```

### Update Security Patches

```bash
# Check for vulnerabilities
pip-audit

# Update vulnerable packages
pip install --upgrade <vulnerable-package>
```

## Performance Optimization

### Reduce Build Time

```bash
# Use Ninja instead of Make
cmake -S . -B build -GNinja
cmake --build build -j$(nproc)

# Or use ccache
cmake -S . -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### Reduce Runtime Dependencies

```bash
# Use minimal Python environment
pip install --no-deps qiskit

# Or use conda for better dependency resolution
conda create -n qallow python=3.10
conda install -c conda-forge qiskit
```

## Support

For dependency issues:
1. Check this guide
2. Run `bash scripts/check_dependencies.sh`
3. Review error messages
4. Open GitHub issue

---

**Next**: Read `docs/KUBERNETES_DEPLOYMENT_GUIDE.md` for cluster deployment.

