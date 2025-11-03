# Qallow Project Setup Guide
# Complete installation and setup instructions

## Quick Start (5 minutes)

### 1. Clone & Navigate
```bash
cd /home/xing/qallow/Qallow
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Core dependencies (CPU-only)
pip install -r requirements.txt

# Or all dependencies (includes dev, web, GPU)
pip install -r requirements.txt -r requirements-dev.txt -r requirements-web.txt
```

### 4. Verify Installation
```bash
python3 run_qallow.py
```

## Detailed Setup Instructions

### Step 1: System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config git \
  python3-pip python3-venv python3-dev \
  libssl-dev libffi-dev \
  libsdl2-dev libsdl2-ttf-dev \
  libglib2.0-dev libgl-dev libxrandr-dev
```

#### macOS
```bash
brew install python3 cmake pkg-config sdl2 sdl2_ttf
```

#### Windows
- Install Python 3.11+ from python.org
- Install Git for Windows
- Install Visual Studio Build Tools 2022

### Step 2: Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate      # Linux/macOS
# or
venv\Scripts\activate         # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install Python Packages

```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install development tools
pip install -r requirements-dev.txt

# Optional: Install web framework dependencies
pip install -r requirements-web.txt

# Optional: Install GPU acceleration (requires CUDA 12.0+)
# pip install -r requirements-gpu.txt
```

### Step 4: Project Structure Setup

```bash
# Create necessary directories
mkdir -p data/logs data/quantum_results
mkdir -p /var/qallow  # For telemetry (or use local directory)
chmod 777 /var/qallow

# Or for local setup (recommended)
export QALLOW_LOG_DIR="./data/logs"
export QALLOW_DATA_DIR="./data/quantum_results"
```

## Running the Project

### Option 1: Run Project Overview
```bash
python3 run_qallow.py
```

### Option 2: Run Tests
```bash
python3 test_quantum_complete.py
```

### Option 3: Run Quantum Algorithms
```bash
python3 -m quantum_algorithms.application_runner
```

### Option 4: Build C/C++ Components
```bash
./build.sh
./qallow_unified run
```

## Component-Specific Setup

### Web Server Setup
```bash
cd server/
# Note: Requires Node.js and npm
npm install
npm start
# Server runs on http://localhost:5000
```

### Native App Setup (Rust)
```bash
cd native_app/
cargo build --release
cargo run --release
```

### Quantum Algorithms Setup
```bash
pip install qiskit>=0.43.0 cirq>=1.2.0 pennylane>=0.31.0
python3 alg/main.py run --quick
```

## Environment Variables (Optional)

Create a `.env` file in the project root:

```bash
# Logging
QALLOW_LOG_DIR="./data/logs"
QALLOW_LOG_LEVEL="INFO"

# Data Storage
QALLOW_DATA_DIR="./data/quantum_results"
QALLOW_TELEMETRY_DIR="/var/qallow"

# GPU Settings
QALLOW_USE_GPU=false
QALLOW_CUDA_DEVICE=0

# Server
QALLOW_SERVER_PORT=5000
QALLOW_SERVER_HOST="0.0.0.0"

# Development
DEBUG=false
VERBOSE=true
```

## Verification Checklist

- [ ] Python 3.10+ installed: `python3 --version`
- [ ] pip working: `pip --version`
- [ ] Virtual environment active: `which python3` shows venv path
- [ ] Core packages installed: `python3 -c "import numpy, scipy, qiskit; print('OK')"`
- [ ] Project runs: `python3 run_qallow.py`
- [ ] Tests pass: `python3 test_quantum_complete.py`

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'numpy'
**Solution:**
```bash
pip install --upgrade numpy scipy
# If still fails, install system dependencies first:
sudo apt-get install -y liblapack-dev libblas-dev gfortran
pip install --force-reinstall numpy scipy
```

### Issue: Failed to build qiskit
**Solution:**
```bash
pip install --upgrade setuptools wheel
pip install --upgrade qiskit qiskit-machine-learning
```

### Issue: Permission denied for /var/qallow
**Solution:**
```bash
# Option 1: Create local directory instead
mkdir -p ./data/telemetry
export QALLOW_TELEMETRY_DIR="./data/telemetry"

# Option 2: Use sudo (not recommended)
sudo mkdir -p /var/qallow && sudo chmod 777 /var/qallow
```

### Issue: CUDA not found
**Solution:**
```bash
# Check if CUDA is installed
nvcc --version

# If not, download from:
# https://developer.nvidia.com/cuda-12-0-download

# Install cupy for GPU support
pip install cupy-cuda12x
```

## System Requirements

### Minimum
- OS: Linux (Ubuntu 20.04+), macOS 11+, or Windows 10+
- Python: 3.10 or higher
- RAM: 4GB
- Disk: 5GB free space

### Recommended
- OS: Ubuntu 22.04 LTS
- Python: 3.11 or 3.12
- RAM: 8GB+
- Disk: 20GB free space
- GPU: NVIDIA GPU with CUDA 12.0+

### GPU-Accelerated
- NVIDIA GPU with compute capability 7.0+ (GeForce RTX 2060+)
- CUDA 12.0+
- cuDNN 8.0+

## Next Steps

1. **Run project overview**: `python3 run_qallow.py`
2. **Read documentation**: Check `README.md`, `QUICKSTART.md`
3. **Run tests**: `python3 test_quantum_complete.py`
4. **Explore components**: Check subdirectories (quantum_algorithms/, native_app/, web-app/)
5. **Build C/C++ project**: `./build.sh`

## Support

- Project: https://github.com/xingxerx/Qallow
- Issues: https://github.com/xingxerx/Qallow/issues
- Documentation: See README.md and other .md files

## Version Information

Created: November 2, 2025
Python: 3.12.3
Pip: 24.0
Last Updated: 2025-11-02
