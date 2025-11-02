# Qallow Project Requirements & Installation Guide

## Overview

This document describes all the requirements and installation files for the Qallow quantum-photonic computing platform.

## Requirements Files

### `requirements.txt` - Core Dependencies (23 packages)
**Purpose:** Essential packages for running Qallow
**Install:** `pip install -r requirements.txt`

**Included:**
- **Scientific Computing:** numpy, scipy, pandas
- **Quantum Frameworks:** qiskit, cirq, pennylane
- **Deep Learning:** tensorflow, torch, scikit-learn
- **Web APIs:** requests, fastapi, uvicorn
- **Data Formats:** pyyaml, python-dotenv, json5
- **Visualization:** matplotlib, plotly, seaborn
- **Utilities:** click, tqdm, Pillow

**Total Size:** ~2-3 GB (with all dependencies)

### `requirements-dev.txt` - Development Tools (15 packages)
**Purpose:** Testing, code quality, and documentation tools
**Install:** `pip install -r requirements-dev.txt`

**Included:**
- **Testing:** pytest, pytest-cov, pytest-asyncio
- **Code Quality:** black, flake8, pylint, mypy
- **Documentation:** sphinx, sphinx-rtd-theme
- **Debugging:** ipython, ipdb, memory-profiler
- **Build Tools:** build, twine, setuptools

**When Needed:**
- Running unit tests
- Contributing to the project
- Generating documentation
- Debugging issues

### `requirements-gpu.txt` - GPU Acceleration (8 packages)
**Purpose:** NVIDIA GPU support for accelerated computing
**Install:** `pip install -r requirements-gpu.txt`
**Prerequisites:** CUDA 12.0 or higher installed

**Included:**
- **GPU Computing:** cupy, numba, pycuda
- **GPU-Accelerated ML:** tensorflow-gpu, torch-cuda, jax[cuda12]
- **GPU Monitoring:** nvidia-ml-py3, py3nvml
- **Distributed Computing:** ray, dask, distributed

**When Needed:**
- Running on NVIDIA GPUs
- Accelerating quantum simulations
- Parallel training of models
- Large-scale computations

### `requirements-web.txt` - Web Framework (18 packages)
**Purpose:** Web server and UI dependencies
**Install:** `pip install -r requirements-web.txt`

**Included:**
- **Web Frameworks:** django, flask, fastapi
- **Async:** websockets, python-socketio
- **API:** pydantic, python-multipart
- **UI:** streamlit, dash, plotly-dash
- **Server:** gunicorn, python-engineio
- **Security:** python-jose, passlib, cryptography

**When Needed:**
- Running web interface
- Building REST APIs
- Real-time data visualization
- Web-based controls

## System Requirements File

### `SYSTEM_REQUIREMENTS.md`
**Purpose:** Lists all system-level (OS) dependencies

**Includes:**
- Build tools (gcc, cmake, make)
- Python development headers
- Graphics libraries (SDL2, OpenGL)
- System libraries (SSL, FFI, compression)
- GPU drivers (optional, for CUDA)

**Total Packages:** ~30 (Ubuntu/Debian)

## Setup Guides

### `SETUP_GUIDE.md`
**Purpose:** Comprehensive setup instructions
**Contents:**
- Quick start (5 minutes)
- Step-by-step detailed setup
- Component-specific setup
- Environment variables
- Verification checklist
- Troubleshooting

### `setup.sh` (Linux/macOS)
**Purpose:** Automated installation script
**Usage:** `bash setup.sh`
**What it does:**
- Detects OS (Linux/macOS/Windows)
- Installs system dependencies
- Creates virtual environment
- Installs Python packages
- Creates directories
- Verifies installation

### `setup.bat` (Windows)
**Purpose:** Automated installation for Windows
**Usage:** Double-click `setup.bat` or run in Command Prompt
**What it does:**
- Checks Python installation
- Creates virtual environment
- Installs Python packages
- Creates directories
- Interactive prompts

## Installation Quick Start

### For Linux/macOS (Automated)
```bash
bash setup.sh
```

### For Windows (Automated)
```cmd
setup.bat
```

### Manual Installation
```bash
# 1. Install system dependencies
sudo apt-get install build-essential cmake python3-pip python3-venv  # Linux
# or
brew install python3 cmake  # macOS

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# 3. Install Python packages
pip install -r requirements.txt

# Optional: Install additional packages
pip install -r requirements-dev.txt    # Development tools
pip install -r requirements-web.txt    # Web framework
pip install -r requirements-gpu.txt    # GPU support (requires CUDA)
```

## Installation Scenarios

### Scenario 1: Testing on CPU (Minimal Setup)
```bash
pip install -r requirements.txt
python3 run_qallow.py
```
**Time:** 5 minutes | **Disk:** 2-3 GB

### Scenario 2: Development Setup
```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 test_quantum_complete.py
pytest tests/
```
**Time:** 10 minutes | **Disk:** 3-4 GB

### Scenario 3: Full Web Stack
```bash
pip install -r requirements.txt -r requirements-web.txt
python3 server/server.py
```
**Time:** 15 minutes | **Disk:** 4-5 GB

### Scenario 4: GPU-Accelerated (Advanced)
```bash
# First ensure CUDA 12.0+ is installed
nvcc --version

# Then install
pip install -r requirements.txt -r requirements-gpu.txt
python3 run_qallow.py --use-gpu
```
**Time:** 20 minutes | **Disk:** 5-8 GB | **Requires:** CUDA 12.0+

## Dependency Summary

| Category | Purpose | Packages | Size |
|----------|---------|----------|------|
| Core | Essential quantum computing | 23 | 2-3 GB |
| Dev | Testing & quality tools | 15 | 500 MB |
| Web | Web framework & UI | 18 | 1-2 GB |
| GPU | NVIDIA acceleration | 8 | 2-3 GB |
| System | OS-level tools | ~30 | 1-2 GB |
| **Total** | **Complete setup** | **~94** | **~10 GB** |

## Verification Commands

After installation, verify with:

```bash
# Python and pip
python3 --version          # Should be 3.10+
pip --version              # Should be 24.0+

# System tools
gcc --version              # Should be 11+
cmake --version            # Should be 3.20+
git --version              # Should be 2.0+

# Python packages
python3 -c "import numpy, scipy, qiskit; print('Core packages OK')"
python3 -c "import pytest, black, sphinx; print('Dev tools OK')"  # If installed
python3 -c "import fastapi, flask, streamlit; print('Web tools OK')"  # If installed
python3 -c "import cupy, numba; print('GPU support OK')"  # If CUDA installed

# Project test
python3 run_qallow.py
python3 test_quantum_complete.py
```

## Troubleshooting

### Issue: pip not found
**Solution:**
```bash
sudo apt-get install -y python3-pip
```

### Issue: scipy fails to install
**Solution:**
```bash
sudo apt-get install -y liblapack-dev libblas-dev gfortran
pip install --force-reinstall scipy
```

### Issue: qiskit installation hangs
**Solution:**
```bash
pip install --upgrade setuptools wheel
pip install --timeout 3600 qiskit
```

### Issue: CUDA not found
**Solution:**
```bash
# Check CUDA installation
nvcc --version

# If not installed, download from:
# https://developer.nvidia.com/cuda-12-0-download

# Verify after installation
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Issue: Virtual environment not activating
**Solution:**
```bash
# Linux/macOS
source venv/bin/activate

# Windows (PowerShell - if cmd doesn't work)
venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat
```

## Updating Requirements

To update to latest versions:

```bash
# Update all core packages
pip install --upgrade -r requirements.txt

# Freeze current versions for reproducibility
pip freeze > requirements-locked.txt
```

## Python Virtual Environment (Recommended)

Using a virtual environment isolates project dependencies:

```bash
# Create
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Delete if needed
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

## Environment Variables

Create a `.env` file for configuration:

```bash
# Python path
PYTHONPATH=/home/xing/qallow/Qallow

# Qallow configuration
QALLOW_LOG_DIR=./data/logs
QALLOW_DATA_DIR=./data/quantum_results
QALLOW_USE_GPU=false
QALLOW_CUDA_DEVICE=0

# Development
DEBUG=false
VERBOSE=true
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
log_dir = os.getenv('QALLOW_LOG_DIR', './data/logs')
```

## Next Steps

1. **Choose installation method:**
   - Automated: Run `setup.sh` or `setup.bat`
   - Manual: Follow `SETUP_GUIDE.md`

2. **Verify installation:**
   - Run `python3 run_qallow.py`
   - Run `python3 test_quantum_complete.py`

3. **Read documentation:**
   - `README.md` - Project overview
   - `QUICKSTART.md` - Quick start guide
   - `SYSTEM_REQUIREMENTS.md` - Detailed requirements

4. **Start using Qallow:**
   - Try quantum algorithms
   - Run web interface
   - Build C/C++ components
   - Explore documentation

## Support

For issues:
1. Check `SETUP_GUIDE.md` troubleshooting section
2. Review error messages carefully
3. Check GitHub issues: https://github.com/xingxerx/Qallow/issues
4. See system requirements in `SYSTEM_REQUIREMENTS.md`

## Version Information

- **Created:** November 2, 2025
- **Python:** 3.10+ (tested with 3.12)
- **Pip:** 24.0+
- **Platform:** Linux, macOS, Windows
- **License:** MIT

---

**Ready to start?** Choose your installation method and follow the guide!
