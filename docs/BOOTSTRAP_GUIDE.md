# Qallow Bootstrap Guide

## Quick Start (1 Command)

```bash
git clone https://github.com/xingxerx/Qallow.git
cd Qallow
chmod +x bootstrap.sh
./bootstrap.sh
```

That's it! The bootstrap script will:
1. Initialize all git submodules
2. Create and activate a Python virtual environment
3. Install all Python dependencies
4. Configure and build the CMake project
5. Run verification tests

---

## What is Bootstrap?

The Qallow bootstrap system makes the codebase **self-provisioning** so that a fresh clone automatically:
- Fetches external dependencies (git submodules, libraries)
- Sets up Python environment with all packages
- Downloads optional assets and data files
- Builds C/CUDA binaries
- Verifies everything works with tests

No manual configuration needed!

---

## Bootstrap Options

### Basic Usage
```bash
./bootstrap.sh
```

### With Options
```bash
# Disable CUDA (CPU-only build)
./bootstrap.sh --no-cuda

# Skip running tests after build
./bootstrap.sh --skip-tests

# Don't set up Python (manual setup later)
./bootstrap.sh --no-python

# Combine options
./bootstrap.sh --no-cuda --skip-tests
```

### Help
```bash
./bootstrap.sh --help
```

---

## What Gets Downloaded

### Git Submodules
Automatically initialized:
- `mcp-memory-service/` - Memory management MCP server
- Any other vendored dependencies

### Python Dependencies

**Base** (`requirements.txt`):
- numpy, scipy, matplotlib
- pyyaml, requests

**Development** (`requirements-dev.txt`):
- pytest, pytest-cov
- black, pylint, mypy

**GPU** (`requirements-gpu.txt`):
- torch/tensorflow (ML acceleration)
- cirq (quantum bridge for Phase 11)

**Web** (`requirements-web.txt`):
- flask, werkzeug (telemetry dashboard)

### Assets & Data

Downloaded on demand via `scripts/fetch_assets.py`:
- Quantum bridge demo data
- Pre-trained ethics baseline
- Elasticity tuning parameters
- Harmonic integration presets
- Convergence patterns database

See `scripts/assets.json` for complete list.

---

## Directory Structure After Bootstrap

```
Qallow/
├── .venv/                    # Python virtual environment
├── build/                    # CMake build directory
│   ├── qallow               # Main binary
│   ├── qallow_test_*        # Test binaries
│   └── ...
├── data/
│   ├── assets/              # Downloaded assets
│   ├── logs/                # Telemetry and logs
│   └── telemetry/           # Performance metrics
├── mcp-memory-service/      # Submodule
└── scripts/
    ├── fetch_assets.py      # Asset downloader
    └── assets.json          # Asset manifest
```

---

## Troubleshooting

### Python not found
```bash
# Install Python 3.8+
sudo apt-get install python3 python3-venv  # Ubuntu/Debian
brew install python3                       # macOS

# Or use --no-python
./bootstrap.sh --no-python
```

### CUDA not found
```bash
# Disable CUDA for CPU-only build
./bootstrap.sh --no-cuda

# Or install CUDA 11.0+ from nvidia.com
```

### Submodule fails
```bash
# Force update submodules
git submodule update --init --recursive --force

# Then run bootstrap
./bootstrap.sh
```

### CMake configuration fails
```bash
# Clean build directory
rm -rf build/

# Try again
./bootstrap.sh
```

### Tests fail after bootstrap
```bash
# Run tests manually to see errors
cd build
ctest --output-on-failure
cd ..

# Check build.log for details
cat build/CMakeOutput.log
```

### Asset download fails
```bash
# Skip assets for now (optional)
# They'll be downloaded on first use

# Or download manually
python3 scripts/fetch_assets.py --list
python3 scripts/fetch_assets.py --force
```

---

## Manual Setup (Without Bootstrap)

If you prefer manual control:

```bash
# 1. Init submodules
git submodule update --init --recursive

# 2. Python setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: pip install -r requirements-dev.txt

# 3. CMake build
mkdir -p build
cd build
cmake -DQALLOW_ENABLE_CUDA=ON ..
cmake --build . --parallel $(nproc)
ctest --output-on-failure
cd ..

# 4. Download assets (optional)
python3 scripts/fetch_assets.py
```

---

## Advanced: Customizing Bootstrap

### Skip Python Entirely
```bash
./bootstrap.sh --no-python
```

### Cache Management
```bash
# Force re-download all assets
python3 scripts/fetch_assets.py --force

# Skip using cache
python3 scripts/fetch_assets.py --no-cache

# List available assets
python3 scripts/fetch_assets.py --list
```

### CI/CD Integration

In GitHub Actions:
```yaml
- name: Bootstrap Qallow
  run: |
    chmod +x bootstrap.sh
    ./bootstrap.sh --skip-tests
    
- name: Run Tests
  run: cd build && ctest --output-on-failure
```

---

## Dependency Manifest

See `DEPENDENCY_MANIFEST.md` for:
- Minimum system requirements
- All external library versions
- Optional features and their dependencies
- System-level vs. vendored libraries

---

## Self-Provisioning Design

The bootstrap system uses these principles:

1. **Reproducibility**: Pin all versions in manifests
2. **Caching**: Avoid re-downloading via SHA256 cache
3. **Graceful Degradation**: Optional assets don't block builds
4. **Auditability**: All URLs centralized in `scripts/assets.json`
5. **Offline Support**: Cache in `~/.cache/qallow/` for offline use

---

## Next Steps After Bootstrap

1. **Activate Python environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Run the main binary**:
   ```bash
   ./build/qallow run unified
   ```

3. **Run tests**:
   ```bash
   cd build && ctest
   ```

4. **View documentation**:
   ```bash
   cat README.md
   cat docs/ARCHITECTURE_SPEC.md
   ```

5. **Modify & rebuild**:
   ```bash
   # Make changes to source files
   cd build && cmake --build .
   ```

---

## Contributing

To update bootstrap:
1. Update `scripts/assets.json` if adding new assets
2. Update `requirements*.txt` if adding Python deps
3. Update `DEPENDENCY_MANIFEST.md` for system-level changes
4. Test the bootstrap with a clean clone:
   ```bash
   git clone https://github.com/xingxerx/Qallow.git test-bootstrap
   cd test-bootstrap
   ./bootstrap.sh
   ```

---

## Support

- Issues? Check `build/CMakeOutput.log` and build errors
- Questions? See README.md and docs/
- Feature requests? Open a GitHub issue
