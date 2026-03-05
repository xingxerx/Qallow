# Bootstrap Integration Guide

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| **First-time setup** | `./bootstrap.sh` | ~5-10 min |
| **CPU-only build** | `./bootstrap.sh --no-cuda` | ~5-10 min |
| **Skip tests** | `./bootstrap.sh --skip-tests` | ~2-3 min |
| **Rebuild after changes** | `cd build && cmake --build .` | <1 min |
| **Run tests** | `cd build && ctest` | ~10 sec |
| **Download assets** | `python3 scripts/fetch_assets.py` | ~30 sec |

---

## How Bootstrap Fits Into Workflow

### For Developers (Fresh Clone)

```bash
# 1. Clone repository
git clone https://github.com/xingxerx/Qallow.git
cd Qallow

# 2. One-command setup
./bootstrap.sh

# 3. Start developing
source .venv/bin/activate
cd build
cmake --build .  # Incremental rebuild after changes
ctest            # Run tests
```

**Time to first successful build**: ~5-10 minutes on typical hardware

### For CI/CD (GitHub Actions)

```yaml
name: Build & Test

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
        with:
          submodules: recursive
      
      - name: Bootstrap Qallow
        run: |
          chmod +x bootstrap.sh
          ./bootstrap.sh --skip-tests  # Skip on CI since we run ctest separately
      
      - name: Run Tests
        run: cd build && ctest --output-on-failure
      
      - name: Upload Artifacts
        uses: actions/upload-artifact@v2
        with:
          name: binaries
          path: build/qallow*
```

### For Docker/Container Builds

```dockerfile
FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git cmake build-essential python3 python3-venv \
    cuda-toolkit-11-0  # optional, for GPU support

WORKDIR /app

# Clone and bootstrap
RUN git clone https://github.com/xingxerx/Qallow.git . && \
    chmod +x bootstrap.sh && \
    ./bootstrap.sh --skip-tests

# Verify
RUN cd build && ctest --output-on-failure

# Set up entry point
ENTRYPOINT ["./build/qallow"]
CMD ["run", "unified"]
```

---

## Bootstrap Phases Explained

The bootstrap script runs 5 phases:

### [1/5] Git Submodules
```bash
git submodule update --init --recursive
```
- Fetches vendored dependencies (e.g., `third_party/cuda-quantum`)
- One-time operation per clone
- ~5-10 sec on typical network

**What it does**: Brings in external repositories pinned at specific commits.

### [2/5] Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # + dev, gpu, web
```
- Creates isolated Python environment
- Installs all dependencies
- ~2-3 min (depending on packages)

**What it does**: Sets up Python sandbox with reproducible versions.

### [3/5] Assets
```bash
python3 scripts/fetch_assets.py
```
- Downloads optional data files
- Verifies SHA256 hashes
- Caches in `~/.cache/qallow/`
- ~30 sec (first time), instant (cached)

**What it does**: Fetches models, datasets, presets if not already cached.

### [4/5] CMake Build
```bash
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build --parallel $(nproc)
```
- Configures project
- Compiles all targets
- Uses all available cores
- ~1-3 min depending on hardware

**What it does**: Builds C/CUDA binaries from source.

### [5/5] Tests
```bash
cd build && ctest --output-on-failure
```
- Runs all unit tests
- Reports results
- ~10 sec on typical hardware

**What it does**: Verifies build succeeded and all systems functional.

---

## Dependency Chain

Bootstrap ensures this dependency order:

```
Git Submodules
    ↓
Python Environment
    ↓
Assets (optional)
    ↓
CMake Configuration
    ↓
Build
    ↓
Tests
```

Each phase depends on previous phases succeeding.

---

## Caching Strategy

### Git Submodules
- **Cached in**: `.git/modules/`
- **Invalidated**: When `.gitmodules` changes
- **Manual reset**: `git submodule update --init --recursive --force`

### Python Environment
- **Cached in**: `.venv/`
- **Invalidated**: When `requirements*.txt` changes
- **Manual reset**: `rm -rf .venv && ./bootstrap.sh`

### Assets
- **Cached in**: `~/.cache/qallow/` (user-level)
- **Verified by**: SHA256 hash
- **Invalidated**: When hash doesn't match
- **Manual reset**: `python3 scripts/fetch_assets.py --force`

### CMake Build
- **Cached in**: `build/` directory
- **Incremental**: Only recompile changed files
- **Manual reset**: `rm -rf build && ./bootstrap.sh`

---

## Troubleshooting by Phase

### Phase 1: Submodules Failed
```bash
# Check .gitmodules
cat .gitmodules

# Manual fix
git submodule update --init --recursive --force

# Retry bootstrap
./bootstrap.sh
```

### Phase 2: Python Failed
```bash
# Check Python version
python3 --version  # Should be 3.8+

# Check pip
python3 -m pip --version

# Manual fix
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Retry bootstrap
./bootstrap.sh
```

### Phase 3: Assets Failed
```bash
# Assets are optional - build can proceed without them

# Check connectivity
curl -I https://github.com

# Retry assets only
python3 scripts/fetch_assets.py --force

# Retry bootstrap
./bootstrap.sh
```

### Phase 4: CMake Failed
```bash
# Check CMake version
cmake --version  # Should be 3.20+

# Check compiler
gcc --version  # Should be 9.0+

# Manual fix
rm -rf build
mkdir -p build
cd build
cmake -DQALLOW_ENABLE_CUDA=ON ..
cmake --build . --parallel $(nproc)
cd ..

# Retry bootstrap
./bootstrap.sh
```

### Phase 5: Tests Failed
```bash
# Check which tests failed
cd build
ctest --output-on-failure

# Individual test
ctest -R test_temporal_memory -VV

# Check logs
cat CMakeOutput.log
cat CMakeError.log
```

---

## Performance Optimization

### Skip Optional Phases

```bash
# Skip tests (faster first-time setup)
./bootstrap.sh --skip-tests

# Skip Python (if already set up)
./bootstrap.sh --no-python

# Skip CUDA (CPU-only)
./bootstrap.sh --no-cuda

# Combine
./bootstrap.sh --no-cuda --skip-tests
```

### Parallel Build
Bootstrap automatically uses all available cores:
```bash
# On 8-core system, uses 8 jobs
# On 16-core system, uses 16 jobs

# Override with environment variable
NPROC=4 ./bootstrap.sh  # Use only 4 cores
```

### Incremental Rebuilds

After first bootstrap, for faster rebuilds:
```bash
# Don't run full bootstrap, just rebuild
cd build
cmake --build . --parallel $(nproc)
ctest --output-on-failure
```

**Typical times**:
- First bootstrap: 5-10 min
- Incremental rebuild: 10-30 sec
- Full clean rebuild: 2-5 min

---

## CI/CD Best Practices

### GitHub Actions
```yaml
# Cache dependencies for faster CI
- uses: actions/cache@v2
  with:
    path: ~/.cache/qallow/
    key: qallow-assets-${{ hashFiles('scripts/assets.json') }}
    restore-keys: qallow-assets-
```

### Matrix Testing
```yaml
strategy:
  matrix:
    os: [ubuntu-20.04, ubuntu-22.04, macos-latest]
    cuda: [true, false]
    python: ['3.8', '3.9', '3.10']

steps:
  - run: ./bootstrap.sh ${{ matrix.cuda && '--cuda' || '--no-cuda' }}
```

### Artifact Management
```yaml
- uses: actions/upload-artifact@v2
  with:
    name: qallow-build-${{ matrix.os }}-${{ matrix.cuda }}
    path: build/qallow*
    retention-days: 30
```

---

## Monitoring Bootstrap Health

### Add to CI/CD monitoring
```bash
# Track bootstrap time
/usr/bin/time -v ./bootstrap.sh

# Track asset cache hit rate
ls -lh ~/.cache/qallow/

# Verify reproducibility
./bootstrap.sh && md5sum build/qallow
# Run again, should produce same checksum
```

---

## Extending Bootstrap

### Add New Asset
1. Add to `scripts/assets.json`
2. Include name, URL, hash, destination
3. Mark as optional if not required
4. Commit manifest
5. Asset automatically fetched on next bootstrap

### Add New Python Dependency
1. Add to `requirements.txt` (base) or specific `requirements-*.txt`
2. Update `DEPENDENCY_MANIFEST.md`
3. Commit both files
4. Next bootstrap installs automatically

### Add New CMake Target
1. Add to `CMakeLists.txt`
2. Bootstrap automatically builds it
3. Include in `ctest` or manual execution

---

## Summary

**Bootstrap design principles:**
- ✅ **One command**: `./bootstrap.sh` does everything
- ✅ **Reproducible**: Pinned versions, verified hashes
- ✅ **Fast**: Caching, incremental builds, parallel jobs
- ✅ **Robust**: Graceful degradation for optional assets
- ✅ **Observable**: Clear progress, error messages
- ✅ **Extensible**: Easy to add assets, dependencies, targets

**After bootstrap, you have:**
- ✅ All git submodules initialized
- ✅ Python venv with all packages
- ✅ Optional assets cached locally
- ✅ CMake-built binaries
- ✅ All tests passing
- ✅ Ready to develop/deploy
