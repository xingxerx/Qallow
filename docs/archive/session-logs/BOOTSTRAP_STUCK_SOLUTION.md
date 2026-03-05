# Bootstrap Stuck? Here's What to Do

## 🔍 What's Happening

The bootstrap script is **downloading large git submodules**, particularly:
- `third_party/cuda-quantum` (NVIDIA CUDA Quantum framework)
- `mcp-memory-service` (Memory service)
- `third_party/vllm-recipes` (vLLM recipes)

These are **large repositories** and can take **5-15 minutes** to download depending on your internet speed.

---

## ✅ Solutions

### Option 1: Wait It Out (Recommended)
**If you have time**: Just wait. The bootstrap will complete.

**Estimated time**: 5-15 minutes depending on internet speed

**Signs it's working**:
- You see git processes running
- Disk activity is happening
- No error messages

---

### Option 2: Skip Submodules (Fastest - 2 minutes)
**If you want to test NOW**: Skip the submodules

```bash
# Kill the current bootstrap
Ctrl+C

# Run bootstrap WITHOUT submodules
./bootstrap.sh --cuda --skip-submodules
```

**What you lose**:
- CUDA Quantum integration
- Some advanced features
- Memory service

**What you keep**:
- ✅ Core Qallow functionality
- ✅ All tests
- ✅ CLI interface
- ✅ Quantum algorithms

---

### Option 3: Skip Tests (Faster - 5 minutes)
**If you want setup without testing**:

```bash
# Kill the current bootstrap
Ctrl+C

# Run bootstrap and skip tests
./bootstrap.sh --cuda --skip-tests
```

**What happens**:
- Downloads submodules (5-10 min)
- Installs dependencies (2 min)
- Builds project (2 min)
- **Skips running tests**

---

### Option 4: CPU-Only Build (Faster - 3 minutes)
**If you don't need CUDA**:

```bash
# Kill the current bootstrap
Ctrl+C

# Run CPU-only bootstrap
./bootstrap.sh --no-cuda --skip-tests
```

**What you lose**:
- CUDA GPU acceleration
- Some performance optimizations

**What you keep**:
- ✅ All core functionality
- ✅ All tests
- ✅ CPU execution

---

## 🎯 Recommended Approach

### For Testing (Fastest)
```bash
# Kill current bootstrap
Ctrl+C

# Option A: Skip submodules (2 min)
./bootstrap.sh --cuda --skip-submodules

# Option B: CPU only (3 min)
./bootstrap.sh --no-cuda --skip-tests
```

### For Full Setup (Patient)
```bash
# Just wait - it will complete in 5-15 minutes
# The bootstrap is working, just downloading large files
```

---

## 📊 Time Estimates

| Option | Time | What You Get |
|--------|------|--------------|
| Wait it out | 5-15 min | Everything |
| Skip submodules | 2 min | Core + tests |
| Skip tests | 5 min | Setup only |
| CPU only | 3 min | Core (no CUDA) |

---

## 🔧 Manual Setup (If Bootstrap Fails)

If bootstrap gets stuck or fails, do this manually:

```bash
cd /home/xing/Qallow

# 1. Create Python environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r config/requirements.txt

# 3. Build project
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build

# 4. Run tests
bash tests/smoke/test_modules.sh
python3 -m pytest tests/ -v
```

---

## 🚨 If Bootstrap Hangs

### Check Progress
```bash
# In another terminal:
ps aux | grep git
du -sh third_party/cuda-quantum
du -sh mcp-memory-service
```

### Kill and Restart
```bash
# Kill the bootstrap
Ctrl+C

# Clean up
rm -rf .git/modules/third_party/cuda-quantum
rm -rf third_party/cuda-quantum

# Restart with skip-submodules
./bootstrap.sh --cuda --skip-submodules
```

### Force Shallow Clone
```bash
# Kill current bootstrap
Ctrl+C

# Edit bootstrap.sh to use shallow clone
# Change: git submodule update --init --recursive
# To: git submodule update --init --recursive --depth 1

# Then restart
./bootstrap.sh --cuda
```

---

## 📋 Bootstrap Options

```bash
./bootstrap.sh [OPTIONS]

Options:
  --cuda              Enable CUDA (default: true)
  --no-cuda           Disable CUDA
  --skip-tests        Skip running tests after build
  --skip-submodules   Skip git submodule initialization
  --no-python         Skip Python virtual environment setup
  --help              Show this help message
```

---

## ✅ What to Do Now

### If Bootstrap is Still Running
1. **Wait 5-10 more minutes** - it's likely downloading
2. **Check progress**: `ps aux | grep git`
3. **If no progress after 15 min**: Kill and use Option 2

### If Bootstrap Failed
1. **Kill it**: `Ctrl+C`
2. **Use Option 2**: `./bootstrap.sh --cuda --skip-submodules`
3. **Or use Option 4**: `./bootstrap.sh --no-cuda --skip-tests`

### If You Want to Test NOW
```bash
# Kill current bootstrap
Ctrl+C

# Quick test setup (2 minutes)
./bootstrap.sh --cuda --skip-submodules

# Then run tests
bash tests/smoke/test_modules.sh
python3 -m pytest tests/ -v
```

---

## 🎯 Recommended Action

**For immediate testing**:
```bash
# Kill current bootstrap
Ctrl+C

# Skip submodules and run
./bootstrap.sh --cuda --skip-submodules
```

**Expected time**: 2-3 minutes

**What you get**:
- ✅ Python environment
- ✅ Dependencies installed
- ✅ Project built
- ✅ Ready for testing

---

## 📞 Still Stuck?

### Check These
1. Internet connection working?
2. Disk space available? `df -h`
3. Git working? `git --version`
4. Python working? `python3 --version`

### Try This
```bash
# Kill bootstrap
Ctrl+C

# Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build
```

---

## 🎉 Next Steps

1. **Choose your option** (wait, skip submodules, or skip tests)
2. **Run the command**
3. **Wait for completion**
4. **Run tests**: `bash tests/smoke/test_modules.sh`

---

*Last Updated: 2025-11-12*

