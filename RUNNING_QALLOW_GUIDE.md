# Running Qallow - Complete Guide

## 🎯 Short Answer: NO ❌

**You do NOT need to run bootstrap every time!**

Bootstrap is a **one-time setup** that:
- ✅ Initializes git submodules
- ✅ Creates Python virtual environment
- ✅ Installs dependencies
- ✅ Builds the project

After bootstrap completes, you only need **2 simple steps** to run the project.

---

## 🚀 After Bootstrap: Running Qallow

### Step 1: Activate Python Environment (10 seconds)
```bash
source .venv/bin/activate
```

### Step 2: Run Qallow (1 second)
```bash
./build/qallow run unified
```

**That's it!** ✅

---

## 📋 Complete Workflow

### First Time (One-time setup)
```bash
# 1. Clone repository
git clone https://github.com/xingxerx/Qallow.git
cd Qallow

# 2. Run bootstrap (5-15 minutes, one time only)
./bootstrap.sh --cuda

# 3. Activate environment
source .venv/bin/activate

# 4. Run project
./build/qallow run unified
```

### Every Time After (10 seconds)
```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run project
./build/qallow run unified
```

---

## 🎯 Common Commands

### Run Main Application
```bash
source .venv/bin/activate
./build/qallow run unified
```

### Run Specific Phase
```bash
source .venv/bin/activate
./build/qallow phase 12 --ticks=8
./build/qallow phase 13 --ticks=8
./build/qallow phase 14 --ticks=8
```

### Run Benchmarks
```bash
source .venv/bin/activate
./build/qallow run bench
```

### Run Tests
```bash
source .venv/bin/activate
bash tests/smoke/test_modules.sh
python3 -m pytest tests/ -v
```

### Run CLI Help
```bash
source .venv/bin/activate
./build/qallow --help
./build/qallow run --help
./build/qallow phase --help
```

---

## 💡 Pro Tips

### Tip 1: Create an Alias
```bash
# Add to ~/.bashrc or ~/.zshrc
alias qallow='source ~/.../Qallow/.venv/bin/activate && ~/.../Qallow/build/qallow'

# Then just use:
qallow run unified
qallow phase 12
```

### Tip 2: Create a Wrapper Script
```bash
# Create run_qallow.sh
#!/bin/bash
cd /home/xing/Qallow
source .venv/bin/activate
./build/qallow "$@"

# Make executable
chmod +x run_qallow.sh

# Use it
./run_qallow.sh run unified
./run_qallow.sh phase 12
```

### Tip 3: Use VS Code Terminal
```bash
# VS Code automatically activates .venv
# Just open terminal and run:
./build/qallow run unified
```

### Tip 4: Keep Terminal Open
```bash
# Activate once, keep terminal open
source .venv/bin/activate

# Then run multiple commands
./build/qallow run unified
./build/qallow phase 12
./build/qallow phase 13
# etc...
```

---

## 🔄 When Do You Need Bootstrap Again?

### You DON'T need bootstrap if:
- ✅ You already ran it once
- ✅ You're just running the project
- ✅ You're running tests
- ✅ You're modifying code

### You DO need bootstrap if:
- ❌ You cloned the repo fresh
- ❌ You deleted `.venv` directory
- ❌ You deleted `build/` directory
- ❌ You want to update dependencies
- ❌ You want to rebuild from scratch

---

## 🛠️ Rebuilding (If Needed)

### Rebuild C/CUDA Only (No Python)
```bash
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build
```

### Rebuild Everything
```bash
# Clean
rm -rf build .venv

# Rebuild
./bootstrap.sh --cuda
source .venv/bin/activate
```

### Update Dependencies Only
```bash
source .venv/bin/activate
pip install -r config/requirements.txt --upgrade
```

---

## 📊 Time Breakdown

| Task | Time | Frequency |
|------|------|-----------|
| Bootstrap (first time) | 5-15 min | Once |
| Activate environment | 10 sec | Every session |
| Run project | 1 sec | Every run |
| **Total per session** | **10 sec** | **Every time** |

---

## 🎯 Quick Reference

### One-Time Setup
```bash
git clone https://github.com/xingxerx/Qallow.git
cd Qallow
./bootstrap.sh --cuda
```

### Every Time You Want to Run
```bash
source .venv/bin/activate
./build/qallow run unified
```

### That's It! ✅

---

## 🚀 Example Workflow

### Day 1: Setup
```bash
# Clone and setup (15 minutes)
git clone https://github.com/xingxerx/Qallow.git
cd Qallow
./bootstrap.sh --cuda

# First run
source .venv/bin/activate
./build/qallow run unified
```

### Day 2: Run Again
```bash
# Just activate and run (10 seconds)
cd Qallow
source .venv/bin/activate
./build/qallow run unified
```

### Day 3: Run Again
```bash
# Same as Day 2
cd Qallow
source .venv/bin/activate
./build/qallow run unified
```

---

## 🔧 Troubleshooting

### Issue: "command not found: qallow"
**Solution**: Activate environment first
```bash
source .venv/bin/activate
./build/qallow --help
```

### Issue: "Python module not found"
**Solution**: Reinstall dependencies
```bash
source .venv/bin/activate
pip install -r config/requirements.txt
```

### Issue: "Build artifacts missing"
**Solution**: Rebuild
```bash
cmake -S . -B build -DQALLOW_ENABLE_CUDA=ON
cmake --build build
```

### Issue: ".venv not found"
**Solution**: Run bootstrap again
```bash
./bootstrap.sh --cuda
source .venv/bin/activate
```

---

## 📚 Related Documentation

- `README.md` - Main documentation
- `docs/BOOTSTRAP_GUIDE.md` - Bootstrap details
- `BOOTSTRAP_STUCK_SOLUTION.md` - If bootstrap hangs
- `START_TESTING_NOW.md` - Testing guide

---

## ✨ Summary

| Question | Answer |
|----------|--------|
| Do I need bootstrap every time? | ❌ NO - only once |
| What do I need every time? | ✅ Activate `.venv` |
| How long does it take? | ⏱️ 10 seconds |
| Can I skip activation? | ❌ No, it's required |
| Can I use it without Python? | ✅ Yes, use `--no-python` |

---

## 🎉 You're Ready!

**After bootstrap completes:**
```bash
source .venv/bin/activate
./build/qallow run unified
```

**That's all you need every time!** 🚀

---

*Last Updated: 2025-11-12*

