# Qallow Quick Start Card

## 🎯 TL;DR

**Bootstrap**: One-time setup (5-15 min)
**Running**: Every time (10 seconds)

---

## 📋 First Time Only

```bash
# 1. Clone
git clone https://github.com/xingxerx/Qallow.git
cd Qallow

# 2. Bootstrap (one time only!)
./bootstrap.sh --cuda

# 3. Activate
source .venv/bin/activate

# 4. Run
./build/qallow run unified
```

**Time**: ~15 minutes (mostly waiting for downloads)

---

## ⚡ Every Time After

```bash
# 1. Activate
source .venv/bin/activate

# 2. Run
./build/qallow run unified
```

**Time**: 10 seconds ✅

---

## 🚀 Common Commands

```bash
# Activate (required every session)
source .venv/bin/activate

# Run main app
./build/qallow run unified

# Run specific phase
./build/qallow phase 12 --ticks=8
./build/qallow phase 13 --ticks=8

# Run benchmarks
./build/qallow run bench

# Run tests
bash tests/smoke/test_modules.sh
python3 -m pytest tests/ -v

# Show help
./build/qallow --help
```

---

## 🔄 When to Re-Bootstrap

| Scenario | Re-Bootstrap? |
|----------|---------------|
| Running project again | ❌ NO |
| Modifying code | ❌ NO |
| Running tests | ❌ NO |
| Fresh clone | ✅ YES |
| Deleted `.venv` | ✅ YES |
| Deleted `build/` | ✅ YES |
| Update dependencies | ✅ YES |

---

## 💡 Pro Tips

### Tip 1: Keep Terminal Open
```bash
source .venv/bin/activate
# Now run multiple commands without re-activating
./build/qallow run unified
./build/qallow phase 12
./build/qallow phase 13
```

### Tip 2: Create Alias
```bash
# Add to ~/.bashrc
alias qallow='source ~/.../Qallow/.venv/bin/activate && ~/.../Qallow/build/qallow'

# Use it
qallow run unified
```

### Tip 3: VS Code
```bash
# VS Code auto-activates .venv in terminal
# Just open terminal and run:
./build/qallow run unified
```

---

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| "command not found" | `source .venv/bin/activate` |
| "Python module not found" | `pip install -r config/requirements.txt` |
| "Build not found" | `cmake -S . -B build && cmake --build build` |
| ".venv not found" | `./bootstrap.sh --cuda` |

---

## 📊 Time Breakdown

| Task | Time | Frequency |
|------|------|-----------|
| Bootstrap | 5-15 min | Once |
| Activate | 10 sec | Every session |
| Run | 1 sec | Every run |

---

## ✅ Checklist

### First Time
- [ ] Clone repository
- [ ] Run bootstrap
- [ ] Activate environment
- [ ] Run project

### Every Time After
- [ ] Activate environment
- [ ] Run project

---

## 🎯 Bootstrap Options

```bash
./bootstrap.sh [OPTIONS]

--cuda              Enable CUDA (default)
--no-cuda           Disable CUDA
--skip-tests        Skip tests after build
--skip-submodules   Skip git submodules
--no-python         Skip Python setup
--help              Show help
```

---

## 📚 Full Guides

- `RUNNING_QALLOW_GUIDE.md` - Complete guide
- `BOOTSTRAP_STUCK_SOLUTION.md` - If bootstrap hangs
- `START_TESTING_NOW.md` - Testing guide
- `README.md` - Main documentation

---

## 🎉 You're Ready!

**First time**: `./bootstrap.sh --cuda`

**Every time after**: `source .venv/bin/activate && ./build/qallow run unified`

---

*Last Updated: 2025-11-12*

