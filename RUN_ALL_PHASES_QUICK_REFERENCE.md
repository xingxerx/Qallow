# 🚀 Run All Phases - Quick Reference Card

## ⚡ Most Common Commands

```bash
# Run all phases 1-20 once
./run_all_phases.sh

# Run all phases continuously
./run_all_phases.sh --loop

# Run with CUDA
./run_all_phases.sh --build cuda

# Run new phases only (16-20)
./run_all_phases.sh --start-phase 16 --end-phase 20

# Run 5 cycles
./run_all_phases.sh --loop-count 5

# Run with CUDA, 3 cycles
./run_all_phases.sh --build cuda --loop-count 3
```

---

## 📊 Phase Groups

### Original System (Phases 1-15)
```bash
./run_all_phases.sh --end-phase 15
```

### New Quantum System (Phases 16-20)
```bash
./run_all_phases.sh --start-phase 16
```

### Quantum Phases Only (13-20)
```bash
./run_all_phases.sh --start-phase 13
```

### Specific Range
```bash
./run_all_phases.sh --start-phase 13 --end-phase 15
```

---

## 🔧 Build Options

### CPU (Default)
```bash
./run_all_phases.sh --build cpu
```

### CUDA (GPU)
```bash
./run_all_phases.sh --build cuda
```

---

## 🔄 Execution Modes

### Single Pass (Default)
```bash
./run_all_phases.sh
```

### N Cycles
```bash
./run_all_phases.sh --loop-count 5
```

### Continuous (Until Ctrl+C)
```bash
./run_all_phases.sh --loop
```

---

## 📁 Logging

### Default Log Location
```
data/logs/phases_YYYYMMDD_HHMMSS.log
```

### View Latest Log
```bash
tail -f data/logs/phases_*.log
```

### Custom Log Directory
```bash
./run_all_phases.sh --log-dir /custom/path
```

---

## 🎯 Phase Descriptions

| Phase | Name | Type |
|-------|------|------|
| 1-10 | Foundation & Ethics | Core |
| 11-15 | Quantum & AGI | Quantum |
| 16 | Rebellion Simulation | Advanced |
| 17 | Memory Persistence | Advanced |
| 18 | Multiplayer Sync | Advanced |
| 19 | Self-Audit | Advanced |
| 20 | Quantum LoreWeave | Advanced |

---

## ✅ Verification

```bash
# Make executable
chmod +x run_all_phases.sh

# Show help
./run_all_phases.sh --help

# Check logs exist
ls -la data/logs/
```

---

## 🚀 Recommended Workflows

### Development Testing
```bash
./run_all_phases.sh --start-phase 13 --end-phase 20 --loop-count 3
```

### Production Benchmark
```bash
./run_all_phases.sh --build cuda --loop-count 10
```

### Continuous Monitoring
```bash
./run_all_phases.sh --loop
```

### Quick Test
```bash
./run_all_phases.sh --start-phase 16 --end-phase 20
```

---

## 📞 Troubleshooting

### Script not found
```bash
ls -la run_all_phases.sh
chmod +x run_all_phases.sh
```

### Permission denied
```bash
chmod +x run_all_phases.sh
```

### Qallow binary not found
```bash
# Build first
cd build && make
```

### Check logs for errors
```bash
tail -100 data/logs/phases_*.log
```

---

## 🟢 Status: PRODUCTION READY

✅ All 20 phases supported  
✅ Continuous execution  
✅ CUDA acceleration  
✅ Comprehensive logging  

---

**Updated**: 2025-10-27  
**System**: Qallow v2.0

