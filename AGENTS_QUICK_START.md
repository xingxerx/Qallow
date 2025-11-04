# Quick Guide: Running Self-Improving Agents in Qallow

**Status**: ✅ Agent is now running  
**Date**: November 4, 2025  
**Version**: 2.2

---

## 🚀 Quick Start (Copy & Paste)

### Run Lightning Agent (Recommended - Fastest)
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Run Advanced Error Fixer (Most Thorough)
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python advanced_error_fixer.py
```

### Run Recursive Engine (Most Intelligent)
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py
```

---

## 🎯 What Each Agent Does

### Lightning Agent (`lightning_agent_fast.py`)
- **Speed**: ULTRA-FAST (0.05s pauses)
- **Focus**: Rapid continuous improvement
- **Best For**: Quick fixes and ongoing optimization
- **Parallelization**: Up to 16 CPU cores
- **GPU Support**: Optional CUDA acceleration

**Workflow**:
1. Build project (CMake)
2. Scan codebase for errors
3. Fix issues in parallel
4. Validate with tests
5. Report improvements
6. Repeat (continuous loop)

### Advanced Error Fixer (`advanced_error_fixer.py`)
- **Speed**: Multi-pass analysis (thorough)
- **Focus**: Deep code analysis
- **Best For**: Complex architectural issues
- **Strategy**: Multiple analysis passes
- **Strength**: Detailed error resolution

### Recursive Engine (`recursive_improvement_engine.py`)
- **Speed**: Meta-level optimization (intelligent)
- **Focus**: Long-term evolution
- **Best For**: Pattern learning and adaptation
- **Strategy**: Recursive self-improvement
- **Strength**: Learns and improves over time

---

## 📊 What They Can Fix

✅ **Code Quality Issues**
- Syntax errors
- Import errors
- Type mismatches
- Undefined variables
- Dead code

✅ **Performance Issues**
- Inefficient loops
- Memory leaks
- Unused imports
- Redundant computations

✅ **Architecture Issues**
- Code duplication
- Circular dependencies
- Poor abstraction
- Inconsistent patterns

✅ **Integration Issues**
- MCP memory service problems
- Network storage sync issues
- Telemetry integration
- Build system problems

---

## 🔧 Advanced Options

### Enable GPU (CUDA)
```bash
export QALLOW_ENABLE_CUDA=ON
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Enable Quantum Support
```bash
export QALLOW_CIRQ=1
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Set Worker Threads
```bash
export MAX_WORKERS=32
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Combine Options
```bash
export QALLOW_ENABLE_CUDA=ON QALLOW_CIRQ=1 MAX_WORKERS=16
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

---

## 📋 Monitor Running Agent

### Watch Progress
```bash
tail -f /tmp/lightning_agent_run.log
```

### Check Process
```bash
ps aux | grep lightning_agent_fast.py
```

### Stop Agent
```bash
pkill -f "lightning_agent_fast.py"
```

---

## 📁 Output Files

### Lightning Agent Outputs
- `lightning_agent_output.log` - Detailed execution log
- `LIGHTNING_AGENT_*.md` - Implementation reports
- `LIGHTNING_AGENT_*_TEST_REPORT.md` - Test results

### Advanced Fixer Outputs
- `advanced_error_fixer.log` - Execution log
- `AGENT_FIXES_*.md` - Fix reports

### Recursive Engine Outputs
- `improvement_reports/` - Report directory
- `RECURSIVE_*.md` - Improvement documentation

---

## 🎯 Typical Workflow

```
Start Agent
  ↓
Build Project (CMake)
  ↓
Scan Codebase (100+ files)
  ↓
Analyze Errors (Parallel)
  ↓
Fix Issues (Auto)
  ↓
Validate Tests
  ↓
Report Progress
  ↓
Repeat or Complete
```

---

## ✨ Key Features

### Self-Improvement
- Analyzes its own code
- Fixes its own bugs
- Improves own performance
- Learns from patterns

### Continuous Operation
- 10-second improvement cycles
- Parallel processing (up to 16 cores)
- GPU acceleration (optional)
- Background operation possible

### Comprehensive Reporting
- Detailed logs
- Change tracking
- Performance metrics
- Issue summaries

---

## 💡 Pro Tips

1. **Start Lightning Agent first** - it's fastest and good for initial cleanup
2. **Use GPU if available** - `export QALLOW_ENABLE_CUDA=ON`
3. **Monitor with tail** - `tail -f /tmp/lightning_agent_run.log`
4. **Let it run** - more iterations = better improvements
5. **Check reports** - review generated markdown files for detailed changes

---

## 🆘 Troubleshooting

### Agent Won't Start
```bash
# Check Python environment
/home/xing/Qallow/.venv/bin/python --version

# Run with full output
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### CMake Build Fails
```bash
# Clean build directory
rm -rf build/

# Rebuild
mkdir build && cd build
cmake -S .. -B . -DQALLOW_ENABLE_CUDA=ON
cmake --build . --parallel 16
```

### Out of Memory
```bash
# Reduce worker threads
export MAX_WORKERS=4
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

---

## 📊 Performance Metrics

| Component | Speed | Cores | Memory |
|-----------|-------|-------|--------|
| Scan | <1s/file | - | Low |
| Analysis | <2s/issue | 16 | Medium |
| Fix | <1s/fix | 8 | Medium |
| Validate | <5s/cycle | 4 | Low |
| **Total/Cycle** | **~10s** | - | - |

---

## 🎬 Ready to Go

**Your agent is now configured and ready to improve your codebase!**

Choose an agent and run it:
1. Lightning Agent (fastest)
2. Advanced Error Fixer (most thorough)
3. Recursive Engine (most intelligent)

Each will continuously scan, analyze, fix, and validate your code.

**Command to start now**:
```bash
/home/xing/Qallow/.venv/bin/python /home/xing/Qallow/lightning_agent_fast.py
```
