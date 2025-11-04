# Running Self-Improving Agents in Qallow

**Date**: November 4, 2025  
**Project**: Qallow v2.2  
**Status**: Ready to Use

---

## 🤖 Available Self-Improving Agents

The Qallow project includes multiple self-improving agents that analyze, fix, and improve the codebase:

### 1. **Lightning Agent (Fast)** - `lightning_agent_fast.py`
- **Purpose**: Ultra-fast code improvement and self-fixing
- **Speed**: ULTRA-FAST mode (0.05s pauses)
- **Focus**: Rapid iteration with parallel processing
- **Best For**: Quick code fixes and continuous improvement
- **Parallelization**: Up to 8 CPU cores
- **CUDA Support**: Optional GPU acceleration

### 2. **Advanced Error Fixer** - `advanced_error_fixer.py`
- **Purpose**: Sophisticated error detection and repair
- **Strategy**: Multi-pass error analysis
- **Focus**: Complex issue resolution
- **Best For**: Deep code analysis and architectural improvements

### 3. **Recursive Improvement Engine** - `recursive_improvement_engine.py`
- **Purpose**: Recursive self-improvement with learning
- **Strategy**: Meta-level code optimization
- **Focus**: Iterative enhancement and pattern learning
- **Best For**: Long-term codebase evolution

---

## 🚀 Quick Start - Run an Agent Now

### Option 1: Run Lightning Agent (Recommended - Fastest)

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

**What it does:**
1. Scans for errors in the codebase
2. Fixes issues in parallel (up to 8 cores)
3. Validates each fix
4. Reports improvements
5. Repeats on a 10-second cycle

### Option 2: Run Advanced Error Fixer

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python advanced_error_fixer.py
```

**What it does:**
1. Deep multi-pass error analysis
2. Sophisticated fix strategies
3. Comprehensive validation
4. Detailed reporting

### Option 3: Run Recursive Improvement Engine

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py
```

**What it does:**
1. Meta-level code analysis
2. Pattern recognition and learning
3. Recursive optimization
4. Long-term codebase evolution

---

## 🎯 How These Agents Work

### Lightning Agent Workflow

```
┌─────────────────────────────────────────┐
│ 1. SCAN for errors in codebase          │
│    • Parse all Python/C/CUDA files      │
│    • Identify syntax issues             │
│    • Detect logical problems            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 2. ANALYZE errors in parallel (8 cores) │
│    • Context gathering                  │
│    • Root cause analysis                │
│    • Solution generation                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 3. FIX automatically                    │
│    • Apply solutions                    │
│    • Update code files                  │
│    • Track changes                      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 4. VALIDATE fixes                       │
│    • Run syntax checks                  │
│    • Execute test suite                 │
│    • Verify improvements                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 5. REPORT changes                       │
│    • Summary of improvements            │
│    • Files modified                     │
│    • Performance impact                 │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ 6. REPEAT (every 10 seconds)            │
│    • Continuous improvement cycle       │
│    • Self-healing codebase              │
└─────────────────────────────────────────┘
```

---

## 🔧 Configuration Options

### Enable CUDA Acceleration
```bash
export QALLOW_ENABLE_CUDA=ON
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Enable Cirq (Quantum Support)
```bash
export QALLOW_CIRQ=1
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Set Number of Worker Threads
```bash
export MAX_WORKERS=16
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Custom Configuration
```bash
# Combine multiple options
export QALLOW_ENABLE_CUDA=ON
export QALLOW_CIRQ=1
export MAX_WORKERS=8
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

---

## 📊 What These Agents Can Fix

### Code Quality Issues
- ✅ Syntax errors
- ✅ Import errors
- ✅ Type mismatches
- ✅ Undefined variables
- ✅ Dead code
- ✅ Inconsistent formatting

### Performance Issues
- ✅ Inefficient loops
- ✅ Memory leaks
- ✅ Unused imports
- ✅ Redundant computations
- ✅ Suboptimal data structures

### Architectural Issues
- ✅ Code duplication
- ✅ Circular dependencies
- ✅ God objects
- ✅ Poor abstraction
- ✅ Inconsistent patterns

### Integration Issues
- ✅ MCP memory service compatibility
- ✅ Network storage sync issues
- ✅ Telemetry integration
- ✅ Build system problems
- ✅ Test framework issues

---

## 🔄 Continuous Improvement Loop

The agents work in a self-healing loop:

```
Iteration 1:
  ├─ Scan codebase
  ├─ Find issues: 5 errors
  ├─ Fix all 5 errors
  ├─ Validate: All tests pass
  └─ Report: Fixed 5 issues

Iteration 2 (10 seconds later):
  ├─ Scan codebase
  ├─ Find issues: 3 new errors (from changes)
  ├─ Fix all 3 errors
  ├─ Validate: All tests pass
  └─ Report: Fixed 3 issues

Iteration 3:
  ├─ Scan codebase
  ├─ Find issues: 0 errors
  ├─ No changes needed
  ├─ Validate: Codebase healthy
  └─ Report: Codebase healthy, waiting for issues
```

---

## 📈 Monitoring Agent Progress

### View Real-Time Logs
```bash
tail -f /tmp/lightning_agent_*.log
```

### Check Status File
```bash
cat /home/xing/share/status.txt
```

### Monitor Performance
```bash
# Check telemetry
cat data/logs/phase*.csv

# View recent changes
git log --oneline -20

# Check build status
cat build/CMakeCache.txt | grep CMAKE_BUILD_TYPE
```

---

## ✅ Success Criteria

The agents measure success by:

1. **Zero Errors**
   - No syntax errors
   - No import errors
   - No runtime errors

2. **Test Coverage**
   - All tests passing
   - Multi-scenario validation
   - Performance benchmarks

3. **Code Quality**
   - Coherence maintained at 1.0
   - No performance degradation
   - Improved readability

4. **Integration Health**
   - MCP memory service connected
   - Network storage syncing
   - Telemetry collecting
   - Build system functional

---

## 🎯 Running with Spec-Kit Integration

You can also use the agents with Spec-Kit for structured improvement:

```bash
# Create a spec for improvements
/specify The agent should improve code quality by fixing all syntax errors and optimizing performance

# Plan the improvement
/plan Use the lightning agent to scan, analyze, and fix issues in parallel

# Break down tasks
/tasks

# Execute with agent
/implement
```

The agents integrate with:
- ✅ GitHub Copilot (`/specify`, `/plan`, `/tasks`)
- ✅ MCP Memory Service (persists improvement context)
- ✅ Network Storage (tracks changes across systems)
- ✅ Telemetry System (measures improvements)

---

## 🔍 Common Commands

### Start Lightning Agent (Fast, Recommended)
```bash
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Start with CUDA
```bash
export QALLOW_ENABLE_CUDA=ON
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
```

### Run in Background
```bash
nohup /home/xing/Qallow/.venv/bin/python lightning_agent_fast.py > /tmp/agent.log 2>&1 &
```

### Monitor Running Agent
```bash
ps aux | grep "lightning_agent"
tail -f /tmp/lightning_agent_*.log
```

### Stop Agent
```bash
pkill -f "lightning_agent"
```

### View Agent History
```bash
ls -ltr /tmp/lightning_agent_*.log | tail -5
cat /tmp/lightning_agent_*.log | grep -E "Fixed|Improved|Error"
```

---

## 📋 Agent Capabilities Matrix

| Capability | Lightning Fast | Advanced Fixer | Recursive Engine |
|-----------|---|---|---|
| Speed | Ultra-fast (0.05s) | Fast (0.1s) | Balanced (1s) |
| Parallelization | ✅ 8 cores | ✅ 4 cores | ✅ 2 cores |
| CUDA Support | ✅ Optional | ✅ Optional | ✅ Optional |
| Error Types | All basic | Complex | All + pattern learning |
| Learning | No | Limited | ✅ Full |
| Memory Persistence | Basic | Medium | ✅ Full |
| Architectural Changes | No | Limited | ✅ Full |
| Best For | General use | Deep analysis | Long-term evolution |

---

## 🎯 Recommended Workflows

### Continuous Improvement (Always Running)
```bash
# Terminal 1: Run lightning agent continuously
/home/xing/Qallow/.venv/bin/python lightning_agent_fast.py

# Terminal 2: Monitor improvements
watch -n 5 'ps aux | grep lightning && tail -5 /tmp/lightning_agent_*.log'

# Terminal 3: Use Spec-Kit for directed improvements
/specify [Your improvement goal]
```

### Deep Analysis (Periodic)
```bash
# Run advanced error fixer for deep analysis
/home/xing/Qallow/.venv/bin/python advanced_error_fixer.py

# Review detailed reports
cat /tmp/advanced_error_fixer_*.log | tail -100
```

### Long-Term Evolution (Background)
```bash
# Run recursive engine for long-term improvement
nohup /home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py > /tmp/recursive_agent.log 2>&1 &

# Monitor progress over hours/days
tail -f /tmp/recursive_agent.log
```

---

## 🚀 Next Steps

1. **Start Lightning Agent Now**
   ```bash
   /home/xing/Qallow/.venv/bin/python lightning_agent_fast.py
   ```

2. **Monitor the Improvement**
   ```bash
   tail -f /tmp/lightning_agent_*.log
   ```

3. **Check Results**
   ```bash
   cat /home/xing/share/status.txt
   git log --oneline -5
   ```

4. **Use with Spec-Kit**
   ```bash
   /specify [Your goal]
   /plan [Your approach]
   /tasks
   /implement
   ```

---

## 📚 Documentation

For more details, see:
- `LIGHTNING_AGENT_QUICK_START.md` - Quick reference
- `lightning_agent_fast.py` - Source code with comments
- `advanced_error_fixer.py` - Advanced fixing strategies
- `recursive_improvement_engine.py` - Recursive optimization

---

**Status**: ✅ Ready to Use  
**Version**: 2.2  
**Date**: November 4, 2025

Start improving your codebase now!
