# Self-Improving Agents - Quick Start Guide

**Project**: Qallow v2.2  
**Date**: November 4, 2025  
**Status**: ✅ Ready to Run

---

## 🤖 What Are These Agents?

Your project has three powerful self-improving agents:

1. **AgentLightning Runner** - Fast, readable code fixer
2. **Recursive Improvement Engine** - Continuous optimization loop
3. **Advanced Error Fixer** - Deep analysis and fixing

Together, they form a system that can:
- Detect errors automatically
- Fix them without human intervention
- Track improvements across iterations
- Optimize performance autonomously
- Learn from previous fixes

---

## 🚀 Quick Start (Choose One)

### ⚡ Option 1: AgentLightning Runner (Recommended for First Run)

**What**: Fast error fixer with visible output  
**Time**: 5-15 minutes  
**Best for**: Quick error fixing and code validation

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python agentlightning_runner.py
```

**What it does**:
1. Builds the entire project
2. Captures all compilation/runtime errors
3. Analyzes error context
4. Auto-fixes errors with intelligent patches
5. Validates each fix works
6. Shows before/after code changes
7. Generates improvement report

**Output**:
- Console log with all changes
- `lightning_agent_output.log` file
- Improvements applied to codebase
- Summary statistics

---

### 🔄 Option 2: Recursive Improvement Engine (Continuous Loop)

**What**: Self-improving loop that iterates autonomously  
**Time**: Runs continuously (configurable)  
**Best for**: Long-term optimization and metric tracking

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py
```

**What it does**:
1. Run 1: Builds and collects baseline metrics
2. Analyzes results (errors, performance, coherence)
3. Applies optimizations
4. Run 2: Rebuilds with improvements
5. Compares metrics (Run 1 vs Run 2)
6. Applies more optimizations
7. Repeats continuously (or N iterations)

**Output**:
- `improvement_reports/` directory with detailed analysis
- Metrics CSV files in `data/logs/`
- Iteration tracking and progress
- Status updates to `/home/xing/share/status.txt`

**Configuration** (optional):
Edit `config/agent.yaml` to set:
```yaml
max_iterations: 5          # Stop after 5 iterations
error_threshold: 10        # Fail if > 10 errors
coherence_target: 1.0      # Maintain perfect coherence
parallel_workers: 8        # Use 8 CPU cores
cuda_enabled: true         # Use GPU acceleration
auto_commit: true          # Auto-commit improvements
```

---

### 🔧 Option 3: Advanced Error Fixer (Deep Analysis)

**What**: Root-cause analysis and intelligent fixing  
**Time**: 10-20 minutes  
**Best for**: Complex, hard-to-debug issues

```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python advanced_error_fixer.py
```

**What it does**:
1. Parses all errors with full context
2. Builds dependency graph
3. Resolves symbol issues
4. Cross-references files
5. Generates intelligent patches
6. Multi-pass fixing (handles cascading errors)
7. Validates the entire system

---

### 📜 Option 4: Using Build Scripts

Pre-configured shell scripts:

```bash
# Run AgentLightning Runner with improvement
./scripts/run_with_improvement.sh

# Run recursive improvement loop
./scripts/run_recursive_improvement.sh

# Generate improvement report only
./scripts/generate_improvement_report.sh
```

---

### 🤖 Option 5: GitHub Copilot Agent (Modern, Spec-Driven)

Since you have spec-kit installed, you can use the modern approach:

```
In Copilot Chat (Ctrl+Shift+I):

/specify Improve code quality and fix all compilation errors
/plan Use AgentLightning Runner for error fixing, recursive improvement for optimization
/tasks
/implement
```

Benefits:
- MCP memory persists context across sessions
- Copilot tracks all improvements
- Creates detailed spec documentation
- Auto-validates against project principles (CONSTITUTION.md)

---

## 📊 Understanding Agent Output

### AgentLightning Runner Output

```
[13:45:22] INFO: Building Qallow project...
[13:45:45] INFO: Build complete. Found 3 errors.
[13:46:01] ERROR: File1.c line 42: 'undefined reference to foo'
[13:46:15] INFO: Fixing... [BEFORE CODE]
[13:46:30] INFO: Applied patch... [AFTER CODE]
[13:46:45] INFO: Validation: ✅ Fix works!
...
[14:00:00] INFO: All 3 errors fixed!
[14:00:15] INFO: Report: lightning_agent_output.log
```

### Recursive Improvement Output

```
Iteration 1:
  - Build time: 45.2s
  - Coherence: 0.95
  - Errors: 5
  - Phase performance: avg 120ms
  
Iteration 2 (with fixes):
  - Build time: 44.8s (+0.4s improvement)
  - Coherence: 0.98
  - Errors: 2
  - Phase performance: avg 118ms (+2ms improvement)
  
...continuing iterations...
```

---

## 🎯 Recommended Workflow

### For First-Time Use:

1. **Start with AgentLightning Runner**
   ```bash
   /home/xing/Qallow/.venv/bin/python agentlightning_runner.py
   ```
   - See what errors exist
   - Understand the fix patterns
   - Validate fixes work

2. **Review the Changes**
   ```bash
   git log --oneline -10
   cat lightning_agent_output.log
   ```

3. **Run Project Once to Verify**
   ```bash
   /home/xing/Qallow/.venv/bin/python .github/workflows/Driver.py
   ```

4. **If All Good, Start Recursive Improvement**
   ```bash
   /home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py
   ```

### For Continuous Improvement:

1. **Background Process** (run overnight)
   ```bash
   nohup /home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py > /tmp/agent_improvement.log 2>&1 &
   ```

2. **Monitor Progress**
   ```bash
   tail -f /tmp/agent_improvement.log
   cat /home/xing/share/status.txt
   ```

3. **Review Results in Morning**
   ```bash
   cat improvement_reports/improvement_latest.md
   ```

---

## 📈 What Gets Tracked

### Metrics
- **Coherence**: State tracking perfection (target: 1.0)
- **Ethics Score**: Aggregate of ethics calculations
- **Phase Stability**: Deviation from baseline
- **Execution Time**: Performance per phase
- **Error Count**: Total errors remaining
- **Build Time**: Compilation speed

### Files Generated
- `improvement_reports/` - Detailed analysis
- `data/logs/` - CSV telemetry files
- `lightning_agent_output.log` - Agent changes
- `/home/xing/share/status.txt` - Live status
- Git commits - Auto-tracked changes

---

## 🔍 Monitoring and Debugging

### View Current Status
```bash
cat /home/xing/share/status.txt
```

### View Recent Agent Changes
```bash
git log --oneline -20
git show HEAD
```

### View Improvement Metrics
```bash
ls improvement_reports/
cat improvement_reports/improvement_latest.md
```

### Monitor Agent in Real-Time
```bash
tail -f lightning_agent_daemon.log
tail -f /tmp/agent_improvement.log
```

### Stop Running Agent
```bash
pkill -f agentlightning_runner.py
pkill -f recursive_improvement_engine.py
```

---

## 🛠️ Configuration

Edit `config/agent.yaml`:

```yaml
# Maximum iterations before stopping
max_iterations: 10

# Fail if error count exceeds this
error_threshold: 50

# Target coherence (1.0 = perfect)
coherence_target: 1.0

# Target execution time (seconds)
performance_goal: 120

# Number of parallel workers
parallel_workers: 8

# Enable GPU acceleration
cuda_enabled: true

# Auto-commit improvements to git
auto_commit: true

# Improvement strategy: default, aggressive, conservative
improvement_strategy: default

# Pause between iterations (seconds)
iteration_pause: 30
```

---

## 💡 Pro Tips

1. **Start with AgentLightning Runner**: Quick way to understand what's broken
2. **Use Recursive for Overnight Runs**: Let it improve while you sleep
3. **Monitor Status File**: Live updates to Windows (Z:\status.txt)
4. **Review Git Commits**: See exactly what changed
5. **Use MCP Memory**: Copilot remembers context across sessions
6. **Check Metrics Trending**: Look at CSV files for performance trends

---

## ✨ What Makes These Agents Smart

### Error Pattern Recognition
- Learns common error types
- Groups similar errors
- Applies pattern-based fixes

### Contextual Analysis
- Reads surrounding code
- Understands dependencies
- Makes informed fixes

### Validation Loop
- Rebuilds after each fix
- Validates fix worked
- Reverts if validation fails

### Metric Tracking
- Tracks performance before/after
- Identifies regressions
- Optimizes for coherence/speed

### Autonomous Operation
- Requires no human input (can run forever)
- Auto-commits changes
- Generates progress reports

---

## 🚀 Running Now

### Quick Test (2 min)
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python agentlightning_runner.py --iterations 1
```

### Full Improvement (15 min)
```bash
cd /home/xing/Qallow
/home/xing/Qallow/.venv/bin/python agentlightning_runner.py
```

### Long-Running (continuous)
```bash
cd /home/xing/Qallow
nohup /home/xing/Qallow/.venv/bin/python recursive_improvement_engine.py &
```

---

## 📚 Next Steps

1. **Run AgentLightning Runner First**: See what it can fix
2. **Review Changes**: Look at git log and diffs
3. **Run Recursive Loop**: Let it optimize overnight
4. **Monitor Metrics**: Check improvement_reports/
5. **Use Copilot Integration**: Combine with /specify /plan /tasks /implement

---

**Status**: ✅ Ready to Run  
**Version**: 2.2  
**Last Updated**: November 4, 2025
