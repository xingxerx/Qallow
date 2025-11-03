# Recursive Improvement Engine for Qallow

## Overview

This system provides **automated, recursive improvement** for the Qallow quantum-AGI platform. Each time you run it, the system:

1. **Builds** the project with CUDA acceleration
2. **Executes** Qallow phases (12-15) with telemetry capture
3. **Detects** errors and issues
4. **Fixes** problems automatically
5. **Optimizes** using Agent Lightning RL training
6. **Repeats** - each iteration improves the project

## Quick Start

### Option 1: Simple Run (Recommended)

```bash
cd /home/xing/Qallow
./run_with_improvement.sh 10 120 cuda
```

**Parameters:**
- `10` - Number of iterations (default)
- `120` - Ticks per phase (default)
- `cuda` - Use CUDA acceleration (or `cpu` for CPU-only)

### Option 2: Advanced Control

```bash
./run_recursive_improvement.sh --iterations 15 --ticks 150 --cuda
```

### Option 3: Direct Python Execution

```bash
python3 recursive_improvement_engine.py --iterations 10 --ticks 120
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ITERATION LOOP (Recursive Improvement)                │
│                                                         │
│  1. BUILD PHASE                                         │
│     - CMake configure with CUDA support               │
│     - Parallel compilation                            │
│     - Extract build errors                            │
│                                                         │
│  2. EXECUTION PHASE                                     │
│     - Run unified Qallow phases (12-15)               │
│     - CUDA kernel execution                           │
│     - Telemetry capture to CSV                        │
│                                                         │
│  3. ERROR DETECTION & FIXING                            │
│     - Parse output for error patterns                 │
│     - Categorize errors (CUDA, memory, etc.)          │
│     - Auto-apply targeted fixes                       │
│                                                         │
│  4. METRICS COLLECTION                                  │
│     - Parse CSV telemetry                             │
│     - Extract: coherence, ethics, stability           │
│                                                         │
│  5. AGENT LIGHTNING TRAINING                            │
│     - Calculate RL reward                             │
│     - Emit Agent Lightning events                     │
│     - Track improvement deltas                        │
│                                                         │
│  6. OPTIMIZATION SUGGESTIONS                            │
│     - Suggest parameter adjustments                   │
│     - Track best iteration                            │
│                                                         │
│  → REPEAT for next iteration                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Components

### 1. `recursive_improvement_engine.py`

Main orchestrator. Coordinates all phases:

```python
from recursive_improvement_engine import RecursiveImprovementEngine

engine = RecursiveImprovementEngine(max_iterations=10)
results = engine.run()

# Access results
for result in results:
    print(f"Iteration {result.iteration}: Reward {result.agent_reward:.4f}")
    if result.improvements_applied:
        print(f"  Applied fixes: {result.improvements_applied}")
```

**Key Classes:**
- `RecursiveImprovementEngine` - Main orchestrator
- `CUDAExecutor` - Builds and runs with CUDA
- `ErrorExtractor` - Detects error patterns
- `MetricsCollector` - Gathers telemetry
- `AgentLightningOptimizer` - RL integration
- `AutoFixer` - Applies targeted fixes

### 2. `advanced_error_fixer.py`

Specialized error detection and fixing:

```python
from advanced_error_fixer import ErrorDetector, ErrorFixer

detector = ErrorDetector()
errors = detector.detect(build_output)

fixer = ErrorFixer()
for error in errors:
    fixer.fix_error(error)
```

**Features:**
- Detects 9+ error types
- Categorizes by severity (CRITICAL, HIGH, MEDIUM, LOW)
- Applies context-aware fixes
- Proactive optimization suggestions

### 3. `run_recursive_improvement.sh`

Bash orchestrator with environment setup:

```bash
./run_recursive_improvement.sh --iterations 10 --ticks 120 --cuda
```

## Error Detection & Fixing

### Supported Error Types

| Category | Severity | Detection | Auto-Fix |
|----------|----------|-----------|----------|
| CUDA not found | CRITICAL | `nvcc` missing | Install toolkit guide |
| CUDA version mismatch | HIGH | Compute capability | Update flags |
| Memory error | CRITICAL | Segfault, OOM | Increase memory |
| Undefined reference | CRITICAL | Linker error | Fix linker flags |
| Missing headers | HIGH | `file not found` | Install dev packages |
| Compilation error | CRITICAL | `error:` in output | Rebuild clean |
| Runtime assertion | HIGH | `assert` failed | Enable debug mode |
| Phase convergence | MEDIUM | `convergence failed` | Increase ticks |
| Ethics calculation | MEDIUM | `ethics error` | Enable debug |

### Example: Auto-fixing Memory Errors

```
Error detected: "segmentation fault: out of memory"
  → Category: memory
  → Severity: CRITICAL
  → Fix: increase_memory
  
Applying fixes:
  1. Set QALLOW_MEMORY_LIMIT=16384
  2. Rebuild with RelWithDebInfo
  3. Re-run phases
```

## Metrics & Rewards

### Collected Metrics

```json
{
  "avg_coherence": 0.92,        // Quantum coherence (target: 0.95)
  "ethics_total": 2.45,          // E = S + C + H (target: 3.0)
  "sustainability": 0.85,
  "compassion": 0.80,
  "harmony": 0.80,
  "phase_drift": 0.15,
  "stability": 0.85,
  "execution_time": 45.2         // seconds
}
```

### RL Reward Calculation

```
Reward = 0.5 * coherence_score 
       + 0.3 * ethics_score 
       + 0.2 * stability_score

Where:
  coherence_score = min(measured / target, 1.0)
  ethics_score    = min(ethics_total / 3.0, 1.0)
  stability_score = 1.0 - min(drift, 1.0)

Improvement bonus:
  bonus = 0.2 * max(delta_coherence, 0)
        + 0.2 * max(delta_ethics, 0)
```

## Output & Reports

### Console Output

```
===============================================================================
ITERATION 1/10
===============================================================================

[Iteration 1] Building with CUDA...
  ✓ Build successful in 42.3s

[Iteration 1] Executing Qallow phases...
  ✓ Execution successful in 35.1s

[Iteration 1] Collecting metrics...
  Coherence: 0.9200
  Ethics: 2.4500
  Stability: 0.8500
  RL Reward: 0.7340

[Iteration 1] Detected 2 errors. Applying fixes...
  ✓ Fixed cuda: "CUDA memory allocation failed"
  ✓ Applied optimization: --integrate-phase13-ticks=200
```

### JSON Report

Each run generates a detailed JSON report:

```json
{
  "timestamp": "2025-11-01T15:30:00.000000",
  "total_iterations": 10,
  "results": [
    {
      "iteration": 1,
      "timestamp": "2025-11-01T15:30:05.000000",
      "success": false,
      "build_time": 42.3,
      "execution_time": 35.1,
      "errors": ["CUDA memory allocation failed"],
      "warnings": ["Phase 13 took longer than expected"],
      "metrics": {
        "avg_coherence": 0.92,
        "ethics_total": 2.45,
        "stability": 0.85,
        "execution_time": 35.1
      },
      "improvements_applied": [
        "Fixed cuda: CUDA memory allocation error",
        "Applied optimization: --integrate-phase13-ticks=200"
      ],
      "agent_reward": 0.7340,
      "notes": ""
    },
    ...
  ]
}
```

**Location:** `improvement_reports/recursive_improvement_YYYYMMDD_HHMMSS.json`

## Agent Lightning Integration

### Automatic Event Emission

The system automatically emits Agent Lightning events:

```python
# At phase start
agl.emit_task_start(
    task_name='qallow_phase_1'
)

# At phase completion
agl.emit_task_complete(
    reward=0.7340,
    metadata={
        'iteration': 1,
        'metrics': {...},
        'errors_count': 2,
        'improvements': [...]
    }
)
```

### Accessing RL Training

To use the collected data for RL training:

```bash
# View LightningStore data
agl store view

# Train with collected traces
agl train --algorithm=PPO --iterations=100

# Monitor training progress
agl dashboard
```

## Environment Variables

### Control Behavior

```bash
# Logging
export QALLOW_LOG_LEVEL=DEBUG      # INFO, DEBUG, WARNING
export QALLOW_TELEMETRY_ENABLED=1  # 0 or 1
export QALLOW_PROFILE_ENABLED=1    # 0 or 1

# CUDA
export CUDA_VISIBLE_DEVICES=0      # GPU device ID
export QALLOW_MEMORY_LIMIT=16384   # MB

# Ethics debugging
export QALLOW_ETHICS_DEBUG=1        # Enable ethics debug
export QALLOW_LOG_ETHICS=1          # Log ethics calculations

# Error handling
export QALLOW_ASSERT_DEBUG=1        # Debug assertions
```

## Workflow Examples

### Example 1: Basic Improvement Loop

```bash
#!/bin/bash
cd /home/xing/Qallow

# Run 5 iterations with default settings
./run_with_improvement.sh 5 120 cuda

# Check the report
cat improvement_reports/recursive_improvement_*.json | jq '.results[] | {iter: .iteration, reward: .agent_reward}'
```

### Example 2: Intensive Optimization

```bash
#!/bin/bash
cd /home/xing/Qallow

# Run 20 iterations with high tick count
./run_recursive_improvement.sh \
    --iterations 20 \
    --ticks 200 \
    --cuda

# Find best iteration
python3 << 'EOF'
import json
from pathlib import Path

reports = sorted(Path('improvement_reports').glob('*.json'), 
                key=lambda x: x.stat().st_mtime, reverse=True)
with open(reports[0]) as f:
    data = json.load(f)
    
best = max(data['results'], key=lambda r: r['agent_reward'])
print(f"Best iteration: #{best['iteration']} with reward {best['agent_reward']:.4f}")
EOF
```

### Example 3: Automated Nightly Improvement

```bash
#!/bin/bash
# Save as: /etc/cron.d/qallow-improvement

# Run at 2 AM daily
0 2 * * * cd /home/xing/Qallow && ./run_with_improvement.sh 10 120 cuda >> /var/log/qallow-improvement.log 2>&1
```

### Example 4: Monitor Real-time Progress

```bash
#!/bin/bash
cd /home/xing/Qallow

# Run in one terminal
./run_with_improvement.sh 10 120 cuda

# In another terminal, monitor logs
watch -n 1 'tail -20 improvement_reports/improvement_*.log'
```

## Troubleshooting

### Build Fails with CUDA Errors

```bash
# Check CUDA installation
nvcc --version

# Fall back to CPU
./run_with_improvement.sh 10 120 cpu

# Or disable CUDA explicitly
./run_recursive_improvement.sh --no-cuda --iterations 10
```

### Memory Issues

```bash
# Increase memory limit
export QALLOW_MEMORY_LIMIT=32768  # 32GB
./run_with_improvement.sh 5 120 cuda

# Reduce phase ticks
./run_with_improvement.sh 5 60 cuda
```

### Agent Lightning Not Detected

```bash
# Install Agent Lightning

# Verify installation

# The system will run without RL if not installed (fallback mode)
```

### Check Latest Report

```bash
# List recent reports
ls -lh improvement_reports/recursive_improvement_*.json | tail -5

# View latest summary
python3 << 'EOF'
import json
from pathlib import Path

latest = sorted(Path('improvement_reports').glob('*.json'), 
               key=lambda x: x.stat().st_mtime, reverse=True)[0]
with open(latest) as f:
    data = json.load(f)
    results = data['results']
    
print(f"Total iterations: {len(results)}")
print(f"Success rate: {sum(1 for r in results if r['success']) / len(results) * 100:.1f}%")
print(f"Avg reward: {sum(r['agent_reward'] for r in results) / len(results):.4f}")

for r in results[-3:]:
    print(f"  Iter {r['iteration']}: reward={r['agent_reward']:.4f}, success={r['success']}")
EOF
```

## Performance Tuning

### For Maximum Performance

```bash
# Use all available CUDA cores
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Multiple GPUs

# Run with high tick count
./run_recursive_improvement.sh --iterations 20 --ticks 300

# Enable profiling
export QALLOW_PROFILE_ENABLED=1
```

### For Quick Testing

```bash
# Minimal iterations with low ticks
./run_recursive_improvement.sh --iterations 3 --ticks 60

# CPU-only for quick feedback
./run_with_improvement.sh 3 60 cpu
```

## Advanced: Custom Error Handlers

Add custom error fixing logic to `advanced_error_fixer.py`:

```python
class CustomErrorFixer(ErrorFixer):
    def _fix_custom_error(self, error: ErrorInfo) -> bool:
        """Custom error fix."""
        logger.info("Applying custom fix...")
        # Your fix logic here
        return True
```

## Monitoring & Metrics

### Track Improvement Over Time

```bash
# Generate improvement trend
python3 << 'EOF'
import json
from pathlib import Path
import statistics

reports = sorted(Path('improvement_reports').glob('*.json'))

for report in reports[-5:]:  # Last 5 reports
    with open(report) as f:
        data = json.load(f)
        rewards = [r['agent_reward'] for r in data['results']]
        print(f"{report.name}:")
        print(f"  Min: {min(rewards):.4f}")
        print(f"  Max: {max(rewards):.4f}")
        print(f"  Avg: {statistics.mean(rewards):.4f}")
        print(f"  Std: {statistics.stdev(rewards):.4f}")
EOF
```

## Summary

The **Recursive Improvement Engine** automates the complete workflow:

✅ **Automated Build** with CUDA acceleration  
✅ **Error Detection** with 9+ pattern types  
✅ **Auto-Fixing** with targeted remedies  
✅ **Metrics Collection** from telemetry  
✅ **RL Training** via Agent Lightning  
✅ **Continuous Loop** until convergence  

Each iteration improves the project, making it a **self-improving system**.

---

**Questions?** Check `QALLOW_LIGHTNING_INTEGRATION.md` for more details on Agent Lightning integration.
