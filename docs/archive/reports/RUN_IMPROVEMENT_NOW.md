# Recursive Improvement Engine - Quick Start

## 🚀 One Command to Improve Everything

```bash
cd /home/xing/Qallow
./run_with_improvement.sh 10 120 cuda
```

This runs 10 iterations with automatic CUDA builds and error fixing.

## What Happens Each Iteration

1. **Build** - Compiles with CUDA support
2. **Execute** - Runs Qallow phases 12-15
3. **Detect** - Finds errors and issues
4. **Fix** - Applies targeted fixes automatically
5. **Measure** - Collects coherence, ethics, stability metrics
6. **Learn** - Agent Lightning RL training
7. **Repeat** - Next iteration with improvements

## Key Features

✅ Recursive improvement - each iteration improves the last  
✅ Automatic error fixing - detects and fixes 9+ error types  
✅ CUDA acceleration - GPU-powered builds and execution  
✅ Agent Lightning integration - RL-based optimization  
✅ Metrics tracking - coherence, ethics, stability scores  
✅ Full reporting - JSON reports with improvement history  

## Output

After completion, check reports:

```bash
cat improvement_reports/recursive_improvement_*.json | jq '.results[] | {iteration, success, reward: .agent_reward}'
```

## Advanced Usage

Different ticks/iterations:

```bash
./run_with_improvement.sh 5 60 cuda      # Quick test: 5 iterations, 60 ticks
./run_with_improvement.sh 20 200 cuda    # Deep optimization: 20 iterations, 200 ticks
./run_with_improvement.sh 10 120 cpu     # CPU-only mode
```

## Components

- `recursive_improvement_engine.py` - Main orchestrator
- `advanced_error_fixer.py` - Error detection & fixing
- `run_recursive_improvement.sh` - Bash orchestrator
- `run_with_improvement.sh` - Quick launcher

## Results Directory

```
improvement_reports/
├── recursive_improvement_20251101_153000.json    # Full results
├── build_20251101_153000.log                     # Build log
├── improvement_20251101_153000.log               # Execution log
└── cmake_config_20251101_153000.log              # CMake log
```

## Monitoring

Real-time monitoring:

```bash
# In terminal 1
./run_with_improvement.sh 10 120 cuda

# In terminal 2
watch -n 1 'tail -30 improvement_reports/improvement_*.log'
```

## See Results

```bash
# Latest improvement summary
python3 -c "import json; from pathlib import Path; data=json.load(open(sorted(Path('improvement_reports').glob('*.json'), key=lambda x: x.stat().st_mtime)[-1])); results=data['results']; print(f'Iterations: {len(results)}, Success: {sum(1 for r in results if r[\"success\"])}, Avg Reward: {sum(r[\"agent_reward\"] for r in results)/len(results):.4f}')"
```

---

For detailed documentation, see: `RECURSIVE_IMPROVEMENT_GUIDE.md`
