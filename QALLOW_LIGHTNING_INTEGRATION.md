# Qallow + Agent Lightning Integration Guide

## Overview

This integration connects **Qallow** (quantum-AGI computing platform) with **Agent Lightning** (Microsoft's RL training framework). It enables automated phase optimization with reinforcement learning.

### What You Can Do

✅ Run Qallow quantum phases with automatic telemetry capture  
✅ Instrument phases with Agent Lightning for RL training  
✅ Calculate performance rewards based on coherence and ethics metrics  
✅ Train RL models to optimize quantum phase parameters  
✅ Track agent decisions and improvements over time  

---

## Architecture

```
Qallow Quantum Phases (C/CUDA)
    ↓
Telemetry CSV Output (metrics, coherence, ethics)
    ↓
QallowPhaseRunner (subprocess wrapper)
    ↓
QallowLightningAgent (Lightning instrumentation)
    ↓
agl.emit_task_start/complete() 
    ↓
LightningStore (central RL hub)
    ↓
RL Algorithms (PPO, GRPO, VERL)
    ↓
Optimized Phase Parameters
```

---

## Components

### 1. **QallowPhaseRunner**
Executes Qallow phases and parses telemetry output.

```python
runner = QallowPhaseRunner()

# Run single phase
metrics = runner.run_phase(phase=13, ticks=120)

# Run unified workflow (phases 12-15)
results = runner.run_unified(phases=[12, 13, 14, 15], ticks=120)
```

**Methods:**
- `run_phase(phase, ticks, additional_args, timeout)` - Execute single phase
- `run_unified(phases, ticks, timeout)` - Execute unified workflow
- `_parse_telemetry(phase, execution_time)` - Parse CSV output

**Returns:**
- `PhaseMetrics`: Aggregated phase data (coherence, ethics, timestamps)

---

### 2. **QallowLightningAgent**
Instruments phase execution with Agent Lightning events.

```python
agent = QallowLightningAgent(runner, agent_id="qallow-optimizer")

# Single phase with RL tracking
metrics, reward = agent.optimize_phase(
    phase=13,
    ticks=120,
    target_coherence=0.95
)

# Unified workflow with RL tracking
results, cumulative_reward = agent.optimize_unified(
    phases=[12, 13, 14, 15],
    ticks=120,
    target_coherence=0.95
)
```

**Emitted Events:**
- `agl.emit_task_start()` - Phase execution started
- `agl.emit_task_complete()` - Phase execution completed with reward

**Reward Formula:**
```
reward = 0.5 * coherence_score + 0.3 * ethics_score + 0.2 * stability_score

Where:
- coherence_score = min(measured_coherence / target_coherence, 1.0)
- ethics_score = (sustainability + compassion + harmony) / 3
- stability_score = 1.0 - min(phase_drift, 1.0)
```

---

## Quick Start

### 1. Prerequisites

```bash
# Ensure Qallow is built
./scripts/build_all.sh

# Ensure Agent Lightning is installed
pip install agentlightning

# Ensure qallow_lightning_integration.py is in workspace root
```

### 2. Run the Demo

```bash
# Activate environment
source .venv/bin/activate

# Run integration demo
python qallow_lightning_integration.py
```

Expected output:
```
======================================================================
Qallow + Agent Lightning Integration Demo
======================================================================

[INFO] Qallow Phase Runner initialized
[INFO] Agent Lightning instrumentation enabled

----------------------------------------------------------------------
Demo 1: Single Phase Optimization (Phase 13)
----------------------------------------------------------------------

✓ Phase 13 Results:
  Coherence:     0.9245
  Ethics Total:  2.7154
  Phase Drift:   0.0156
  Reward:        0.8924
  Exec Time:     12.34s

...
```

---

## Advanced Usage

### Training with Different RL Algorithms

```bash
# Terminal 1: Start LightningStore with PPO
agl store --algorithm=ppo

# Terminal 2: Run multiple optimization passes
for i in {1..10}; do
    python qallow_lightning_integration.py
    sleep 2
done

# Monitor training
agl store --monitor
```

### Custom Phase Parameters

```python
from qallow_lightning_integration import QallowPhaseRunner, QallowLightningAgent

runner = QallowPhaseRunner()
agent = QallowLightningAgent(runner)

# Phase 14 with custom parameters
metrics, reward = agent.optimize_phase(
    phase=14,
    ticks=200,
    target_coherence=0.92
)

# Phase 15 with extended execution
metrics, reward = agent.optimize_phase(
    phase=15,
    ticks=500,
    target_coherence=0.98
)
```

### Batch Optimization

```python
from qallow_lightning_integration import QallowPhaseRunner, QallowLightningAgent

runner = QallowPhaseRunner()
agent = QallowLightningAgent(runner)

# Optimize each phase sequentially
phases = [12, 13, 14, 15]
rewards = {}

for phase in phases:
    metrics, reward = agent.optimize_phase(phase=phase, ticks=150)
    rewards[phase] = reward
    print(f"Phase {phase}: Reward = {reward:.4f}")

# Calculate average performance
avg_reward = sum(rewards.values()) / len(rewards)
print(f"\nAverage Reward: {avg_reward:.4f}")
```

---

## Integration Points

### Phase 12: Elasticity & Harmonics
- **Metric**: `phase12_energy`, `phase12_coherence`
- **Reward Trigger**: Coherence stability and energy efficiency
- **RL Optimization**: Elasticity parameters and harmonic frequency

### Phase 13: Harmonic Acceleration
- **Metrics**: `avg_coherence`, `phase_drift`, `ethics_total`
- **Reward Trigger**: Coherence improvement with ethical consistency
- **RL Optimization**: Acceleration rates and ethics weighting

### Phase 14: Coherence & Determinism
- **Metrics**: `phase14_entanglement`, `phase14_alignment`, `phase14_flux`
- **Reward Trigger**: Entanglement stability and alignment precision
- **RL Optimization**: Deterministic coherence targets and flux management

### Phase 15: Convergence & Lock-in
- **Metrics**: `phase15_convergence`, `phase15_entropy`, `global_coherence`
- **Reward Trigger**: Final convergence success and entropy minimization
- **RL Optimization**: Convergence threshold and lock-in timing

---

## Telemetry Files

The integration reads from these Qallow log files:

| Phase(s) | Log File | Key Metrics |
|----------|----------|------------|
| 13 | `data/logs/phase13.csv` | coherence, phase_drift, ethics_total, ethics components |
| 14-15 | `data/logs/lattice_integrations.csv` | entanglement, alignment, convergence, global_coherence |

---

## Training Results Interpretation

### Reward Breakdown

- **0.9 - 1.0**: Excellent - Phase optimization near optimal
- **0.7 - 0.9**: Good - Phase running well with minor improvements possible
- **0.5 - 0.7**: Fair - Phase has room for optimization
- **0.0 - 0.5**: Poor - Phase needs significant tuning

### Metrics to Watch

| Metric | Good Range | Interpretation |
|--------|-----------|----------------|
| `avg_coherence` | > 0.90 | Quantum coherence maintained well |
| `phase_drift` | < 0.05 | Phase stability is high |
| `ethics_total` | > 2.5 | Ethics metrics well-balanced |
| `sustainability` | > 0.80 | Sustainability index high |
| `compassion` | > 0.90 | Compassion scoring high |
| `harmony` | > 0.99 | System harmony excellent |

---

## Troubleshooting

### Issue: "qallow binary not found"
**Solution**: Build Qallow first
```bash
./scripts/build_all.sh
```

### Issue: "No data in log file"
**Solution**: Ensure Qallow phase ran successfully
```bash
./build/qallow_unified_cpu phase 13 --ticks=100
ls -la data/logs/phase13.csv
```

### Issue: Agent Lightning events not being recorded
**Solution**: Start LightningStore server in separate terminal
```bash
agl store
```

### Issue: Timeout errors
**Solution**: Increase timeout or reduce tick count
```python
metrics = runner.run_phase(phase=13, ticks=50, timeout=600)
```

---

## Next Steps

1. **Baseline Measurement**: Run the demo to establish baseline performance metrics

2. **RL Training**: Start LightningStore and run multiple optimization iterations
   ```bash
   agl store --algorithm=ppo &
   for i in {1..50}; do python qallow_lightning_integration.py; done
   ```

3. **Analysis**: Review accumulated agent traces and reward signals

4. **Parameter Tuning**: Adjust target_coherence, ticks, and phase selection based on results

5. **Production Integration**: Embed `QallowLightningAgent` into your application

---

## API Reference

### PhaseMetrics
```python
@dataclass
class PhaseMetrics:
    phase_number: int        # Phase 12-15
    tick_count: int          # Total execution ticks
    avg_coherence: float     # Final average coherence
    phase_drift: float       # Phase deviation
    ethics_total: float      # Combined ethics score
    sustainability: float    # Sustainability component
    compassion: float        # Compassion component
    harmony: float           # Harmony component
    execution_time: float    # Wall-clock time (seconds)
    log_file: str            # Source telemetry file
```

### QallowPhaseRunner
```python
runner = QallowPhaseRunner(qallow_binary="./build/qallow_unified_cpu")

# Single phase
metrics = runner.run_phase(
    phase: int,
    ticks: int = 120,
    additional_args: List[str] = None,
    timeout: int = 300
) -> PhaseMetrics

# Unified workflow
results = runner.run_unified(
    phases: List[int] = None,  # [12, 13, 14, 15]
    ticks: int = 120,
    timeout: int = 600
) -> Dict[int, PhaseMetrics]
```

### QallowLightningAgent
```python
agent = QallowLightningAgent(
    runner: QallowPhaseRunner,
    agent_id: str = "qallow-optimizer"
)

# Single phase with RL
metrics, reward = agent.optimize_phase(
    phase: int,
    ticks: int = 120,
    target_coherence: float = 0.95
) -> Tuple[PhaseMetrics, float]

# Unified with RL
results, cumulative_reward = agent.optimize_unified(
    phases: List[int] = None,  # [12, 13, 14, 15]
    ticks: int = 120,
    target_coherence: float = 0.95
) -> Tuple[Dict[int, PhaseMetrics], float]
```

---

## Resources

- **Agent Lightning GitHub**: https://github.com/microsoft/agent-lightning
- **Agent Lightning Docs**: https://microsoft.github.io/agent-lightning/
- **Qallow Architecture**: `docs/ARCHITECTURE_SPEC.md`
- **Qallow README**: `README.md`

---

**Status**: ✅ Production Ready  
**Last Updated**: November 2025  
**Integration Version**: 1.0
