# Qallow Unified CLI - Quick Reference Card

## Command Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    qallow <GROUP> [SUBCOMMAND] [OPTIONS]        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Main Command Groups

### 1️⃣ RUN - Workflow Execution
```
qallow run              # Default: execute unified VM
qallow run vm           # Same as above
qallow run bench        # Benchmark profile
qallow run live         # Live data ingestion
qallow run unified      # Phases 12-15 pipeline
qallow run accelerator  # Phase 13 accelerator
```

### 2️⃣ SYSTEM - Build & Maintenance
```
qallow system build     # Compile CPU + CUDA
qallow system clear     # Clean build artifacts
qallow system verify    # Health checks
```

### 3️⃣ PHASE - Individual Phases
```
qallow phase 11         # Coherence bridge (quantum)
qallow phase 12         # Elasticity simulation
qallow phase 13         # Harmonic propagation
qallow phase 14         # Coherence-lattice integration
qallow phase 15         # Convergence & lock-in
```

### 4️⃣ MIND - Cognitive Pipeline
```
qallow mind pipeline    # Cognitive modules
qallow mind bench       # Benchmarking
```

---

## 📋 Cheat Sheet

### Build & Setup
```bash
qallow system build                    # Build project
qallow system clear                    # Clean everything
qallow system verify                   # Health check
```

### Execute Phases
```bash
qallow run                             # Execute unified VM
qallow run unified                     # Phases 12-15 pipeline
qallow phase 12 --ticks=100           # Phase 12
qallow phase 13 --nodes=16 --ticks=500  # Phase 13 custom
qallow phase 14 --tune_qaoa           # Phase 14 with QAOA tuning
```

### Advanced
```bash
qallow run vm --integrate phase11       # Add quantum phase
qallow run vm --self-audit             # Enable auditing
qallow run bench --dashboard=off       # Benchmark, no dashboard
```

### Help
```bash
qallow help                            # Main help
qallow run help                        # Run group help
qallow phase help                      # Phase help
```

---

## 📊 Phase Options Quick Ref

### Phase 12 (Elasticity)
```bash
qallow phase 12 --ticks=100 --eps=0.0001 --audit-tag=demo
```
**Options:** `--ticks`, `--eps`, `--log`, `--audit-tag`

### Phase 13 (Harmonic)
```bash
qallow phase 13 --nodes=16 --ticks=500 --k=0.002 --audit-tag=test
```
**Options:** `--nodes`, `--ticks`, `--k`, `--log`, `--audit-tag`

### Phase 14 (Coherence-Lattice)
```bash
qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981
```
**Options:** `--ticks`, `--nodes`, `--target_fidelity`, `--tune_qaoa`, `--alpha`, `--export`

### Phase 15 (Convergence)
```bash
qallow phase 15 --ticks=400 --eps=1e-5 --export=/tmp/results.json
```
**Options:** `--ticks`, `--eps`, `--export`

---

## 🔄 Before vs After

### ❌ Old (Multiple Separate Commands)
```bash
qallow build
qallow phase11 --ticks=100
qallow phase12 --ticks=100  
qallow phase13 --ticks=100
qallow bench
qallow clear
```

### ✅ New (Unified Interface)
```bash
qallow system build
qallow phase 11 --ticks=100
qallow phase 12 --ticks=100
qallow phase 13 --ticks=100
qallow run bench
qallow system clear
```

---

## 🎯 Common Workflows

### Development
```bash
qallow system build
qallow system verify
qallow run unified
python3 recursive_improvement_engine.py
```

### Benchmarking
```bash
qallow system build
qallow run bench --dashboard=50
qallow phase 13 --nodes=256 --ticks=600 --audit-tag=bench
```

### Quantum Integration
```bash
export QALLOW_cirq=1
qallow run vm --integrate phase11 --integrate-phase11-hardware
```

### Production
```bash
qallow system build
qallow system verify
qallow run vm --self-audit --export-pocket-map /tmp/pockets.json
```

---

## 🔗 Integration with AgentLightning Runner

### Auto-Improvement Loop
```bash
# Runs in background, continuously improving codebase
python3 recursive_improvement_engine.py

# In another terminal, run phases as needed
qallow run unified
qallow phase 13 --nodes=32 --ticks=400
```

### Features
- Auto-detects issues
- Auto-fixes problems  
- Runs tests
- Benchmarks performance
- Iterates continuously

---

## 📚 Environment Variables

```bash
# Quantum
export QALLOW_cirq=1

# Logging
export QALLOW_LOG=data/logs/telemetry.csv
export QALLOW_LOG_DIR=/var/log/qallow

# UI
export QALLOW_DASHBOARD_INTERVAL=100

# Mind
export QALLOW_MIND_STEPS=50
```

---

## ⚠️ Deprecated (Still Work - Use New Syntax)

```bash
# Old → New
qallow build       → qallow system build
qallow clear       → qallow system clear
qallow verify      → qallow system verify
qallow bench       → qallow run bench
qallow live        → qallow run live
qallow accelerator → qallow run accelerator
qallow phase11     → qallow phase 11
qallow phase12     → qallow phase 12
qallow phase13     → qallow phase 13
```

---

## 📖 Full Documentation

See `UNIFIED_CLI.md` for:
- Complete command reference
- All options for each command
- Advanced features
- Troubleshooting
- Usage patterns
- Examples

---

## 🎓 Quick Examples

```bash
# 1. Setup
qallow system build && qallow system verify

# 2. Run everything (phases 12-15)
qallow run unified

# 3. Run specific phase with custom params
qallow phase 13 --nodes=32 --ticks=600 --k=0.003

# 4. Enable benchmarking
qallow run bench

# 5. Add quantum phase (requires QALLOW_cirq=1)
qallow run vm --integrate phase11

# 6. Auto-improve the codebase
python3 recursive_improvement_engine.py
qallow run unified

# 7. Get help
qallow help
qallow run help
qallow phase help
```

---

**Key Takeaway:** One command `qallow` with grouped subcommands replaces 10+ separate commands. Cleaner, easier, better! 🚀
