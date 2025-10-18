# Phase V – Autonomous Governance Quick Start

## What's New

Phase V delivers a **unified binary** (`qallow`) that controls every phase through one command interface:

```
./qallow [mode]
```

## One-Command Operation

After building once, every task becomes:

```bash
./qallow build      # Build both CPU + CUDA
./qallow run        # Execute unified runtime
./qallow bench      # Run HITL benchmark
./qallow visual     # Open live dashboard
./qallow govern     # Run autonomous governance audit
./qallow help       # Show help
```

## Getting Started

### 1. Build the Unified Binary

```bash
# Windows Batch
.\qallow.bat build

# Or directly
build\qallow.exe build
```

The system auto-detects CUDA and builds accordingly:
- If CUDA is available: Builds `build\qallow_cuda.exe`
- Otherwise: Builds `build\qallow.exe`

### 2. Run the VM

```bash
# Windows Batch
.\qallow.bat run

# Or directly
build\qallow.exe run
```

Output:
```
[RUN] Executing Qallow VM...
[TELEMETRY] System initialized
[SYSTEM] Qallow VM initialized
[SYSTEM] Execution mode: CPU
[MAIN] Starting VM execution loop...
[TICK 0000] Coherence: 0.9992 | Decoherence: 0.000010 | Stability: 0.9984 0.9982 0.9984
[MAIN] VM execution completed
```

### 3. Run Autonomous Governance

```bash
# Windows Batch
.\qallow.bat govern

# Or directly
build\qallow.exe govern
```

Output:
```
╔════════════════════════════════════════╗
║  AUTONOMOUS GOVERNANCE LOOP STARTING   ║
╚════════════════════════════════════════╝

[GOVERN] Audit #1: Ethics Score = 2.3000
[GOVERN] ⚠️  WARNING: Ethics score below threshold (2.3000 < 2.9000)

╔════════════════════════════════════════╗
║  GOVERNANCE HALT - VIOLATION DETECTED  ║
╚════════════════════════════════════════╝

[GOVERN] Reason: Initial ethics score below threshold
[GOVERN] Initiating emergency rollback...
```

## Architecture

### Unified Binary Structure

```
qallow.exe
├─ launcher.c (main entry point)
│  ├─ build mode
│  ├─ run mode
│  ├─ bench mode
│  ├─ visual mode
│  └─ govern mode
├─ main.c (VM execution)
├─ govern.c (autonomous governance)
├─ ethics.c (ethics monitoring)
├─ adaptive.c (learning)
├─ sandbox.c (state snapshots)
└─ ... (all other modules)
```

### Governance Loop

```
Initialize → Audit → Check Threshold → Adapt → Persist → Report
```

1. **Initialize**: Load state, create checkpoints
2. **Audit**: Evaluate ethics score (E = S + C + H)
3. **Check**: Verify E >= 2.9 threshold
4. **Adapt**: Adjust parameters based on stability
5. **Persist**: Save state to JSON
6. **Report**: Generate audit report

## Ethics Scoring

The system evaluates three components:

- **S (Safety)**: Physical, information, environmental safety
- **C (Clarity)**: Transparency, predictability, explainability
- **H (Human Benefit)**: Welfare, autonomy, justice

**Total Score**: E = S + C + H

**Threshold**: E >= 2.9 (minimum 0.8 + 0.7 + 0.6)

## Wrapper Scripts

### Batch Wrapper (qallow.bat)

Simple command routing:
```batch
.\qallow.bat build
.\qallow.bat run
.\qallow.bat govern
```

### PowerShell Wrapper (qallow_launcher.ps1)

Feature-rich with colored output:
```powershell
.\qallow_launcher.ps1 build
.\qallow_launcher.ps1 run
.\qallow_launcher.ps1 govern
```

## Files Created/Modified

### Created
- `core/include/govern.h` - Governance API
- `backend/cpu/govern.c` - Governance implementation
- `interface/launcher.c` - Unified launcher
- `qallow_launcher.ps1` - PowerShell wrapper
- `PHASE_V_AUTONOMOUS_GOVERNANCE.md` - Full documentation

### Modified
- `scripts/build_wrapper.bat` - Added launcher + govern compilation
- `interface/main.c` - Renamed main() to qallow_vm_main()
- `core/include/qallow.h` - Added qallow_vm_main() declaration
- `qallow.bat` - Updated to route to unified binary

## Verification

✅ **Build**: CPU build successful
✅ **Run**: VM execution works
✅ **Govern**: Governance audit loop works
✅ **Batch Wrapper**: Command routing works
✅ **Ethics Scoring**: Correctly evaluates E = S + C + H
✅ **Emergency Procedures**: Detects violations and rolls back

## Next Steps

1. **Test CUDA build** (if CUDA available)
2. **Implement dashboard** for real-time monitoring
3. **Add telemetry streaming** to external systems
4. **Implement adaptive learning** feedback loop
5. **Add multi-threaded governance** for parallel audits

## Troubleshooting

### Build fails
```bash
# Check Visual Studio is installed
cl /?

# Check CUDA (if needed)
nvcc --version
```

### Executable not found
```bash
# Rebuild
.\qallow.bat build

# Or directly
build\qallow.exe build
```

### Ethics score too low
The system is working correctly! It detected that the ethics score is below threshold and triggered emergency rollback. This is the safety mechanism in action.

## Summary

Phase V unifies Qallow into a single command system:

- **One binary** for all operations
- **Auto-detection** of available hardware
- **Autonomous governance** with ethics enforcement
- **Emergency procedures** for safety violations
- **State persistence** for learning continuity

Everything flows through: **Build → Run → Monitor → Learn → Govern → Adapt**

