# Phase V – Command Reference Card

## Quick Commands

### Build
```bash
# Build with toolchain detection
./qallow build

# Direct execution
build\qallow.exe build
```

### Run
```bash
# Execute VM (auto-selects CPU/CUDA)
./qallow run

# Direct execution
build\qallow.exe run
```

### Benchmark
```bash
# Run HITL benchmark
./qallow bench

# Direct execution
build\qallow.exe bench
```

### Governance
```bash
# Run autonomous governance audit
./qallow govern

# Direct execution
build\qallow.exe govern
```

### Dashboard
```bash
# Open live dashboard
./qallow visual

# Direct execution
build\qallow.exe visual
```

### Help
```bash
# Show help message
./qallow help

# Direct execution
build\qallow.exe help
```

## Wrapper Scripts

### Windows Batch (qallow.bat)
```batch
.\qallow.bat build
.\qallow.bat run
.\qallow.bat bench
.\qallow.bat govern
.\qallow.bat visual
.\qallow.bat help
```

### PowerShell (qallow_launcher.ps1)
```powershell
.\qallow_launcher.ps1 build
.\qallow_launcher.ps1 run
.\qallow_launcher.ps1 bench
.\qallow_launcher.ps1 govern
.\qallow_launcher.ps1 visual
.\qallow_launcher.ps1 help
```

## Command Modes Explained

### build
**Purpose**: Compile the Qallow system
**Behavior**: 
- Detects CUDA availability
- Compiles CPU version if CUDA unavailable
- Compiles CUDA version if available
- Includes launcher + governance core
**Output**: `build\qallow.exe` or `build\qallow_cuda.exe`

### run
**Purpose**: Execute the Qallow VM
**Behavior**:
- Initializes all subsystems
- Runs VM execution loop
- Monitors ethics and safety
- Generates reports
**Output**: VM execution results, ethics report, sandbox report

### bench
**Purpose**: Run HITL benchmark
**Behavior**:
- Executes benchmark suite
- Logs runtime metrics
- Records ethics scores
- Generates performance report
**Output**: Benchmark results, performance metrics

### govern
**Purpose**: Run autonomous governance audit
**Behavior**:
- Evaluates ethics score (E = S + C + H)
- Checks safety threshold (E >= 2.9)
- Adapts system parameters
- Creates safety checkpoints
- Performs emergency rollback if needed
**Output**: Governance audit report, ethics summary

### visual
**Purpose**: Open live dashboard
**Behavior**:
- Launches dashboard interface
- Streams real-time metrics
- Shows ethics scores
- Displays system status
**Output**: Dashboard interface (placeholder)

### help
**Purpose**: Display help message
**Behavior**:
- Shows all available commands
- Displays usage examples
- Lists command descriptions
**Output**: Help text

## Ethics Scoring

### Formula
```
E = S + C + H

Where:
  S = Safety Score (0.0 - 1.0)
  C = Clarity Score (0.0 - 1.0)
  H = Human Benefit Score (0.0 - 1.0)
```

### Components

**Safety (S)**
- Physical safety
- Information security
- Environmental impact

**Clarity (C)**
- Transparency
- Predictability
- Explainability
- Auditability

**Human Benefit (H)**
- Welfare
- Autonomy
- Justice

### Thresholds
- Minimum Safety: 0.8
- Minimum Clarity: 0.7
- Minimum Human Benefit: 0.6
- Minimum Total (E): 2.1
- **Governance Threshold: 2.9**

## Governance Loop

```
1. Initialize
   └─ Load adaptive state
   └─ Create initial checkpoint

2. Audit
   └─ Evaluate ethics score
   └─ Check threshold

3. Adapt
   └─ Adjust learning rate
   └─ Modify thread count
   └─ Reinforce learning

4. Persist
   └─ Save adaptive state
   └─ Create checkpoints

5. Report
   └─ Generate audit report
   └─ Print governance summary
```

## Output Examples

### Build Output
```
[BUILD] Detecting toolchain...
[BUILD] CUDA detected - building CUDA-accelerated version...
[BUILD] Compiling unified launcher and governance core...
[CUDA] Compiling CUDA-enabled version...
[SUCCESS] CUDA build completed: build\qallow_cuda.exe
```

### Run Output
```
[RUN] Executing Qallow VM...
[TELEMETRY] System initialized
[SYSTEM] Qallow VM initialized
[SYSTEM] Execution mode: CPU
[MAIN] Starting VM execution loop...
[TICK 0000] Coherence: 0.9992 | Decoherence: 0.000010
[MAIN] VM execution completed
```

### Govern Output
```
[GOVERN] Starting autonomous governance loop...
[GOVERN] Audit #1: Ethics Score = 2.3000
[GOVERN] ⚠️  WARNING: Ethics score below threshold
[GOVERN] Initiating emergency rollback...
[GOVERN] Final Ethics Score: 2.3000
[GOVERN] Status: ✗ FAIL
```

## File Locations

### Executables
- `build\qallow.exe` - CPU version
- `build\qallow_cuda.exe` - CUDA version

### Wrappers
- `qallow.bat` - Batch wrapper
- `qallow_launcher.ps1` - PowerShell wrapper

### Output Files
- `qallow_stream.csv` - Telemetry stream
- `qallow_bench.log` - Benchmark log
- `adapt_state.json` - Adaptive state

### Documentation
- `PHASE_V_AUTONOMOUS_GOVERNANCE.md` - Full documentation
- `PHASE_V_QUICKSTART.md` - Quick start guide
- `PHASE_V_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `PHASE_V_COMMAND_REFERENCE.md` - This file

## Troubleshooting

### Build fails
```bash
# Check Visual Studio
cl /?

# Check CUDA (if needed)
nvcc --version

# Rebuild
.\qallow.bat build
```

### Executable not found
```bash
# Rebuild
build\qallow.exe build

# Or use wrapper
.\qallow.bat build
```

### Ethics score too low
This is expected behavior! The system is working correctly:
- Ethics score evaluated
- Threshold checked
- Violation detected
- Emergency rollback triggered

This demonstrates the safety mechanism in action.

### Command not recognized
```bash
# Use full path
.\qallow.bat [command]

# Or direct execution
build\qallow.exe [command]
```

## Performance Metrics

- **Build Time**: ~2 seconds (CPU)
- **Governance Audit**: ~0.05 seconds
- **VM Execution**: ~1 millisecond
- **Ethics Evaluation**: <1 millisecond
- **State Persistence**: <1 millisecond

## System Requirements

- **OS**: Windows 10/11
- **Compiler**: Visual Studio 2022 Build Tools
- **RAM**: 512 MB minimum
- **Disk**: 50 MB for build artifacts
- **CUDA** (optional): CUDA 13.0+ for GPU acceleration

## Summary

Phase V provides a unified command interface for all Qallow operations:

```
./qallow [mode]
```

Where mode is one of:
- `build` - Compile
- `run` - Execute
- `bench` - Benchmark
- `govern` - Governance audit
- `visual` - Dashboard
- `help` - Help

Everything is controlled through this single command system.

