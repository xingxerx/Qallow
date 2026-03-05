# 🎯 Button Operations Reference

## Quick Reference - What Each Button Does

### ▶️ Start VM
```
Starts the Qallow Quantum-Photonic AGI System
├─ Initializes selected build (CPU/CUDA)
├─ Loads selected phase (13/14/15)
├─ Sets execution environment
└─ Output: "🚀 Starting Qallow VM with CPU build on Phase 14"
```

### ⏹️ Stop VM
```
Gracefully stops the running VM
├─ Calculates uptime
├─ Reports final metrics
├─ Preserves execution history
└─ Output: "⏹️ VM stopped gracefully (uptime: 45s, steps: 1250)"
```

### ⏸️ Pause
```
Pauses VM execution
├─ Captures current metrics
├─ Allows state inspection
├─ Can resume later
└─ Output: "⏸️ VM paused (step: 1250, reward: 3.45)"
```

### 🔄 Reset
```
Resets all execution metrics
├─ Clears terminal output
├─ Clears telemetry
├─ Prepares for new run
└─ Output: "🔄 System reset (cleared 1250 steps)"
```

### 📈 Export Metrics
```
Exports metrics to JSON file
├─ Creates: qallow_metrics_export.json
├─ Includes: timestamp, state, metrics, telemetry count
└─ Output: "✓ Metrics exported to qallow_metrics_export.json"
```

### 💾 Save Config
```
Saves configuration to JSON file
├─ Creates: qallow_phase_config.json
├─ Includes: phase config, build, phase, current metrics
└─ Output: "✓ Configuration saved to qallow_phase_config.json"
```

### 📋 View Logs
```
Displays audit logs in console
├─ Shows: last 50 entries
├─ Format: "✅ [HH:MM:SS] SUCCESS - Component: Message"
└─ Output: Formatted audit log display
```

### 📦 Build Selection
```
Switches between CPU and CUDA builds
├─ CPU - Optimized for CPU processing
├─ CUDA - Optimized for GPU acceleration
└─ Output: "📦 Build selected: CPU (optimized for CPU processing)"
```

### 📍 Phase Selection
```
Switches between execution phases
├─ Phase 13 - Quantum Circuit Optimization
├─ Phase 14 - Photonic Integration
├─ Phase 15 - AGI Synthesis
└─ Output: "📍 Phase selected: Phase 14 - Photonic Integration"
```

---

## Files Generated

| Button | File | Format |
|--------|------|--------|
| Any | qallow.log | Text log |
| Export Metrics | qallow_metrics_export.json | JSON |
| Save Config | qallow_phase_config.json | JSON |

---

## State Changes

### Start VM
- vm_running: false → true
- current_step: any → 0
- mind_started_at: None → now

### Stop VM
- vm_running: true → false
- Preserves: current_step, reward, energy, risk

### Pause
- vm_running: true → false
- Preserves all metrics

### Reset
- current_step: any → 0
- reward: any → 0.0
- energy: any → 0.0
- risk: any → 0.0
- telemetry: cleared
- terminal_output: cleared

### Build Selection
- selected_build: CPU ↔ CUDA

### Phase Selection
- selected_phase: Phase13 ↔ Phase14 ↔ Phase15

---

## Error Conditions

| Error | Cause | Solution |
|-------|-------|----------|
| VM already running | Start clicked twice | Stop VM first |
| VM not running | Stop clicked when stopped | Start VM first |
| Cannot reset while running | Reset clicked during execution | Stop VM first |
| Cannot change build while running | Build changed during execution | Stop VM first |
| Cannot change phase while running | Phase changed during execution | Stop VM first |

---

## Workflow Example

```
1. Select Build: CPU
   └─ Output: "📦 Build selected: CPU"

2. Select Phase: Phase 14
   └─ Output: "📍 Phase selected: Phase 14 - Photonic Integration"

3. Click Start VM
   └─ Output: "🚀 Starting Qallow VM with CPU build on Phase 14"

4. Wait for execution...

5. Click Export Metrics
   └─ Creates: qallow_metrics_export.json
   └─ Output: "✓ Metrics exported"

6. Click Save Config
   └─ Creates: qallow_phase_config.json
   └─ Output: "✓ Configuration saved"

7. Click View Logs
   └─ Output: Formatted audit log

8. Click Stop VM
   └─ Output: "⏹️ VM stopped gracefully (uptime: 45s)"

9. Click Reset
   └─ Output: "🔄 System reset (cleared 1250 steps)"
```

---

## Logging

All operations logged to:
- **Console**: Immediate feedback
- **qallow.log**: Persistent record
- **Audit Logs**: In-memory history

---

## Status

✅ All 9 buttons functional
✅ All operations working
✅ All files generated
✅ All logs recorded
✅ Production ready

---

**Last Updated**: 2025-10-25
**Version**: 1.0.0
**Status**: PRODUCTION READY

