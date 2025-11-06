# Qallow Unified CLI - Implementation Summary

**Status:** ✅ COMPLETE

## What Was Done

The Qallow project now has a **unified, consolidated command-line interface** that eliminates separate phase commands and streamlines all operations into a single, cohesive structure.

---

## Old Structure (Pre-Consolidation)

```bash
# Separate commands for each operation
qallow build
qallow clear
qallow verify
qallow bench
qallow live
qallow accelerator
qallow phase11 --ticks=100
qallow phase12 --ticks=100
qallow phase13 --ticks=100
```

**Problems:**
- ❌ Too many top-level commands
- ❌ Inconsistent naming conventions
- ❌ No logical grouping
- ❌ Hard to discover related operations
- ❌ Difficult to maintain

---

## New Unified Structure

```bash
qallow <GROUP> [SUBCOMMAND] [OPTIONS]
```

### Command Groups

**1. `qallow run` - Execution**
```bash
qallow run                           # Execute unified VM
qallow run vm                        # Same as above
qallow run bench                     # Benchmark profile
qallow run live                      # Live ingestion profile
qallow run unified                   # Phases 12-15 pipeline
qallow run accelerator               # Phase 13 accelerator
qallow run entangle                  # Generate entanglement data
```

**2. `qallow system` - Build & Maintenance**
```bash
qallow system build                  # Compile CPU + CUDA
qallow system clear                  # Clean build artifacts
qallow system verify                 # Health checks
```

**3. `qallow phase` - Individual Phases**
```bash
qallow phase 11                      # Coherence bridge (quantum)
qallow phase 12                      # Elasticity simulation
qallow phase 13                      # Harmonic propagation
qallow phase 14                      # Coherence-lattice integration
qallow phase 15                      # Convergence & lock-in
qallow phase 16 ... 20               # Advanced phases
```

**4. `qallow mind` - Cognitive Pipeline**
```bash
qallow mind pipeline                 # Cognitive modules
qallow mind bench                    # Benchmarking suite
```

---

## Key Features

### 1. Unified Phase Runner
- Single entry point for all phases
- Auto-detection of phase number
- Consistent parameter passing

```bash
# Before (multiple separate commands)
qallow phase11 --ticks=400
qallow phase12 --ticks=100
qallow phase13 --ticks=200

# After (unified interface)
qallow phase 11 --ticks=400
qallow phase 12 --ticks=100
qallow phase 13 --ticks=200
```

### 2. Integrated Phase Pipeline
- Run multiple phases sequentially
- Unified parameter overrides
- Coordinated telemetry

```bash
# Run phases 12-15 with default settings
qallow run unified

# Run with custom phase parameters
qallow run vm --integrate \
  --integrate-phase12-ticks=150 \
  --integrate-phase13-ticks=200 \
  --integrate-phase13-k=0.003
```

### 3. Command Grouping
Four logical groups organize all functionality:
- `run` - All execution modes
- `system` - Build and maintenance
- `phase` - Individual phase runners
- `mind` - Cognitive pipelines

### 4. Help System
```bash
qallow help                          # Main help
qallow run help                      # Run group help
qallow system help                   # System group help
qallow phase help                    # Phase group help
qallow mind help                     # Mind group help
```

### 5. Backward Compatibility
Old commands still work with deprecation warnings:
```bash
qallow build          # → qallow system build ⚠️
qallow phase12        # → qallow phase 12 ⚠️
qallow bench          # → qallow run bench ⚠️
```

---

## Usage Examples

### Basic Execution
```bash
# Build project
qallow system build

# Run unified VM
qallow run

# Verify system health
qallow system verify
```

### Phase Execution
```bash
# Run individual phases
qallow phase 12 --ticks=100 --eps=0.0001
qallow phase 13 --nodes=16 --ticks=500 --k=0.002
qallow phase 14 --ticks=600 --target_fidelity=0.981

# Run integrated pipeline (phases 12-15)
qallow run unified
```

### Advanced Workflows
```bash
# With AgentLightning Runner auto-improvement
python3 recursive_improvement_engine.py
qallow run unified

# With quantum hardware integration
export QALLOW_QISKIT=1
qallow run vm --integrate phase11 --integrate-phase11-hardware

# With auto-auditing and export
qallow run vm --self-audit --export-pocket-map /tmp/pockets.json
```

---

## Implementation Details

### Files Modified
- `interface/launcher.c` - Command dispatcher with unified routing
- `interface/main.c` - Phase runner implementations (unchanged)

### Command Dispatcher Logic

**Unified command routing:**
1. Parse first argument as GROUP (`run`, `system`, `phase`, `mind`)
2. Parse second argument as SUBCOMMAND (if present)
3. Route to appropriate handler
4. Pass remaining arguments as options

**Phase routing:**
- Single `qallow phase <N>` dispatcher handles phases 11-20
- Auto-detects phase number and routes to runner
- Consistent argument passing

### Backward Compatibility

**Deprecation handling:**
- Old commands detected by dispatcher
- Print deprecation warning
- Route to new command path
- Full functionality preserved

```c
// Example: Legacy command routing
if (strcmp(command, "phase12") == 0) {
    printf("[INFO] `qallow phase12` is deprecated; use `qallow phase 12`.\n");
    return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase12", 
                                 qallow_phase12_runner);
}
```

---

## Benefits

✅ **Reduced Cognitive Load**
- Single entry point
- Logical grouping
- Easier to remember

✅ **Better Discoverability**
- `qallow help` shows all groups
- Group-specific help available
- Consistent naming

✅ **Maintenance & Extensibility**
- Central dispatcher makes adding commands easy
- Grouped functionality is easier to maintain
- Clear patterns for future expansion

✅ **Backward Compatible**
- Old commands still work
- Deprecation notices guide users to new syntax
- Smooth migration path

✅ **Unified Development Experience**
- AgentLightning Runner integration seamless
- All operations accessible through one interface
- Consistent parameter style

---

## Migration Guide

### For Users

**Update your scripts from:**
```bash
qallow phase12 --ticks=100
qallow build
qallow bench
```

**To:**
```bash
qallow phase 12 --ticks=100
qallow system build
qallow run bench
```

**Old commands still work!**
```bash
# These are still valid (with deprecation notices):
qallow phase12 --ticks=100  # ⚠️ Deprecated
qallow build               # ⚠️ Deprecated
qallow bench               # ⚠️ Deprecated

# But prefer the new syntax:
qallow phase 12 --ticks=100
qallow system build
qallow run bench
```

### For Developers

When adding new commands:

1. **Choose appropriate group:**
   - `run` - Execution workflows
   - `system` - Build/maintenance
   - `phase` - Quantum simulation phases
   - `mind` - Cognitive pipelines

2. **Add handler function:**
   ```c
   static int qallow_handle_<group>_group(int argc, char** argv, int arg_offset) {
       // Implementation
   }
   ```

3. **Register in main dispatcher:**
   ```c
   if (strcmp(command, "<group>") == 0) {
       return qallow_handle_<group>_group(argc, argv, arg_offset + 1);
   }
   ```

4. **Update help text:**
   ```c
   static void qallow_print_<group>_help(void) {
       // Help message
   }
   ```

---

## Documentation

Complete documentation available in: `UNIFIED_CLI.md`

**Topics covered:**
- Quick start guide
- All command groups with examples
- Phase-specific options
- Advanced features (acceleration, auditing, ML integration)
- Environment variables
- Troubleshooting
- Usage patterns (development, benchmarking, production)

---

## Quick Reference

```
qallow run      - Execute workflows (vm, bench, live, unified, accelerator)
qallow system   - Build and maintain (build, clear, verify)
qallow phase    - Run phases (11-20 with consistent interface)
qallow mind     - Cognitive pipeline (pipeline, bench)
qallow help     - Show help (global and group-specific)
```

**Most Common Commands:**
```bash
qallow system build                        # Build
qallow run                                 # Execute
qallow phase 13 --nodes=16 --ticks=500    # Run phase 13
qallow run unified                         # Run phases 12-15
```

---

## Verification

The new unified CLI has been implemented in `interface/launcher.c` with:

✅ Four command groups with logical subcommands
✅ Unified phase runner (phases 11-20)
✅ Integrated pipeline support
✅ Comprehensive help system
✅ Backward compatibility with deprecation warnings
✅ Clear command structure: `qallow <group> [subcommand] [options]`

All code is already in production in `launcher.c`. See `UNIFIED_CLI.md` for complete reference.
