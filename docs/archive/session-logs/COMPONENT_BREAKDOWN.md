# Qallow Components - Detailed Breakdown

## 1. Process Manager (`native_app/src/backend/process_manager.rs`)

### What It Does
Manages the lifecycle of subprocess execution for running quantum phases and system commands.

### Key Structures
```rust
pub struct ProcessManager {
    process: Option<Child>,           // Current subprocess
    output_tx: Sender<String>,        // Output channel sender
    output_rx: Receiver<String>,      // Output channel receiver
    metadata: Option<ProcessMetadata>, // Phase metadata
    retry_count: u32,                 // Retry counter
    max_retries: u32,                 // Max retry limit
    persistent_mode: bool,            // Auto-restart on exit
}
```

### Main Methods
- `start_vm()` - Start a specific phase
- `start_vm_unified()` - Start unified pipeline (phases 13, 14, 15)
- `stop_vm()` - Gracefully stop running process
- `is_running()` - Check if process is actually running (FIXED)
- `get_output()` - Retrieve process output
- `poll_exit()` - Check if process exited

### The Fix
**Before**: `is_running()` only checked if `self.process.is_some()`
**After**: Uses `try_wait()` to verify process actually finished
```rust
pub fn is_running(&mut self) -> bool {
    if let Some(ref mut child) = self.process {
        match child.try_wait() {
            Ok(Some(_)) => {
                self.process = None;
                self.metadata = None;
                false  // Process finished
            }
            Ok(None) => true,  // Still running
            Err(_) => {
                self.process = None;
                false
            }
        }
    } else {
        false
    }
}
```

---

## 2. Phase 11 - Cirq Quantum Bridge (`python/quantum/cirq_phase11.py`)

### What It Does
Implements quantum coherence bridge using Google Cirq for quantum circuit simulation.

### Key Functions
- `build_ansatz()` - Creates parameterized quantum circuits
- `run_phase11()` - Executes quantum simulation
- `main()` - CLI entry point

### Features
- **Simulators**: Ideal (perfect) and Noisy (realistic)
- **Qubits**: Configurable (default 4)
- **Ticks**: Simulation iterations (default 64)
- **Output**: CSV telemetry data

### Example Usage
```bash
# Ideal simulator
python3 cirq_phase11.py --ticks=64 --simulator=ideal

# Noisy simulator
python3 cirq_phase11.py --ticks=64 --simulator=noisy --qubits=8
```

### Output
- Quantum circuit diagrams
- Measurement histograms
- Fidelity metrics
- CSV logs in `data/logs/phase11.csv`

---

## 3. SDL GUI (`interface/qallow_ui.c`)

### What It Does
Provides a graphical interface for controlling Qallow using SDL2 and TTF fonts.

### Key Components
- **Window**: 1200x800 SDL2 window
- **Buttons**: 8 interactive buttons for different actions
- **Status Display**: Real-time status and help text
- **Font Rendering**: TrueType font for text

### Buttons
| Button | Key | Action |
|--------|-----|--------|
| Build CUDA | B | Builds CUDA pipeline |
| Run Binary | R | Runs Qallow binary |
| Run Accelerator | A | Runs Phase 13 accelerator |
| Phase 11 (Cirq) | 0 | Executes Phase 11 |
| Phase 14 | 1 | Executes Phase 14 |
| Phase 15 | 2 | Executes Phase 15 |
| Phase 16 | 3 | Executes Phase 16 |
| Stop | S | Stops current command |

### The Fix
**Font Path Issue**:
- **Before**: `/usr/share/fonts/TTF/DejaVuSans.ttf` (doesn't exist)
- **After**: `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` (correct)

When font fails to load, button text rendering is skipped, making buttons invisible.

---

## 4. Native Rust App (`native_app/src/main.rs`)

### What It Does
Provides a modern FLTK-based GUI for Qallow with real-time monitoring.

### Key Components
- **Main Window**: FLTK window with multiple panels
- **Control Panel**: Buttons for phase execution
- **Matrix View**: Visualization of quantum state matrices
- **Telemetry Panel**: Real-time metrics display
- **Terminal**: Command output display

### Architecture
```
MainWindow
├── wind (FLTK Window)
├── control_panel (ControlPanel)
│   └── buttons (PhaseButtons)
├── matrix_view (MatrixView)
│   └── table (FLTK Table)
├── telemetry_panel (TelemetryPanel)
└── terminal (Terminal)
```

### Process Flow
1. Initialize FLTK app
2. Create process manager
3. Setup button callbacks
4. Enter event loop
5. Handle UI messages
6. Update telemetry

---

## 5. Unified Binary (`build/qallow_unified_cuda`)

### What It Does
Main executable that orchestrates the entire Qallow system.

### Execution Flow
1. **Quantum Framework** (Python)
   - Runs 6 quantum algorithms
   - Generates quantum report
   - Exports results to JSON

2. **Qallow VM** (C/CUDA)
   - Initializes 1000 ticks
   - Monitors overlay stability
   - Tracks ethics metrics
   - Detects reality drift
   - Measures quantum coherence

3. **Output**
   - Real-time dashboard
   - Telemetry data
   - Ethics audit log
   - Performance metrics

### Command Structure
```bash
./qallow_unified_cuda <command> [options]

Commands:
  run       - Execute unified VM
  phase     - Run specific phase
  system    - Build/clean operations
  mind      - Cognitive pipeline
```

---

## 6. Quantum Algorithms

### Implemented (✅)
- **Hello Quantum**: Basic circuit with Hadamard and CNOT
- **Bell State**: Quantum entanglement demonstration
- **Deutsch**: Function classification (constant/balanced)
- **Grover's**: Quantum search algorithm
- **VQE**: Variational quantum eigensolver

### Needs Fix (❌)
- **Shor's**: Factoring algorithm (missing `gcd` import)

---

## 7. Ethics Monitoring

### Metrics Tracked
- **Safety (S)**: System safety score (0.98)
- **Clarity (C)**: Decision clarity (1.00)
- **Human (H)**: Human feedback integration (1.00)
- **Reality Drift**: Deviation from expected behavior (0.020)

### Formula
```
E = S + C + H - Δ
E = 0.98 + 1.00 + 1.00 - 0.020 = 2.96
```

### Status
- ✅ PASS: All metrics within acceptable ranges
- ✅ Drift Guard: OK (0.020 < 0.250 limit)

---

## 8. Overlay Stability

### Components
- **Orbital**: Quantum state orbital stability
- **River**: Data flow stability
- **Mycelial**: Network connectivity
- **Global**: Overall system stability

### Typical Values
- Orbital: 0.95-0.96
- River: 0.99+
- Mycelial: 0.99+
- Global: 0.98+

---

## 9. GPU Acceleration

### CUDA Features
- Photonic simulation kernels
- Quantum circuit acceleration
- Parallel processing
- Memory optimization

### Fallback
- CPU implementations available
- Automatic fallback if CUDA unavailable
- Same API for both modes

---

**Last Updated**: 2025-11-11

