# Qallow Native App - Complete Guide

## Overview

The Qallow Native App is a Rust/FLTK desktop application that serves as the unified engine for the Quantum-Photonic AGI System. It provides a complete interface for managing phases, builds, metrics, and system control.

## Running the Application

```bash
cd /root/Qallow/native_app
cargo run
```

The application will:
1. Load configuration from `qallow_config.json`
2. Load previous state from `qallow_state.json`
3. Initialize logging to `qallow.log`
4. Display the native desktop window

## Application Architecture

### Core Modules

- **main.rs** - Application entry point, event loop, and button integration
- **ui/mod.rs** - Main UI layout and tab management
- **button_handlers.rs** - Button event handlers connected to backend functionality
- **backend/process_manager.rs** - VM process lifecycle management
- **backend/metrics_collector.rs** - System and process metrics collection
- **codebase_manager.rs** - Codebase integration and management
- **models.rs** - Data structures for state, metrics, and logs
- **config.rs** - Configuration management
- **logging.rs** - File-based logging with rotation
- **shutdown.rs** - Graceful shutdown handling
- **error_recovery.rs** - Error tracking and recovery

### UI Tabs

1. **Dashboard** - Overview of system status and key metrics
2. **Metrics** - Real-time system and process metrics
3. **Terminal** - Live output from running processes
4. **Audit Log** - Complete audit trail of all operations
5. **Control Panel** - VM control and configuration
6. **Settings** - Application preferences and configuration
7. **Help** - Documentation and keyboard shortcuts

## Control Panel Features

### VM Control Buttons

#### ▶️ Start VM
- **Function**: Starts the Qallow VM with the selected build and phase
- **Prerequisites**: VM must not be running
- **Actions**:
  - Initializes process manager with current configuration
  - Sets `vm_running` flag to true
  - Records start time
  - Logs to terminal and audit log
  - Saves state automatically

#### ⏹️ Stop VM
- **Function**: Gracefully stops the running VM
- **Prerequisites**: VM must be running
- **Actions**:
  - Sends SIGTERM to process (graceful shutdown)
  - Waits up to 30 seconds for graceful termination
  - Falls back to SIGKILL if needed
  - Sets `vm_running` flag to false
  - Records stop event in audit log

#### ⏸️ Pause
- **Function**: Pauses VM execution
- **Prerequisites**: VM must be running
- **Actions**:
  - Pauses the running process
  - Maintains state for resume
  - Logs pause event

#### 🔄 Reset
- **Function**: Resets system state
- **Prerequisites**: VM must not be running
- **Actions**:
  - Clears all metrics and telemetry
  - Resets step counter and rewards
  - Clears terminal output
  - Logs reset event

### Build Selection

- **CPU** - Run on CPU (slower, no GPU required)
- **CUDA** - Run on NVIDIA GPU (faster, requires CUDA 12.0+)

### Phase Configuration

Configure the following parameters before starting:
- **Phase**: Select from Phase 13, 14, or 15
- **Ticks**: Number of iterations (default: 1000)
- **Fidelity**: Quantum fidelity level (default: 0.981)
- **Epsilon**: Error tolerance (default: 5e-6)

### Quick Actions

#### 📈 Export Metrics
- Exports current metrics to `qallow_metrics_export.json`
- Includes all system and process metrics
- Useful for analysis and reporting

#### 💾 Save Config
- Saves current phase configuration to `qallow_phase_config.json`
- Preserves settings for future runs
- Includes all phase parameters

#### 📋 View Logs
- Displays audit log in terminal
- Shows all operations with timestamps
- Includes component and severity information

## State Management

### Automatic State Persistence

The application automatically:
- Saves state to `qallow_state.json` on graceful shutdown
- Loads previous state on startup
- Maintains state during runtime in memory

### State Contents

- VM running status
- Selected build and phase
- Terminal output (last 1000 lines)
- Metrics (CPU, GPU, memory, coherence)
- Audit logs (last 1000 entries)
- Phase configuration
- Telemetry data
- Current step and rewards

## Logging

### Log Files

- **qallow.log** - Main application log
- **qallow_config.json** - Application configuration
- **qallow_state.json** - Application state (auto-saved)
- **qallow_phase_config.json** - Phase configuration (manual save)
- **qallow_metrics_export.json** - Exported metrics

### Log Levels

- **INFO** - General information
- **SUCCESS** - Successful operations
- **WARNING** - Warnings and non-critical issues
- **ERROR** - Errors and failures

## Keyboard Shortcuts

- **Ctrl+C** - Graceful shutdown
- **Ctrl+S** - Save configuration
- **Ctrl+E** - Export metrics
- **Ctrl+L** - View logs
- **Ctrl+Q** - Quit application

## Error Handling

The application includes comprehensive error handling:

- **Graceful Degradation** - Continues operation even if some features fail
- **Error Recovery** - Automatic retry with exponential backoff
- **Error Logging** - All errors logged with context
- **User Feedback** - Errors displayed in terminal and logs

## Codebase Integration

The native app integrates with the Qallow codebase:

- **Phases** - Automatically detects available phases
- **Builds** - Supports CPU and CUDA builds
- **Metrics** - Collects from `/proc` filesystem
- **Git Integration** - Tracks commits and branches
- **Build Management** - Can trigger builds from UI

## Performance

- **Memory Usage** - ~50-100 MB base
- **CPU Usage** - <5% idle, scales with workload
- **Startup Time** - <2 seconds
- **UI Responsiveness** - 60 FPS

## Troubleshooting

### VM Won't Start
1. Check if another instance is running
2. Verify build is available (CPU or CUDA)
3. Check logs for detailed error message
4. Try resetting system state

### High Memory Usage
1. Clear terminal output (reset)
2. Export and clear metrics
3. Reduce log retention
4. Restart application

### Metrics Not Updating
1. Verify VM is running
2. Check `/proc` filesystem access
3. Review logs for collection errors
4. Try manual metrics export

## Development

### Building from Source

```bash
cd /root/Qallow/native_app
cargo build --release
```

### Running Tests

```bash
cargo test
```

### Code Structure

```
native_app/src/
├── main.rs                 # Entry point
├── ui/                     # UI modules
│   ├── mod.rs             # Main UI layout
│   ├── dashboard.rs       # Dashboard tab
│   ├── metrics.rs         # Metrics tab
│   ├── terminal.rs        # Terminal tab
│   ├── audit_log.rs       # Audit log tab
│   ├── control_panel.rs   # Control panel tab
│   ├── settings.rs        # Settings tab
│   └── help.rs            # Help tab
├── backend/               # Backend modules
│   ├── mod.rs            # Backend exports
│   ├── process_manager.rs # VM process management
│   ├── metrics_collector.rs # Metrics collection
│   └── api_client.rs     # API client
├── button_handlers.rs     # Button event handlers
├── codebase_manager.rs    # Codebase integration
├── models.rs              # Data structures
├── config.rs              # Configuration
├── logging.rs             # Logging
├── shutdown.rs            # Shutdown handling
├── error_recovery.rs      # Error recovery
└── shortcuts.rs           # Keyboard shortcuts
```

## Future Enhancements

- [ ] Real-time metrics dashboard
- [ ] Advanced phase configuration UI
- [ ] Build output streaming
- [ ] Remote execution support
- [ ] Multi-instance management
- [ ] Performance profiling
- [ ] Advanced error recovery
- [ ] Custom theme support

