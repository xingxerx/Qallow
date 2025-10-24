# 🚀 Qallow Native Desktop Application

A high-performance native desktop application for the Qallow Quantum-Photonic AGI System, built with **Rust** and **FLTK** for a modern, responsive user interface.

## Features

### 📊 Dashboard
- Real-time system metrics visualization
- Overlay stability monitoring (Orbital, River, Mycelial, Global)
- Ethics monitoring (Safety, Clarity, Human scores)
- Coherence tracking with decoherence measurements
- System component status indicators
- GPU acceleration information

### 📈 Metrics
- Real-time performance monitoring
- Phase status tracking
- GPU/CPU memory usage
- Network statistics
- Auto-refresh capability
- Detailed metrics table

### 💻 Terminal
- Live output streaming from Qallow VM
- Real-time error logging
- Timestamp tracking for each line
- Scrollable history
- Clear, Copy, Export buttons

### 🔍 Audit Log
- Comprehensive event logging
- Filterable by log level (INFO, SUCCESS, WARNING, ERROR)
- Component-based tracking
- Color-coded severity levels
- Search functionality

### ⚙️ Control Panel
- Start/Stop VM buttons
- Phase configuration (Phase 13, 14, 15)
- Parameter adjustment (Ticks, Fidelity, Epsilon)
- Build selection (CPU vs CUDA)
- Quick actions (Export, Save, Reset)
- System information display

## System Requirements

### Minimum
- **OS**: Linux (Ubuntu 20.04+), macOS 10.13+, or Windows 10+
- **Rust**: 1.70.0 or higher
- **RAM**: 4 GB
- **Disk Space**: 500 MB

### Recommended
- **OS**: Ubuntu 22.04 LTS or later
- **Rust**: 1.75.0 or higher
- **RAM**: 8 GB
- **Disk Space**: 2 GB
- **GPU**: NVIDIA with CUDA support

## Installation

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Clone and Build

```bash
cd /root/Qallow/native_app
cargo build --release
```

### 3. Run the Application

```bash
cargo run --release
```

Or run the compiled binary:

```bash
./target/release/qallow-native
```

## Usage

### Starting the Application

```bash
./target/release/qallow-native
```

The application will open with the Dashboard tab active.

### Running Qallow VM

1. Go to **Control Panel** tab
2. Select build type (CPU or CUDA)
3. Select phase (13, 14, or 15)
4. Adjust parameters if needed
5. Click **▶️ Start VM**
6. Monitor execution in **Terminal** tab

### Monitoring Execution

- **Dashboard**: View real-time metrics
- **Metrics**: Monitor performance details
- **Terminal**: Watch live output
- **Audit Log**: Review events
- **Control Panel**: Manage system

## Architecture

### Project Structure

```
native_app/
├── Cargo.toml              # Project configuration
├── src/
│   ├── main.rs             # Application entry point
│   ├── models.rs           # Data models
│   ├── utils.rs            # Utility functions
│   ├── ui/
│   │   ├── mod.rs          # UI module
│   │   ├── dashboard.rs    # Dashboard component
│   │   ├── terminal.rs     # Terminal component
│   │   ├── metrics.rs      # Metrics component
│   │   ├── audit_log.rs    # Audit log component
│   │   └── control_panel.rs # Control panel component
│   └── backend/
│       ├── mod.rs          # Backend module
│       ├── process_manager.rs  # Process management
│       ├── metrics_collector.rs # Metrics collection
│       └── api_client.rs    # API client
├── README.md               # This file
└── QUICKSTART.md           # Quick start guide
```

### Technology Stack

- **GUI Framework**: FLTK-rs (Fast Light Toolkit)
- **Async Runtime**: Tokio
- **Serialization**: Serde + serde_json
- **Process Management**: std::process
- **Threading**: crossbeam-channel
- **HTTP Client**: reqwest

## Building for Distribution

### Linux

```bash
cargo build --release
# Binary: target/release/qallow-native
```

### macOS

```bash
cargo build --release --target x86_64-apple-darwin
# Binary: target/x86_64-apple-darwin/release/qallow-native
```

### Windows

```bash
cargo build --release --target x86_64-pc-windows-msvc
# Binary: target/x86_64-pc-windows-msvc/release/qallow-native.exe
```

## Configuration

### Environment Variables

```bash
# Set custom Qallow binary path
export QALLOW_BIN_PATH=/custom/path/to/qallow

# Enable debug logging
export RUST_LOG=debug

# Set web dashboard URL
export QALLOW_API_URL=http://localhost:5000
```

### Configuration File

Create `~/.qallow/config.json`:

```json
{
  "qallow_bin_path": "/root/Qallow/build/qallow",
  "api_url": "http://localhost:5000",
  "auto_refresh_interval": 5000,
  "theme": "dark"
}
```

## Troubleshooting

### Application won't start

```bash
# Check Rust installation
rustc --version
cargo --version

# Check dependencies
cargo check

# Run with debug output
RUST_LOG=debug cargo run --release
```

### Qallow VM won't start

```bash
# Check if binary exists
ls -la /root/Qallow/build/qallow*

# Check permissions
chmod +x /root/Qallow/build/qallow*

# Test binary directly
/root/Qallow/build/qallow phase 14 --ticks=100
```

### Terminal output not showing

- Check if Qallow binary path is correct
- Verify process is running: `ps aux | grep qallow`
- Check system logs for errors

### Metrics not updating

- Ensure web dashboard is running: `cd /root/Qallow/ui && python3 dashboard.py`
- Check API URL in configuration
- Verify network connectivity

## Performance

- **Startup Time**: ~2-3 seconds
- **Memory Usage**: ~50-100 MB
- **CPU Usage**: <5% idle
- **Responsiveness**: Real-time updates

## Development

### Running Tests

```bash
cargo test
```

### Building Documentation

```bash
cargo doc --open
```

### Code Formatting

```bash
cargo fmt
```

### Linting

```bash
cargo clippy
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:

1. Check QUICKSTART.md
2. Review troubleshooting section
3. Check Qallow documentation
4. Open an issue on GitHub

---

**Version**: 1.0.0  
**Last Updated**: 2025-10-23  
**Status**: Production Ready ✅

